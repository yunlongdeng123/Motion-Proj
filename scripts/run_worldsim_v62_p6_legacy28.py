#!/usr/bin/env python3
"""Run the artifact-bounded V6.2 CPSC-Lite legacy28 mechanism benchmark."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from motion_proj.worldsim_v61.me1_oracle import (
    _camera_contract,
    _independent_eval_factors,
    _load_grid,
    _raycast,
    method_decision,
    occupancy_gate,
)
from motion_proj.worldsim_v61.me3_predicted import (
    predicted_method_factors,
    raycast_predicted_conservative,
    resample_irwm_classes,
)
from motion_proj.worldsim_v61.occupancy import OCCUPIED, VoxelGridSpec
from motion_proj.worldsim_v62.cpsc_lite import CPSCLite
from motion_proj.worldsim_v62.legacy_bridge import (
    fit_query_weighted_class_prototypes,
    infer_legacy_tristate_grids,
)


TASK_ID = "WS-V62-P6-LEGACY28-ME-01"
PRIMARY_ARM = "B5-CPSC-LITE-PRECONFORMAL"
RUNNABLE_ARMS = (
    "B0-IRWM-ARGMAX",
    "B1-HARD-CLIP",
    "B3-EVIDENTIAL-NO-PROJECTION",
    PRIMARY_ARM,
)


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(_safe(row), sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _arm_summary(
    rows: list[dict[str, Any]], arm: str, r10_accepts: set[str]
) -> dict[str, Any]:
    selected = [row for row in rows if row["arm"] == arm]
    accepted = [row for row in selected if row["decision"] == "ACCEPT"]
    conflicts = [
        float(row["independent_eval_geometry_factors"]["free_space_conflict"])
        for row in accepted
    ]
    total_mask = sum(int(row["mask_pixel_count"]) for row in selected)
    accepted_ids = {row["case_id"] for row in accepted}
    new_rows = [row for row in accepted if row["case_id"] not in r10_accepts]
    return {
        "arm": arm,
        "denominator": len(selected),
        "accept_count": len(accepted),
        "abstain_count": sum(row["decision"] == "ABSTAIN" for row in selected),
        "reject_count": sum(row["decision"] == "REJECT" for row in selected),
        "false_safe_count": sum(bool(row["false_safe"]) for row in accepted),
        "accepted_case_ids": sorted(accepted_ids),
        "accepted_mask_area_yield": sum(
            int(row["mask_pixel_count"]) for row in accepted
        )
        / max(1, total_mask),
        "accepted_free_conflict_mean": float(np.mean(conflicts)) if conflicts else 1.0,
        "accepted_free_conflict_worst": max(conflicts) if conflicts else 1.0,
        "r10_retained_count": len(accepted_ids & r10_accepts),
        "new_actor_accept_count": sum(
            row["hole_type"] == "actor_removal_hole" for row in new_rows
        ),
        "new_static_or_disocclusion_accept_count": sum(
            row["hole_type"] != "actor_removal_hole" for row in new_rows
        ),
    }


def run(config_path: Path, repo_root: Path, run_root: Path) -> Path:
    started = time.monotonic()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["task_id"] != TASK_ID:
        raise ValueError("P6 task identity drift")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__legacy28-s0-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    o_eval_read = False
    try:
        sources = {name: Path(value) for name, value in config["sources"].items()}
        cases = _read_jsonl(sources["r9"] / "CASES.jsonl")
        if len(cases) != 28:
            raise ValueError("legacy denominator is not 28")
        r9_rows = {
            row["case_id"]: row
            for row in _read_jsonl(
                sources["r9"] / "verifier_worker/PER_CASE_ARMS.jsonl"
            )
        }
        r10_rows = {
            row["case_id"]: row
            for row in _read_jsonl(sources["r10"] / "FACTORIZED_DECISIONS.jsonl")
        }
        me1_rows = {
            row["case_id"]: row
            for row in _read_jsonl(sources["me1"] / "PER_CASE.jsonl")
        }
        b0_rows = {
            row["case_id"]: row
            for row in _read_jsonl(sources["me3r"] / "METHOD_DECISIONS.jsonl")
        }
        case_ids = {row["case_id"] for row in cases}
        if not all(case_ids == set(rows) for rows in (r9_rows, r10_rows, me1_rows, b0_rows)):
            raise ValueError("legacy case identities are not aligned")

        p5_config = yaml.safe_load(
            Path(config["p5"]["config"]).read_text(encoding="utf-8")
        )
        prototypes = fit_query_weighted_class_prototypes(
            Path(p5_config["inputs"]["p2_run"]),
            Path(p5_config["inputs"]["p4_run"]),
            [str(value) for value in p5_config["split"]["train_scenes"]],
            [int(value) for value in p5_config["inputs"]["target_frames"]],
        )
        np.savez_compressed(
            run_dir / "CLASS_PROTOTYPES.npz",
            class_query_count=prototypes["counts"],
            mean_prior_logits=prototypes["logits"],
            mean_bev_features=prototypes["bev"],
        )
        device = torch.device(f"cuda:{int(config['resources']['gpu'])}")
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        checkpoint = torch.load(
            Path(config["p5"]["best_model"]),
            map_location=device,
            weights_only=True,
        )
        model = CPSCLite(
            int(checkpoint["prior_feature_dimension"]),
            int(checkpoint["query_feature_dimension"]),
            hidden_width=int(p5_config["model"]["hidden_width"]),
            decoder_layers=int(p5_config["model"]["query_decoder_layers"]),
            residual_blocks=int(p5_config["model"]["residual_blocks"]),
            projection_iterations=int(p5_config["model"]["projection_iterations"]),
            dropout=float(p5_config["model"]["dropout"]),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        spec = VoxelGridSpec(
            frame="target_lidar",
            origin_m=tuple(float(value) for value in config["target_grid"]["origin_m"]),
            voxel_size_m=float(config["target_grid"]["voxel_size_m"]),
            shape=tuple(int(value) for value in config["target_grid"]["shape"]),
        )
        method_rays: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}
        state_grids: dict[tuple[str, str, int], np.ndarray] = {}
        diagnostic_rows: list[dict[str, Any]] = []
        for scene in config["cohort"]["scenes"]:
            scene_root = Path(config["raw_evidence"][scene])
            for frame in config["cohort"]["target_frames"]:
                key = (scene, int(frame))
                with np.load(
                    sources["me3r"]
                    / f"predictions/{scene}/f{int(frame):03d}/IRWM_CLASS.npz",
                    allow_pickle=False,
                ) as source:
                    adapted = resample_irwm_classes(
                        np.asarray(source["class_label"], dtype=np.uint8), spec
                    )
                method_grid = _load_grid(
                    sources["me0"]
                    / f"evidence/{scene}/f{int(frame):03d}/O_method.npz"
                )
                grids, diagnostics = infer_legacy_tristate_grids(
                    model,
                    predicted_class=np.asarray(adapted["predicted_class"]),
                    source_valid=np.asarray(adapted["source_valid"]),
                    method_semantics=np.asarray(method_grid["semantics"]),
                    actor_grid=np.asarray(method_grid["actor_grid"]),
                    prototypes=prototypes,
                    batch_size=int(config["resources"]["query_batch_size"]),
                    device=device,
                )
                diagnostics.update({"scene": scene, "frame": int(frame)})
                diagnostic_rows.append(diagnostics)
                t_lidar_camera, intrinsics = _camera_contract(
                    scene_root, int(frame), config["projection"]
                )
                for arm, semantics in grids.items():
                    actor_grid = np.where(
                        (semantics == OCCUPIED)
                        & (np.asarray(method_grid["actor_grid"]) >= 0),
                        np.asarray(method_grid["actor_grid"]),
                        -1,
                    ).astype(np.int32)
                    relative = Path(f"states/{arm}/{scene}/f{int(frame):03d}.npz")
                    path = run_dir / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        path,
                        semantics=semantics,
                        actor_grid=actor_grid,
                        source_valid=np.asarray(adapted["source_valid"]),
                        grid_origin_m=spec.origin,
                        voxel_size_m=np.asarray(spec.voxel_size_m),
                    )
                    state_grids[(arm, *key)] = semantics
                    method_rays[(arm, *key)] = raycast_predicted_conservative(
                        {
                            "semantics": semantics,
                            "actor_grid": actor_grid,
                            "origin": spec.origin,
                            "voxel_size": spec.voxel_size_m,
                        },
                        t_lidar_camera,
                        intrinsics,
                        config["projection"],
                    )
                with np.load(
                    sources["me3r"]
                    / f"projections/{scene}/f{int(frame):03d}/PREDICTED.npz",
                    allow_pickle=False,
                ) as projection:
                    method_rays[("B0-IRWM-ARGMAX", *key)] = {
                        name: np.asarray(projection[name]) for name in projection.files
                    }
        _write_jsonl(run_dir / "GRID_DIAGNOSTICS.jsonl", diagnostic_rows)

        method_rows: list[dict[str, Any]] = []
        candidate_root = run_dir / "candidates"
        for case in cases:
            case_id = case["case_id"]
            key = (case["scene"], int(case["frame_index"]))
            with np.load(
                sources["r9"] / f"verifier_inputs/{case_id}.npz",
                allow_pickle=False,
            ) as payload:
                mask = np.asarray(payload["mask"], dtype=bool)
            photo = r9_rows[case_id]["P1"]["decision"]
            for arm in RUNNABLE_ARMS:
                if arm == "B0-IRWM-ARGMAX":
                    source_row = b0_rows[case_id]
                    factors = source_row["predicted_geometry_factors"]
                    with np.load(
                        sources["me3r"] / source_row["compiled_proposal_path"],
                        allow_pickle=False,
                    ) as compiled:
                        candidates = np.asarray(
                            compiled["occupied_voxel_linear"], dtype=np.int64
                        )
                        actor_ids = np.asarray(
                            compiled["actor_instance_ids"], dtype=np.int32
                        ).tolist()
                    decision = source_row["decisions"]["P1-IRWM-PREDICTED"]
                else:
                    factors, candidates, actor_ids = predicted_method_factors(
                        mask,
                        method_rays[(arm, *key)],
                        case,
                        float(config["method_gate"]["minimum_surface_coverage"]),
                    )
                    decision = method_decision(photo, bool(factors["passed"]))
                relative = Path(f"candidates/{arm}/{case_id}.npz")
                path = run_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    path,
                    occupied_voxel_linear=np.asarray(candidates, dtype=np.int64),
                    actor_instance_ids=np.asarray(actor_ids, dtype=np.int32),
                )
                method_rows.append(
                    {
                        "arm": arm,
                        "case_id": case_id,
                        "scene": case["scene"],
                        "frame_index": int(case["frame_index"]),
                        "frontend": case["frontend"],
                        "hole_type": case["hole_type"],
                        "mask_pixel_count": int(mask.sum()),
                        "photo_decision": photo,
                        "method_geometry_factors": factors,
                        "candidate_path": str(relative),
                        "decision": decision,
                    }
                )
        _write_jsonl(run_dir / "METHOD_DECISIONS.jsonl", method_rows)

        # Hidden evaluator is loaded only after all arm decisions and candidates exist.
        o_eval_read = True
        eval_grids: dict[tuple[str, int], dict[str, Any]] = {}
        eval_rays: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        for scene in config["cohort"]["scenes"]:
            scene_root = Path(config["raw_evidence"][scene])
            for frame in config["cohort"]["target_frames"]:
                key = (scene, int(frame))
                grid = _load_grid(
                    sources["me0"]
                    / f"evidence/{scene}/f{int(frame):03d}/O_eval.npz"
                )
                t_lidar_camera, intrinsics = _camera_contract(
                    scene_root, int(frame), config["projection"]
                )
                eval_grids[key] = grid
                eval_rays[key] = _raycast(
                    grid, t_lidar_camera, intrinsics, config["projection"]
                )

        final_rows: list[dict[str, Any]] = []
        for row in method_rows:
            key = (row["scene"], int(row["frame_index"]))
            with np.load(run_dir / row["candidate_path"], allow_pickle=False) as payload:
                candidates = np.asarray(
                    payload["occupied_voxel_linear"], dtype=np.int64
                )
            with np.load(
                sources["r9"] / f"verifier_inputs/{row['case_id']}.npz",
                allow_pickle=False,
            ) as payload:
                mask = np.asarray(payload["mask"], dtype=bool)
            factors = _independent_eval_factors(
                mask,
                candidates,
                method_rays[(row["arm"], *key)],
                eval_grids[key],
                eval_rays[key],
                int(config["projection"]["observed_support_dilation_voxels"]),
            )
            factors["passed"] = occupancy_gate(
                factors, config["independent_eval_gate"]
            )
            final_rows.append(
                {
                    **row,
                    "independent_eval_geometry_factors": factors,
                    "false_safe": row["decision"] == "ACCEPT"
                    and not bool(factors["passed"]),
                    "method_decisions_written_before_eval": True,
                }
            )
        _write_jsonl(run_dir / "PER_CASE.jsonl", final_rows)

        r10_accepts = {
            case_id
            for case_id, row in r10_rows.items()
            if row["overall_decision"] == "ACCEPT"
        }
        arm_summaries = [
            _arm_summary(final_rows, arm, r10_accepts) for arm in RUNNABLE_ARMS
        ]
        summary_by_arm = {row["arm"]: row for row in arm_summaries}
        primary = summary_by_arm[PRIMARY_ARM]
        b1 = summary_by_arm["B1-HARD-CLIP"]

        oracle_surface_count = 0
        retained_surface_count = 0
        for case_id, row in me1_rows.items():
            if row["decisions"]["O2-OCC-GEOMETRY"] != "ACCEPT":
                continue
            with np.load(
                sources["me1"] / row["compiled_proposal_path"], allow_pickle=False
            ) as proposal:
                linear = np.asarray(proposal["occupied_voxel_linear"], dtype=np.int64)
            state = state_grids[
                (PRIMARY_ARM, row["scene"], int(row["frame_index"]))
            ].reshape(-1)
            oracle_surface_count += int(linear.size)
            retained_surface_count += int(np.count_nonzero(state[linear] == OCCUPIED))
        safe_occ_retention = retained_surface_count / max(1, oracle_surface_count)
        valid_count = sum(int(row["source_valid_count"]) for row in diagnostic_rows)
        source_valid_unknown_count = sum(
            float(row["source_valid_unknown_fraction"][PRIMARY_ARM])
            * int(row["source_valid_count"])
            for row in diagnostic_rows
        )
        source_valid_unknown_fraction = source_valid_unknown_count / max(1, valid_count)
        hard_count = sum(int(row["hard_constraint_count"]) for row in diagnostic_rows)
        hard_violations = sum(
            int(row["b5_hard_violation_count"]) for row in diagnostic_rows
        )
        thresholds = config["primary_gate"]
        checks = {
            "accept_at_least_5_of_28": primary["accept_count"]
            >= int(thresholds["minimum_accept_count"]),
            "false_safe_zero": primary["false_safe_count"]
            <= int(thresholds["maximum_false_safe_count"]),
            "r10_three_of_three_retained": primary["r10_retained_count"]
            == len(r10_accepts)
            == 3,
            "at_least_one_new_actor": primary["new_actor_accept_count"] >= 1,
            "at_least_one_new_static_or_disocclusion": primary[
                "new_static_or_disocclusion_accept_count"
            ]
            >= 1,
            "accepted_mask_area_at_least_0_12": primary["accepted_mask_area_yield"]
            >= float(thresholds["minimum_accepted_mask_area_yield"]),
            "accepted_free_conflict_mean_at_most_0_05": primary[
                "accepted_free_conflict_mean"
            ]
            <= float(thresholds["maximum_accepted_free_conflict"]),
            "accepted_free_conflict_worst_at_most_0_05": primary[
                "accepted_free_conflict_worst"
            ]
            <= float(thresholds["maximum_accepted_free_conflict"]),
            "safe_occ_retention_at_least_half": safe_occ_retention
            >= float(thresholds["minimum_safe_occ_retention"]),
            "source_valid_unknown_below_frozen_upper": source_valid_unknown_fraction
            <= float(thresholds["maximum_source_valid_unknown_fraction"]),
            "hard_projection_violations_zero": hard_violations == 0,
        }
        primary_passed = all(checks.values())
        b1_stop_triggered = (
            b1["accept_count"] >= int(thresholds["minimum_accept_count"])
            and b1["false_safe_count"] == 0
        )
        if b1_stop_triggered:
            decision = "projection_only_control_sufficient_pivot_from_learned_claim"
        elif primary_passed:
            decision = "cpsc_lite_preconformal_passed_enter_fresh_development"
        else:
            decision = "cpsc_lite_family_rejected_mechanism_recovery_only"
        gate = {
            "task_id": TASK_ID,
            "primary_arm": PRIMARY_ARM,
            "checks": checks,
            "primary_passed": primary_passed,
            "projection_only_stop_triggered": b1_stop_triggered,
            "decision": decision,
        }
        _write_json(run_dir / "P6_GATE.json", gate)
        _write_jsonl(run_dir / "ARM_SUMMARIES.jsonl", arm_summaries)
        metrics = {
            "task_id": TASK_ID,
            "arm_summaries": arm_summaries,
            "safe_occ_retention": safe_occ_retention,
            "oracle_surface_voxel_count": oracle_surface_count,
            "retained_surface_voxel_count": retained_surface_count,
            "source_valid_unknown_fraction": source_valid_unknown_fraction,
            "hard_constraint_count": hard_count,
            "hard_violation_count": hard_violations,
            "arm_availability": config["arm_availability"],
        }
        _write_json(run_dir / "METRICS.json", metrics)
        elapsed = time.monotonic() - started
        peak_gpu = torch.cuda.max_memory_allocated(device) / 1024**3
        output_bytes = sum(
            path.stat().st_size for path in run_dir.rglob("*") if path.is_file()
        )
        resource = {
            "wall_seconds": elapsed,
            "peak_gpu_memory_gib": peak_gpu,
            "output_bytes_before_closeout": output_bytes,
            "irwm_inference_started": False,
            "training_started": False,
            "o_eval_read_after_method_freeze": o_eval_read,
            "confirmation_read": False,
            "exact_once_test_read": False,
            "disk_free_gib": shutil.disk_usage(run_root).free / 1024**3,
        }
        _write_json(run_dir / "RESOURCE_AUDIT.json", resource)
        summary = {
            "task_id": TASK_ID,
            "source_commit": source_commit,
            "status": "done" if primary_passed or b1_stop_triggered else "rejected",
            "decision": decision,
            "primary_arm": PRIMARY_ARM,
            "primary_accept_count": primary["accept_count"],
            "primary_false_safe_count": primary["false_safe_count"],
            "primary_accepted_mask_area_yield": primary["accepted_mask_area_yield"],
            "b1_accept_count": b1["accept_count"],
            "b1_false_safe_count": b1["false_safe_count"],
            "safe_occ_retention": safe_occ_retention,
            "source_valid_unknown_fraction": source_valid_unknown_fraction,
            "failure_ledger_delta": "none_if_pass_else_required",
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        _write_json(
            run_dir / "MANIFEST.json",
            {
                "task_id": TASK_ID,
                "source_commit": source_commit,
                "config_path": str(config_path),
                "identity_policy": config["identity_policy"],
                "sources": {name: str(path) for name, path in sources.items()},
                "artifacts": [
                    "CLASS_PROTOTYPES.npz",
                    "GRID_DIAGNOSTICS.jsonl",
                    "METHOD_DECISIONS.jsonl",
                    "PER_CASE.jsonl",
                    "ARM_SUMMARIES.jsonl",
                    "P6_GATE.json",
                    "METRICS.json",
                    "RESOURCE_AUDIT.json",
                    "SUMMARY.json",
                ],
                "hash_checksum_fingerprint": False,
            },
        )
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "task_id": TASK_ID,
                "status": summary["status"],
                "canonical": True,
                "run_uri": f"run://worldsim_v62/{TASK_ID}/{run_dir.name}",
            },
        )
        return run_dir
    except Exception as error:
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "task_id": TASK_ID,
                "status": "failed",
                "canonical": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "o_eval_read": o_eval_read,
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v62")
    )
    args = parser.parse_args()
    run_dir = run(args.config, args.repo_root.resolve(), args.run_root)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
