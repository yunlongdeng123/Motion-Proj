#!/usr/bin/env python3
"""在逐 actor G0 上隔离 Gaussian sampling 与 alpha mixing 机制。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import cv2
import numpy as np
from scipy.ndimage import binary_dilation
import torch
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v32.inpainting_adapter import completion_points_from_view
from motion_proj.worldsim_v33.spatial_delta import atomic_save_npz
from motion_proj.worldsim_v4.repair_assets import temporary_repair_composition
from motion_proj.worldsim_v4.repair_builders import completion_points_to_repair_asset
from motion_proj.worldsim_v5.geometry_repair import (
    GeometryRepairError,
    depth_error_summary,
    fit_inverse_depth_surface,
)
from scripts.run_worldsim_v5_m2_actor_geometry import _load_requests, _mask
from scripts.run_worldsim_v5_m2_staged_geometry_diagnostic import (
    build_all_rigid_erase_delta,
)
from scripts.run_worldsim_v5_m2_surface_ablation import _runtime
from scripts.worldsim_v5_forensics_common import (
    atomic_json,
    copy_source_snapshot,
    inventory_files,
    prepare_formal_run,
    sha256_file,
    utc_now,
    verify_file,
    write_events,
    write_resolved_config,
)


TASK_ID = "WS-V5-M2-GEOMETRY-FIRST-REPAIR-01"
SCHEMA_VERSION = "worldsim_v5_m2_gaussianization_forensics_v1"
ARM_IDS = ("BASE", "OPAQUE", "DENSE", "DENSE_OPAQUE")


class M2GaussianizationError(RuntimeError):
    """Gaussianization forensic 输入、回放或正式运行合同失败。"""


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise M2GaussianizationError("Gaussianization config schema 漂移")
    arms = tuple(arm["id"] for arm in payload["gaussianization_arms"])
    if (
        payload.get("task_id") != TASK_ID
        or payload.get("status") != "running"
        or payload.get("phase") != "per_actor_g0_gaussianization_factor_forensics"
        or arms != ARM_IDS
        or payload["request_protocol"]["unit"] != "one_actor_one_view_one_hole"
        or payload["surface"]["model"] != "G0_ROBUST_PLANE"
    ):
        raise M2GaussianizationError("Gaussianization task/arms/request 漂移")
    for name in (
        "validation_quality_read",
        "heldout_quality_read",
        "test_quality_read",
        "parameter_search_performed",
        "method_arm_selection_performed",
        "router_refit_performed",
    ):
        if payload["scope"].get(name) is not False:
            raise M2GaussianizationError(f"Gaussianization restriction 漂移: {name}")
    if payload["reference"]["independent_geometry_claim_allowed"] is not False:
        raise M2GaussianizationError("model proxy 不得声明独立 GT")
    return payload


def effective_opacity_gain(
    base_opacity: np.ndarray, composed_opacity: np.ndarray
) -> np.ndarray:
    base = np.asarray(base_opacity, dtype=np.float64)
    composed = np.asarray(composed_opacity, dtype=np.float64)
    if base.shape != composed.shape:
        raise M2GaussianizationError("opacity shape 漂移")
    denominator = np.maximum(1.0 - np.clip(base, 0.0, 1.0), 1e-6)
    gain = (composed - base) / denominator
    return np.clip(gain, 0.0, 1.0)


def mechanism_decision(
    rows: list[Mapping[str, Any]], gate: Mapping[str, Any]
) -> dict[str, Any]:
    done = [row for row in rows if row.get("status") == "done"]
    candidates: dict[str, Any] = {}
    supported: list[str] = []
    for arm in ARM_IDS[1:]:
        deltas = [
            float(row["arms"][arm]["post_geometry_error"]["mae_m"])
            - float(row["arms"]["BASE"]["post_geometry_error"]["mae_m"])
            for row in done
        ]
        improved = sum(
            value <= -float(gate["minimum_post_mae_improvement_m"])
            for value in deltas
        )
        mean_delta = float(np.mean(deltas)) if deltas else math.nan
        median_delta = float(np.median(deltas)) if deltas else math.nan
        passed = (
            len(done) >= int(gate["minimum_evaluable_request_count"])
            and improved >= int(gate["minimum_improvement_request_count"])
            and mean_delta < float(gate["require_mean_delta_below_m"])
            and median_delta < float(gate["require_median_delta_below_m"])
        )
        if passed:
            supported.append(arm)
        candidates[arm] = {
            "improvement_request_count": improved,
            "mean_candidate_minus_base_post_mae_m": mean_delta,
            "median_candidate_minus_base_post_mae_m": median_delta,
            "minimum_delta_m": float(np.min(deltas)) if deltas else math.nan,
            "maximum_delta_m": float(np.max(deltas)) if deltas else math.nan,
            "mechanism_gate_passed": passed,
        }
    if not supported:
        conclusion = "frozen_density_opacity_arms_do_not_explain_gaussianization_failure"
    elif supported == ["OPAQUE"]:
        conclusion = "alpha_background_mixing_is_primary_gaussianization_mechanism"
    elif supported == ["DENSE"]:
        conclusion = "sampling_density_is_primary_gaussianization_mechanism"
    elif supported == ["DENSE_OPAQUE"]:
        conclusion = "density_opacity_interaction_is_primary_gaussianization_mechanism"
    else:
        conclusion = "multiple_gaussianization_factors_have_broad_mechanism_support"
    return {
        "conclusion": conclusion,
        "evaluable_request_count": len(done),
        "baseline_replay_exact_count": sum(
            row.get("baseline_replay_exact") is True for row in done
        ),
        "supported_diagnostic_arms": supported,
        "candidate_deltas": candidates,
        "method_arm_selected": False,
        "reference_scope": "model_derived_proxy_not_independent_ground_truth",
    }


def _terminal(
    run_dir: Path,
    *,
    status: str,
    source_head: str,
    summary_sha256: str | None,
    manifest_sha256: str | None,
    reason: str | None,
) -> None:
    atomic_json(
        run_dir / "status.json",
        {
            "schema_version": "worldsim_v5_m2_gaussianization_status_v1",
            "task_id": TASK_ID,
            "task_status": "running",
            "status": status,
            "source_commit": source_head,
            "summary_sha256": summary_sha256,
            "manifest_sha256": manifest_sha256,
            "reason": reason,
            "finished_at_utc": utc_now(),
        },
    )


def _prior_rows(config: Mapping[str, Any]) -> tuple[Path, dict[tuple[int, int, int], dict[str, Any]]]:
    path = Path(config["inputs"]["g0_actor_diagnostics"]["path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = {
        (int(row["frame"]), int(row["camera_id"]), int(row["actor_id"])): row
        for row in payload["rows"]
    }
    if len(rows) != int(config["request_protocol"]["expected_request_count"]):
        raise M2GaussianizationError("r005 request denominator 漂移")
    return path.parent.parent, rows


def _arm_state(
    *,
    arm: Mapping[str, Any],
    fit: Any,
    inpaint: np.ndarray,
    target: np.ndarray,
    base: Mapping[str, Any],
    trainer: Any,
    dataset: Any,
    device: torch.device,
    frame: int,
    camera: int,
    actor_id: int,
    erase_delta: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    from scripts.run_worldsim_v32_s2_3dgic import render_snapshot

    points = completion_points_from_view(
        rgb=inpaint,
        depth=fit.depth,
        mask=fit.valid,
        observed_cross_view=np.zeros_like(target),
        intrinsics=np.asarray(base["intrinsics"]),
        camera_to_world=np.asarray(base["camera_to_world"]),
        stride=int(arm["stride"]),
        scale_multiplier=float(arm["scale_multiplier"]),
        minimum_scale_m=float(config["gaussianization_limits"]["minimum_scale_m"]),
        maximum_scale_m=float(config["gaussianization_limits"]["maximum_scale_m"]),
    )
    asset = completion_points_to_repair_asset(
        points,
        candidate_id=(
            f"scene0471_f{frame:03d}_c{camera}_g0_robust_plane_a{actor_id:03d}"
            f"{'' if arm['id'] == 'BASE' else '_' + str(arm['id']).lower()}"
        ),
        method="DONOR",
        provenance="native_scene_donor",
        features_rest_shape=tuple(trainer.models["Background"]._features_rest.shape[1:]),
        opacity=float(arm["opacity"]),
        target_frame=frame,
        target_camera_id=camera,
    )
    started = time.perf_counter()
    with temporary_repair_composition(
        trainer.models, erase_delta=erase_delta, asset=asset
    ) as composition:
        rendered = render_snapshot(
            trainer=trainer,
            dataset=dataset,
            frame=frame,
            camera_id=camera,
            device=device,
        )
    sampled = np.zeros(target.shape, dtype=bool)
    pixels = np.asarray(points.source_pixels_xy, dtype=np.int64)
    sampled[pixels[:, 1], pixels[:, 0]] = True
    return {
        "post": np.asarray(rendered["depth"], dtype=np.float32),
        "background_depth": np.asarray(rendered["background_depth"], dtype=np.float32),
        "background_opacity": np.asarray(rendered["background_opacity"], dtype=np.float32),
        "sampled": sampled,
        "gaussian_count": int(points.means.shape[0]),
        "scale_min_m": float(np.min(points.scales)),
        "scale_mean_m": float(np.mean(points.scales)),
        "scale_max_m": float(np.max(points.scales)),
        "compute_seconds": time.perf_counter() - started,
        "composition_audit": composition,
    }


def run(config_path: Path, run_dir: Path, device_name: str) -> dict[str, Any]:
    config = load_config(config_path)
    source_head = prepare_formal_run(run_dir, TASK_ID, PROJECT)
    resolved = write_resolved_config(run_dir, config)
    events: list[dict[str, Any]] = [
        {"event": "run_started", "at_utc": utc_now(), "source_commit": source_head}
    ]
    write_events(run_dir, events)
    try:
        inputs = {
            name: verify_file(binding["path"], binding["sha256"])
            for name, binding in config["inputs"].items()
        }
        g0_summary = json.loads(Path(inputs["g0_actor_summary"]["path"]).read_text())
        g3_summary = json.loads(Path(inputs["g3_surface_summary"]["path"]).read_text())
        if (
            g0_summary.get("status") != "done"
            or g0_summary.get("conclusion")
            != config["unlock_binding"]["g0_source_conclusion"]
            or g0_summary.get("mechanism", {}).get("gaussianization_primary") is not True
            or g3_summary.get("status") != "done"
            or g3_summary.get("conclusion")
            != config["unlock_binding"]["surface_sequence_terminal_conclusion"]
            or g3_summary.get("selection", {}).get("gate_passed") is not False
        ):
            raise M2GaussianizationError("r005/r009 unlock binding 漂移")
        mask_root, requests = _load_requests(config)
        protocol = config["request_protocol"]
        if len(requests) != int(protocol["expected_request_count"]):
            raise M2GaussianizationError("actor request denominator 漂移")
        accepted_count = sum(request.get("accepted") is True for request in requests)
        rejected_count = sum(request.get("accepted") is not True for request in requests)
        if (
            accepted_count != int(protocol["expected_accepted_mask_count"])
            or rejected_count != int(protocol["expected_rejected_mask_count"])
        ):
            raise M2GaussianizationError("actor accepted/rejected denominator 漂移")
        prior_root, prior_rows = _prior_rows(config)
        checkpoint = Path(inputs["formal_checkpoint"]["path"])
        checkpoint_before = sha256_file(checkpoint)
        if not torch.cuda.is_available():
            raise M2GaussianizationError("Gaussianization forensic 需要 CUDA")
        device = torch.device(device_name)
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        _, dataset, trainer = _runtime(config, device)
        from scripts.run_worldsim_v32_s2_3dgic import render_snapshot

        erase_delta = build_all_rigid_erase_delta(trainer.models["RigidNodes"])
        grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for request in requests:
            grouped[(int(request["frame"]), int(request["camera_id"]))].append(request)
        rows: list[dict[str, Any]] = []
        surface = config["surface"]
        arms = {str(arm["id"]): arm for arm in config["gaussianization_arms"]}
        for view_index, ((frame, camera), view_requests) in enumerate(sorted(grouped.items())):
            base = render_snapshot(
                trainer=trainer, dataset=dataset, frame=frame, camera_id=camera, device=device
            )
            reference = np.asarray(base["background_depth"], dtype=np.float32)
            base_opacity = np.asarray(base["background_opacity"], dtype=np.float32)
            for request in sorted(view_requests, key=lambda item: int(item["actor_id"])):
                actor_id = int(request["actor_id"])
                key = (frame, camera, actor_id)
                row: dict[str, Any] = {"frame": frame, "camera_id": camera, "actor_id": actor_id}
                prior = prior_rows[key]
                if request["accepted"] is not True:
                    if prior.get("status") != "abstain":
                        raise M2GaussianizationError("r005 rejected request 漂移")
                    row.update({"status": "abstain", "reason": "ABSTAIN_SAM_MASK_REJECTED"})
                    rows.append(row)
                    continue
                try:
                    target = binary_dilation(
                        _mask(mask_root, request),
                        iterations=int(protocol["target_mask_dilation_pixels"]),
                    )
                    inner = binary_dilation(
                        target, iterations=int(protocol["support_ring_inner_pixels"])
                    )
                    outer = binary_dilation(
                        target, iterations=int(protocol["support_ring_outer_pixels"])
                    )
                    support = outer & ~inner & ~np.asarray(base["dynamic_mask"], bool)
                    fit = fit_inverse_depth_surface(
                        depth=reference,
                        support_mask=support,
                        target_mask=target,
                        intrinsics=np.asarray(base["intrinsics"]),
                        model="G0_ROBUST_PLANE",
                        minimum_support_points=int(surface["minimum_support_points"]),
                        huber_delta=float(surface["huber_delta"]),
                        maximum_iterations=int(surface["maximum_iterations"]),
                        minimum_depth_m=float(surface["minimum_depth_m"]),
                        maximum_depth_m=float(surface["maximum_depth_m"]),
                    )
                    inpaint = cv2.inpaint(
                        cv2.cvtColor(np.asarray(base["groundtruth"], np.uint8), cv2.COLOR_RGB2BGR),
                        target.astype(np.uint8) * 255,
                        3.0,
                        cv2.INPAINT_TELEA,
                    )
                    inpaint = cv2.cvtColor(inpaint, cv2.COLOR_BGR2RGB)
                    states = {
                        arm_id: _arm_state(
                            arm=arms[arm_id],
                            fit=fit,
                            inpaint=inpaint,
                            target=target,
                            base=base,
                            trainer=trainer,
                            dataset=dataset,
                            device=device,
                            frame=frame,
                            camera=camera,
                            actor_id=actor_id,
                            erase_delta=erase_delta,
                            config=config,
                        )
                        for arm_id in ARM_IDS
                    }
                    rollback = render_snapshot(
                        trainer=trainer,
                        dataset=dataset,
                        frame=frame,
                        camera_id=camera,
                        device=device,
                    )
                    if not np.array_equal(rollback["rgb"], base["rgb"]):
                        raise M2GaussianizationError("Gaussianization arm rollback 非 exact")
                    baseline_common = (
                        target
                        & states["BASE"]["sampled"]
                        & np.isfinite(reference)
                        & (reference > 1e-4)
                        & np.isfinite(fit.depth)
                        & np.isfinite(states["BASE"]["post"])
                        & (states["BASE"]["post"] > 1e-4)
                    )
                    common = baseline_common.copy()
                    for arm_id in ARM_IDS[1:]:
                        state = states[arm_id]
                        common &= np.isfinite(state["post"]) & (state["post"] > 1e-4)
                    baseline_pre = np.full(target.shape, np.nan, dtype=np.float32)
                    baseline_pre[states["BASE"]["sampled"]] = fit.depth[
                        states["BASE"]["sampled"]
                    ]
                    prior_artifact = prior_root / prior["artifact"]["path"]
                    verify_file(prior_artifact, prior["artifact"]["sha256"])
                    with np.load(prior_artifact, allow_pickle=False) as old:
                        replay_exact = (
                            np.array_equal(old["target_mask"].astype(bool), target)
                            and np.array_equal(old["support_mask"].astype(bool), support)
                            and np.array_equal(
                                old["common_evaluation_mask"].astype(bool), baseline_common
                            )
                            and np.array_equal(
                                old["reference_depth"],
                                reference.astype(np.float16),
                                equal_nan=True,
                            )
                            and np.array_equal(
                                old["raw_surface_depth"], fit.depth.astype(np.float16), equal_nan=True
                            )
                            and np.array_equal(
                                old["pre_gaussianization_depth"],
                                baseline_pre.astype(np.float16),
                                equal_nan=True,
                            )
                            and np.array_equal(
                                old["post_gaussianization_render_depth"],
                                states["BASE"]["post"].astype(np.float16),
                                equal_nan=True,
                            )
                        )
                    if not replay_exact:
                        raise M2GaussianizationError(f"r005 BASE replay 非 exact: {key}")
                    arm_metrics: dict[str, Any] = {}
                    for arm_id, state in states.items():
                        gain = effective_opacity_gain(base_opacity, state["background_opacity"])
                        arm_metrics[arm_id] = {
                            "post_geometry_error": depth_error_summary(
                                state["post"], reference, common
                            ),
                            "representation_gap": depth_error_summary(
                                state["post"], fit.depth, common
                            ),
                            "effective_opacity_gain_mean": float(np.mean(gain[common])),
                            "effective_opacity_gain_median": float(np.median(gain[common])),
                            "effective_opacity_gain_above_0_01_fraction": float(
                                np.mean(gain[common] > 0.01)
                            ),
                            "total_vs_background_depth_mae_m": float(
                                np.mean(np.abs(state["post"][common] - state["background_depth"][common]))
                            ),
                            "sampled_pixel_count": int(state["sampled"].sum()),
                            "gaussian_count": state["gaussian_count"],
                            "scale_min_m": state["scale_min_m"],
                            "scale_mean_m": state["scale_mean_m"],
                            "scale_max_m": state["scale_max_m"],
                            "compute_seconds": state["compute_seconds"],
                            "frozen_parameters": dict(arms[arm_id]),
                            "composition_audit": state["composition_audit"],
                        }
                    artifact = run_dir / "artifacts/requests" / f"f{frame:03d}_c{camera}_a{actor_id:03d}.npz"
                    arrays: dict[str, np.ndarray] = {
                        "target_mask": target.astype(np.int8),
                        "support_mask": support.astype(np.int8),
                        "common_evaluation_mask": common.astype(np.int8),
                        "reference_depth": reference.astype(np.float16),
                        "raw_surface_depth": fit.depth.astype(np.float16),
                    }
                    for arm_id, state in states.items():
                        arrays[f"{arm_id.lower()}_post_depth"] = state["post"].astype(np.float16)
                        arrays[f"{arm_id.lower()}_background_opacity"] = state[
                            "background_opacity"
                        ].astype(np.float16)
                    atomic_save_npz(artifact, arrays)
                    row.update(
                        {
                            "status": "done",
                            "target_pixels": int(target.sum()),
                            "baseline_evaluation_pixels": int(baseline_common.sum()),
                            "common_evaluation_pixels": int(common.sum()),
                            "raw_geometry_error": depth_error_summary(
                                fit.depth, reference, common
                            ),
                            "baseline_replay_exact": True,
                            "reference": {
                                "source": config["reference"]["source"],
                                "independent_geometry_claim_allowed": False,
                            },
                            "arms": arm_metrics,
                            "artifact": {
                                "path": str(artifact.relative_to(run_dir)),
                                "sha256": sha256_file(artifact),
                            },
                        }
                    )
                except GeometryRepairError as error:
                    row.update({"status": "abstain", "reason": str(error)})
                rows.append(row)
            print(
                f"M2 Gaussianization {view_index + 1}/{len(grouped)} frame={frame} camera={camera} requests={len(view_requests)}",
                flush=True,
            )
        decision = mechanism_decision(rows, config["mechanism_gate"])
        if decision["baseline_replay_exact_count"] != decision["evaluable_request_count"]:
            raise M2GaussianizationError("r005 BASE replay denominator 不完整")
        diagnostics = {
            "schema_version": "worldsim_v5_m2_gaussianization_diagnostics_v1",
            "task_id": TASK_ID,
            "status": "done",
            "scene": config["scene"]["name"],
            "request_unit": "one_actor_one_view_one_hole",
            "rows": rows,
            "mechanism_decision": decision,
            "method_arm_selected": False,
            "validation_quality_read": False,
            "heldout_quality_read": False,
            "test_quality_read": False,
            "parameter_search_performed": False,
            "router_refit_performed": False,
        }
        diagnostics_path = run_dir / "artifacts/diagnostics.json"
        atomic_json(diagnostics_path, diagnostics)
        checkpoint_after = sha256_file(checkpoint)
        if checkpoint_after != checkpoint_before:
            raise M2GaussianizationError("formal checkpoint 被 Gaussianization forensic 修改")
        snapshot = copy_source_snapshot(
            run_dir,
            [
                config_path,
                PROJECT / "scripts/run_worldsim_v5_m2_gaussianization_forensics.py",
                PROJECT / "motion_proj/worldsim_v5/geometry_repair.py",
                PROJECT / "tests/test_run_worldsim_v5_m2_gaussianization_forensics.py",
            ],
            PROJECT,
        )
        summary = {
            "schema_version": "worldsim_v5_m2_gaussianization_summary_v1",
            "task_id": TASK_ID,
            "task_status": "running",
            "status": "done",
            "phase": config["phase"],
            "scene": config["scene"]["name"],
            "source_commit": source_head,
            "conclusion": decision["conclusion"],
            "request_unit": "one_actor_one_view_one_hole",
            "request_count": len(rows),
            "evaluable_request_count": decision["evaluable_request_count"],
            "abstain_request_count": len(rows) - decision["evaluable_request_count"],
            "mechanism_decision": decision,
            "method_arm_selected": False,
            "reference_scope": "model_derived_proxy_not_independent_ground_truth",
            "checkpoint_sha256_before": checkpoint_before,
            "checkpoint_sha256_after": checkpoint_after,
            "diagnostics_sha256": sha256_file(diagnostics_path),
            "duration_seconds": time.perf_counter() - started,
            "peak_gpu_memory_mib": int(torch.cuda.max_memory_allocated(device) / 1024**2),
            "validation_quality_read": False,
            "heldout_quality_read": False,
            "test_quality_read": False,
            "parameter_search_performed": False,
            "router_refit_performed": False,
        }
        summary_path = run_dir / "summary.json"
        atomic_json(summary_path, summary)
        atomic_json(
            run_dir / "fingerprint.json",
            {
                "schema_version": "worldsim_v5_m2_gaussianization_fingerprint_v1",
                "task_id": TASK_ID,
                "source_commit": source_head,
                "source_clean": True,
                "resolved_config": resolved,
                "inputs": inputs,
                "runtime": {
                    "drivestudio_commit": config["runtime"]["drivestudio_commit"],
                    "drivestudio_status": config["runtime"]["drivestudio_expected_status"],
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(device),
                },
                "source_snapshot": snapshot,
            },
        )
        events.append({"event": "run_done", "at_utc": utc_now(), **decision})
        write_events(run_dir, events)
        manifest = {
            "schema_version": "worldsim_v5_m2_gaussianization_run_manifest_v1",
            "task_id": TASK_ID,
            "status": "done",
            "inventory": inventory_files(run_dir, {"manifest.json", "status.json"}),
        }
        manifest_path = run_dir / "manifest.json"
        atomic_json(manifest_path, manifest)
        _terminal(
            run_dir,
            status="done",
            source_head=source_head,
            summary_sha256=sha256_file(summary_path),
            manifest_sha256=sha256_file(manifest_path),
            reason=None,
        )
        return summary
    except Exception as error:
        events.append(
            {
                "event": "run_blocked",
                "at_utc": utc_now(),
                "reason": f"{type(error).__name__}: {error}",
            }
        )
        write_events(run_dir, events)
        _terminal(
            run_dir,
            status="blocked",
            source_head=source_head,
            summary_sha256=None,
            manifest_sha256=None,
            reason=f"{type(error).__name__}: {error}",
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run(args.config.resolve(), args.run_dir.resolve(), args.device)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
