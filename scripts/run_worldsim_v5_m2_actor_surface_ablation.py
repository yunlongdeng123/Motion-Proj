#!/usr/bin/env python3
"""在 per-actor requests 上隔离比较冻结的 raw surface arms。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
from scipy.ndimage import binary_dilation
import torch
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v33.spatial_delta import atomic_save_npz
from motion_proj.worldsim_v5.geometry_repair import (
    GeometryRepairError,
    depth_error_summary,
    fit_inverse_depth_surface,
    geometry_reference_confidence,
)
from scripts.run_worldsim_v5_m2_actor_geometry import _load_requests, _mask
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
SCHEMA_VERSION = "worldsim_v5_m2_actor_surface_ablation_v1"
ARMS = ("G0_ROBUST_PLANE", "G1_PIECEWISE_PLANE")
SCHEMA_BINDINGS = {
    SCHEMA_VERSION: (
        "per_actor_g0_g1_raw_surface_ablation",
        ARMS,
    ),
    "worldsim_v5_m2_actor_surface_ablation_v2": (
        "per_actor_g0_g2_raw_surface_ablation",
        ("G0_ROBUST_PLANE", "G2_MOVING_LEAST_SQUARES"),
    ),
    "worldsim_v5_m2_actor_surface_ablation_v3": (
        "per_actor_g0_g3_raw_surface_ablation",
        ("G0_ROBUST_PLANE", "G3_ROBUST_QUADRATIC"),
    ),
}


class M2ActorSurfaceError(RuntimeError):
    """per-actor raw surface ablation 合约失败。"""


def _raw_depth_payload(
    states: Mapping[str, Mapping[str, Any]], arms: tuple[str, str]
) -> dict[str, np.ndarray]:
    return {
        f"{model.split('_', 1)[0].lower()}_raw_depth": states[model][
            "fit"
        ].depth.astype(np.float16)
        for model in arms
    }


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") not in SCHEMA_BINDINGS:
        raise M2ActorSurfaceError("M2 actor surface config schema 漂移")
    phase, arms = SCHEMA_BINDINGS[payload["schema_version"]]
    if (
        payload.get("task_id") != TASK_ID
        or payload.get("status") != "running"
        or payload.get("phase") != phase
        or tuple(payload["surface"]["matched_models"]) != arms
        or payload["request_protocol"]["unit"] != "one_actor_one_view_one_hole"
        or payload["request_protocol"]["union_mask_for_geometry_forbidden"] is not True
        or payload["scope"]["gaussianization_started"] is not False
    ):
        raise M2ActorSurfaceError("M2 actor surface task/arms/scope 漂移")
    for name in (
        "validation_quality_read",
        "heldout_quality_read",
        "test_quality_read",
        "parameter_search_performed",
        "router_refit_performed",
    ):
        if payload["scope"].get(name) is not False:
            raise M2ActorSurfaceError(f"M2 actor surface restriction 漂移: {name}")
    if payload["reference"]["independent_geometry_claim_allowed"] is not False:
        raise M2ActorSurfaceError("model proxy 不得声明独立 GT")
    if payload["schema_version"] == SCHEMA_VERSION:
        if payload["unlock_binding"]["source_g1_unlocked"] is not True:
            raise M2ActorSurfaceError("G1 未被 r005 gate 解锁")
    elif payload["unlock_binding"]["source_next_model_unlocked"] != arms[1]:
        raise M2ActorSurfaceError("G2 未按冻结序列解锁")
    return payload


def selection(
    rows: list[Mapping[str, Any]],
    gate: Mapping[str, Any],
    arms: tuple[str, str] = ARMS,
) -> dict[str, Any]:
    baseline, candidate = arms
    candidate_slug = candidate.split("_", 1)[0].lower()
    done = [row for row in rows if row.get("status") == "done"]
    deltas = [
        float(row["arms"][candidate]["raw_geometry_error"]["mae_m"])
        - float(row["arms"][baseline]["raw_geometry_error"]["mae_m"])
        for row in done
    ]
    improvement_m = gate.get(
        "minimum_candidate_raw_improvement_m",
        gate.get("minimum_g1_raw_improvement_m"),
    )
    improvement_count = gate.get(
        "minimum_candidate_raw_improvement_request_count",
        gate.get("minimum_g1_raw_improvement_request_count"),
    )
    mean_limit = gate.get(
        "require_mean_candidate_raw_delta_below_m",
        gate.get("require_mean_g1_raw_delta_below_m"),
    )
    median_limit = gate.get(
        "require_median_candidate_raw_delta_below_m",
        gate.get("require_median_g1_raw_delta_below_m"),
    )
    improved = sum(
        value <= -float(improvement_m) for value in deltas
    )
    mean_delta = float(np.mean(deltas)) if deltas else math.nan
    median_delta = float(np.median(deltas)) if deltas else math.nan
    passed = (
        len(done) >= int(gate["minimum_evaluable_request_count"])
        and improved >= int(improvement_count)
        and mean_delta < float(mean_limit)
        and median_delta < float(median_limit)
    )
    label = {
        "G1_PIECEWISE_PLANE": "g1_piecewise_surface",
        "G2_MOVING_LEAST_SQUARES": "g2_moving_least_squares_surface",
        "G3_ROBUST_QUADRATIC": "g3_quadratic_surface",
    }.get(candidate, candidate.lower())
    return {
        "conclusion": (
            f"{label}_supported_on_model_proxy"
            if passed
            else f"{label}_rejected_on_model_proxy"
        ),
        "gate_passed": passed,
        "evaluable_request_count": len(done),
        f"{candidate_slug}_raw_improvement_request_count": improved,
        f"mean_{candidate_slug}_minus_g0_raw_mae_m": mean_delta,
        f"median_{candidate_slug}_minus_g0_raw_mae_m": median_delta,
        "minimum_delta_m": float(np.min(deltas)) if deltas else math.nan,
        "maximum_delta_m": float(np.max(deltas)) if deltas else math.nan,
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
    schema_suffix: str = "v1",
) -> None:
    atomic_json(
        run_dir / "status.json",
        {
            "schema_version": f"worldsim_v5_m2_actor_surface_status_{schema_suffix}",
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


def run(config_path: Path, run_dir: Path, device_name: str) -> dict[str, Any]:
    config = load_config(config_path)
    arms = tuple(config["surface"]["matched_models"])
    schema_suffix = config["schema_version"].rsplit("_", 1)[-1]
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
        unlock = config["unlock_binding"]
        prior_key = unlock.get("summary_input_key", "g0_actor_summary")
        prior = json.loads(Path(inputs[prior_key]["path"]).read_text())
        prior_valid = (
            prior.get("status") == "done"
            and prior.get("conclusion") == unlock["source_conclusion"]
            and int(prior.get("evaluable_request_count", -1))
            == int(unlock["source_evaluable_request_count"])
        )
        if config["schema_version"] == SCHEMA_VERSION:
            prior_valid = prior_valid and (
                prior.get("mechanism", {}).get("g1_unlocked_for_next_development_run")
                is True
            )
        else:
            prior_valid = prior_valid and (
                prior.get("selection", {}).get("gate_passed")
                is bool(unlock["source_selection_gate_passed"])
            )
        if not prior_valid:
            raise M2ActorSurfaceError("raw surface unlock binding 漂移")
        mask_root, requests = _load_requests(config)
        protocol = config["request_protocol"]
        if len(requests) != int(protocol["expected_request_count"]):
            raise M2ActorSurfaceError("actor request denominator 漂移")
        accepted = sum(bool(row["accepted"]) for row in requests)
        if (
            accepted != int(protocol["expected_accepted_mask_count"])
            or len(requests) - accepted != int(protocol["expected_rejected_mask_count"])
        ):
            raise M2ActorSurfaceError("actor accepted/rejected denominator 漂移")
        checkpoint = Path(inputs["formal_checkpoint"]["path"])
        checkpoint_before = sha256_file(checkpoint)
        if not torch.cuda.is_available():
            raise M2ActorSurfaceError("M2 actor surface 需要 CUDA base render")
        device = torch.device(device_name)
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        _, dataset, trainer = _runtime(config, device)
        from scripts.run_worldsim_v32_s2_3dgic import render_snapshot

        grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for request in requests:
            grouped[(int(request["frame"]), int(request["camera_id"]))].append(request)
        rows: list[dict[str, Any]] = []
        surface_cfg = config["surface"]
        for view_index, ((frame, camera), view_requests) in enumerate(sorted(grouped.items())):
            base = render_snapshot(
                trainer=trainer, dataset=dataset, frame=frame, camera_id=camera, device=device
            )
            reference = np.asarray(base["background_depth"], dtype=np.float32)
            for request in sorted(view_requests, key=lambda row: int(row["actor_id"])):
                actor_id = int(request["actor_id"])
                row: dict[str, Any] = {"frame": frame, "camera_id": camera, "actor_id": actor_id}
                if request["accepted"] is not True:
                    row.update({"status": "abstain", "reason": "ABSTAIN_SAM_MASK_REJECTED"})
                    rows.append(row)
                    continue
                try:
                    target = binary_dilation(
                        _mask(mask_root, request),
                        iterations=int(protocol["target_mask_dilation_pixels"]),
                    )
                    inner = binary_dilation(target, iterations=int(protocol["support_ring_inner_pixels"]))
                    outer = binary_dilation(target, iterations=int(protocol["support_ring_outer_pixels"]))
                    support = outer & ~inner & ~np.asarray(base["dynamic_mask"], bool)
                    states: dict[str, Any] = {}
                    for model in arms:
                        arm_started = time.perf_counter()
                        fit = fit_inverse_depth_surface(
                            depth=reference,
                            support_mask=support,
                            target_mask=target,
                            intrinsics=np.asarray(base["intrinsics"]),
                            model=model,
                            minimum_support_points=int(surface_cfg["minimum_support_points"]),
                            huber_delta=float(surface_cfg["huber_delta"]),
                            maximum_iterations=int(surface_cfg["maximum_iterations"]),
                            minimum_depth_m=float(surface_cfg["minimum_depth_m"]),
                            maximum_depth_m=float(surface_cfg["maximum_depth_m"]),
                            mls_neighbor_count=int(surface_cfg.get("mls_neighbor_count", 128)),
                            mls_bandwidth_pixels=float(surface_cfg.get("mls_bandwidth_pixels", 18.0)),
                            mls_minimum_weight=float(surface_cfg.get("mls_minimum_weight", 1e-5)),
                        )
                        states[model] = {
                            "fit": fit,
                            "compute_seconds": time.perf_counter() - arm_started,
                        }
                    common = target & np.isfinite(reference) & (reference > 1e-4)
                    for state in states.values():
                        common &= np.isfinite(state["fit"].depth) & (state["fit"].depth > 1e-4)
                    arm_metrics = {
                        model: {
                            "raw_geometry_error": depth_error_summary(
                                state["fit"].depth, reference, common
                            ),
                            "surface_fit": state["fit"].audit(),
                            "compute_seconds": state["compute_seconds"],
                        }
                        for model, state in states.items()
                    }
                    lidar = np.asarray(base["measured_lidar_depth"], dtype=np.float32)
                    lidar_valid = support & np.isfinite(lidar) & (lidar > 1e-4) & np.isfinite(reference) & (reference > 1e-4)
                    lidar_count = int(lidar_valid.sum())
                    lidar_mae = float(np.mean(np.abs(lidar[lidar_valid] - reference[lidar_valid]))) if lidar_count else None
                    confidence = geometry_reference_confidence(
                        observed_reference_pixels=lidar_count,
                        target_pixels=int(target.sum()),
                        lidar_agreement_mae_m=lidar_mae,
                        agreement_scale_m=float(config["reference"]["confidence"]["agreement_scale_m"]),
                    )
                    artifact = run_dir / "artifacts/requests" / f"f{frame:03d}_c{camera}_a{actor_id:03d}.npz"
                    atomic_save_npz(
                        artifact,
                        {
                            "target_mask": target.astype(np.int8),
                            "support_mask": support.astype(np.int8),
                            "common_evaluation_mask": common.astype(np.int8),
                            "reference_depth": reference.astype(np.float16),
                            **_raw_depth_payload(states, arms),
                        },
                    )
                    row.update(
                        {
                            "status": "done",
                            "target_pixels": int(target.sum()),
                            "common_evaluation_pixels": int(common.sum()),
                            "reference": {
                                "source": config["reference"]["source"],
                                "confidence": confidence,
                                "observed_reference_pixels": lidar_count,
                                "lidar_agreement_mae_m": lidar_mae,
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
                f"M2 actor {arms[0].split('_', 1)[0]}/{arms[1].split('_', 1)[0]} raw {view_index + 1}/{len(grouped)} frame={frame} camera={camera} requests={len(view_requests)}",
                flush=True,
            )
        decision = selection(rows, config["selection_gate"], arms)
        diagnostics = {
            "schema_version": f"worldsim_v5_m2_actor_surface_diagnostics_{schema_suffix}",
            "task_id": TASK_ID,
            "status": "done",
            "scene": config["scene"]["name"],
            "request_unit": "one_actor_one_view_one_hole",
            "gaussianization_started": False,
            "rows": rows,
            "selection": decision,
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
            raise M2ActorSurfaceError("formal checkpoint 被 raw surface ablation 修改")
        snapshot = copy_source_snapshot(
            run_dir,
            [
                config_path,
                PROJECT / "scripts/run_worldsim_v5_m2_actor_surface_ablation.py",
                PROJECT / "motion_proj/worldsim_v5/geometry_repair.py",
                PROJECT / "tests/test_run_worldsim_v5_m2_actor_surface_ablation.py",
            ],
            PROJECT,
        )
        summary = {
            "schema_version": f"worldsim_v5_m2_actor_surface_summary_{schema_suffix}",
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
            "selection": decision,
            "gaussianization_started": False,
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
                "schema_version": f"worldsim_v5_m2_actor_surface_fingerprint_{schema_suffix}",
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
            "schema_version": f"worldsim_v5_m2_actor_surface_run_manifest_{schema_suffix}",
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
            schema_suffix=schema_suffix,
        )
        return summary
    except Exception as error:
        events.append({"event": "run_blocked", "at_utc": utc_now(), "reason": f"{type(error).__name__}: {error}"})
        write_events(run_dir, events)
        _terminal(
            run_dir,
            status="blocked",
            source_head=source_head,
            summary_sha256=None,
            manifest_sha256=None,
            reason=f"{type(error).__name__}: {error}",
            schema_suffix=schema_suffix,
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.run_dir.resolve(), args.device), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
