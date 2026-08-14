#!/usr/bin/env python3
"""在冻结 M2 development views 上执行 G0/G3 matched surface ablation。"""

from __future__ import annotations

import argparse
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
    fit_inverse_depth_surface,
    geometry_reference_confidence,
    staged_geometry_metrics,
)
from scripts.run_worldsim_v5_m2_staged_geometry_diagnostic import (
    _git,
    _load_mask_rows,
    _mask,
    build_all_rigid_erase_delta,
)
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
SCHEMA_VERSION = "worldsim_v5_m2_surface_ablation_v1"
ARMS = ("G0_ROBUST_PLANE", "G3_ROBUST_QUADRATIC")


class M2SurfaceAblationError(RuntimeError):
    """冻结输入或 matched G0/G3 合约失败。"""


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise M2SurfaceAblationError("M2 surface ablation config schema 漂移")
    if (
        payload.get("task_id") != TASK_ID
        or payload.get("status") != "running"
        or payload.get("phase") != "g3_surface_ablation"
        or tuple(payload["surface"]["matched_models"]) != ARMS
    ):
        raise M2SurfaceAblationError("M2 surface ablation task/arms 漂移")
    if payload["reference"]["independent_geometry_claim_allowed"] is not False:
        raise M2SurfaceAblationError("model proxy 不得声明独立 geometry GT")
    for name in (
        "validation_quality_read",
        "heldout_quality_read",
        "test_quality_read",
        "parameter_search_performed",
        "router_refit_performed",
    ):
        if payload["scope"].get(name) is not False:
            raise M2SurfaceAblationError(f"M2 surface restriction 漂移: {name}")
    unlock = payload["unlock_binding"]
    if (
        unlock.get("source_g3_unlocked") is not True
        or int(unlock.get("source_g3_unlock_failure_view_count", -1)) < 3
    ):
        raise M2SurfaceAblationError("G3 未被前序冻结 gate 解锁")
    return payload


def surface_selection(rows: list[Mapping[str, Any]], gate: Mapping[str, Any]) -> dict[str, Any]:
    evaluable = [row for row in rows if row.get("status") == "done"]
    deltas = []
    for row in evaluable:
        g0 = row["arms"][ARMS[0]]["staged_metrics"]
        g3 = row["arms"][ARMS[1]]["staged_metrics"]
        deltas.append(
            {
                "frame": int(row["frame"]),
                "camera_id": int(row["camera_id"]),
                "raw_mae_delta_m": float(
                    g3["raw_geometry_error"]["mae_m"]
                    - g0["raw_geometry_error"]["mae_m"]
                ),
                "post_render_mae_delta_m": float(
                    g3["post_gaussianization_render_error"]["mae_m"]
                    - g0["post_gaussianization_render_error"]["mae_m"]
                ),
            }
        )
    minimum = float(gate["minimum_g3_raw_improvement_m"])
    improved = sum(row["raw_mae_delta_m"] <= -minimum for row in deltas)
    mean_raw = float(np.mean([row["raw_mae_delta_m"] for row in deltas])) if deltas else math.nan
    mean_post = (
        float(np.mean([row["post_render_mae_delta_m"] for row in deltas]))
        if deltas
        else math.nan
    )
    passed = (
        len(evaluable) >= int(gate["minimum_evaluable_views"])
        and improved >= int(gate["minimum_g3_raw_improvement_view_count"])
        and mean_raw < float(gate["require_mean_g3_raw_delta_below_m"])
        and mean_post <= float(gate["maximum_mean_g3_post_render_regression_m"])
    )
    return {
        "conclusion": (
            "g3_surface_supported_on_model_proxy"
            if passed
            else "g3_surface_rejected_on_model_proxy"
        ),
        "gate_passed": passed,
        "evaluable_view_count": len(evaluable),
        "g3_raw_improvement_view_count": improved,
        "mean_g3_minus_g0_raw_mae_m": mean_raw,
        "mean_g3_minus_g0_post_render_mae_m": mean_post,
        "view_deltas": deltas,
        "reference_scope": "model_derived_proxy_not_independent_ground_truth",
    }


def _runtime(config: Mapping[str, Any], device: torch.device):
    checkout = Path(config["runtime"]["drivestudio_checkout"])
    if _git(checkout, "rev-parse", "HEAD") != config["runtime"]["drivestudio_commit"]:
        raise M2SurfaceAblationError("DriveStudio commit 漂移")
    if _git(checkout, "status", "--short") != str(
        config["runtime"]["drivestudio_expected_status"]
    ).strip():
        raise M2SurfaceAblationError("DriveStudio frozen patch status 漂移")
    from scripts.run_worldsim_v5_m1_unary_diagnostic import _build_runtime

    return _build_runtime(config, device)


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
            "schema_version": "worldsim_v5_m2_surface_ablation_status_v1",
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


def _arm(
    *,
    model: str,
    trainer: Any,
    dataset: Any,
    device: torch.device,
    frame: int,
    camera: int,
    base: Mapping[str, Any],
    target: np.ndarray,
    support: np.ndarray,
    inpaint: np.ndarray,
    erase_delta: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    candidate_suffix: str = "",
) -> dict[str, Any]:
    from scripts.run_worldsim_v32_s2_3dgic import render_snapshot

    surface_cfg = config["surface"]
    gaussian_cfg = config["gaussianization"]
    reference = np.asarray(base["background_depth"], dtype=np.float32)
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
    )
    points = completion_points_from_view(
        rgb=inpaint,
        depth=fit.depth,
        mask=fit.valid,
        observed_cross_view=np.zeros_like(target),
        intrinsics=np.asarray(base["intrinsics"]),
        camera_to_world=np.asarray(base["camera_to_world"]),
        stride=int(gaussian_cfg["stride"]),
        scale_multiplier=float(gaussian_cfg["scale_multiplier"]),
        minimum_scale_m=float(gaussian_cfg["minimum_scale_m"]),
        maximum_scale_m=float(gaussian_cfg["maximum_scale_m"]),
    )
    asset = completion_points_to_repair_asset(
        points,
        candidate_id=(
            f"scene0471_f{frame:03d}_c{camera}_{model.lower()}"
            f"{'_' + candidate_suffix if candidate_suffix else ''}"
        ),
        method="DONOR",
        provenance="native_scene_donor",
        features_rest_shape=tuple(trainer.models["Background"]._features_rest.shape[1:]),
        opacity=float(gaussian_cfg["opacity"]),
        target_frame=frame,
        target_camera_id=camera,
    )
    pre = np.full(reference.shape, np.nan, dtype=np.float32)
    pixels = np.asarray(points.source_pixels_xy, dtype=np.int64)
    pre[pixels[:, 1], pixels[:, 0]] = fit.depth[pixels[:, 1], pixels[:, 0]]
    with temporary_repair_composition(
        trainer.models, erase_delta=erase_delta, asset=asset
    ) as composition:
        rendered = render_snapshot(
            trainer=trainer, dataset=dataset, frame=frame, camera_id=camera, device=device
        )
    rollback = render_snapshot(
        trainer=trainer, dataset=dataset, frame=frame, camera_id=camera, device=device
    )
    if not np.array_equal(rollback["rgb"], base["rgb"]):
        raise M2SurfaceAblationError(f"{model} rollback render 非 exact")
    return {
        "fit": fit,
        "raw": fit.depth,
        "pre": pre,
        "post": np.asarray(rendered["depth"], dtype=np.float32),
        "gaussian_count": int(points.means.shape[0]),
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
        prior = json.loads(Path(inputs["g0_staged_summary"]["path"]).read_text())
        if (
            prior.get("status") != "done"
            or prior.get("conclusion") != config["unlock_binding"]["source_conclusion"]
            or prior.get("mechanism_decision", {}).get("g3_unlocked_for_next_development_run")
            is not True
            or prior.get("reference_scope") != config["unlock_binding"]["source_reference_scope"]
        ):
            raise M2SurfaceAblationError("r002 G3 unlock binding 漂移")
        mask_root, mask_rows = _load_mask_rows(config)
        checkpoint = Path(inputs["formal_checkpoint"]["path"])
        checkpoint_before = sha256_file(checkpoint)
        if not torch.cuda.is_available():
            raise M2SurfaceAblationError("M2 surface ablation 需要 CUDA")
        device = torch.device(device_name)
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        _, dataset, trainer = _runtime(config, device)
        from scripts.run_worldsim_v32_s2_3dgic import render_snapshot

        erase_delta = build_all_rigid_erase_delta(trainer.models["RigidNodes"])
        rows: list[dict[str, Any]] = []
        view_cfg = config["view_protocol"]
        for index, mask_row in enumerate(mask_rows):
            frame, camera = int(mask_row["frame"]), int(mask_row["camera_id"])
            row: dict[str, Any] = {"frame": frame, "camera_id": camera}
            if mask_row.get("mask_quality_accepted") is not True:
                row.update({"status": "abstain", "reason": "ABSTAIN_SAM_MASK_UNAVAILABLE"})
                rows.append(row)
                print(f"M2 G0/G3 {index + 1}/{len(mask_rows)} frame={frame} camera={camera} abstain", flush=True)
                continue
            try:
                base = render_snapshot(
                    trainer=trainer, dataset=dataset, frame=frame, camera_id=camera, device=device
                )
                target = binary_dilation(
                    _mask(mask_root, mask_row),
                    iterations=int(view_cfg["target_mask_dilation_pixels"]),
                )
                inner = binary_dilation(target, iterations=int(view_cfg["support_ring_inner_pixels"]))
                outer = binary_dilation(target, iterations=int(view_cfg["support_ring_outer_pixels"]))
                support = outer & ~inner & ~np.asarray(base["dynamic_mask"], bool)
                inpaint = cv2.inpaint(
                    cv2.cvtColor(np.asarray(base["groundtruth"], np.uint8), cv2.COLOR_RGB2BGR),
                    target.astype(np.uint8) * 255,
                    3.0,
                    cv2.INPAINT_TELEA,
                )
                inpaint = cv2.cvtColor(inpaint, cv2.COLOR_BGR2RGB)
                arm_state = {
                    model: _arm(
                        model=model,
                        trainer=trainer,
                        dataset=dataset,
                        device=device,
                        frame=frame,
                        camera=camera,
                        base=base,
                        target=target,
                        support=support,
                        inpaint=inpaint,
                        erase_delta=erase_delta,
                        config=config,
                    )
                    for model in ARMS
                }
                reference = np.asarray(base["background_depth"], dtype=np.float32)
                common = target & np.isfinite(reference) & (reference > 1e-4)
                for state in arm_state.values():
                    common &= (
                        np.isfinite(state["raw"])
                        & np.isfinite(state["pre"])
                        & np.isfinite(state["post"])
                        & (state["post"] > 1e-4)
                    )
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
                arms: dict[str, Any] = {}
                arrays: dict[str, np.ndarray] = {
                    "target_mask": target.astype(np.int8),
                    "support_mask": support.astype(np.int8),
                    "common_evaluation_mask": common.astype(np.int8),
                    "reference_depth": reference.astype(np.float16),
                }
                for model, state in arm_state.items():
                    metrics = staged_geometry_metrics(
                        raw_surface_depth=state["raw"],
                        pre_gaussianization_depth=state["pre"],
                        post_gaussianization_render_depth=state["post"],
                        reference_depth=reference,
                        evaluation_mask=common,
                    )
                    arms[model] = {
                        "staged_metrics": metrics,
                        "surface_fit": state["fit"].audit(),
                        "gaussian_count": state["gaussian_count"],
                        "composition_audit": state["composition_audit"],
                    }
                    prefix = model.lower()
                    arrays[f"{prefix}_raw_depth"] = state["raw"].astype(np.float16)
                    arrays[f"{prefix}_pre_depth"] = state["pre"].astype(np.float16)
                    arrays[f"{prefix}_post_depth"] = state["post"].astype(np.float16)
                artifact = run_dir / "artifacts/views" / f"f{frame:03d}_c{camera}.npz"
                atomic_save_npz(artifact, arrays)
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
                        "arms": arms,
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
                f"M2 G0/G3 {index + 1}/{len(mask_rows)} frame={frame} camera={camera} {row['status']}",
                flush=True,
            )
        selection = surface_selection(rows, config["selection_gate"])
        diagnostics = {
            "schema_version": "worldsim_v5_m2_surface_ablation_diagnostics_v1",
            "task_id": TASK_ID,
            "status": "done",
            "scene": config["scene"]["name"],
            "rows": rows,
            "selection": selection,
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
            raise M2SurfaceAblationError("formal checkpoint 被 surface ablation 修改")
        snapshot = copy_source_snapshot(
            run_dir,
            [
                config_path,
                PROJECT / "scripts/run_worldsim_v5_m2_surface_ablation.py",
                PROJECT / "scripts/run_worldsim_v5_m2_staged_geometry_diagnostic.py",
                PROJECT / "motion_proj/worldsim_v5/geometry_repair.py",
                PROJECT / "tests/test_run_worldsim_v5_m2_surface_ablation.py",
            ],
            PROJECT,
        )
        summary = {
            "schema_version": "worldsim_v5_m2_surface_ablation_summary_v1",
            "task_id": TASK_ID,
            "task_status": "running",
            "status": "done",
            "phase": config["phase"],
            "scene": config["scene"]["name"],
            "source_commit": source_head,
            "conclusion": selection["conclusion"],
            "selection": selection,
            "view_count": len(rows),
            "evaluable_view_count": selection["evaluable_view_count"],
            "abstain_view_count": len(rows) - selection["evaluable_view_count"],
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
                "schema_version": "worldsim_v5_m2_surface_ablation_fingerprint_v1",
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
        events.append({"event": "run_done", "at_utc": utc_now(), **selection})
        write_events(run_dir, events)
        manifest = {
            "schema_version": "worldsim_v5_m2_surface_ablation_manifest_v1",
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
        events.append({"event": "run_blocked", "at_utc": utc_now(), "reason": f"{type(error).__name__}: {error}"})
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
    print(json.dumps(run(args.config.resolve(), args.run_dir.resolve(), args.device), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
