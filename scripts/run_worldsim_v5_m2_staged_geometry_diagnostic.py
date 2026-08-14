#!/usr/bin/env python3
"""在 fresh development base 上采集 V5 M2 分阶段几何诊断。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
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
from motion_proj.worldsim_v33.spatial_delta import (
    ERASE_SCHEMA_VERSION,
    MODEL_RIGID,
    atomic_save_npz,
    sha256_arrays,
)
from motion_proj.worldsim_v4.repair_assets import temporary_repair_composition
from motion_proj.worldsim_v4.repair_builders import completion_points_to_repair_asset
from motion_proj.worldsim_v5.geometry_repair import (
    GeometryRepairError,
    fit_inverse_depth_surface,
    geometry_reference_confidence,
    staged_geometry_metrics,
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
SCHEMA_VERSION = "worldsim_v5_m2_geometry_first_development_v1"


class M2StagedGeometryError(RuntimeError):
    """冻结输入、运行时或分阶段几何合约失败。"""


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise M2StagedGeometryError("M2 staged geometry config schema 漂移")
    if (
        payload.get("task_id") != TASK_ID
        or payload.get("status") != "running"
        or payload.get("phase") != "staged_geometry_contract_smoke"
    ):
        raise M2StagedGeometryError("M2 staged geometry task/phase/status 漂移")
    scope = payload["scope"]
    for name in (
        "validation_quality_read",
        "heldout_quality_read",
        "test_quality_read",
        "parameter_search_performed",
        "router_refit_performed",
    ):
        if scope.get(name) is not False:
            raise M2StagedGeometryError(f"M2 development restriction 漂移: {name}")
    views = payload["view_protocol"]
    if int(views["expected_view_count"]) != len(views["frames"]) * len(views["cameras"]):
        raise M2StagedGeometryError("M2 frozen view denominator 漂移")
    if payload["surface"]["active_models"] != ["G0_ROBUST_PLANE"]:
        raise M2StagedGeometryError("M2 首轮只允许 G0")
    if payload["reference"]["independent_geometry_claim_allowed"] is not False:
        raise M2StagedGeometryError("model-derived geometry reference 不得升级为独立 GT")
    return payload


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def _runtime(config: Mapping[str, Any], device: torch.device):
    checkout = Path(config["runtime"]["drivestudio_checkout"])
    if _git(checkout, "rev-parse", "HEAD") != config["runtime"]["drivestudio_commit"]:
        raise M2StagedGeometryError("DriveStudio commit 漂移")
    if _git(checkout, "status", "--short") != str(
        config["runtime"]["drivestudio_expected_status"]
    ).strip():
        raise M2StagedGeometryError("DriveStudio frozen patch status 漂移")
    from scripts.run_worldsim_v5_m1_unary_diagnostic import _build_runtime

    return _build_runtime(config, device)


def build_all_rigid_erase_delta(rigid: Any) -> dict[str, np.ndarray]:
    count = int(rigid._means.shape[0])
    if count <= 0:
        raise M2StagedGeometryError("ABSTAIN_NO_RIGID_GAUSSIANS")
    gaussian_ids = rigid.point_ids.detach().cpu().numpy().reshape(-1).astype(np.int64)
    if gaussian_ids.size != count or np.unique(gaussian_ids).size != count:
        gaussian_ids = np.arange(10**12, 10**12 + count, dtype=np.int64)
    selector = {
        "model_code": np.full(count, MODEL_RIGID, dtype=np.int8),
        "source_flat_indices": np.arange(count, dtype=np.int64),
        "gaussian_ids": gaussian_ids,
        "selection_score": np.ones(count, dtype=np.float32),
    }
    return {
        "schema_version": np.asarray(ERASE_SCHEMA_VERSION, dtype="<U64"),
        "instance_id": np.asarray(-1, dtype=np.int32),
        "instance_token": np.asarray("moving_rigid_union", dtype="<U64"),
        "mask_hash": np.asarray(sha256_arrays(selector), dtype="<U64"),
        "selection_policy": np.asarray("hard_assignment_all", dtype="<U64"),
        "minimum_background_instance_opacity": np.asarray(-1.0, dtype=np.float32),
        **selector,
    }


def mechanism_conclusion(
    rows: list[Mapping[str, Any]], gates: Mapping[str, Any]
) -> dict[str, Any]:
    evaluable = [row for row in rows if row.get("status") == "done"]
    gaussian_gate = gates["gaussianization_primary"]
    builder_gate = gates["candidate_builder_primary"]
    unlock_gate = gates["unlock_g3_if"]
    gaussian_count = sum(
        float(row["staged_metrics"]["gaussianization_delta_mae_m"])
        >= float(gaussian_gate["minimum_post_minus_pre_mae_m"])
        for row in evaluable
    )
    raw_count = sum(
        float(row["staged_metrics"]["raw_geometry_error"]["mae_m"])
        >= float(builder_gate["minimum_raw_mae_m"])
        for row in evaluable
    )
    unlock_count = sum(
        float(row["staged_metrics"]["raw_geometry_error"]["mae_m"])
        >= float(unlock_gate["g0_raw_failure_mae_m"])
        for row in evaluable
    )
    if len(evaluable) < int(gates["minimum_evaluable_views"]):
        conclusion = "insufficient_evaluable_views_keep_g3_locked"
    elif gaussian_count >= int(gaussian_gate["minimum_view_count"]):
        conclusion = "gaussianization_is_primary_mechanism_on_model_proxy"
    elif raw_count >= int(builder_gate["minimum_view_count"]):
        conclusion = "g0_candidate_builder_is_primary_mechanism_on_model_proxy"
    else:
        conclusion = "g0_and_gaussianization_not_primary_on_model_proxy"
    return {
        "conclusion": conclusion,
        "evaluable_view_count": len(evaluable),
        "gaussianization_primary_view_count": gaussian_count,
        "candidate_builder_primary_view_count": raw_count,
        "g3_unlock_failure_view_count": unlock_count,
        "g3_unlocked_for_next_development_run": (
            len(evaluable) >= int(gates["minimum_evaluable_views"])
            and unlock_count >= int(unlock_gate["minimum_g0_raw_failure_views"])
        ),
    }


def _load_mask_rows(config: Mapping[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    manifest_path = Path(config["inputs"]["sam_mask_manifest"]["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("scene") != config["scene"]["name"]
        or manifest.get("heldout_quality_read") is not False
    ):
        raise M2StagedGeometryError("SAM mask manifest scene/provenance 漂移")
    expected = {
        (int(frame), int(camera))
        for frame in config["view_protocol"]["frames"]
        for camera in config["view_protocol"]["cameras"]
    }
    rows = [
        row
        for row in manifest["views"]
        if row.get("split") == "evaluation"
        and (int(row["frame"]), int(row["camera_id"])) in expected
    ]
    actual = {(int(row["frame"]), int(row["camera_id"])) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise M2StagedGeometryError("SAM frozen view denominator 不完整")
    if any(row.get("mask_quality_accepted") is not True for row in rows):
        raise M2StagedGeometryError("冻结 M2 view 含未通过质量门的 mask")
    return manifest_path.parent.parent, sorted(
        rows, key=lambda row: (int(row["frame"]), int(row["camera_id"]))
    )


def _mask(mask_root: Path, row: Mapping[str, Any]) -> np.ndarray:
    path = mask_root / str(row["mask"]["path"])
    verify_file(path, str(row["mask"]["sha256"]))
    with np.load(path, allow_pickle=False) as payload:
        accepted = bool(int(payload["mask_quality_accepted"].item()))
        binary = payload["binary"].astype(bool)
    if not accepted:
        raise M2StagedGeometryError(f"mask quality flag 漂移: {path}")
    return binary


def _aggregate(rows: list[Mapping[str, Any]], field: str) -> dict[str, float]:
    values = [float(row["staged_metrics"][field]["mae_m"]) for row in rows if row["status"] == "done"]
    if not values:
        return {"mean_mae_m": math.nan, "median_mae_m": math.nan}
    return {"mean_mae_m": float(np.mean(values)), "median_mae_m": float(np.median(values))}


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
            "schema_version": "worldsim_v5_m2_staged_geometry_status_v1",
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
        formal = json.loads(Path(inputs["formal_summary"]["path"]).read_text())
        sam = json.loads(Path(inputs["sam_summary"]["path"]).read_text())
        if (
            formal.get("status") != "done"
            or formal.get("scene") != config["scene"]["name"]
            or formal.get("checkpoint", {}).get("sha256")
            != inputs["formal_checkpoint"]["sha256"]
            or formal.get("validation_quality_read") is not False
            or formal.get("test_quality_read") is not False
            or sam.get("status") != "done"
            or sam.get("scene") != config["scene"]["name"]
            or sam.get("heldout_quality_read") is not False
        ):
            raise M2StagedGeometryError("formal base 或 SAM provenance 漂移")
        mask_root, mask_rows = _load_mask_rows(config)
        checkpoint = Path(inputs["formal_checkpoint"]["path"])
        checkpoint_before = sha256_file(checkpoint)
        if not torch.cuda.is_available():
            raise M2StagedGeometryError("M2 staged geometry 需要 CUDA")
        device = torch.device(device_name)
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        _, dataset, trainer = _runtime(config, device)
        from scripts.run_worldsim_v32_s2_3dgic import render_snapshot

        erase_delta = build_all_rigid_erase_delta(trainer.models["RigidNodes"])
        view_rows: list[dict[str, Any]] = []
        surface_cfg = config["surface"]
        gaussian_cfg = config["gaussianization"]
        view_cfg = config["view_protocol"]
        rest_shape = tuple(trainer.models["Background"]._features_rest.shape[1:])
        for view_index, mask_row in enumerate(mask_rows):
            frame = int(mask_row["frame"])
            camera = int(mask_row["camera_id"])
            row: dict[str, Any] = {"frame": frame, "camera_id": camera}
            try:
                base = render_snapshot(
                    trainer=trainer, dataset=dataset, frame=frame, camera_id=camera, device=device
                )
                original = _mask(mask_root, mask_row)
                target = binary_dilation(
                    original, iterations=int(view_cfg["target_mask_dilation_pixels"])
                )
                inner = binary_dilation(
                    target, iterations=int(view_cfg["support_ring_inner_pixels"])
                )
                outer = binary_dilation(
                    target, iterations=int(view_cfg["support_ring_outer_pixels"])
                )
                support = outer & ~inner & ~np.asarray(base["dynamic_mask"], bool)
                reference = np.asarray(base["background_depth"], dtype=np.float32)
                fit = fit_inverse_depth_surface(
                    depth=reference,
                    support_mask=support,
                    target_mask=target,
                    intrinsics=np.asarray(base["intrinsics"]),
                    model="G0_ROBUST_PLANE",
                    minimum_support_points=int(surface_cfg["minimum_support_points"]),
                    huber_delta=float(surface_cfg["huber_delta"]),
                    maximum_iterations=int(surface_cfg["maximum_iterations"]),
                    minimum_depth_m=float(surface_cfg["minimum_depth_m"]),
                    maximum_depth_m=float(surface_cfg["maximum_depth_m"]),
                )
                lidar = np.asarray(base["measured_lidar_depth"], dtype=np.float32)
                lidar_valid = (
                    support
                    & np.isfinite(lidar)
                    & (lidar > 1e-4)
                    & np.isfinite(reference)
                    & (reference > 1e-4)
                )
                lidar_count = int(lidar_valid.sum())
                lidar_mae = (
                    float(np.mean(np.abs(lidar[lidar_valid] - reference[lidar_valid])))
                    if lidar_count
                    else None
                )
                confidence = geometry_reference_confidence(
                    observed_reference_pixels=lidar_count,
                    target_pixels=int(target.sum()),
                    lidar_agreement_mae_m=lidar_mae,
                    agreement_scale_m=float(config["reference"]["confidence"]["agreement_scale_m"]),
                )
                inpaint = cv2.inpaint(
                    cv2.cvtColor(np.asarray(base["groundtruth"], np.uint8), cv2.COLOR_RGB2BGR),
                    target.astype(np.uint8) * 255,
                    3.0,
                    cv2.INPAINT_TELEA,
                )
                inpaint = cv2.cvtColor(inpaint, cv2.COLOR_BGR2RGB)
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
                    candidate_id=f"scene0471_f{frame:03d}_c{camera}_g0",
                    method="DONOR",
                    provenance="native_scene_donor",
                    features_rest_shape=rest_shape,
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
                        trainer=trainer,
                        dataset=dataset,
                        frame=frame,
                        camera_id=camera,
                        device=device,
                    )
                rollback = render_snapshot(
                    trainer=trainer, dataset=dataset, frame=frame, camera_id=camera, device=device
                )
                if not np.array_equal(rollback["rgb"], base["rgb"]):
                    raise M2StagedGeometryError("candidate composition rollback render 非 exact")
                post = np.asarray(rendered["depth"], dtype=np.float32)
                common = (
                    target
                    & np.isfinite(fit.depth)
                    & np.isfinite(pre)
                    & np.isfinite(post)
                    & np.isfinite(reference)
                    & (post > 1e-4)
                    & (reference > 1e-4)
                )
                metrics = staged_geometry_metrics(
                    raw_surface_depth=fit.depth,
                    pre_gaussianization_depth=pre,
                    post_gaussianization_render_depth=post,
                    reference_depth=reference,
                    evaluation_mask=common,
                )
                artifact = run_dir / "artifacts/views" / f"f{frame:03d}_c{camera}.npz"
                atomic_save_npz(
                    artifact,
                    {
                        "target_mask": target.astype(np.int8),
                        "support_mask": support.astype(np.int8),
                        "common_evaluation_mask": common.astype(np.int8),
                        "reference_depth": reference.astype(np.float16),
                        "raw_surface_depth": fit.depth.astype(np.float16),
                        "pre_gaussianization_depth": pre.astype(np.float16),
                        "post_gaussianization_render_depth": post.astype(np.float16),
                    },
                )
                row.update(
                    {
                        "status": "done",
                        "target_pixels": int(target.sum()),
                        "common_evaluation_pixels": int(common.sum()),
                        "reference": {
                            "reference_source": config["reference"]["source"],
                            "reference_confidence": confidence,
                            "observed_reference_pixels": lidar_count,
                            "lidar_agreement_mae_m": lidar_mae,
                            "independent_geometry_claim_allowed": False,
                        },
                        "raw_surface_model": fit.audit(),
                        "geometry_support": {
                            "support_pixels": int(support.sum()),
                            "fit_support_pixels": fit.support_count,
                        },
                        "lidar_support": {
                            "pixel_count": lidar_count,
                            "agreement_mae_m": lidar_mae,
                        },
                        "multi_view_depth_support": {
                            "available": False,
                            "reason": "not_collected_in_first_staged_contract_smoke",
                        },
                        "surface_fit_residual": {
                            "median_m": fit.surface_fit_residual_median_m,
                            "p90_m": fit.surface_fit_residual_p90_m,
                        },
                        "extrapolation_distance": {
                            "mean_normalized_image": fit.extrapolation_distance_mean,
                            "p95_normalized_image": fit.extrapolation_distance_p95,
                        },
                        "occlusion_uncertainty": 1.0 - confidence,
                        "staged_metrics": metrics,
                        "gaussian_count": int(points.means.shape[0]),
                        "composition_audit": composition,
                        "compatibility_asset_label": {
                            "method": "DONOR",
                            "provenance": "native_scene_donor",
                            "scientific_candidate": "GEOMETRY_FIRST_G0",
                        },
                        "artifact": {
                            "path": str(artifact.relative_to(run_dir)),
                            "sha256": sha256_file(artifact),
                        },
                    }
                )
            except GeometryRepairError as error:
                row.update({"status": "abstain", "reason": str(error)})
            view_rows.append(row)
            print(
                f"M2 staged geometry {view_index + 1}/{len(mask_rows)} "
                f"frame={frame} camera={camera} status={row['status']}",
                flush=True,
            )
        decision = mechanism_conclusion(view_rows, config["mechanism_gates"])
        diagnostics = {
            "schema_version": "worldsim_v5_m2_staged_geometry_diagnostics_v1",
            "task_id": TASK_ID,
            "status": "done",
            "scene": config["scene"]["name"],
            "reference_scope": "model_derived_proxy_not_independent_ground_truth",
            "views": view_rows,
            "mechanism_decision": decision,
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
            raise M2StagedGeometryError("formal checkpoint 被 staged diagnostic 修改")
        source_snapshot = copy_source_snapshot(
            run_dir,
            [
                config_path,
                PROJECT / "scripts/run_worldsim_v5_m2_staged_geometry_diagnostic.py",
                PROJECT / "scripts/worldsim_v5_forensics_common.py",
                PROJECT / "motion_proj/worldsim_v5/geometry_repair.py",
                PROJECT / "motion_proj/worldsim_v4/repair_assets.py",
                PROJECT / "motion_proj/worldsim_v4/repair_builders.py",
                PROJECT / "motion_proj/worldsim_v32/inpainting_adapter.py",
                PROJECT / "tests/test_worldsim_v5_geometry_repair.py",
                PROJECT / "tests/test_run_worldsim_v5_m2_staged_geometry_diagnostic.py",
            ],
            PROJECT,
        )
        done_rows = [row for row in view_rows if row["status"] == "done"]
        summary = {
            "schema_version": "worldsim_v5_m2_staged_geometry_summary_v1",
            "task_id": TASK_ID,
            "task_status": "running",
            "status": "done",
            "phase": config["phase"],
            "scene": config["scene"]["name"],
            "source_commit": source_head,
            "conclusion": decision["conclusion"],
            "reference_scope": "model_derived_proxy_not_independent_ground_truth",
            "view_count": len(view_rows),
            "evaluable_view_count": len(done_rows),
            "abstain_view_count": len(view_rows) - len(done_rows),
            "mechanism_decision": decision,
            "aggregate": {
                "raw_geometry_error": _aggregate(view_rows, "raw_geometry_error"),
                "pre_gaussianization_geometry_error": _aggregate(
                    view_rows, "pre_gaussianization_geometry_error"
                ),
                "post_gaussianization_render_error": _aggregate(
                    view_rows, "post_gaussianization_render_error"
                ),
                "representation_gap": _aggregate(view_rows, "representation_gap"),
            },
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
        fingerprint = {
            "schema_version": "worldsim_v5_m2_staged_geometry_fingerprint_v1",
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
            "source_snapshot": source_snapshot,
        }
        atomic_json(run_dir / "fingerprint.json", fingerprint)
        events.append({"event": "run_done", "at_utc": utc_now(), **decision})
        write_events(run_dir, events)
        manifest = {
            "schema_version": "worldsim_v5_m2_staged_geometry_manifest_v1",
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
