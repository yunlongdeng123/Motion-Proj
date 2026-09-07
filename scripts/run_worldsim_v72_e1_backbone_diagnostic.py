"""E1：在相同 RGB 与标定上运行两个官方基座并保存分层诊断。"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_proj.worldsim_v72.data.nuscenes_camera import NuScenesCameraIndex
from motion_proj.worldsim_v72.data.splits import load_data_roles, require_role_access
from motion_proj.worldsim_v72.eas_vggt.alignment import (
    aligned_camera_center_rmse_m,
    common_surface_mask,
    nearest_pixel_indices,
)
from motion_proj.worldsim_v72.eas_vggt.backbones import Pi3XBackbone, VGGTBackbone
from motion_proj.worldsim_v72.eas_vggt.cache import save_backbone_geometry
from motion_proj.worldsim_v72.eas_vggt.preprocess import resize_camera_window


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _git_commit(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def _pairwise_scale_stress(predicted: np.ndarray, calibrated: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=np.float64)
    calibrated = np.asarray(calibrated, dtype=np.float64)
    pred_distances: list[float] = []
    gt_distances: list[float] = []
    for left in range(len(predicted)):
        for right in range(left + 1, len(predicted)):
            pred_distances.append(float(np.linalg.norm(predicted[left] - predicted[right])))
            gt_distances.append(float(np.linalg.norm(calibrated[left] - calibrated[right])))
    pred = np.asarray(pred_distances)
    gt = np.asarray(gt_distances)
    scale = float(np.dot(pred, gt) / max(float(np.dot(pred, pred)), 1.0e-12))
    return float(np.sqrt(np.mean((scale * pred - gt) ** 2)))


def _projected_lidar_metrics(
    points_world: np.ndarray,
    window: Any,
    geometry: Any,
    aligned_points_world: np.ndarray,
    surface_mask: np.ndarray,
    *,
    depth_tolerance_m: float,
    maximum_lidar_points: int,
) -> dict[str, Any]:
    points_world = np.asarray(points_world, dtype=np.float64)
    if len(points_world) > maximum_lidar_points:
        indices = np.linspace(0, len(points_world) - 1, maximum_lidar_points, dtype=np.int64)
        points_world = points_world[indices]
    homogeneous = np.concatenate([points_world, np.ones((len(points_world), 1))], axis=1)
    residuals: list[np.ndarray] = []
    depth_errors: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    for index, frame in enumerate(window.frames):
        camera_from_world = np.linalg.inv(frame.world_from_camera_opencv)
        camera_points = homogeneous @ camera_from_world.T
        positive = camera_points[:, 2] > 0.5
        projected = camera_points[:, :3] @ frame.intrinsics_px.T
        uv = projected[:, :2] / np.maximum(projected[:, 2:3], 1.0e-8)
        width, height = map(int, frame.original_size_wh)
        inside = positive & (uv[:, 0] >= 0.0) & (uv[:, 0] < width) & (uv[:, 1] >= 0.0) & (uv[:, 1] < height)
        if not np.any(inside):
            continue
        original_uv1 = np.concatenate([uv[inside], np.ones((inside.sum(), 1))], axis=1)
        model_uv = original_uv1 @ geometry.model_from_original_px[index].T
        x = np.clip(np.rint(model_uv[:, 0]).astype(np.int64), 0, geometry.points_reference.shape[2] - 1)
        y = np.clip(np.rint(model_uv[:, 1]).astype(np.int64), 0, geometry.points_reference.shape[1] - 1)
        # 同一相机像素只保留最近的 LiDAR 投影，避免把后方点误算为可见表面。
        camera_depth = camera_points[inside, 2]
        visible = nearest_pixel_indices(
            x,
            y,
            camera_depth,
            image_width=geometry.points_reference.shape[2],
        )
        x = x[visible]
        y = y[visible]
        keep = surface_mask[index, y, x]
        if not np.any(keep):
            continue
        lidar_world = points_world[inside][visible][keep]
        predicted_world = aligned_points_world[index, y[keep], x[keep]].astype(np.float64)
        lidar_camera = np.concatenate([lidar_world, np.ones((len(lidar_world), 1))], axis=1) @ camera_from_world.T
        predicted_camera = np.concatenate([predicted_world, np.ones((len(predicted_world), 1))], axis=1) @ camera_from_world.T
        residuals.append(np.linalg.norm(predicted_world - lidar_world, axis=1))
        depth_errors.append(predicted_camera[:, 2] - lidar_camera[:, 2])
        confidences.append(geometry.confidence[index, y[keep], x[keep]])
    if not residuals:
        return {"correspondence_count": 0}
    residual = np.concatenate(residuals)
    depth_error = np.concatenate(depth_errors)
    confidence = np.concatenate(confidences)
    early = depth_error < -depth_tolerance_m
    hit = np.abs(depth_error) <= depth_tolerance_m
    late = depth_error > depth_tolerance_m
    return {
        "correspondence_count": int(len(residual)),
        "median_surface_residual_m": float(np.median(residual)),
        "p90_surface_residual_m": float(np.quantile(residual, 0.9)),
        "median_absolute_depth_error_m": float(np.median(np.abs(depth_error))),
        "early_rate": float(np.mean(early)),
        "hit_rate": float(np.mean(hit)),
        "late_rate": float(np.mean(late)),
        "median_retained_confidence": float(np.median(confidence)),
    }


def _make_backbone(name: str, config: Mapping[str, Any]) -> Any:
    common = {
        "repository_root": Path(config["repository_root"]),
        "checkpoint": Path(config["checkpoint"]),
        "device": str(config.get("device", "cuda")),
    }
    if name == "vggt":
        return VGGTBackbone(**common)
    if name == "pi3x":
        return Pi3XBackbone(**common)
    raise ValueError(f"未知基座: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["task_id"] != "WS-V72-E1-VGGT-EVIDENCE-IO-01":
        raise ValueError("E1 config task_id 不匹配")
    run_dir = Path(config["runs_root"]) / "worldsim_v72" / config["task_id"] / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {
        "schema_version": "worldsim_v72.e1_backbone_diagnostic.v1",
        "task_id": config["task_id"],
        "run_id": args.run_id,
        "status": "running",
        "git_commit": _git_commit(REPO_ROOT),
        "failure_ledger_refs": list(config["failure_ledger_refs"]),
        "failure_ledger_delta": "pending",
        "role": str(config["data"]["role"]),
        "selection_uses_quality": False,
        "target_access": False,
        "source_test_read": False,
        "external_test_read": False,
        "backbones": list(config["backbones"]),
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "data_index"})
    try:
        roles = load_data_roles(Path(config["data"]["roles"]))
        role = str(config["data"]["role"])
        log_ids = require_role_access(roles, "nuscenes", role)
        index = NuScenesCameraIndex(Path(config["data"]["dataset_root"]))
        windows = list(
            index.iter_windows(
                log_ids,
                role=role,
                camera_channels=config["data"]["camera_channels"],
                maximum_windows=int(config["data"]["maximum_windows"]),
            )
        )
        if not windows:
            raise RuntimeError("冻结角色中没有 payload 完整的多相机窗口")
        rows: list[dict[str, Any]] = []
        for window in windows:
            images, image_transforms = resize_camera_window(
                window,
                maximum_pixels=int(config["preprocess"]["maximum_pixels"]),
                patch_multiple=int(config["preprocess"]["patch_multiple"]),
            )
            lidar_world = index.sensor_points_world(window.frames[0].sample_id)
            calibrated_poses = np.stack([frame.world_from_camera_opencv for frame in window.frames])
            for name, backbone_config in config["backbones"].items():
                _write_json(run_dir / "status.json", {"status": "running", "phase": f"infer_{name}", "window_id": window.window_id})
                torch.cuda.reset_peak_memory_stats()
                inference_started = time.monotonic()
                geometry = _make_backbone(name, backbone_config).infer(window, images, image_transforms)
                inference_seconds = time.monotonic() - inference_started
                cache_path = run_dir / "cache" / name / f"{window.window_id}.npz"
                save_backbone_geometry(geometry, cache_path)
                predicted_centers = geometry.reference_from_camera_opencv[:, :3, 3]
                calibrated_centers = calibrated_poses[:, :3, 3]
                similarity, camera_rmse = aligned_camera_center_rmse_m(
                    geometry.reference_from_camera_opencv, calibrated_poses
                )
                aligned_points = similarity.apply(geometry.points_reference)
                surface_mask = common_surface_mask(
                    geometry.confidence,
                    geometry.valid_mask,
                    confidence_quantile=float(config["diagnostic"]["confidence_quantile"]),
                )
                lidar_metrics = _projected_lidar_metrics(
                    lidar_world,
                    window,
                    geometry,
                    aligned_points,
                    surface_mask,
                    depth_tolerance_m=float(config["diagnostic"]["depth_tolerance_m"]),
                    maximum_lidar_points=int(config["diagnostic"]["maximum_lidar_points"]),
                )
                rows.append(
                    {
                        "backbone": name,
                        "backbone_id": geometry.backbone_id,
                        "window_id": window.window_id,
                        "window_fingerprint": window.fingerprint,
                        "log_id": window.log_id,
                        "scene_id": window.scene_id,
                        "camera_count": len(window.frames),
                        "model_height": int(geometry.points_reference.shape[1]),
                        "model_width": int(geometry.points_reference.shape[2]),
                        "native_scale_status": geometry.scale_status,
                        "native_valid_fraction": float(np.mean(geometry.valid_mask)),
                        "native_pairwise_camera_stress_m": _pairwise_scale_stress(predicted_centers, calibrated_centers),
                        "alignment_scale": float(similarity.scale),
                        "aligned_camera_center_rmse_m": camera_rmse,
                        "surface_retained_fraction": float(np.mean(surface_mask)),
                        "feature_shape": list(geometry.feature_grid.shape),
                        "inference_seconds": inference_seconds,
                        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
                        "cache_path": str(cache_path),
                        "lidar_observation_diagnostic": lidar_metrics,
                    }
                )
                del geometry, aligned_points, surface_mask
                gc.collect()
                torch.cuda.empty_cache()
        _write_jsonl(run_dir / "metrics.jsonl", rows)
        summary = {
            **manifest,
            "status": "done",
            "failure_ledger_delta": "none",
            "window_count": len(windows),
            "window_ids": [window.window_id for window in windows],
            "window_fingerprints": [window.fingerprint for window in windows],
            "rows": rows,
            "metric_contract": dict(config["diagnostic"]),
            "resources": {
                "wall_seconds": time.monotonic() - started,
                "peak_process_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
            },
        }
        _write_json(run_dir / "summary.json", summary)
        _write_json(run_dir / "manifest.json", {**manifest, "status": "done", "failure_ledger_delta": "none", "summary": "summary.json"})
        _write_json(run_dir / "status.json", {"status": "done", "phase": "complete"})
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    except Exception as error:
        _write_json(run_dir / "status.json", {"status": "failed", "phase": "runtime", "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
