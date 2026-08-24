#!/usr/bin/env python3
"""Materialize full native IR-WM logits and BEV latent for one scene."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import pickle
import platform
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _build_lidar2img(
    scene_root: Path, frame: int, camera_ids: list[int]
) -> list[np.ndarray]:
    global_from_lidar = np.loadtxt(scene_root / f"lidar_pose/{frame:03d}.txt")
    matrices = []
    for camera_id in camera_ids:
        global_from_camera = np.loadtxt(
            scene_root / f"extrinsics/{frame:03d}_{camera_id}.txt"
        )
        camera_from_lidar = np.linalg.inv(global_from_camera) @ global_from_lidar
        fx, fy, cx, cy = np.loadtxt(
            scene_root / f"intrinsics/{camera_id}.txt"
        ).reshape(-1)[:4]
        viewpad = np.eye(4, dtype=np.float64)
        viewpad[:3, :3] = np.asarray(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        matrices.append(viewpad @ camera_from_lidar)
    return matrices


def _quaternion_yaw_degrees(values: list[float]) -> float:
    w, x, y, z = (float(value) for value in values)
    yaw = math.degrees(
        math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    )
    return yaw + 360.0 if yaw < 0.0 else yaw


def _load_scene_infos(
    metadata_path: Path, scene: str, indices: list[int]
) -> dict[int, dict[str, Any]]:
    with metadata_path.open("rb") as stream:
        payload = pickle.load(stream)
    scene_infos = payload["infos"][scene]
    selected = {}
    for index in sorted(set(indices)):
        info = scene_infos[index]
        can_bus = np.asarray(info["can_bus"], dtype=np.float64).copy()
        rotation = np.asarray(info["ego2global_rotation"], dtype=np.float64)
        translation = np.asarray(info["ego2global_translation"], dtype=np.float64)
        yaw_degrees = _quaternion_yaw_degrees(rotation.tolist())
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        can_bus[-2] = math.radians(yaw_degrees)
        can_bus[-1] = yaw_degrees
        selected[index] = {
            "can_bus_absolute": can_bus,
            "scene_token": str(info["scene_token"]),
            "timestamp": int(info["timestamp"]),
        }
    return selected


def _build_frame_metas(
    scene_root: Path,
    scene: str,
    frames: list[int],
    camera_ids: list[int],
    infos: list[dict[str, Any]],
    padded_shape: tuple[int, int, int],
) -> list[dict[str, Any]]:
    metas = []
    previous_position: np.ndarray | None = None
    previous_angle: float | None = None
    for frame, info in zip(frames, infos):
        can_bus = info["can_bus_absolute"].copy()
        absolute_position = can_bus[:3].copy()
        absolute_angle = float(can_bus[-1])
        if previous_position is None:
            can_bus[:3] = 0.0
            can_bus[-1] = 0.0
            prev_bev_exists = False
        else:
            can_bus[:3] = absolute_position - previous_position
            can_bus[-1] = absolute_angle - float(previous_angle)
            prev_bev_exists = True
        previous_position = absolute_position
        previous_angle = absolute_angle
        lidar_pose = np.loadtxt(scene_root / f"lidar_pose/{frame:03d}.txt")
        metas.append(
            {
                "scene_name": scene,
                "scene_token": info["scene_token"],
                "timestamp": info["timestamp"],
                "lidar2img": _build_lidar2img(scene_root, frame, camera_ids),
                "lidar2global_rotation": lidar_pose[:3, :3],
                "can_bus": can_bus,
                "prev_bev_exists": prev_bev_exists,
                "img_shape": [padded_shape for _ in camera_ids],
                "ori_shape": [padded_shape for _ in camera_ids],
                "pad_shape": [padded_shape for _ in camera_ids],
                "flip": False,
                "box_mode_3d": None,
                "box_type_3d": None,
            }
        )
    return metas


def _load_images(
    mmcv: Any,
    scene_root: Path,
    frames: list[int],
    camera_ids: list[int],
    plan: dict[str, Any],
) -> tuple[np.ndarray, tuple[int, int, int]]:
    images_by_frame = []
    for frame in frames:
        images = []
        for camera_id in camera_ids:
            image = mmcv.imread(
                str(scene_root / f"images/{frame:03d}_{camera_id}.jpg"), "unchanged"
            )
            if image.shape[:2] != tuple(plan["native_shape"]):
                raise RuntimeError(
                    f"native image shape differs at frame={frame}, camera={camera_id}: {image.shape}"
                )
            image = mmcv.imnormalize(
                image.astype(np.float32),
                np.asarray(plan["image_mean_bgr"], dtype=np.float32),
                np.asarray(plan["image_std"], dtype=np.float32),
                to_rgb=False,
            )
            image = mmcv.impad_to_multiple(
                image, int(plan["pad_size_divisor"]), pad_val=0
            )
            images.append(image.transpose(2, 0, 1))
        images_by_frame.append(np.stack(images, axis=0))
    image_array = np.stack(images_by_frame, axis=0)
    return image_array, (int(image_array.shape[-2]), int(image_array.shape[-1]), 3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))

    official_repo = Path(plan["official_repo"]).resolve()
    os.chdir(official_repo)
    sys.path.insert(0, str(official_repo))

    import mmcv
    import torch
    from mmcv import Config
    from mmdet3d.models import build_model

    import projects.mmdet3d_plugin  # noqa: F401

    torch.manual_seed(int(plan["seed"]))
    np.random.seed(int(plan["seed"]))
    torch.cuda.set_device(int(plan["gpu"]))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()

    cfg = Config.fromfile(plan["official_config"])
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg"))
    checkpoint = torch.load(plan["checkpoint_path"], map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if next(iter(state_dict)).startswith("module."):
        state_dict = OrderedDict((key[7:], value) for key, value in state_dict.items())
    incompatible = model.load_state_dict(state_dict, strict=False)
    model = model.cuda().eval()

    scene_root = Path(plan["scene_root"])
    camera_ids = [int(value) for value in plan["camera_ids"]]
    all_metadata_indices = [
        int(index) for target in plan["targets"] for index in target["metadata_indices"]
    ]
    scene_infos = _load_scene_infos(
        Path(plan["temporal_metadata_path"]), plan["scene"], all_metadata_indices
    )
    target_rows = []
    output_rows = []
    for target in plan["targets"]:
        frames = [int(value) for value in target["frames"]]
        metadata_indices = [int(value) for value in target["metadata_indices"]]
        image_array, padded_shape = _load_images(
            mmcv, scene_root, frames, camera_ids, plan
        )
        imgs = torch.from_numpy(image_array[None]).cuda()
        frame_metas = _build_frame_metas(
            scene_root,
            plan["scene"],
            frames,
            camera_ids,
            [scene_infos[index] for index in metadata_indices],
            padded_shape,
        )

        torch.cuda.synchronize()
        inference_started = time.monotonic()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            prev_bev, _ = model.obtain_history_bev(imgs[:, :-1], [frame_metas])
            ref_bev = model.obtain_ref_bev(
                imgs[:, -1], [frame_metas[-1]], prev_bev
            )
            decoder_layers = len(model.future_pred_head.bev_pred_head)
            current_bev = ref_bev.unsqueeze(0).repeat(decoder_layers, 1, 1, 1)
            logits = model.future_pred_head.forward_head(current_bev.unsqueeze(0))
        torch.cuda.synchronize()
        inference_seconds = time.monotonic() - inference_started

        expected_shape = tuple(int(value) for value in plan["raw_logits_shape"])
        if tuple(logits.shape) != expected_shape or not bool(torch.isfinite(logits).all()):
            raise RuntimeError(f"IR-WM current logits contract failed: {tuple(logits.shape)}")
        grid_x, grid_y, grid_z = (int(value) for value in plan["grid_shape"])
        final_logits = logits[0, -1, 0]
        logits_grid = (
            final_logits.view(grid_y, grid_x, grid_z, int(plan["class_count"]))
            .transpose(0, 1)
            .contiguous()
        )
        probabilities = logits_grid.float().softmax(dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        top2 = probabilities.topk(k=2, dim=-1).values
        margin = top2[..., 0] - top2[..., 1]
        argmax = probabilities.argmax(dim=-1)
        latent_dimension = int(ref_bev.shape[-1])
        bev_grid = (
            ref_bev[0]
            .view(grid_y, grid_x, latent_dimension)
            .transpose(0, 1)
            .contiguous()
        )

        target_frame = int(target["target_frame"])
        unit_dir = args.output_dir / f"f{target_frame:03d}"
        unit_dir.mkdir(parents=True, exist_ok=False)
        arrays = {
            "NATIVE_LOGITS.npy": logits_grid.to(torch.float16).cpu().numpy(),
            "BEV_LATENT.npy": bev_grid.to(torch.float16).cpu().numpy(),
            "ARGMAX.npy": argmax.to(torch.uint8).cpu().numpy(),
            "ENTROPY.npy": entropy.to(torch.float16).cpu().numpy(),
            "MARGIN.npy": margin.to(torch.float16).cpu().numpy(),
            "SOURCE_VALID.npy": np.ones((grid_x, grid_y, grid_z), dtype=bool),
        }
        for filename, array in arrays.items():
            np.save(unit_dir / filename, array, allow_pickle=False)
        meta = {
            "schema_version": "worldsim_v63.native_sidecar.v1",
            "scene": plan["scene"],
            "target_frame": target_frame,
            "input_frames": frames,
            "metadata_indices": metadata_indices,
            "native_grid_shape": [grid_x, grid_y, grid_z],
            "source_origin_m": plan["source_origin_m"],
            "source_voxel_size_m": plan["source_voxel_size_m"],
            "class_count": int(plan["class_count"]),
            "latent_dimension": latent_dimension,
            "target_lidar_pose": np.loadtxt(
                scene_root / f"lidar_pose/{target_frame:03d}.txt"
            ).tolist(),
            "source_grid_identity": "irwm_current_native_200x200x16",
            "prototype_used": False,
            "target_evidence_read": False,
        }
        _write_json(unit_dir / "META.json", meta)
        bytes_written = sum(path.stat().st_size for path in unit_dir.iterdir())
        reloaded_logits = np.load(unit_dir / "NATIVE_LOGITS.npy", mmap_mode="r")
        reloaded_bev = np.load(unit_dir / "BEV_LATENT.npy", mmap_mode="r")
        if reloaded_logits.shape != (grid_x, grid_y, grid_z, int(plan["class_count"])):
            raise RuntimeError("fresh-process-compatible logits shape failed")
        if reloaded_bev.shape != (grid_x, grid_y, latent_dimension):
            raise RuntimeError("fresh-process-compatible BEV shape failed")
        target_rows.append(
            {
                "target_frame": target_frame,
                "input_frames": frames,
                "metadata_indices": metadata_indices,
                "inference_seconds": inference_seconds,
                "bytes": bytes_written,
                "native_logits_shape": list(reloaded_logits.shape),
                "bev_latent_shape": list(reloaded_bev.shape),
                "finite": bool(np.isfinite(reloaded_logits).all())
                and bool(np.isfinite(reloaded_bev).all()),
                "prototype_used": False,
            }
        )
        output_rows.append(
            {
                "target_frame": target_frame,
                "path": f"f{target_frame:03d}",
                "bytes": bytes_written,
            }
        )
        del (
            imgs,
            prev_bev,
            ref_bev,
            current_bev,
            logits,
            final_logits,
            logits_grid,
            probabilities,
            entropy,
            margin,
            argmax,
            bev_grid,
        )
        torch.cuda.empty_cache()

    report = {
        "schema_version": "worldsim_v63.p2_native_scene_report.v1",
        "scene": plan["scene"],
        "partition": plan["partition"],
        "target_count": len(target_rows),
        "target_rows": target_rows,
        "outputs": output_rows,
        "backend_identity": plan["backend_identity"],
        "current_state_extraction": "official_scene_encoder_final_decoder_occupancy_head_and_ref_bev",
        "model_load_missing_keys": list(incompatible.missing_keys),
        "model_load_unexpected_keys": list(incompatible.unexpected_keys),
        "prototype_used": False,
        "target_evidence_read": False,
        "occupancy_ground_truth_read": False,
        "training_started": False,
        "future_decoder_started": False,
        "planning_head_started": False,
        "calibration_quality_read": False,
        "confirmation_content_read": False,
        "exact_once_test_read": False,
        "wall_seconds": time.monotonic() - started,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "gpu_name": torch.cuda.get_device_name(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "environment_versions": {
            name: importlib.metadata.version(name)
            for name in (
                "mmcv-full",
                "mmdet",
                "mmsegmentation",
                "mmdet3d",
                "detectron2",
            )
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.report, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
