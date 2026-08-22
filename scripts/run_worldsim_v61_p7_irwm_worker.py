#!/usr/bin/env python3
"""WorldSim V6.1 P7：IR-WM 当前 Occupancy 的隔离 GPU worker。"""

from __future__ import annotations

import argparse
import hashlib
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_lidar2img(scene_root: Path, frame: int, camera_ids: list[int]) -> list[np.ndarray]:
    """按 DriveStudio camera-to-world / lidar-to-world 合同生成投影矩阵。"""
    t_global_lidar = np.loadtxt(scene_root / f"lidar_pose/{frame:03d}.txt")
    matrices = []
    for camera_id in camera_ids:
        t_global_camera = np.loadtxt(scene_root / f"extrinsics/{frame:03d}_{camera_id}.txt")
        camera_from_lidar = np.linalg.inv(t_global_camera) @ t_global_lidar
        values = np.loadtxt(scene_root / f"intrinsics/{camera_id}.txt").reshape(-1)
        fx, fy, cx, cy = values[:4]
        viewpad = np.eye(4, dtype=np.float64)
        viewpad[:3, :3] = np.asarray(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        matrices.append(viewpad @ camera_from_lidar)
    return matrices


def quaternion_yaw_degrees(values: list[float]) -> float:
    """nuScenes [w,x,y,z] quaternion 的 z-yaw，规范化到 [0,360)。"""
    w, x, y, z = (float(value) for value in values)
    yaw = math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    return yaw + 360.0 if yaw < 0.0 else yaw


def load_scene_infos(metadata_path: Path, scene: str, indices: list[int]) -> list[dict[str, Any]]:
    """只从官方 metadata 提取 CAN bus/ego pose，不读取 Occupancy annotation。"""
    with metadata_path.open("rb") as stream:
        payload = pickle.load(stream)
    scene_infos = payload["infos"][scene]
    selected = []
    for index in indices:
        info = scene_infos[index]
        can_bus = np.asarray(info["can_bus"], dtype=np.float64).copy()
        rotation = np.asarray(info["ego2global_rotation"], dtype=np.float64)
        translation = np.asarray(info["ego2global_translation"], dtype=np.float64)
        yaw_degrees = quaternion_yaw_degrees(rotation.tolist())
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        can_bus[-2] = math.radians(yaw_degrees)
        can_bus[-1] = yaw_degrees
        selected.append(
            {
                "can_bus_absolute": can_bus,
                "scene_token": str(info["scene_token"]),
                "timestamp": int(info["timestamp"]),
            }
        )
    return selected


def build_frame_metas(
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
                "lidar2img": build_lidar2img(scene_root, frame, camera_ids),
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))

    for record in plan["frozen_files"]:
        path = Path(record["path"])
        if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"冻结输入漂移: {path}")

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
    frames = [int(value) for value in plan["frames"]]
    metadata_indices = [int(value) for value in plan["metadata_indices"]]
    camera_ids = [int(value) for value in plan["camera_ids"]]
    images_by_frame: list[np.ndarray] = []
    for frame in frames:
        images = []
        for camera_id in camera_ids:
            image = mmcv.imread(str(scene_root / f"images/{frame:03d}_{camera_id}.jpg"), "unchanged")
            if image.shape[:2] != tuple(plan["native_shape"]):
                raise RuntimeError(f"原生图像尺寸漂移: frame={frame}, camera={camera_id}")
            image = mmcv.imnormalize(
                image.astype(np.float32),
                np.asarray(plan["image_mean_bgr"], dtype=np.float32),
                np.asarray(plan["image_std"], dtype=np.float32),
                to_rgb=False,
            )
            image = mmcv.impad_to_multiple(image, int(plan["pad_size_divisor"]), pad_val=0)
            images.append(image.transpose(2, 0, 1))
        images_by_frame.append(np.stack(images, axis=0))
    image_array = np.stack(images_by_frame, axis=0)
    padded_shape = (int(image_array.shape[-2]), int(image_array.shape[-1]), 3)
    imgs = torch.from_numpy(image_array[None]).cuda()

    infos = load_scene_infos(Path(plan["temporal_metadata_path"]), plan["scene"], metadata_indices)
    frame_metas = build_frame_metas(
        scene_root, plan["scene"], frames, camera_ids, infos, padded_shape
    )

    torch.cuda.synchronize()
    inference_started = time.monotonic()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        prev_bev, _ = model.obtain_history_bev(imgs[:, :-1], [frame_metas])
        ref_bev = model.obtain_ref_bev(imgs[:, -1], [frame_metas[-1]], prev_bev)
        decoder_layers = len(model.future_pred_head.bev_pred_head)
        current_bev = ref_bev.unsqueeze(0).repeat(decoder_layers, 1, 1, 1)
        logits = model.future_pred_head.forward_head(current_bev.unsqueeze(0))
    torch.cuda.synchronize()
    inference_seconds = time.monotonic() - inference_started

    expected_shape = tuple(int(value) for value in plan["raw_logits_shape"])
    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(f"IR-WM current logits 尺寸漂移: {tuple(logits.shape)} != {expected_shape}")
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("IR-WM current logits 非 finite")
    final_logits = logits[0, -1, 0]
    grid_h, grid_w, grid_d = (int(value) for value in plan["grid_shape"])
    class_label = final_logits.argmax(dim=-1).view(grid_w, grid_h, grid_d).transpose(0, 1)
    class_label = class_label.to(torch.uint8).cpu().numpy()
    occupied = class_label >= int(plan["occupied_class_min"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        class_label=class_label,
        occupied_mask=occupied,
        pc_range_m=np.asarray(plan["pc_range_m"], dtype=np.float32),
        grid_size_m=np.asarray(float(plan["grid_size_m"]), dtype=np.float32),
        input_frames=np.asarray(frames, dtype=np.int32),
        metadata_indices=np.asarray(metadata_indices, dtype=np.int32),
        target_lidar_pose=np.loadtxt(scene_root / f"lidar_pose/{frames[-1]:03d}.txt"),
    )
    unique, counts = np.unique(class_label, return_counts=True)
    report = {
        "schema_version": "worldsim_v61.p7_irwm_worker_report.v1",
        "scene": plan["scene"],
        "frames": frames,
        "history_frame_count": len(frames) - 1,
        "camera_count": len(camera_ids),
        "camera_ids": camera_ids,
        "model_load_missing_keys": list(incompatible.missing_keys),
        "model_load_unexpected_keys": list(incompatible.unexpected_keys),
        "raw_logits_shape": list(logits.shape),
        "class_label_shape": list(class_label.shape),
        "finite_logits": True,
        "class_histogram": {str(int(key)): int(value) for key, value in zip(unique, counts)},
        "occupied_voxel_count": int(occupied.sum()),
        "free_voxel_count": int(np.count_nonzero(class_label == int(plan["empty_class"]))),
        "current_state_extraction": "official_scene_encoder_plus_official_final_decoder_occupancy_head",
        "future_decoder_started": False,
        "planning_head_started": False,
        "occupancy_ground_truth_read": False,
        "o_method_or_o_eval_read": False,
        "training_started": False,
        "confirmation_content_read": False,
        "inference_seconds": inference_seconds,
        "wall_seconds": time.monotonic() - started,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "gpu_name": torch.cuda.get_device_name(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "environment_versions": {
            name: importlib.metadata.version(name)
            for name in ("mmcv-full", "mmdet", "mmsegmentation", "mmdet3d", "detectron2")
        },
        "output_sha256": sha256_file(args.output),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
