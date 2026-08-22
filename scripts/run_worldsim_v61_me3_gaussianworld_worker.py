#!/usr/bin/env python3
"""WorldSim V6.1 ME-3：单场景 GaussianWorld streaming GPU worker。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))

    for record in plan["frozen_scene_files"]:
        path = Path(record["path"])
        if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"冻结场景输入漂移: {path}")

    official_repo = Path(plan["official_repo"]).resolve()
    os.chdir(official_repo)
    sys.path.insert(0, str(official_repo))

    import mmcv
    import torch
    from mmengine import Config

    from dataset.transform_3d import ImageAug3D, NormalizeMultiviewImage, PadMultiViewImage
    from model import build_model

    seed = int(plan["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.set_device(int(plan["gpu"]))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()

    cfg = Config.fromfile(plan["official_config"])
    cfg.model.backbone.init_cfg.checkpoint = plan["backbone_path"]
    model = build_model(cfg.model)
    checkpoint = torch.load(plan["checkpoint_path"], map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if next(iter(state_dict)).startswith("module."):
        state_dict = OrderedDict((key[7:], value) for key, value in state_dict.items())
    incompatible = model.load_state_dict(state_dict, strict=False)
    model = model.cuda().eval()

    scene_root = Path(plan["scene_root"])
    camera_ids = [int(value) for value in plan["camera_ids"]]
    stream_frames = [int(value) for value in plan["stream_frames"]]
    target_frames = {int(value) for value in plan["target_frames"]}
    grid_shape = [int(value) for value in plan["grid_shape"]]
    dummy_label = torch.full(
        (1, 1, *grid_shape),
        int(plan["empty_class"]),
        dtype=torch.long,
        device="cuda",
    )
    history_anchor = None
    previous_lidar_pose: np.ndarray | None = None
    frame_rows: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    for frame in stream_frames:
        images = []
        for camera_id in camera_ids:
            image = mmcv.imread(
                str(scene_root / f"images/{frame:03d}_{camera_id}.jpg"), "unchanged"
            )
            if image.shape[:2] != tuple(plan["native_shape"]):
                raise RuntimeError(f"原生图像尺寸漂移: camera={camera_id}, frame={frame}")
            images.append(image.astype(np.float32))
        transformed: dict[str, Any] = {"img": images}
        for transform in (
            ImageAug3D(
                final_dim=plan["final_shape"],
                resize_lim=[1.0, 1.0],
                flip=False,
                is_train=False,
            ),
            NormalizeMultiviewImage(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True,
            ),
            PadMultiViewImage(size_divisor=32),
        ):
            transformed = transform(transformed)
        image_array = np.stack(
            [image.transpose(2, 0, 1) for image in transformed["img"]], axis=0
        )
        imgs = torch.from_numpy(image_array[None]).cuda()
        current_lidar_pose = np.loadtxt(scene_root / f"lidar_pose/{frame:03d}.txt")
        meta = {
            "scene_name": plan["scene"],
            "lidar2img": [build_lidar2img(scene_root, frame, camera_ids)],
            "img_aug_matrix": np.stack(transformed["img_aug_matrix"], axis=0)[None],
            "img_shape": transformed["img_shape"],
            "fill_num": -1,
        }
        if previous_lidar_pose is not None:
            meta["lidar2global"] = [previous_lidar_pose, current_lidar_pose]
        torch.cuda.synchronize()
        inference_started = time.monotonic()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            result = model(
                imgs=imgs, metas=[meta], label=dummy_label, history_anchor=history_anchor
            )
        torch.cuda.synchronize()
        inference_seconds = time.monotonic() - inference_started
        history_anchor = result["history_anchor"]
        previous_lidar_pose = current_lidar_pose
        logits = result["ce_input"]
        expected = (1, int(plan["class_count"]), *grid_shape)
        if tuple(logits.shape) != expected or not bool(torch.isfinite(logits).all()):
            raise RuntimeError(f"GaussianWorld 输出合同失败: {tuple(logits.shape)}")
        class_label = logits.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()
        unique, counts = np.unique(class_label, return_counts=True)
        row = {
            "frame": frame,
            "inference_seconds": inference_seconds,
            "history_input_present": frame != stream_frames[0],
            "history_output_present": history_anchor is not None,
            "class_histogram": {
                str(int(key)): int(value) for key, value in zip(unique, counts)
            },
        }
        frame_rows.append(row)
        if frame in target_frames:
            relative = Path(f"f{frame:03d}/GAUSSIANWORLD_CLASS.npz")
            output_path = args.output_dir / relative
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output_path,
                class_label=class_label,
                pc_range_m=np.asarray(plan["pc_range_m"], dtype=np.float32),
                grid_size_m=np.asarray(float(plan["grid_size_m"]), dtype=np.float32),
                lidar_pose=current_lidar_pose.astype(np.float64),
                input_frame=np.asarray(frame, dtype=np.int32),
            )
            outputs.append(
                {
                    "frame": frame,
                    "path": str(relative),
                    "sha256": sha256_file(output_path),
                    "occupied_voxel_count": int(
                        np.count_nonzero(
                            (class_label >= int(plan["occupied_class_min"]))
                            & (class_label <= int(plan["occupied_class_max"]))
                        )
                    ),
                    "empty_voxel_count": int(
                        np.count_nonzero(class_label == int(plan["empty_class"]))
                    ),
                    "noise_voxel_count": int(
                        np.count_nonzero(class_label == int(plan["noise_class"]))
                    ),
                }
            )
    if {row["frame"] for row in outputs} != target_frames:
        raise RuntimeError("target frame 输出不完整")
    report = {
        "schema_version": "worldsim_v61.me3_gaussianworld_worker_report.v1",
        "scene": plan["scene"],
        "stream_frames": stream_frames,
        "target_frames": sorted(target_frames),
        "camera_count": len(camera_ids),
        "model_load_missing_keys": list(incompatible.missing_keys),
        "model_load_unexpected_keys": list(incompatible.unexpected_keys),
        "frame_rows": frame_rows,
        "outputs": outputs,
        "dummy_label_role": "shape_only_not_truth",
        "surroundocc_label_read": False,
        "training_started": False,
        "confirmation_content_read": False,
        "wall_seconds": time.monotonic() - started,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "gpu_name": torch.cuda.get_device_name(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "environment_versions": {
            name: importlib.metadata.version(name)
            for name in ("mmcv", "mmdet", "mmsegmentation", "mmdet3d", "spconv-cu117")
        },
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
