#!/usr/bin/env python3
"""WorldSim V6.1 ME-3R：单场景两个 target 的 IR-WM GPU worker。"""

from __future__ import annotations

import argparse
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

from run_worldsim_v61_p7_irwm_worker import (
    build_frame_metas,
    load_scene_infos,
    sha256_file,
    write_json,
)


def load_images(
    mmcv: Any,
    scene_root: Path,
    frames: list[int],
    camera_ids: list[int],
    plan: dict[str, Any],
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """严格复用 P7 的 BGR normalization 与 padding。"""
    images_by_frame: list[np.ndarray] = []
    for frame in frames:
        images = []
        for camera_id in camera_ids:
            image = mmcv.imread(
                str(scene_root / f"images/{frame:03d}_{camera_id}.jpg"), "unchanged"
            )
            if image.shape[:2] != tuple(plan["native_shape"]):
                raise RuntimeError(
                    f"原生图像尺寸漂移: frame={frame}, camera={camera_id}"
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
    camera_ids = [int(value) for value in plan["camera_ids"]]
    target_rows: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    for target in plan["targets"]:
        frames = [int(value) for value in target["frames"]]
        metadata_indices = [int(value) for value in target["metadata_indices"]]
        image_array, padded_shape = load_images(
            mmcv, scene_root, frames, camera_ids, plan
        )
        imgs = torch.from_numpy(image_array[None]).cuda()
        infos = load_scene_infos(
            Path(plan["temporal_metadata_path"]), plan["scene"], metadata_indices
        )
        frame_metas = build_frame_metas(
            scene_root, plan["scene"], frames, camera_ids, infos, padded_shape
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
            raise RuntimeError(f"IR-WM current logits 合同失败: {tuple(logits.shape)}")
        final_logits = logits[0, -1, 0]
        grid_h, grid_w, grid_d = (int(value) for value in plan["grid_shape"])
        class_label = (
            final_logits.argmax(dim=-1)
            .view(grid_w, grid_h, grid_d)
            .transpose(0, 1)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        occupied = (class_label >= int(plan["occupied_class_min"])) & (
            class_label <= int(plan["occupied_class_max"])
        )
        target_frame = frames[-1]
        relative = Path(f"f{target_frame:03d}/IRWM_CLASS.npz")
        output_path = args.output_dir / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            class_label=class_label,
            occupied_mask=occupied,
            pc_range_m=np.asarray(plan["pc_range_m"], dtype=np.float32),
            grid_size_m=np.asarray(float(plan["grid_size_m"]), dtype=np.float32),
            input_frames=np.asarray(frames, dtype=np.int32),
            metadata_indices=np.asarray(metadata_indices, dtype=np.int32),
            target_lidar_pose=np.loadtxt(
                scene_root / f"lidar_pose/{target_frame:03d}.txt"
            ),
        )
        unique, counts = np.unique(class_label, return_counts=True)
        target_rows.append(
            {
                "target_frame": target_frame,
                "input_frames": frames,
                "metadata_indices": metadata_indices,
                "inference_seconds": inference_seconds,
                "class_histogram": {
                    str(int(key)): int(value) for key, value in zip(unique, counts)
                },
            }
        )
        outputs.append(
            {
                "target_frame": target_frame,
                "path": str(relative),
                "sha256": sha256_file(output_path),
                "occupied_voxel_count": int(occupied.sum()),
                "free_voxel_count": int(
                    np.count_nonzero(class_label == int(plan["empty_class"]))
                ),
            }
        )
        del imgs, ref_bev, current_bev, logits, final_logits
        torch.cuda.empty_cache()

    expected_targets = sorted(int(value) for value in plan["target_frames"])
    if sorted(row["target_frame"] for row in outputs) != expected_targets:
        raise RuntimeError("IR-WM target outputs 不完整")
    report = {
        "schema_version": "worldsim_v61.me3r_irwm_worker_report.v1",
        "scene": plan["scene"],
        "target_frames": expected_targets,
        "camera_count": len(camera_ids),
        "camera_ids": camera_ids,
        "model_load_missing_keys": list(incompatible.missing_keys),
        "model_load_unexpected_keys": list(incompatible.unexpected_keys),
        "target_rows": target_rows,
        "outputs": outputs,
        "current_state_extraction": "official_scene_encoder_plus_official_final_decoder_occupancy_head",
        "future_decoder_started": False,
        "planning_head_started": False,
        "occupancy_ground_truth_read": False,
        "o_method_or_o_eval_read": False,
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
            for name in (
                "mmcv-full", "mmdet", "mmsegmentation", "mmdet3d", "detectron2"
            )
        },
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
