#!/usr/bin/env python3
"""在冻结 DriveStudio 环境中提取单帧三相机 logged-LiDAR support。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--streetgs-config", required=True, type=Path)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--frame-index", required=True, type=int)
    parser.add_argument("--camera-ids", required=True)
    parser.add_argument("--camera-downscale", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    camera_ids = [int(value) for value in args.camera_ids.split(",")]
    backup = args.streetgs_config.resolve().parent / "backup"
    sys.path.insert(0, str(args.repo_root.resolve()))
    sys.path.insert(0, str(backup))
    sys.path.append(str(args.upstream_root.resolve()))
    from datasets.driving_dataset import DrivingDataset
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(args.streetgs_config)
    cfg.data.preload_device = "cpu"
    dataset = DrivingDataset(data_cfg=cfg.data)
    support: dict[str, np.ndarray] = {}
    for camera_id in camera_ids:
        image_infos, camera_infos = dataset.full_image_set.get_image(
            args.frame_index * len(camera_ids) + camera_id, args.camera_downscale
        )
        prefix = f"cam{camera_id}"
        support[f"{prefix}_lidar_depth"] = image_infos["lidar_depth_map"].detach().cpu().numpy().astype(np.float32)
        support[f"{prefix}_dynamic_mask"] = image_infos["dynamic_masks"].detach().cpu().numpy().astype(bool)
        support[f"{prefix}_intrinsics"] = camera_infos["intrinsics"].detach().cpu().numpy().astype(np.float32)
        support[f"{prefix}_camera_to_world"] = camera_infos["camera_to_world"].detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(args.output, **support)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
