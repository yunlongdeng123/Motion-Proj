#!/usr/bin/env python3
"""在冻结 DriveStudio 环境中重复提取多帧三相机 logged-LiDAR support。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def _bundle_hash(values: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        value = np.ascontiguousarray(values[name])
        digest.update(name.encode() + b"\0" + str(value.dtype).encode() + b"\0")
        digest.update(json.dumps(value.shape).encode() + b"\0" + value.tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--streetgs-config", required=True, type=Path)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--camera-ids", required=True)
    parser.add_argument("--camera-downscale", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args()

    frames = [int(value) for value in args.frames.split(",")]
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

    def extract() -> dict[str, np.ndarray]:
        support: dict[str, np.ndarray] = {}
        for frame_index in frames:
            for camera_id in camera_ids:
                image_infos, camera_infos = dataset.full_image_set.get_image(
                    frame_index * len(camera_ids) + camera_id, args.camera_downscale
                )
                prefix = f"frame{frame_index:03d}_cam{camera_id}"
                support[f"{prefix}_lidar_depth"] = image_infos["lidar_depth_map"].detach().cpu().numpy().astype(np.float32)
                support[f"{prefix}_dynamic_mask"] = image_infos["dynamic_masks"].detach().cpu().numpy().astype(bool)
                support[f"{prefix}_intrinsics"] = camera_infos["intrinsics"].detach().cpu().numpy().astype(np.float32)
                support[f"{prefix}_camera_to_world"] = camera_infos["camera_to_world"].detach().cpu().numpy().astype(np.float32)
        return support

    first = extract()
    second = extract()
    first_hash = _bundle_hash(first)
    second_hash = _bundle_hash(second)
    np.savez_compressed(args.output, **first)
    args.audit.write_text(json.dumps({
        "schema_version": "worldsim_v6.r63_support_worker_audit.v1",
        "frame_indices": frames, "camera_ids": camera_ids,
        "array_count": len(first), "first_bundle_sha256": first_hash,
        "second_bundle_sha256": second_hash, "repeat_exact": first_hash == second_hash,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
