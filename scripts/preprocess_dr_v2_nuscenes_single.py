#!/usr/bin/env python
"""以单进程运行 DriveStudio 的精确 10 Hz nuScenes 转换器。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--scene-index", type=int, required=True)
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=Path("/root/autodl-tmp/third_party/drivestudio"),
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream_root))
    from datasets.nuscenes.nuscenes_preprocess import NuScenesProcessor

    processor = NuScenesProcessor(
        load_dir=args.data_root,
        save_dir=args.target_dir,
        split="v1.0-trainval",
        interpolate_N=4,
        process_keys=["images", "lidar", "calib", "dynamic_masks", "objects"],
        process_id_list=[args.scene_index],
        workers=1,
    )
    # 上游在 workers==1 时会转发到 convert_one() 并静默丢失插值；直接调用其
    # 插值实现既保持算法不变，也避免单场景额外启用多进程 worker。
    processor.convert_one_interpolated(args.scene_index)


if __name__ == "__main__":
    main()
