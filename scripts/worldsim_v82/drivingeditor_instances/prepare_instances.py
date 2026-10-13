#!/usr/bin/env python3
"""在新目录准备真实 Waymo rigid instances；只用 CPU，不改原输入。"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True

if __package__:
    from .data import prepare_scene
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data import prepare_scene


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path, help="Untouched existing prepared FRONT scene")
    parser.add_argument("--output", required=True, type=Path, help="New isolated output outside source scene")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--annotations", type=Path, help="Genuine per-frame laser_labels audit directory or consolidated JSON")
    source.add_argument("--tfrecord", type=Path, help="Actual raw Waymo TFRecord; labels/pose only, no LiDAR reextraction")
    parser.add_argument("--pb-root", type=Path, help="Existing pure protobuf Waymo package root")
    parser.add_argument("--crc-library", type=Path, help="Existing TFRecord CRC32C reader library")
    parser.add_argument("--waymo-reader", type=Path, help="Existing pure protobuf/NumPy waymo_numpy_cpu.py for isolated deployment")
    parser.add_argument("--normalization", type=Path, help="Original loader transform_matrix.txt; defaults inside scene")
    parser.add_argument("--selection", type=Path, help="Actual reviewed DEVA/SAM physical-track mapping; omissions remain mask_unknown")
    parser.add_argument("--track-id", action="append", help="Optional explicit raw vehicle track; default all vehicles, stationary included")
    parser.add_argument("--box-padding-m", type=float, default=0, help="Explicit OBB padding in physical meters")
    parser.add_argument("--scale-prior-m", type=float, default=.02, help="Only single-point Gaussian size prior; never a synthetic XYZ seed")
    args = parser.parse_args()
    report = prepare_scene(args.scene, args.output, annotations=args.annotations, tfrecord=args.tfrecord,
                           pb_root=args.pb_root, crc_library=args.crc_library, waymo_reader=args.waymo_reader, normalization_path=args.normalization,
                           selection_path=args.selection, instance_ids=args.track_id,
                           padding_m=args.box_padding_m, scale_prior_m=args.scale_prior_m)
    print(json.dumps({key: report[key] for key in ("output", "instance_count", "real_instance_points", "geometry_ready", "full_mask_supervision_ready",
                                                 "world_scale", "source_raw_window", "GPU_used", "old_assets_modified")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
