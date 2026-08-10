#!/usr/bin/env python
"""在 train-only 帧中只读搜索 S2 boundary 目标的跨视图支持。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from motion_proj.worldsim_v32.depth_guided_unseen_mask import splat_rgbd_to_target
from scripts.lift_worldsim_v32_semantics import build_runtime
from scripts.run_worldsim_v32_s2_3dgic import (
    load_binary_mask,
    render_snapshot,
)


def count_splat(source, target, mask, *, absolute, relative, minimum_opacity):
    valid = (
        ~source["dynamic_mask"]
        & ~source["egocar_mask"]
        & np.isfinite(source["background_depth"])
        & (source["background_depth"] > 1e-4)
        & (source["background_opacity"] >= minimum_opacity)
    )
    splat = splat_rgbd_to_target(
        source_depth=source["background_depth"],
        source_rgb=source["groundtruth"],
        source_valid=valid,
        source_intrinsics=source["intrinsics"],
        source_camera_to_world=source["camera_to_world"],
        target_depth=target["depth"],
        target_mask=mask,
        target_intrinsics=target["intrinsics"],
        target_camera_to_world=target["camera_to_world"],
        absolute_depth_tolerance_m=absolute,
        relative_depth_tolerance=relative,
        stride=1,
    )
    return int(splat.observed.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    torch.cuda.set_device(0)
    torch.cuda.init()
    device = torch.device("cuda:0")
    dataset, trainer = build_runtime(config, device)
    spec = config["targets"]["boundary_support"]
    target = render_snapshot(
        trainer=trainer,
        dataset=dataset,
        frame=int(spec["frame"]),
        camera_id=int(spec["camera_id"]),
        device=device,
        hide_actor=int(spec["rigid_model_index"]),
    )
    mask = load_binary_mask(Path(spec["mask"]))
    heldout = set(int(value) for value in config["scene"]["heldout_frames"])
    rows = []
    for frame in range(21, 40):
        if frame in heldout or frame == int(spec["frame"]):
            continue
        for camera_id in range(int(dataset.pixel_source.num_cams)):
            source = render_snapshot(
                trainer=trainer,
                dataset=dataset,
                frame=frame,
                camera_id=camera_id,
                device=device,
            )
            rows.append(
                {
                "frame": frame,
                "camera_id": camera_id,
                "strict": count_splat(
                    source,
                    target,
                    mask,
                    absolute=0.5,
                    relative=0.1,
                    minimum_opacity=0.5,
                ),
                "relaxed": count_splat(
                    source,
                    target,
                    mask,
                    absolute=2.0,
                    relative=0.3,
                    minimum_opacity=0.25,
                ),
                "geometric_overlap": count_splat(
                    source,
                    target,
                    mask,
                    absolute=100.0,
                    relative=10.0,
                    minimum_opacity=0.1,
                ),
                }
            )
            print(json.dumps(rows[-1]), flush=True)
    payload = {
        "target_frame": int(spec["frame"]),
        "camera_id": int(spec["camera_id"]),
        "target_mask_pixels": int(mask.sum()),
        "heldout_excluded": True,
        "rows": rows,
        "top_strict": sorted(rows, key=lambda row: row["strict"], reverse=True)[:12],
        "top_relaxed": sorted(rows, key=lambda row: row["relaxed"], reverse=True)[:12],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
