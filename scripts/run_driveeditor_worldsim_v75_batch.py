#!/usr/bin/env python3
"""DriveEditor 非交互批处理入口；仅在满足官方 GPU 配置后运行。"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--input-index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decoding-t", type=int, default=1)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    os.chdir(source_root)
    sys.path.insert(0, str(source_root))
    from interactive_gui import GradioShow, set_seed  # noqa: PLC0415

    input_index = json.loads(args.input_index.read_text(encoding="utf-8"))
    selected = set(args.case_id)
    input_rows = [
        row for row in input_index["cases"] if not selected or row["case_id"] in selected
    ]
    if not input_rows:
        raise RuntimeError("no DriveEditor cases selected")

    show = GradioShow(num_frames=10, step=args.steps)
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for index, input_row in enumerate(input_rows):
        with Path(input_row["pickle"]).open("rb") as handle:
            item = pickle.load(handle)[0]
        case_id = str(item["name"])
        operation = (
            "Deletion"
            if case_id.startswith("CFB-REMOVE")
            else "Insertion"
            if case_id.startswith("CFB-INSERT")
            else "Repositioning"
        )
        show.data = item
        show.camera_intrinsic = item["camera_intrinsic"]
        show.box, show.im, show.im_with_box, show.im_result = [], [], [], []
        show.selected_object_idx = 0
        for frame in item["data"]:
            image = cv2.resize(frame["im"], show.out_size[::-1], interpolation=cv2.INTER_LINEAR)
            if operation in ("Deletion", "Replacement"):
                box = frame["box"]
            else:
                box = frame[f"box_{operation}"]
            show.im.append(image)
            show.im_with_box.append(np.copy(image))
            show.box.append(box)
        set_seed(args.seed + index)
        temporary_video = show.predict(args.decoding_t, False, operation)
        case_root = args.output_root / case_id
        case_root.mkdir(parents=True, exist_ok=True)
        output_video = case_root / "counterfactual.mp4"
        shutil.copyfile(temporary_video, output_video)
        results.append(
            {
                "case_id": case_id,
                "operation": operation,
                "seed": args.seed + index,
                "counterfactual_video": str(output_video.resolve()),
            }
        )
    result_path = args.output_root / "run.json"
    result_path.write_text(
        json.dumps(
            {
                "schema_version": "worldsim_v75_driveeditor_run_v1",
                "model_execution_run": True,
                "results": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(result_path)


if __name__ == "__main__":
    main()
