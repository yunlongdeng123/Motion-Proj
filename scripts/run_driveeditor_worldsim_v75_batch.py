#!/usr/bin/env python3
"""DriveEditor 非交互批处理入口；仅在满足官方 GPU 配置后运行。"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def _operation(case_id: str) -> str:
    return (
        "Deletion"
        if case_id.startswith("CFB-REMOVE")
        else "Insertion"
        if case_id.startswith("CFB-INSERT")
        else "Repositioning"
    )


def _load_and_validate(input_row: dict) -> tuple[dict, str, dict]:
    pickle_path = Path(input_row["pickle"])
    with pickle_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise RuntimeError(f"{pickle_path}: expected one-item list containing a mapping")
    item = payload[0]
    case_id = str(item.get("name"))
    if case_id != str(input_row["case_id"]):
        raise RuntimeError(f"{pickle_path}: case id mismatch {case_id!r}")
    operation = _operation(case_id)
    if input_row.get("operation") != operation:
        raise RuntimeError(
            f"{pickle_path}: operation mismatch {input_row.get('operation')!r} != {operation!r}"
        )
    intrinsic = np.asarray(item.get("camera_intrinsic"))
    if intrinsic.shape != (3, 3):
        raise RuntimeError(f"{pickle_path}: camera intrinsic shape is {intrinsic.shape}")
    frames = item.get("data")
    expected_frames = int(input_row.get("frame_count", 10))
    if not isinstance(frames, list) or len(frames) != expected_frames:
        raise RuntimeError(f"{pickle_path}: expected {expected_frames} frames")
    for frame_index, frame in enumerate(frames):
        image = np.asarray(frame.get("im"))
        if image.shape != (900, 1600, 3) or image.dtype != np.uint8:
            raise RuntimeError(
                f"{pickle_path}: frame {frame_index} image is {image.shape}/{image.dtype}"
            )
        if frame.get("box") is None:
            raise RuntimeError(f"{pickle_path}: frame {frame_index} lacks factual box")
        if operation in ("Repositioning", "Insertion") and frame.get(f"box_{operation}") is None:
            raise RuntimeError(
                f"{pickle_path}: frame {frame_index} lacks box_{operation}"
            )
    reference = np.asarray(item.get("im"))
    mask = np.asarray(item.get("mask"))
    if reference.ndim != 3 or reference.shape[-1] != 3 or reference.dtype != np.uint8:
        raise RuntimeError(f"{pickle_path}: invalid reference image")
    if mask.shape != reference.shape[:2] or mask.dtype != np.uint8:
        raise RuntimeError(f"{pickle_path}: invalid reference mask")
    summary = {
        "case_id": case_id,
        "operation": operation,
        "pickle": str(pickle_path.resolve()),
        "frame_count": len(frames),
        "image_shape": list(np.asarray(frames[0]["im"]).shape),
        "reference_shape": list(reference.shape),
        "mask_shape": list(mask.shape),
    }
    return item, operation, summary


def _save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _video_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot open output video: {path}")
        count = 0
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            count += 1
        return count
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--input-index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decoding-t", type=int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只在当前 DriveEditor 环境反序列化并验证输入，不加载模型或调用 GPU",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    args.output_root = args.output_root.resolve()
    input_index = json.loads(args.input_index.read_text(encoding="utf-8"))
    selected = set(args.case_id)
    input_rows = [
        row for row in input_index["cases"] if not selected or row["case_id"] in selected
    ]
    if not input_rows:
        raise RuntimeError("no DriveEditor cases selected")
    found = {str(row["case_id"]) for row in input_rows}
    missing = sorted(selected - found)
    if missing:
        raise RuntimeError(f"requested DriveEditor cases are absent: {missing}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        validated = [_load_and_validate(row)[2] for row in input_rows]
        result_path = args.output_root / "dry-run.json"
        result_path.write_text(
            json.dumps(
                {
                    "schema_version": "worldsim_v75_driveeditor_dry_run_v1",
                    "model_execution_run": False,
                    "gpu_operations_run": False,
                    "numpy_version": np.__version__,
                    "validated_case_count": len(validated),
                    "cases": validated,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(result_path)
        return

    os.chdir(source_root)
    sys.path.insert(0, str(source_root))
    from interactive_gui import GradioShow, set_seed  # noqa: PLC0415

    show = GradioShow(num_frames=10, step=args.steps)
    results = []
    for index, input_row in enumerate(input_rows):
        case_id = str(input_row["case_id"])
        case_root = args.output_root / case_id
        case_root.mkdir(parents=True, exist_ok=True)
        output_video = case_root / "counterfactual.mp4"
        item_result = case_root / "result.json"
        if item_result.is_file():
            prior = json.loads(item_result.read_text(encoding="utf-8"))
            if (prior.get("status") == "generation_complete"
                    and prior.get("steps") == args.steps
                    and prior.get("seed") == args.seed + index
                    and output_video.is_file()
                    and _video_frame_count(output_video) == 10):
                print(json.dumps({"verified_skip": case_id}), flush=True)
                results.append(prior)
                continue
            raise RuntimeError(f"existing incomplete or mismatched result; preserve for audit: {item_result}")
        item, operation, _ = _load_and_validate(input_row)
        result = {
            "case_id": case_id, "approved_case_id": input_row.get("approved_case_id", case_id),
            "window_index": input_row.get("window_index"), "operation": operation,
            "seed": args.seed + index, "steps": args.steps,
            "sequential_cfg": os.environ.get("DRIVEEDITOR_SEQUENTIAL_CFG") == "1",
            "input_pickle": str(Path(input_row["pickle"]).resolve()),
            "counterfactual_video": str(output_video.resolve()), "status": "started",
        }
        _save_json(item_result, result)
        began = time.monotonic()
        try:
            show.data = item
            show.camera_intrinsic = item["camera_intrinsic"]
            show.box, show.im, show.im_with_box, show.im_result = [], [], [], []
            show.selected_object_idx = 0
            for frame in item["data"]:
                image = cv2.resize(frame["im"], show.out_size[::-1], interpolation=cv2.INTER_LINEAR)
                box = frame["box"] if operation in ("Deletion", "Replacement") else frame[f"box_{operation}"]
                show.im.append(image)
                show.im_with_box.append(np.copy(image))
                show.box.append(box)
            set_seed(args.seed + index)
            temporary_video = show.predict(args.decoding_t, False, operation)
            shutil.copyfile(temporary_video, output_video)
            decoded = _video_frame_count(output_video)
            if decoded != 10:
                raise RuntimeError(f"expected 10 decoded frames, got {decoded}")
            result.update(status="generation_complete", decoded_frames=decoded)
        except BaseException as exc:
            result.update(status="failed_stopped", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            result["wall_s"] = time.monotonic() - began
            _save_json(item_result, result)
        results.append(result)
        _save_json(args.output_root / "run.json", {
            "schema_version": "worldsim_v75_driveeditor_run_v2", "model_execution_run": True,
            "completed_windows": len(results), "requested_windows": len(input_rows), "results": results,
        })
        print(json.dumps({"window_complete": case_id, "wall_s": result["wall_s"]}), flush=True)
    result_path = args.output_root / "run.json"
    result_path.write_text(
        json.dumps(
            {
                "schema_version": "worldsim_v75_driveeditor_run_v2",
                "model_execution_run": True,
                "completed_windows": len(results),
                "requested_windows": len(input_rows),
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
