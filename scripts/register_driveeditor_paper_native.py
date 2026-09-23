#!/usr/bin/env python3
"""Register a verified DriveEditor 10-frame paper-native case video."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2


def video_info(path: Path) -> tuple[int, float, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = 0
    while capture.read()[0]:
        frames += 1
    capture.release()
    return frames, fps, width, height


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-index", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()
    index = json.loads(args.input_index.read_text(encoding="utf-8"))
    if index.get("window_stride_frames") != 10 or len(index["cases"]) != 1:
        raise RuntimeError("paper-native registration requires one non-overlapping 10-frame input")
    row = index["cases"][0]
    if row["approved_case_id"] != args.case_id or row["reference_scope"] != "window":
        raise RuntimeError("wrong approved case or reference scope")
    role = row["evaluation_role"]
    if role not in {"factual_identity_reconstruction", "counterfactual_edit"}:
        raise RuntimeError("unknown native evaluation role")
    source_root = args.run_root / row["case_id"]
    source_video = source_root / "counterfactual.mp4"
    result = json.loads((source_root / "result.json").read_text(encoding="utf-8"))
    if (result.get("status") != "generation_complete" or result.get("decoded_frames") != 10 or
            result.get("steps") != 25 or result.get("approved_case_id") != args.case_id or
            result.get("operation") != row["operation"] or result.get("paper_iterative_conditioning")):
        raise RuntimeError("native inference result does not match the approved 10-frame input")
    if video_info(source_video) != (10, 10.0, 1024, 576):
        raise RuntimeError("native video is not 10 frames at 10 Hz / 1024x576")
    target_root = args.output_root / args.case_id
    target_root.mkdir(parents=True, exist_ok=True)
    target_video = target_root / ("factual-reconstruction.mp4" if role == "factual_identity_reconstruction"
                                  else "counterfactual.mp4")
    if not target_video.exists():
        os.link(source_video, target_video)
    elif not os.path.samefile(source_video, target_video):
        raise RuntimeError(f"preserving differing native video: {target_video}")
    protocol = {
        "schema_version": "driveeditor_paper_native_10frame_v1",
        "approved_case_id": args.case_id,
        "evaluation_role": role,
        "operation": row["operation"],
        "reference_scope": row["reference_scope"],
        "reference_frame": row["reference_frame"],
        "window_index": row["window_index"],
        "window_source_frames": row["window_source_frames"],
        "decoded_frames": 10,
        "fps": 10,
        "source_video_sha256": sha256(source_video),
        "single_gpu_sequential_cfg": result["sequential_cfg"],
        "method_caveat": "DriveEditor object reconstruction/editing, not independent full-scene 3D reconstruction",
    }
    protocol_path = target_root / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise RuntimeError(f"preserving differing native protocol: {protocol_path}")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case_id": args.case_id, "role": role, "frames": 10,
                      "video": str(target_video)}))


if __name__ == "__main__":
    main()
