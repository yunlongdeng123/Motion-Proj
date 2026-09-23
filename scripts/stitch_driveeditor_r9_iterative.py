#!/usr/bin/env python3
"""Assemble an audited 100-frame DriveEditor iterative-condition chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import cv2
import imageio.v2 as imageio


def video_info(path: Path) -> tuple[int, float, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot decode {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    return count, fps, width, height


def last_rgb_frame_sha256(path: Path) -> str:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot decode previous segment: {path}")
    last = None
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        last = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cap.release()
    if last is None:
        raise RuntimeError(f"no previous segment frame: {path}")
    return hashlib.sha256(last.tobytes()).hexdigest()


def link_once(source: Path, target: Path) -> None:
    if not target.exists():
        os.link(source, target)
    elif not os.path.samefile(source, target):
        raise RuntimeError(f"preserving differing existing file: {target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--input-index", type=Path, required=True)
    parser.add_argument("--window-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()
    manifest = json.loads((args.manifest_root / "candidates.json").read_text(encoding="utf-8"))
    row = next((r for r in manifest["cases"] if r["case_id"] == args.case_id), None)
    if row is None or row["approval"]["status"] != "approved":
        raise RuntimeError("case not in approved manifest")
    inputs = json.loads(args.input_index.read_text(encoding="utf-8"))
    if inputs["schema_version"] != "driveeditor_r9_iterative_overlap_v1":
        raise RuntimeError("not a one-frame-overlap iterative input index")
    windows = [r for r in inputs["cases"] if r["approved_case_id"] == args.case_id]
    if len(windows) != 11 or [r["window_index"] for r in windows] != list(range(11)):
        raise RuntimeError("11 contiguous windows are required for exactly 100 output frames")
    if any(a["window_source_frames"][-1] != b["window_source_frames"][0]
           for a, b in zip(windows, windows[1:])):
        raise RuntimeError("windows do not overlap by exactly one source frame")
    parts = []
    for index, window in enumerate(windows):
        root = args.window_root / window["case_id"]
        video = root / "counterfactual.mp4"
        result = json.loads((root / "result.json").read_text(encoding="utf-8"))
        if result.get("status") != "generation_complete" or result.get("steps") != 25:
            raise RuntimeError(f"unverified native window: {video}")
        if result.get("paper_iterative_conditioning") is not True:
            raise RuntimeError(f"window was not in iterative mode: {video}")
        if result.get("used_previous_segment_condition") != (index > 0):
            raise RuntimeError(f"wrong previous-frame conditioning state: {video}")
        if index and not result.get("previous_generated_last_frame_sha256"):
            raise RuntimeError(f"previous generated condition was not hashed: {video}")
        if index and result["previous_generated_last_frame_sha256"] != last_rgb_frame_sha256(parts[-1]):
            raise RuntimeError(f"previous generated condition does not match preceding segment: {video}")
        if video_info(video) != (10, 10.0, 1024, 576):
            raise RuntimeError(f"invalid native 10-frame video: {video}")
        parts.append(video)

    case_root = args.output_root / args.case_id
    case_root.mkdir(parents=True, exist_ok=True)
    result_path = case_root / "result.json"
    if result_path.exists():
        raise RuntimeError(f"preserving existing assembled result: {result_path}")
    original = args.manifest_root / "cases" / args.case_id / "original-nuscenes.mp4"
    if video_info(original) != (100, 10.0, 1600, 900):
        raise RuntimeError("approved original video is not 100 frames at 10 Hz")
    link_once(original, case_root / "original-nuscenes.mp4")
    link_once(original, case_root / "factual.mp4")
    final = case_root / "counterfactual.mp4"
    if final.exists():
        raise RuntimeError(f"preserving existing counterfactual: {final}")
    frame_to_segment = []
    with imageio.get_writer(str(final), fps=10, codec="libx264", macro_block_size=None,
                            ffmpeg_params=["-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"]) as writer:
        for segment_index, video in enumerate(parts):
            cap = cv2.VideoCapture(str(video))
            native_index = 0
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                if segment_index == 0 or native_index > 0:
                    writer.append_data(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    frame_to_segment.append({"segment": segment_index, "native_frame": native_index})
                native_index += 1
            cap.release()
            if native_index != 10:
                raise RuntimeError(f"segment changed during assembly: {video}")
    if len(frame_to_segment) != 100 or video_info(final) != (100, 10.0, 1024, 576):
        raise RuntimeError("iterative 100-frame assembly verification failed")
    shutil.copyfile(final, case_root / "counterfactual-raw.mp4")
    result = {
        "schema_version": "driveeditor_r9_paper_style_iterative_v1",
        "case_id": args.case_id,
        "status": "generation_complete",
        "paper_iterative_conditioning": True,
        "generated_frames": 100,
        "native_segments": 11,
        "native_frames_per_segment": 10,
        "overlap_frames_per_transition": 1,
        "discarded_overlap_frames": 10,
        "segment_boundary_output_frames": [10 + 9 * k for k in range(10)],
        "first_five_frames_replaced_with_original": False,
        "factual_role": "original observation; DriveEditor did not generate factual",
        "window_boundary_caveat": "paper-style final-frame conditioning; 11 x 10-frame segments exceeds the paper's published 39-frame illustration",
        "native_steps": 25,
        "single_gpu_sequential_cfg": True,
        "source_window_root": str(args.window_root.resolve()),
        "counterfactual": str(final.resolve()),
        "frame_to_segment": frame_to_segment,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case_id": args.case_id, "frames": 100, "segments": 11,
                      "output": str(final)}))


if __name__ == "__main__":
    main()
