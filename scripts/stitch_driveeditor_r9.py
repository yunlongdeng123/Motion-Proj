#!/usr/bin/env python3
"""Assemble verified native DriveEditor windows for approved 10 s cases."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg


def video_info(path: Path) -> tuple[int, float, int, int]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot decode {path}")
        count = 0
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            count += 1
        return count, capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()


def link_once(source: Path, target: Path) -> None:
    if not target.exists():
        os.link(source, target)
    elif not os.path.samefile(source, target):
        raise RuntimeError(f"existing file differs: {target}")


def run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args],
                            text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:])


def assemble(row: dict, window_roots: list[Path], output_root: Path,
             manifest_root: Path, offscreen_windows: set[tuple[str, int]]) -> dict:
    cid = row["case_id"]
    case = row["case"]
    case_root = output_root / cid
    case_root.mkdir(parents=True, exist_ok=True)
    original = manifest_root / "cases" / cid / "original-nuscenes.mp4"
    if video_info(original)[:2] != (100, 10.0):
        raise RuntimeError(f"original is not 100 frames at 10 Hz: {cid}")
    link_once(original, case_root / "original-nuscenes.mp4")
    link_once(original, case_root / "factual.mp4")
    if case["target"]["role"] == "ego":
        result = {"case_id": cid, "status": "unsupported_ego", "generated_frames": 0,
                  "factual_role": "original observation; DriveEditor did not generate factual"}
        (case_root / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    parts = []
    passthrough = []
    for index in range(10):
        if (cid, index) in offscreen_windows:
            # An off-screen target cannot condition this native editor at the
            # window keyframe. Preserve the approved factual video there and
            # count those frames as ungenerated model coverage.
            video = case_root / f"passthrough-w{index:02d}.mp4"
            if not video.is_file():
                filter_graph = (f"trim=start_frame={index * 10}:end_frame={(index + 1) * 10},"
                                "setpts=PTS-STARTPTS,scale=1024:576")
                run_ffmpeg(["-i", str(original), "-vf", filter_graph, "-frames:v", "10",
                            "-r", "10", "-an", "-c:v", "libx264", "-crf", "18",
                            "-pix_fmt", "yuv420p", str(video)])
            passthrough.append(index)
        else:
            available = []
            for window_root in window_roots:
                part_root = window_root / f"{cid}__w{index:02d}"
                record = part_root / "result.json"
                candidate = part_root / "counterfactual.mp4"
                if record.is_file() and candidate.is_file():
                    outcome = json.loads(record.read_text(encoding="utf-8"))
                    if outcome.get("status") == "generation_complete" and outcome.get("steps") == 25:
                        available.append(candidate)
            if len(available) > 1:
                raise RuntimeError(f"multiple generated outputs for {cid} window {index}: {available}")
            if not available:
                return {"case_id": cid, "status": "partial", "ready_windows": len(parts),
                        "passthrough_windows": passthrough}
            video = available[0]
        if video_info(video) != (10, 10.0, 1024, 576):
            raise RuntimeError(f"window video failed 10-frame/10-Hz check: {video}")
        parts.append(video)
    result_path = case_root / "result.json"
    if result_path.is_file():
        prior = json.loads(result_path.read_text(encoding="utf-8"))
        if prior.get("status") in {"generation_complete", "assembled_with_passthrough"} and video_info(case_root / "counterfactual.mp4")[:2] == (100, 10.0):
            return prior
        raise RuntimeError(f"preserving incomplete prior result: {result_path}")
    raw = case_root / "counterfactual-raw.mp4"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", prefix="driveeditor-r9-", delete=False) as handle:
        concat_list = Path(handle.name)
        for part in parts:
            handle.write("file '" + str(part.resolve()).replace("'", "'\\''") + "'\n")
    try:
        run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(raw)])
    finally:
        concat_list.unlink(missing_ok=True)
    if video_info(raw) != (100, 10.0, 1024, 576):
        raise RuntimeError(f"stitched raw video failed verification: {raw}")
    final = case_root / "counterfactual.mp4"
    gated = case["intervention"]["family"] in {"actor_removal", "actor_insertion"}
    if gated:
        filter_graph = ("[0:v]trim=start_frame=0:end_frame=5,setpts=PTS-STARTPTS,scale=1024:576[p];"
                        "[1:v]trim=start_frame=5:end_frame=100,setpts=PTS-STARTPTS[q];"
                        "[p][q]concat=n=2:v=1:a=0[v]")
        run_ffmpeg(["-i", str(original), "-i", str(raw), "-filter_complex", filter_graph,
                    "-map", "[v]", "-r", "10", "-an", "-c:v", "libx264", "-crf", "18",
                    "-pix_fmt", "yuv420p", str(final)])
    else:
        shutil.copyfile(raw, final)
    if video_info(final) != (100, 10.0, 1024, 576):
        raise RuntimeError(f"final video failed verification: {final}")
    generated_windows = 10 - len(passthrough)
    result = {"case_id": cid,
              "status": "assembled_with_passthrough" if passthrough else "generation_complete",
              "generated_frames": generated_windows * 10,
              "native_windows": generated_windows, "passthrough_windows": passthrough,
              "native_window_duration_s": 1.0, "output_fps": 10,
              "native_steps": 25, "single_gpu_sequential_cfg": True,
              "factual_role": "original observation; DriveEditor did not generate factual",
              "counterfactual_raw": str(raw), "counterfactual": str(final),
              "first_five_frames_replaced_with_original": gated,
              "prefix_compositing_reason": "DriveEditor has no time-gated insertion/deletion; raw output is retained for audit" if gated else None,
              "window_boundary_caveat": "independent 1 s windows; continuity must be scored, not presumed"}
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--window-root", type=Path, required=True, action="append")
    parser.add_argument("--visibility-audit", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.manifest_root / "candidates.json").read_text(encoding="utf-8"))
    assert manifest["case_count"] == 24 and all(r["approval"]["status"] == "approved" for r in manifest["cases"])
    offscreen_windows = set()
    if args.visibility_audit:
        audit = json.loads(args.visibility_audit.read_text(encoding="utf-8"))
        assert audit["schema_version"] == "driveeditor_r9_visibility_audit_v1"
        offscreen_windows = {(row["case_id"], int(row["window"])) for row in audit["windows"]
                             if not row["keyframe_visible"]}
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = [assemble(row, args.window_root, args.output_root, args.manifest_root,
                        offscreen_windows) for row in manifest["cases"]]
    (args.output_root / "assembly-status.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": sum(r["status"] == "generation_complete" for r in results),
                      "with_passthrough": sum(r["status"] == "assembled_with_passthrough" for r in results),
                      "partial": sum(r["status"] == "partial" for r in results),
                      "unsupported": sum(r["status"] == "unsupported_ego" for r in results)}))


if __name__ == "__main__":
    main()
