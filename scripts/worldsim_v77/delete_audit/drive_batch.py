"""Run one frozen DriveEditor deletion checkpoint on the 70 val audit clips."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def deadline(*_):
    raise TimeoutError("DriveEditor single 10-frame window exceeded 240 seconds")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--ready-only", action="store_true")
    p.add_argument("--clip-id")
    args = p.parse_args()
    root = args.root
    run_lock = open(root / "drive_batch.lock", "a")
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    mask_state = read(root / "mask_state.json")
    if not args.ready_only:
        assert mask_state["state"] == "complete"
    selection = read(root / "selection.json")
    mask_done = {row["clip_id"] for row in mask_state["completed"]}
    clips = [row for row in selection["clips"] if row["clip_id"] in mask_done and
             (not args.clip_id or row["clip_id"] == args.clip_id)]
    assert clips
    sys.path.insert(0, "/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77")
    from repair_drive import Engine, set_seed, torch
    torch.set_num_threads(4)
    signal.signal(signal.SIGALRM, deadline)
    state_file = root / "drive_state.json"
    state = read(state_file) if state_file.exists() else {
        "task_id": selection["task_id"], "run_id": selection["run_id"],
        "state": "loading", "pid": os.getpid(), "windows": [], "clips_complete": [],
        "human_verdict": None,
    }
    state["state"] = "loading"
    state["pid"] = os.getpid()
    state["updated_unix"] = time.time()
    write(state_file, state)
    finished_clips = set(state["clips_complete"])
    finished_windows = {(x["clip_id"], x["start"]) for x in state["windows"]}
    try:
        engine = Engine()
        state["state"] = "running"
        state.pop("error", None)
        write(state_file, state)
        for clip in clips:
            clip_id = clip["clip_id"]
            if clip_id in finished_clips:
                continue
            dest = root / "clips" / clip_id
            for folder in ["native", "background", "windows"]:
                (dest / folder).mkdir(exist_ok=True)
            prev = None
            written = {int(path.stem) for path in (dest / "background").glob("*.png")}
            for start in [0, 9, 18]:
                ids = [min(start + j, 25) for j in range(10)]
                valid = min(10, 26 - start)
                if (clip_id, start) in finished_windows:
                    assert all((dest / "background" / f"{i:05d}.png").exists() for i in ids[:valid])
                    prev = np.array(Image.open(dest / "background" / f"{ids[valid - 1]:05d}.png").convert("RGB"))
                    continue
                images = [np.array(Image.open(dest / "rgb" / f"{i:05d}.jpg").convert("RGB")) for i in ids]
                masks = [np.array(Image.open(dest / "model_mask" / f"{i:05d}.png")) > 0 for i in ids]
                assert all(im.shape == (576, 1024, 3) for im in images)
                engine.im = images
                engine.masks = masks
                engine.previous_segment_last_frame = prev
                engine.im_result = []
                set_seed(42)
                torch.cuda.reset_peak_memory_stats()
                t0 = time.monotonic()
                generated = any(mask.any() for mask in masks)
                if generated:
                    signal.alarm(240)
                    try:
                        engine.predict(1, False, "Deletion")
                    finally:
                        signal.alarm(0)
                    assert len(engine.im_result) == 10
                else:
                    engine.im_result = [im.copy() for im in images]
                raw_folder = dest / "windows" / f"{start:05d}"
                raw_folder.mkdir(exist_ok=True)
                checks = []
                for j, (i, image, raw, mask) in enumerate(zip(ids, images, engine.im_result, masks)):
                    assert raw.dtype == np.uint8 and raw.shape == image.shape
                    alpha = np.array(Image.open(dest / "alpha" / f"{i:05d}.png")).astype(np.float32) / 255
                    write_mask = np.array(Image.open(dest / "write_mask" / f"{i:05d}.png")) > 0
                    protect = np.array(Image.open(dest / "protect" / f"{i:05d}.png")) > 0
                    assert np.all(alpha[write_mask] == 1) and np.all(alpha[protect] == 0)
                    composite = np.rint(raw * alpha[..., None] + image * (1 - alpha[..., None])).astype(np.uint8)
                    check = {"frame": i, "outside_changed": int(np.count_nonzero(composite[~mask] != image[~mask])),
                             "protected_changed": int(np.count_nonzero(composite[protect] != image[protect])),
                             "write_diff_native": int(np.count_nonzero(composite[write_mask] != raw[write_mask]))}
                    assert all(value == 0 for key, value in check.items() if key != "frame")
                    checks.append(check)
                    Image.fromarray(raw).save(raw_folder / f"{j:02d}.png")
                    if j < valid and i not in written:
                        Image.fromarray(raw).save(dest / "native" / f"{i:05d}.png")
                        Image.fromarray(composite).save(dest / "background" / f"{i:05d}.png")
                        written.add(i)
                prev = np.array(Image.open(dest / "background" / f"{ids[valid - 1]:05d}.png").convert("RGB"))
                row = {"clip_id": clip_id, "start": start, "source_frames": ids, "generated": generated,
                       "skip_reason": None if generated else "all_masks_empty_input_RGB_retained",
                       "previous_condition": bool(engine.used_previous_segment_condition) if generated else False,
                       "seconds": round(time.monotonic() - t0, 2),
                       "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                       "contracts": checks}
                state["windows"].append(row)
                state["updated_unix"] = time.time()
                finished_windows.add((clip_id, start))
                write(state_file, state)
                print("DRIVE_WINDOW", clip_id, start, generated, row["seconds"], flush=True)
            assert written == set(range(26)), (clip_id, sorted(written))
            state["clips_complete"].append(clip_id)
            finished_clips.add(clip_id)
            write(state_file, state)
            print("DRIVE_CLIP", clip_id, len(finished_clips), "/70", flush=True)
        state["state"] = "complete" if len(state["clips_complete"]) == 70 else "partial"
        write(state_file, state)
    except Exception as exc:
        state["state"] = "failed_engineering"
        state["error"] = repr(exc)
        write(state_file, state)
        raise
    print("DRIVE_ALL_DONE", len(state["clips_complete"]), len(state["windows"]), flush=True)


if __name__ == "__main__":
    main()
