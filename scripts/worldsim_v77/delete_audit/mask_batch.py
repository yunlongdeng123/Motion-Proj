"""Frozen v77 SAM2 mask/write preparation for the nuScenes val DELETE audit."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def hull_mask(polygon, pad=0):
    output = np.zeros((576, 1024), dtype=np.uint8)
    if polygon:
        cv2.fillConvexPoly(output, np.asarray(polygon, dtype=np.int32), 1)
    if pad:
        output = cv2.dilate(output, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1)))
    return output.astype(bool)


def largest(mask):
    count, labels, stat, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return labels == 1 + int(np.argmax(stat[1:, cv2.CC_STAT_AREA])) if count > 1 else mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--ready-only", action="store_true")
    p.add_argument("--clip-id")
    args = p.parse_args()
    root = args.root
    if not args.ready_only:
        assert read(root / "pub_rgb_check.json")["all_selected_rgb_available"]
    geometry = read(root / "clip_geometry.json")
    assert len(geometry["clips"]) == 70
    clips = [clip for clip in geometry["clips"] if not args.clip_id or clip["clip_id"] == args.clip_id]
    assert clips
    ready = []
    for clip in clips:
        links = []
        for frame in clip["frames"]:
            name = frame["filename"]
            alias = root / "rgb" / name
            if alias.is_file():
                links.append((alias, None))
                continue
            places = [root / "pub_by_shard" / f"{shard:02d}" / name for shard in range(1, 11)]
            found = [path for path in places if path.is_file()]
            links.append((alias, found[0] if len(found) == 1 else None))
        if any(not alias.is_file() and source is None for alias, source in links):
            if args.ready_only:
                continue
            raise FileNotFoundError(clip["clip_id"])
        for alias, source in links:
            if source is not None and not alias.is_file():
                alias.parent.mkdir(parents=True, exist_ok=True)
                alias.symlink_to(source)
        ready.append(clip)
    print("MASK_READY", len(ready), "of", len(clips), flush=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    torch.manual_seed(42)
    import sys
    sys.path.insert(0, "/root/autodl-tmp/third_party/worldsim_v32/sam2")
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "/root/autodl-tmp/third_party/worldsim_v32/sam2/checkpoints/sam2.1_hiera_large.pt",
        device="cuda",
    )
    state_file = root / "mask_state.json"
    state = read(state_file) if state_file.exists() else {"task_id": "WS-V77-DELETE-AUDIT-20260928", "run_id": "r1",
        "state": "running", "pid": os.getpid(), "prompt_policy_revision": "p2_uniform_instance_separability_before_GPU",
        "completed": [], "human_verdict": None}
    assert state["prompt_policy_revision"] == "p2_uniform_instance_separability_before_GPU"
    done = {x["clip_id"] for x in state["completed"]}
    write(state_file, state)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for clip in ready:
            clip_id = clip["clip_id"]
            if clip_id in done:
                continue
            t0 = time.monotonic()
            dest = root / "clips" / clip_id
            rgb_dir = dest / "rgb"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            for frame in clip["frames"]:
                target_jpg = rgb_dir / f"{frame['frame']:05d}.jpg"
                if not target_jpg.is_file():
                    source = root / "rgb" / frame["filename"]
                    assert source.is_file(), source
                    image = Image.open(source).convert("RGB").resize((1024, 576), Image.Resampling.BILINEAR)
                    image.save(target_jpg, quality=96)
            assert len(list(rgb_dir.glob("*.jpg"))) == 26
            prompt = clip["prompt_frame"]
            box = np.asarray(clip["frames"][prompt]["target"]["box_xyxy"], dtype=np.float32)
            track = predictor.init_state(video_path=str(rgb_dir), offload_video_to_cpu=True, offload_state_to_cpu=True)
            predictor.add_new_points_or_box(track, frame_idx=prompt, obj_id=1, box=box)
            raw = {}
            for reverse in [False, True]:
                for frame, _, logit in predictor.propagate_in_video(track, start_frame_idx=prompt, reverse=reverse):
                    raw[int(frame)] = (logit[0, 0] > 0).cpu().numpy()
            assert set(raw) == set(range(26)), (clip_id, sorted(raw))
            folders = ["sam", "core", "write_mask", "model_mask", "protect", "alpha"]
            for folder in folders:
                (dest / folder).mkdir(exist_ok=True)
            stats = []
            for fr in clip["frames"]:
                i = fr["frame"]
                target = hull_mask(fr["target"]["hull"] if fr["target"] else None, pad=3)
                core = largest(raw[i] & target)
                write_mask = cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
                write_mask &= target
                model = np.zeros((576, 1024), dtype=bool)
                if write_mask.any():
                    yy, xx = np.where(write_mask)
                    model[max(0, int(yy.min()) - 8):min(576, int(yy.max()) + 25),
                          max(0, int(xx.min()) - 8):min(1024, int(xx.max()) + 9)] = True
                neighbor_union = np.zeros_like(model)
                for neighbor in fr["neighbors"]:
                    neighbor_union |= hull_mask(neighbor["hull"])
                protect = neighbor_union & model & ~write_mask
                distance = cv2.distanceTransform(model.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                fade = np.clip(distance / 8, 0, 1)
                alpha = fade * fade * (3 - 2 * fade)
                alpha[write_mask] = 1
                alpha[protect] = 0
                arrays = {"sam": raw[i] * 255, "core": core * 255, "write_mask": write_mask * 255,
                          "model_mask": model * 255, "protect": protect * 255, "alpha": np.rint(alpha * 255)}
                for folder, values in arrays.items():
                    Image.fromarray(values.astype(np.uint8)).save(dest / folder / f"{i:05d}.png")
                stats.append({"frame": i, "gt_target_pixels": int(target.sum()), "sam_pixels": int(raw[i].sum()),
                              "core_pixels": int(core.sum()), "write_pixels": int(write_mask.sum()),
                              "model_pixels": int(model.sum()), "protect_pixels": int(protect.sum()),
                              "sam_other_GT_pixels": int((raw[i] & neighbor_union).sum()),
                              "write_other_GT_pixels": int((write_mask & neighbor_union).sum()),
                              "gt_target_visible": fr["target"] is not None,
                              "sam_outside_gt_pixels": int((raw[i] & ~target).sum())})
            write(dest / "mask_stats.json", stats)
            review = Image.open(rgb_dir / f"{prompt:05d}.jpg").convert("RGB")
            draw = ImageDraw.Draw(review)
            x0, y0, x1, y1 = box
            draw.rectangle((x0, y0, x1, y1), outline=(255, 220, 30), width=4)
            draw.text((max(0, x0), max(0, y0 - 16)), f"{clip_id} target {clip['instance_token'][:8]}", fill=(255, 220, 30))
            review.save(dest / "input_review.jpg", quality=94)
            record = {"clip_id": clip_id, "state": "complete", "prompt_frame": prompt,
                      "seconds": round(time.monotonic() - t0, 2),
                      "gt_target_visible_frames": sum(x["gt_target_visible"] for x in stats),
                      "core_nonempty_frames": sum(x["core_pixels"] > 0 for x in stats),
                      "model_nonempty_frames": sum(x["model_pixels"] > 0 for x in stats),
                      "frames_write_other_GT_overlap": [x["frame"] for x in stats if x["write_other_GT_pixels"] > 0],
                      "max_write_other_GT_pixels": max(x["write_other_GT_pixels"] for x in stats),
                      "empty_core_with_visible_GT": [x["frame"] for x in stats if x["gt_target_visible"] and x["core_pixels"] == 0]}
            state["completed"].append(record)
            write(state_file, state)
            print("MASK", record, flush=True)
            del track, raw
            torch.cuda.empty_cache()
    state["state"] = "complete" if len(state["completed"]) == 70 else "partial"
    write(state_file, state)
    print("MASK_PASS_DONE", len(state["completed"]), state["state"], flush=True)


if __name__ == "__main__":
    main()
