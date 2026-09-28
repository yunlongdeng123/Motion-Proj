"""Encode source/mask/native/final DELETE videos for the frozen val audit."""
from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw


KINDS = ["original", "model_input", "native", "delete"]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


class VideoWriter:
    def __init__(self, path):
        self.target = path
        self.temp = path.with_suffix(".encoding.mp4")
        self.container = av.open(str(self.temp), "w", options={"movflags": "+faststart"})
        self.stream = self.container.add_stream("libx264", rate=10)
        self.stream.width = 1024
        self.stream.height = 576
        self.stream.pix_fmt = "yuv420p"
        self.stream.thread_count = 1
        self.stream.options = {"crf": "20", "preset": "fast", "g": "10", "bf": "0"}

    def write(self, rgb, index):
        assert rgb.shape == (576, 1024, 3) and rgb.dtype == np.uint8
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts = index
        frame.time_base = Fraction(1, 10)
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        self.temp.replace(self.target)


def box_image(rgb, frame, clip_id, instance_token):
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)
    target = frame["target"]
    if target:
        box = target["box_xyxy"]
        # 小目标使用细框，避免审核标记本身遮住车辆。
        short_side = min(box[2] - box[0], box[3] - box[1])
        width = 1 if short_side < 32 else 2 if short_side < 80 else 3
        draw.rectangle(box, outline=(255, 213, 0), width=width)
        label = f"{clip_id} target {instance_token[:8]}"
    else:
        label = f"{clip_id} target outside projected view"
    draw.rectangle((3, 3, min(400, 12 + 7 * len(label)), 30), fill=(22, 29, 38))
    draw.text((10, 10), label, fill=(255, 224, 25))
    return np.asarray(image)


def mask_image(original, model, write_mask, protect):
    output = original.astype(np.float32)
    output[model] = output[model] * .65 + np.array([30, 130, 255]) * .35
    output[write_mask] = output[write_mask] * .5 + np.array([255, 175, 10]) * .5
    output[protect] = output[protect] * .45 + np.array([20, 240, 170]) * .55
    image = Image.fromarray(np.rint(output).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    draw.rectangle((3, 3, 600, 30), fill=(22, 29, 38))
    draw.text((10, 10), "Blue: model hole   Amber: target write   Green: neighbor protect", fill="white")
    return np.asarray(image)


def review_crop(box, original, final):
    if box is None:
        region = (0, 0, 1024, 576)
    else:
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        width = max(256, (x1 - x0) * 2.2)
        height = max(144, (y1 - y0) * 2.2)
        width = max(width, height * 16 / 9)
        height = width * 9 / 16
        width, height = min(1024, width), min(576, height)
        left = min(max(0, cx - width / 2), 1024 - width)
        top = min(max(0, cy - height / 2), 576 - height)
        region = tuple(map(int, (left, top, left + width, top + height)))
    a = Image.fromarray(original).crop(region).resize((640, 360), Image.Resampling.BICUBIC)
    b = Image.fromarray(final).crop(region).resize((640, 360), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (1280, 390), (20, 29, 39))
    canvas.paste(a, (0, 30))
    canvas.paste(b, (640, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), "Original with target box", fill=(255, 220, 25))
    draw.text((652, 8), "DELETE final, same crop", fill=(255, 220, 25))
    return canvas


def validate_video(path):
    with av.open(str(path)) as container:
        count = 0
        last = -1.0
        for frame in container.decode(video=0):
            assert (frame.width, frame.height) == (1024, 576)
            assert frame.time > last
            last = frame.time
            count += 1
    assert count == 26 and abs(last - 2.5) < 1e-6, (path, count, last)
    return {"decoded_frames": count, "fps": 10, "last_pts_s": last, "bytes": path.stat().st_size}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--ready-only", action="store_true")
    args = p.parse_args()
    root = args.root
    state = read(root / "drive_state.json")
    if not args.ready_only:
        assert state["state"] == "complete"
    completed = set(state["clips_complete"])
    geometry = {c["clip_id"]: c for c in read(root / "clip_geometry.json")["clips"]}
    selection = read(root / "selection.json")
    review = root / "review"
    review.mkdir(exist_ok=True)
    manifest_path = review / "review_manifest.json"
    previous = read(manifest_path) if manifest_path.exists() else {"task_id": selection["task_id"],
        "run_id": selection["run_id"], "clips": [], "human_verdict": None}
    indexed = {row["clip_id"]: row for row in previous["clips"]}
    for spec in selection["clips"]:
        clip_id = spec["clip_id"]
        if clip_id not in completed or clip_id in indexed:
            continue
        source = root / "clips" / clip_id
        dest = review / "assets" / clip_id
        dest.mkdir(parents=True, exist_ok=True)
        frames = geometry[clip_id]["frames"]
        writers = {kind: VideoWriter(dest / f"{kind}.mp4") for kind in KINDS}
        prompt = geometry[clip_id]["prompt_frame"]
        posters = None
        pair = None
        for j, frame in enumerate(frames):
            original = np.array(Image.open(source / "rgb" / f"{j:05d}.jpg").convert("RGB"))
            native = np.array(Image.open(source / "native" / f"{j:05d}.png").convert("RGB"))
            final = np.array(Image.open(source / "background" / f"{j:05d}.png").convert("RGB"))
            model = np.array(Image.open(source / "model_mask" / f"{j:05d}.png")) > 0
            target_write = np.array(Image.open(source / "write_mask" / f"{j:05d}.png")) > 0
            protect = np.array(Image.open(source / "protect" / f"{j:05d}.png")) > 0
            original_boxed = box_image(original, frame, clip_id, spec["instance_token"])
            masked = mask_image(original_boxed, model, target_write, protect)
            arrays = {"original": original_boxed, "model_input": masked, "native": native, "delete": final}
            for kind, image in arrays.items():
                writers[kind].write(image, j)
            if j == prompt:
                posters = {kind: Image.fromarray(image) for kind, image in arrays.items()}
                pair = review_crop(frame["target"]["box_xyxy"] if frame["target"] else None, original_boxed, final)
        for writer in writers.values():
            writer.close()
        assert posters and pair
        for kind, image in posters.items():
            image.save(dest / f"{kind}_poster.jpg", quality=92)
        pair.save(dest / "one_frame_review.jpg", quality=91)
        videos = {kind: {"path": f"assets/{clip_id}/{kind}.mp4",
                         "poster": f"assets/{clip_id}/{kind}_poster.jpg",
                         **validate_video(dest / f"{kind}.mp4")}
                  for kind in KINDS}
        stats = read(source / "mask_stats.json")
        row = {"clip_id": clip_id, "scene": spec["scene"], "instance_token": spec["instance_token"],
               "category": spec["category"], "camera": spec["camera"], "start_keyframe": spec["start_keyframe"],
               "prompt_frame": prompt, "input_difficulty_proxy": spec["input_difficulty_proxy"],
               "difficulty_factors": spec["difficulty_factors"], "size_bucket": spec["size_bucket"],
               "occlusion_proxy": spec["occlusion_proxy"], "behind_vehicle_proxy": spec["behind_vehicle_proxy"],
               "GT_target_visible_frames": sum(fr["target"] is not None for fr in frames),
               "core_nonempty_frames": sum(x["core_pixels"] > 0 for x in stats),
               "model_nonempty_frames": sum(x["model_pixels"] > 0 for x in stats),
               "empty_core_with_visible_GT": [x["frame"] for x in stats if x["gt_target_visible"] and not x["core_pixels"]],
               "frames_write_other_GT_overlap": [x["frame"] for x in stats if x["write_other_GT_pixels"] > 0],
               "max_write_other_GT_pixels": max(x["write_other_GT_pixels"] for x in stats),
               "input_qualification": "pending_assistant_one_frame_review",
               "videos": videos, "poster": f"assets/{clip_id}/original_poster.jpg",
               "one_frame_review": f"assets/{clip_id}/one_frame_review.jpg",
               "assistant_one_frame_issue": "unreviewed", "assistant_review_note": "",
               "driveeditor_factual_reconstruction": "not_run_no_native_identity_task",
               "human_verdict": None}
        indexed[clip_id] = row
        previous["clips"] = [indexed[x["clip_id"]] for x in selection["clips"] if x["clip_id"] in indexed]
        write(manifest_path, previous)
        print("ENCODED", clip_id, len(indexed), "/70", flush=True)
    print("REVIEW_ENCODED", len(indexed), flush=True)


if __name__ == "__main__":
    main()
