#!/usr/bin/env python3
"""银车 DEVA154 ↔ Waymo track 的只读 CPU 关联提议；所有帧交由主线程审查。"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from google.protobuf.json_format import MessageToDict

sys.dont_write_bytecode = True


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def mask_bbox(body):
    y, x = np.nonzero(body)
    require(len(x) > 0, "Reviewed DEVA154 must have real nonzero pixels in every declared visible frame")
    return np.array([x.min(), y.min(), x.max() + 1, y.max() + 1], dtype=np.float64)


def projected_bbox(box):
    # 原生 projected_lidar_labels 为图像轴对齐框：length 沿 x，width 沿 y。
    return np.array([box.center_x - box.length / 2, box.center_y - box.width / 2,
                     box.center_x + box.length / 2, box.center_y + box.width / 2], dtype=np.float64)


def box_iou(first, second):
    intersection = np.maximum(np.minimum(first[2:], second[2:]) - np.maximum(first[:2], second[:2]), 0)
    area = float(np.prod(intersection))
    first_area, second_area = float(np.prod(first[2:] - first[:2])), float(np.prod(second[2:] - second[:2]))
    return area / (first_area + second_area - area) if first_area + second_area > area else 0.


def compare(body, target_bbox, candidate_bbox):
    height, width = body.shape
    x0, y0 = np.maximum(np.floor(candidate_bbox[:2]).astype(np.int64), [0, 0])
    x1, y1 = np.minimum(np.ceil(candidate_bbox[2:]).astype(np.int64), [width, height])
    covered = int(body[y0:y1, x0:x1].sum()) if x1 > x0 and y1 > y0 else 0
    return {"mask_bbox_iou": float(box_iou(target_bbox, candidate_bbox)),
            "mask_pixel_coverage": covered / int(body.sum()), "covered_mask_pixels": covered}


def letterbox(image, width, height):
    image = image.copy()
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (18, 22, 28))
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return canvas


def font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def diagnostic(rgb, body, target_bbox, proposed, frame_id, raw_index, track):
    height, width = rgb.shape[:2]
    box = np.array(proposed["projected_bbox_xyxy"]) if proposed else target_bbox
    union_min, union_max = np.minimum(target_bbox[:2], box[:2]), np.maximum(target_bbox[2:], box[2:])
    padding = np.maximum((union_max - union_min) * .12, 24)
    minimum = np.maximum(np.floor(union_min - padding).astype(int), [0, 0])
    maximum = np.minimum(np.ceil(union_max + padding).astype(int), [width, height])
    crop = tuple([*minimum, *maximum])
    original = Image.fromarray(rgb)
    overlay_rgb = rgb.copy()
    overlay_rgb[body] = (rgb[body].astype(np.float64) * .55 + np.array([30, 255, 80]) * .45).astype(np.uint8)
    overlay = Image.fromarray(overlay_rgb)
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(tuple(target_bbox), outline=(30, 255, 80), width=4)
    if proposed:
        draw.rectangle(tuple(box), outline=(255, 155, 20), width=5)
    pane_w, pane_h, header = 640, 427, 82
    canvas = Image.new("RGB", (pane_w * 3, pane_h + header), (18, 22, 28))
    canvas.paste(letterbox(original, pane_w, pane_h), (0, header))
    canvas.paste(letterbox(original.crop(crop), pane_w, pane_h), (pane_w, header))
    canvas.paste(letterbox(overlay.crop(crop), pane_w, pane_h), (pane_w * 2, header))
    draw = ImageDraw.Draw(canvas)
    small = font(19)
    score = f"IoU={proposed['mask_bbox_iou']:.3f} coverage={proposed['mask_pixel_coverage']:.3f}" if proposed else "NO FRONT projected label"
    draw.text((12, 6), f"{frame_id} | raw {raw_index} | proposed raw track {track} | {score}", font=font(23), fill=(244, 246, 250))
    draw.text((12, 42), "Actual FRONT RGB", font=small, fill=(244, 246, 250))
    draw.text((pane_w + 12, 42), "Actual target crop (no overlay)", font=small, fill=(244, 246, 250))
    draw.text((pane_w * 2 + 12, 42), "Green: exact DEVA154 | Orange: native projected track box", font=small, fill=(244, 246, 250))
    return canvas


def run(scene, output, reader_path, raw=None, pb_root=None, crc_library=None):
    started = time.monotonic()
    scene, output = Path(scene).resolve(), Path(output).resolve()
    require(scene != output and scene not in output.parents and not output.exists(), "Proposal must use new isolated output, never existing scene/selection")
    manifest = load_json(scene / "preparation_manifest.json")
    rows = manifest.get("frames", [])
    require(manifest.get("author_file_index") == 17 and len(rows) == 49
            and [row["source_index"] for row in rows] == list(range(49)), "Actual scene17 raw0..48 source mapping required")
    review_path = scene / "selected_actor_0.json"
    review = load_json(review_path)
    require(review.get("status") == "visually_reviewed" and review.get("default_instance_ids") == [154], "Actual inherited DEVA154 visual review required")
    require(all(review.get("per_frame_visibility", {}).get(row["id"]) == "visible"
                and review.get("per_frame_instance_ids", {}).get(row["id"], [154]) == [154] for row in rows),
            "Actual49 reviewed visible DEVA154 mappings required")
    raw = Path(raw or manifest["raw_path"])
    require(raw.is_file() and raw.name == Path(manifest["raw_path"]).name, "Actual original TFRecord identity required")
    reader_path = Path(reader_path)
    require(reader_path.is_file(), "Existing pure protobuf CPU reader required")
    spec = importlib.util.spec_from_file_location("waymo_silver_association_cpu_reader", reader_path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    pb = reader.protobuf(pb_root or reader.PB_ROOT)
    audits, aggregates = [], {}
    expected_context = manifest["segment_context_name"]
    for index, payload, _ in reader.records(raw, 48, crc_library or reader.CRC_LIBRARY):
        frame = pb.Frame.FromString(payload)
        row = rows[index]
        require(frame.context.name == expected_context and int(frame.timestamp_micros) == row["timestamp_micros"], "Actual raw/prepared identity/timestamp differs")
        actual_pose = np.asarray(frame.pose.transform, dtype=np.float64).reshape(4, 4)
        require(np.allclose(actual_pose, np.loadtxt(scene / "pose" / (row["id"] + ".txt")), atol=1e-7, rtol=0), "Actual raw/prepared pose differs")
        laser = {label.id: label for label in frame.laser_labels if label.type == 1}
        front = [camera for camera in frame.projected_lidar_labels if camera.name == 1]
        require(len(front) == 1, "Actual native FRONT projected labels missing/duplicated")
        labels = np.array(Image.open(scene / "mask/mask_0/Annotations" / (row["id"] + ".png")))
        require(labels.shape == (1280, 1920) and labels.dtype == np.uint8, "Actual native DEVA PNG dimensions/type invalid")
        body = labels == 154
        target = mask_bbox(body)
        candidates = []
        for label in front[0].labels:
            if label.type != 1:
                continue
            # 只在当前真实 laser ID 集合中接受匹配，不能猜测不在标注中的后缀 ID。
            track = label.id if label.id in laser else label.id[:-len("_FRONT")] if label.id.endswith("_FRONT") else None
            require(track in laser, "Native FRONT projected vehicle ID must bind exact real laser track")
            box = projected_bbox(label.box)
            require(np.isfinite(box).all() and np.all(box[2:] > box[:2]), "Actual projected vehicle box invalid")
            score = compare(body, target, box)
            source = laser[track]
            world_center = actual_pose @ np.array([source.box.center_x, source.box.center_y, source.box.center_z, 1.])
            entry = {"raw_track_id": track, "projected_label_id": label.id, "projected_bbox_xyxy": box.tolist(), **score,
                     "laser_box_source": MessageToDict(source, preserving_proto_field_name=True), "world_center_xyz_m": world_center[:3].tolist()}
            candidates.append(entry)
            aggregates.setdefault(track, []).append({"frame_id": row["id"], "source_index": index, "timestamp_micros": int(frame.timestamp_micros), **entry})
        candidates.sort(key=lambda entry: (-entry["mask_bbox_iou"], -entry["mask_pixel_coverage"], entry["raw_track_id"]))
        require(bool(candidates), "Actual FRONT vehicle candidates absent; cannot invent silver track")
        audits.append({"frame_id": row["id"], "source_index": index, "timestamp_micros": int(frame.timestamp_micros),
                       "DEVA154_pixels": int(body.sum()), "DEVA154_bbox_xyxy": target.tolist(), "candidates": candidates})
    rank = []
    for track, observations in aggregates.items():
        ious = np.array([row["mask_bbox_iou"] for row in observations])
        centers = np.array([row["world_center_xyz_m"] for row in observations])
        times = np.array([row["timestamp_micros"] for row in observations], dtype=np.float64) / 1e6
        speeds = np.linalg.norm(np.diff(centers, axis=0), axis=1) / np.diff(times) if len(times) > 1 else np.array([])
        rank.append({"raw_track_id": track, "FRONT_projected_frames": len(observations), "mean_mask_bbox_iou_all49": float(ious.sum() / 49),
                     "median_mask_bbox_iou_present": float(np.median(ious)), "min_mask_bbox_iou_present": float(ious.min()),
                     "mean_mask_pixel_coverage_all49": float(sum(row["mask_pixel_coverage"] for row in observations) / 49),
                     "top1_frames": sum(frame["candidates"][0]["raw_track_id"] == track for frame in audits),
                     "world_displacement_m": float(np.linalg.norm(centers[-1] - centers[0])),
                     "median_world_speed_mps": float(np.median(speeds)) if len(speeds) else None})
    rank.sort(key=lambda row: (-row["mean_mask_bbox_iou_all49"], -row["mean_mask_pixel_coverage_all49"], row["raw_track_id"]))
    primary = rank[0]["raw_track_id"]
    output.mkdir(parents=True)
    (output / "frames").mkdir()
    cards = []
    for row in audits:
        name = row["frame_id"]
        rgb = np.array(Image.open(scene / "image/image_0" / (name + ".png")).convert("RGB"))
        require(rgb.shape == (1280, 1920, 3), "Actual original FRONT RGB required for visual review")
        body = np.array(Image.open(scene / "mask/mask_0/Annotations" / (name + ".png"))) == 154
        proposed = next((item for item in row["candidates"] if item["raw_track_id"] == primary), None)
        card = diagnostic(rgb, body, np.array(row["DEVA154_bbox_xyxy"]), proposed, name, row["source_index"], primary)
        card.save(output / "frames" / (name + ".png"))
        cards.append(card.resize((960, 255), Image.Resampling.LANCZOS))
        row["primary_candidate_in_frame"] = proposed
        row["root_visual_reviewed"] = False
    pages = []
    for start in range(0, 49, 8):
        page_name = f"contact_{start:02d}_{min(start + 7, 48):02d}.png"
        subset = cards[start:start + 8]
        page = Image.new("RGB", (1920, ((len(subset) + 1) // 2) * 255), (18, 22, 28))
        for index, card in enumerate(subset):
            page.paste(card, ((index % 2) * 960, (index // 2) * 255))
        page.save(output / page_name)
        pages.append({"path": str(output / page_name), "frame_ids": [row["frame_id"] for row in audits[start:start + 8]]})
    proposal = {"status": "REAL49_TRACK_MASK_ASSOCIATION_PROPOSAL_REVIEW_REQUIRED", "source_scene": str(scene), "raw_path": str(raw),
                "context_name": expected_context, "reviewed_mask_source": str(review_path), "DEVA_instance_id": 154,
                "raw_frame_window": [0, 48], "frame_count": 49, "proposed_raw_track_id": primary, "ranking": rank,
                "ranking_policy": "mean native projected-box vs actual DEVA154 mask-bbox IoU over all49; missing projected label score0; coverage tie-break",
                "projected_id_binding": "actual raw laser ID membership; native projected label raw_id or raw_id + _FRONT",
                "source_static_or_instance_NPZ_modified": False, "source_selection_modified": False, "source_DEVA_modified": False,
                "GT_boxes_used_to_create_masks": False, "GT_boxes_use": "identity proposal and diagnostic rectangle only, no mask/point creation",
                "human_verdict": None, "root_visual_review_complete": False, "approved_mapping": None,
                "GPU_used": False, "Torch_imported": "torch" in sys.modules, "TF_imported": any(name.startswith("tensorflow") for name in sys.modules),
                "physical_moving_evidence": "world displacement and speed use actual frame.pose @ real laser box centers, never ego-frame-only motion",
                "contact_pages": pages, "frames": audits, "execution": {"interpreter": sys.executable, "pid": os.getpid(), "command": sys.argv,
                   "completed_utc": datetime.now(timezone.utc).isoformat(), "wall_seconds": time.monotonic() - started,
                   "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}},
                "primary_source": "https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/dataset.proto"}
    require(not proposal["Torch_imported"] and not proposal["TF_imported"], "CPU association unexpectedly imported model framework")
    save_json_new(output / "silver_track_binding.proposal.json", proposal)
    print(json.dumps({"status": proposal["status"], "output": str(output), "primary": rank[0], "runner_up": rank[1] if len(rank) > 1 else None,
                      "contact_pages": len(pages), "source_masks_modified": False, "root_review_complete": False}, ensure_ascii=False), flush=True)
    return proposal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--waymo-reader", type=Path, required=True)
    parser.add_argument("--tfrecord", type=Path)
    parser.add_argument("--pb-root", type=Path)
    parser.add_argument("--crc-library", type=Path)
    args = parser.parse_args()
    run(args.scene, args.output, args.waymo_reader, args.tfrecord, args.pb_root, args.crc_library)


if __name__ == "__main__":
    main()
