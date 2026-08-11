#!/usr/bin/env python3
"""使用冻结 Grounding DINO 评测 M5 渲染的非目标感知一致性。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from motion_proj.dynamic_editing_v2.pilot_metrics import canonical_sha256
from motion_proj.dynamic_editing_v2.stress_metrics import box_iou, safe_mean


CAMERA_NAMES = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")


def atomic_json(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def snapshot_fingerprint(root: Path) -> tuple[str, list[dict]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest(), rows


def detect(processor, model, image_path: Path, config: dict) -> list[dict]:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    inputs = processor(
        images=image, text=config["prompt"], return_tensors="pt"
    ).to("cuda")
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=float(config["box_threshold"]),
        text_threshold=float(config["text_threshold"]),
        target_sizes=[image.size[::-1]],
    )[0]
    labels = result.get("text_labels", result.get("labels", []))
    rows = []
    for box, score, label in zip(result["boxes"], result["scores"], labels):
        rows.append(
            {
                "box_xyxy": [float(value) for value in box.detach().cpu().tolist()],
                "score": float(score.detach().cpu()),
                "label": str(label),
            }
        )
    return rows


def target_overlap(detection: dict, mask: np.ndarray) -> bool:
    box = detection["box_xyxy"]
    x0 = max(0, min(mask.shape[1] - 1, int(np.floor(box[0]))))
    y0 = max(0, min(mask.shape[0] - 1, int(np.floor(box[1]))))
    x1 = max(x0 + 1, min(mask.shape[1], int(np.ceil(box[2]))))
    y1 = max(y0 + 1, min(mask.shape[0], int(np.ceil(box[3]))))
    return bool(mask[y0:y1, x0:x1].any())


def match_detections(reference: list[dict], candidate: list[dict]) -> dict:
    pairs = []
    for reference_index, first in enumerate(reference):
        for candidate_index, second in enumerate(candidate):
            pairs.append(
                (
                    box_iou(np.asarray(first["box_xyxy"]), np.asarray(second["box_xyxy"])),
                    reference_index,
                    candidate_index,
                )
            )
    used_reference: set[int] = set()
    used_candidate: set[int] = set()
    matches = []
    for iou, reference_index, candidate_index in sorted(pairs, reverse=True):
        if iou < 0.10:
            break
        if reference_index in used_reference or candidate_index in used_candidate:
            continue
        used_reference.add(reference_index)
        used_candidate.add(candidate_index)
        first = reference[reference_index]
        second = candidate[candidate_index]
        matches.append(
            {
                "iou": iou,
                "reference_label": first["label"],
                "candidate_label": second["label"],
                "class_changed": first["label"] != second["label"],
                "confidence_abs_change": abs(first["score"] - second["score"]),
            }
        )
    return {
        "reference_count": len(reference),
        "candidate_count": len(candidate),
        "match_count": len(matches),
        "matched_iou_mean": safe_mean(row["iou"] for row in matches),
        "confidence_abs_change_mean": safe_mean(
            row["confidence_abs_change"] for row in matches
        ),
        "class_change_rate": (
            sum(row["class_changed"] for row in matches) / len(matches)
            if matches
            else None
        ),
        "false_disappearance_rate": (
            (len(reference) - len(used_reference)) / len(reference)
            if reference
            else 0.0
        ),
        "new_detection_rate": (
            (len(candidate) - len(used_candidate)) / max(1, len(candidate))
        ),
        "matches": matches,
    }


def main() -> None:
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-frames", type=int, nargs="+")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    import yaml

    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    perception = protocol["perception"]
    snapshot = Path(
        snapshot_download(
            repo_id=perception["model"],
            revision=perception["revision"],
            local_files_only=True,
        )
    )
    fingerprint, snapshot_files = snapshot_fingerprint(snapshot)
    if fingerprint != perception["snapshot_fingerprint"]:
        raise RuntimeError(
            f"Grounding DINO snapshot fingerprint changed: {fingerprint}"
        )
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        snapshot, local_files_only=True
    ).cuda().eval()
    scene_report = json.loads(
        (args.scene_output / "report.json").read_text(encoding="utf-8")
    )
    rows = []
    original_cache = {}
    sample_frames = args.sample_frames or perception["sample_frames"]
    for frame in sample_frames:
        for camera in range(3):
            stem = f"frame_{frame:03d}_camera_{camera}"
            path = args.scene_output / "original/rgb" / f"{stem}.png"
            original_cache[(frame, camera)] = detect(processor, model, path, perception)

    for role, actor in scene_report["actors"].items():
        if role not in scene_report["available_actor_roles"]:
            continue
        actor_key = f"{role}__{actor['instance_token'][:12]}"
        actor_dir = args.scene_output / "actors" / actor_key
        for edit in ("lateral", "speed", "stop_restart", "delete"):
            for frame in sample_frames:
                for camera in range(3):
                    stem = f"frame_{frame:03d}_camera_{camera}"
                    source = cv2.imread(
                        str(actor_dir / "masks/source" / f"{stem}.png"),
                        cv2.IMREAD_GRAYSCALE,
                    ) > 0
                    edited = cv2.imread(
                        str(actor_dir / "masks" / f"edited_{edit}" / f"{stem}.png"),
                        cv2.IMREAD_GRAYSCALE,
                    ) > 0
                    target = cv2.dilate(
                        (source | edited).astype(np.uint8), np.ones((15, 15), np.uint8)
                    ).astype(bool)
                    original = original_cache[(frame, camera)]
                    candidate = detect(
                        processor,
                        model,
                        actor_dir / edit / "rgb" / f"{stem}.png",
                        perception,
                    )
                    original_target = [row for row in original if target_overlap(row, target)]
                    candidate_target = [row for row in candidate if target_overlap(row, target)]
                    original_non_target = [row for row in original if not target_overlap(row, target)]
                    candidate_non_target = [row for row in candidate if not target_overlap(row, target)]
                    matching = match_detections(original_non_target, candidate_non_target)
                    matching.update(
                        {
                            "scene": scene_report["scene"],
                            "role": role,
                            "instance_token": actor["instance_token"],
                            "edit": edit,
                            "frame": frame,
                            "camera": camera,
                            "camera_name": CAMERA_NAMES[camera],
                            "target_mask_pixels": int(target.sum()),
                            "original_target_detection_count": len(original_target),
                            "candidate_target_detection_count": len(candidate_target),
                            "expected_target_change": (
                                "target detection should disappear"
                                if edit == "delete"
                                else "target detection may move with edited footprint"
                            ),
                        }
                    )
                    rows.append(matching)

    rows_path = args.output_dir / "perception_rows.jsonl"
    with rows_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    sequences = []
    for role, actor in scene_report["actors"].items():
        for edit in ("lateral", "speed", "stop_restart", "delete"):
            if role not in scene_report["available_actor_roles"]:
                sequences.append(
                    {
                        "scene": scene_report["scene"],
                        "role": role,
                        "instance_token": actor["instance_token"],
                        "edit": edit,
                        "status": "ABSTAIN",
                        "metrics": None,
                        "failure_codes": [],
                        "abstain_reason": actor.get(
                            "availability", "actor_checkpoint_slice_unavailable"
                        ),
                    }
                )
                continue
            selected = [
                row for row in rows if row["role"] == role and row["edit"] == edit
            ]
            mean_iou = safe_mean(row["matched_iou_mean"] for row in selected)
            false_disappearance = safe_mean(
                row["false_disappearance_rate"] for row in selected
            )
            per_camera = {}
            for camera_name in CAMERA_NAMES:
                camera_rows = [
                    row for row in selected if row["camera_name"] == camera_name
                ]
                per_camera[camera_name] = {
                    "row_count": len(camera_rows),
                    "matched_iou_mean": safe_mean(
                        row["matched_iou_mean"] for row in camera_rows
                    ),
                    "false_disappearance_rate_mean": safe_mean(
                        row["false_disappearance_rate"] for row in camera_rows
                    ),
                    "confidence_abs_change_mean": safe_mean(
                        row["confidence_abs_change_mean"] for row in camera_rows
                    ),
                }
            camera_iou_values = [
                row["matched_iou_mean"]
                for row in per_camera.values()
                if row["matched_iou_mean"] is not None
            ]
            metrics = {
                "row_count": len(selected),
                "camera_coverage": sorted(
                    {row["camera_name"] for row in selected}
                ),
                "per_camera": per_camera,
                "multicamera_matched_iou_range": (
                    max(camera_iou_values) - min(camera_iou_values)
                    if camera_iou_values
                    else None
                ),
                "matched_iou_mean": mean_iou,
                "false_disappearance_rate_mean": false_disappearance,
                "confidence_abs_change_mean": safe_mean(
                    row["confidence_abs_change_mean"] for row in selected
                ),
                "class_change_rate_mean": safe_mean(
                    row["class_change_rate"] for row in selected
                ),
                "new_detection_rate_mean": safe_mean(
                    row["new_detection_rate"] for row in selected
                ),
                "target_detection_delta_mean": safe_mean(
                    row["candidate_target_detection_count"]
                    - row["original_target_detection_count"]
                    for row in selected
                ),
            }
            failed = (
                (mean_iou is not None and mean_iou < protocol["thresholds"]["perception_match_iou_min"])
                or (
                    false_disappearance is not None
                    and false_disappearance
                    > protocol["thresholds"]["perception_false_disappearance_max"]
                )
            )
            sequences.append(
                {
                    "scene": scene_report["scene"],
                    "role": role,
                    "instance_token": actor["instance_token"],
                    "edit": edit,
                    "metrics": metrics,
                    "failure_codes": ["NON_TARGET_PERCEPTION_DRIFT"] if failed else [],
                }
            )
    checks = {
        "sample_frames_unique_and_in_range": len(sample_frames)
        == len(set(sample_frames))
        and all(0 <= frame < int(scene_report["frames"]) for frame in sample_frames),
        "rows_complete": len(rows)
        == len(scene_report["available_actor_roles"])
        * 4
        * len(sample_frames)
        * len(CAMERA_NAMES),
        "eight_sequences": len(sequences) == 8,
        "all_available_sequences_cover_three_cameras": all(
            row["metrics"] is None
            or row["metrics"]["camera_coverage"] == sorted(CAMERA_NAMES)
            for row in sequences
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"M5 感知 validator 失败: "
            f"{[key for key, value in checks.items() if not value]}"
        )
    report = {
        "schema_version": 1,
        "task_id": "DR-V2-M5-STRESS-3SCENE-01",
        "scene": scene_report["scene"],
        "status": "done",
        "model": perception["model"],
        "revision": perception["revision"],
        "snapshot": str(snapshot),
        "snapshot_fingerprint": fingerprint,
        "snapshot_files": snapshot_files,
        "prompt": perception["prompt"],
        "box_threshold": perception["box_threshold"],
        "text_threshold": perception["text_threshold"],
        "sample_frames": sample_frames,
        "rows": len(rows),
        "sequences": sequences,
        "checks": checks,
        "tracker": {
            "status": "ABSTAIN",
            "reason": "frozen CoTracker3 is a point tracker, not an object-ID tracker; IDF1/ID-switch would be invalid",
        },
        "multicamera_scope": "三相机分别报告 detector preservation；不同视场之间没有可靠对象真值对应，故不伪造跨相机 ID 匹配",
        "claim_scope": "task-aligned evaluator only; not a safety claim",
    }
    report["report_payload_sha256"] = canonical_sha256(report)
    atomic_json(args.output_dir / "report.json", report)
    print(
        json.dumps(
            {
                "status": "done",
                "scene": report["scene"],
                "rows": len(rows),
                "sequences": len(sequences),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
