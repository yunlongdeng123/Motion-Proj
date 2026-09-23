#!/usr/bin/env python3
"""Conservative 2D detector diagnostics; never substitutes planned boxes for observed ones."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import (SSDLite320_MobileNet_V3_Large_Weights,
                                          ssdlite320_mobilenet_v3_large)


CLASS_IDS = {
    "vehicle.car": {3},
    "vehicle.bus.rigid": {6, 8},
    "vehicle.trailer": {8},
    "vehicle.construction": {8},
}


def frame_at(cap: cv2.VideoCapture, index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"cannot decode frame {index}")
    return cv2.resize(frame, (1024, 576), interpolation=cv2.INTER_AREA)


def scaled_box(box: list[float] | None) -> np.ndarray | None:
    return np.asarray([box[0] * .64, box[1] * .64, box[2] * .64, box[3] * .64], dtype=float) if box else None


def iou(a: np.ndarray, b: np.ndarray) -> float:
    area_a = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    area_b = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    width = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    overlap = width * height
    return overlap / (area_a + area_b - overlap) if area_a + area_b > overlap else 0.


def detect(model, frame: np.ndarray, allowed_labels: set[int]) -> list[dict]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.
    with torch.no_grad():
        prediction = model([tensor])[0]
    objects = []
    for box, score, label in zip(prediction["boxes"], prediction["scores"], prediction["labels"]):
        confidence = float(score)
        if confidence < .5:
            break
        if int(label) in allowed_labels:
            objects.append({"bbox_xyxy": box.tolist(), "confidence": confidence,
                            "class_id": int(label)})
    return objects


def associate(objects: list[dict], expected: np.ndarray | None) -> dict | None:
    if expected is None:
        return None
    center = (expected[:2] + expected[2:]) / 2
    diag = max(float(np.linalg.norm(expected[2:] - expected[:2])), 12.)
    candidates = []
    for obj in objects:
        box = np.asarray(obj["bbox_xyxy"])
        distance = float(np.linalg.norm((box[:2] + box[2:]) / 2 - center))
        overlap = iou(box, expected)
        if distance <= 1.5 * diag and overlap >= .05:
            candidates.append((overlap + .2 * obj["confidence"] - .1 * distance / diag, obj, overlap))
    if not candidates:
        return None
    _, best, overlap = max(candidates, key=lambda item: item[0])
    return {**best, "iou_to_planned_box": round(overlap, 5)}


def color_distance(first_frame: np.ndarray, first_box: list[float],
                   second_frame: np.ndarray, second_box: list[float]) -> float | None:
    def hist(frame, box):
        left, top, right, bottom = map(int, box)
        left, right = max(0, left), min(frame.shape[1], right)
        top, bottom = max(0, top), min(frame.shape[0], bottom)
        if right - left < 8 or bottom - top < 8:
            return None
        crop = frame[top:bottom, left:right]
        crop = crop[int(.1 * crop.shape[0]):max(int(.9 * crop.shape[0]), 1),
                    int(.1 * crop.shape[1]):max(int(.9 * crop.shape[1]), 1)]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [18, 16], [0, 180, 0, 256])
        return cv2.normalize(h, h).astype(np.float32)
    a, b = hist(first_frame, first_box), hist(second_frame, second_box)
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)) if a is not None and b is not None else None


def evaluate(model, case_root: Path, geometry: dict, auto: dict, step: int, cls: str) -> dict:
    source = cv2.VideoCapture(str(case_root / "factual.mp4"))
    edited = cv2.VideoCapture(str(case_root / "counterfactual.mp4"))
    if not source.isOpened() or not edited.isOpened():
        raise RuntimeError("missing paired video")
    family = geometry["family"]
    allowed = CLASS_IDS.get(cls)
    if not allowed:
        raise RuntimeError(f"no detector class mapping for {cls}")
    indices = sorted(set(range(0, 100, step)) | {99})
    rows, endpoint_errors, color_distances = [], [], []
    try:
        for index in indices:
            if index < geometry["event_frame_in_clip"] or not auto["frames"][index]["native_model_frame"]:
                continue
            geo = geometry["frames"][index]
            factual_box, desired_box = scaled_box(geo["factual_bbox_xyxy"]), scaled_box(geo["desired_bbox_xyxy"])
            if family == "actor_removal":
                desired_box = factual_box
            if factual_box is None and desired_box is None:
                continue
            factual_im, edited_im = frame_at(source, index), frame_at(edited, index)
            factual_match = associate(detect(model, factual_im, allowed), factual_box)
            edited_match = associate(detect(model, edited_im, allowed), desired_box)
            error = None
            if family not in {"actor_removal"} and desired_box is not None and edited_match:
                observed = np.asarray(edited_match["bbox_xyxy"])
                error = float(np.linalg.norm((observed[:2] + observed[2:]) / 2 -
                                             (desired_box[:2] + desired_box[2:]) / 2))
                endpoint_errors.append((index, error))
            if factual_match and edited_match and family in {"actor_speed_change", "actor_lateral_relocation"}:
                distance = color_distance(factual_im, factual_match["bbox_xyxy"],
                                          edited_im, edited_match["bbox_xyxy"])
                if distance is not None:
                    color_distances.append(distance)
            rows.append({"frame": index,
                         "factual_detected_near_gt": factual_match,
                         "generated_detected_near_plan": edited_match,
                         "generated_center_error_px": round(error, 3) if error is not None else None})
    finally:
        source.release()
        edited.release()
    factual_hits = sum(row["factual_detected_near_gt"] is not None for row in rows)
    generated_hits = sum(row["generated_detected_near_plan"] is not None for row in rows)
    valid = len(rows)
    if not valid:
        raise RuntimeError("no sampled visible/postevent native frames")
    # Matched-only ADE is deliberately withheld when a detector does not
    # verify most frames: otherwise the hardest failures vanish from the mean.
    reliable = factual_hits / valid >= .7 and generated_hits / valid >= .7
    pixel_ade = float(np.mean([error for _, error in endpoint_errors])) if reliable and endpoint_errors else None
    pixel_fde = next((error for frame, error in endpoint_errors if frame == 99), None) if reliable else None
    return {
        "schema_version": "cfbench_r9_ssdlite_2d_detector_diagnostics_v1",
        "model": "torchvision SSDLite320_MobileNet_V3_Large COCO DEFAULT",
        "torchvision_version": torchvision.__version__,
        "confidence_threshold": .5,
        "matching": "class-gated IoU>=0.05 and center distance<=1.5 expected diagonal; independent per frame",
        "sample_step_frames": step,
        "sampled_postevent_native_frames": valid,
        "factual_target_detection_coverage": round(factual_hits / valid, 4),
        "generated_planned_region_detection_coverage": round(generated_hits / valid, 4),
        "pixel_ADE_matched_only": round(pixel_ade, 3) if pixel_ade is not None else None,
        "pixel_FDE_if_final_detected": round(pixel_fde, 3) if pixel_fde is not None else None,
        "pixel_ADE_FDE_reason_if_null": None if reliable else "detector coverage <70% in factual or generated frames; missing detections cannot silently disappear from ADE",
        "normalized_crop_HSV_Bhattacharyya_distance": round(float(np.mean(color_distances)), 4) if len(color_distances) >= 5 else None,
        "object_identity_caveat": "HSV color is not identity; independent class detections can switch actors, especially in crowded scenes",
        "outcome_caveat": "detection in planned region is a 2D proxy; absence may mean detector failure and not actual deletion",
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--sample-step", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT).cpu().eval()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    classes = {row["case_id"]: row["case"]["target"].get("class_name") for row in manifest["cases"]}
    selected = set(args.case_id)
    for case_root in sorted(args.cases_root.iterdir()):
        if not case_root.is_dir() or selected and case_root.name not in selected:
            continue
        auto_path = case_root / "auto-metrics.json"
        geometry_path = args.geometry_root / f"{case_root.name}.json"
        if not auto_path.is_file() or not geometry_path.is_file():
            continue
        auto = json.loads(auto_path.read_text(encoding="utf-8"))
        geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
        if not geometry["geometry_available"]:
            continue
        detector = evaluate(model, case_root, geometry, auto, args.sample_step,
                            classes[case_root.name])
        detector_path = case_root / "detector-metrics.json"
        detector_path.write_text(json.dumps(detector, indent=2) + "\n", encoding="utf-8")
        auto["detector_diagnostics"] = {
            key: value for key, value in detector.items() if key != "rows"
        }
        auto_path.write_text(json.dumps(auto, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"case_id": case_root.name,
                          "factual_detection_coverage": detector["factual_target_detection_coverage"],
                          "generated_detection_coverage": detector["generated_planned_region_detection_coverage"],
                          "pixel_ADE": detector["pixel_ADE_matched_only"]}), flush=True)


if __name__ == "__main__":
    main()
