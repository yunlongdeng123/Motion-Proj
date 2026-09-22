#!/usr/bin/env python3
"""把 frozen non-ego cases 编译为 DriveEditor 官方 pickle 输入格式。"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from nuscenes.utils.data_classes import Box
from PIL import Image, ImageDraw
from pyquaternion import Quaternion

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion_proj.cfbench.geometry import pose_at, project_box, shifted_pose  # noqa: E402
from motion_proj.cfbench.schema import validate_manifest  # noqa: E402


def _track(scene_root: Path, actor_key: str) -> tuple[dict[int, np.ndarray], dict[int, list[float]], dict[str, Any]]:
    info = json.loads((scene_root / "instances" / "instances_info.json").read_text())
    row = info[str(actor_key)]
    annotations = row["frame_annotations"]
    frames = [int(value) for value in annotations["frame_idx"]]
    poses = {
        frame: np.asarray(value, dtype=np.float64)
        for frame, value in zip(frames, annotations["obj_to_world"])
    }
    sizes = {
        frame: list(map(float, value))
        for frame, value in zip(frames, annotations["box_size"])
    }
    return poses, sizes, row


def _size_at(sizes: dict[int, list[float]], frame: float) -> list[float] | None:
    return sizes.get(int(round(frame)))


def _counterfactual(
    case: dict[str, Any],
    poses: dict[int, np.ndarray],
    sizes: dict[int, list[float]],
    frame: int,
) -> tuple[np.ndarray | None, list[float] | None]:
    event = int(case["anchor"]["event_frame"])
    family = case["intervention"]["family"]
    control = case["intervention"]["counterfactual"]
    source_frame = float(frame)
    if family == "actor_speed_change":
        source_frame = event + (frame - event) * float(control["speed_scale"])
    pose = pose_at(poses, source_frame)
    size = _size_at(sizes, source_frame)
    if pose is None or size is None:
        return None, None
    if family == "actor_lateral_relocation":
        progress = float(np.clip((frame - event) / 18.0, 0.0, 1.0))
        smooth = progress * progress * (3.0 - 2.0 * progress)
        pose = shifted_pose(pose, [0.0, float(control["lateral_offset_m"]) * smooth, 0.0])
    elif family == "actor_insertion":
        pose = shifted_pose(pose, case["target"]["proposal_offset_actor_frame_m"])
    return pose, size


def _choose_camera(
    case: dict[str, Any], scene_root: Path, poses: dict[int, np.ndarray], sizes: dict[int, list[float]]
) -> int:
    event = int(case["anchor"]["event_frame"])
    scores: list[tuple[int, int, float]] = []
    for camera in range(6):
        counterfactual_visible = 0
        source_visible = 0
        area = 0.0
        for frame in range(event, event + 10):
            source = project_box(scene_root, frame, camera, poses[frame], sizes[frame])
            source_visible += source is not None
            cf_pose, cf_size = _counterfactual(case, poses, sizes, frame)
            edited = (
                project_box(scene_root, frame, camera, cf_pose, cf_size)
                if cf_pose is not None and cf_size is not None
                else None
            )
            counterfactual_visible += edited is not None
            if edited is not None:
                area += float(edited["area_px2"])
        scores.append((counterfactual_visible, source_visible, area))
    camera = max(range(6), key=lambda value: scores[value])
    if scores[camera][0] != 10 or scores[camera][1] != 10:
        raise RuntimeError(f"{case['case_id']}: no camera sees source and edited box for all 10 frames: {scores}")
    return camera


def _intrinsic(scene_root: Path, camera: int) -> np.ndarray:
    fx, fy, cx, cy = np.loadtxt(scene_root / "intrinsics" / f"{camera}.txt")[:4]
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _box(
    scene_root: Path,
    frame: int,
    camera: int,
    pose: np.ndarray,
    size: list[float],
    class_name: str,
    token: str,
) -> Box:
    object_to_camera = np.linalg.inv(np.loadtxt(scene_root / "extrinsics" / f"{frame:03d}_{camera}.txt")) @ pose
    length, width, height = map(float, size[:3])
    return Box(
        center=object_to_camera[:3, 3],
        size=[width, length, height],
        orientation=Quaternion(matrix=object_to_camera[:3, :3]),
        name=class_name,
        token=token,
    )


def _component_mask(mask: np.ndarray, bbox: list[float]) -> tuple[np.ndarray, dict[str, Any]]:
    left, top, right, bottom = [int(round(value)) for value in bbox]
    left, top = max(0, left), max(0, top)
    right, bottom = min(mask.shape[1], right), min(mask.shape[0], bottom)
    clipped = np.zeros_like(mask, dtype=np.uint8)
    clipped[top:bottom, left:right] = (mask[top:bottom, left:right] > 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(clipped, connectivity=8)
    candidates = [index for index in range(1, count) if stats[index, cv2.CC_STAT_AREA] >= 16]
    if not candidates:
        clipped[top:bottom, left:right] = 1
        return clipped, {
            "fallback_to_projected_box": True,
            "candidate_component_count": 0,
            "component_area_px2": int(clipped.sum()),
        }
    center_x = int(np.clip(round((left + right) / 2.0), 0, mask.shape[1] - 1))
    center_y = int(np.clip(round((top + bottom) / 2.0), 0, mask.shape[0] - 1))
    center_label = int(labels[center_y, center_x])
    if center_label in candidates:
        label = center_label
    else:
        label = min(
            candidates,
            key=lambda index: float(
                (centroids[index][0] - center_x) ** 2 + (centroids[index][1] - center_y) ** 2
            ),
        )
    selected = (labels == label).astype(np.uint8)
    width, height = max(1, right - left), max(1, bottom - top)
    offset = math.sqrt(
        ((float(centroids[label][0]) - center_x) / width) ** 2
        + ((float(centroids[label][1]) - center_y) / height) ** 2
    )
    return selected, {
        "fallback_to_projected_box": False,
        "candidate_component_count": len(candidates),
        "component_area_px2": int(stats[label, cv2.CC_STAT_AREA]),
        "component_center_offset_normalized": offset,
    }


def _reference_crop(
    scene_root: Path,
    frame: int,
    camera: int,
    pose: np.ndarray,
    size: list[float],
    class_name: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    image = np.asarray(Image.open(scene_root / "images" / f"{frame:03d}_{camera}.jpg").convert("RGB"))
    projected = project_box(scene_root, frame, camera, pose, size)
    if projected is None:
        raise RuntimeError("reference object is not visible")
    mask_kind = "human" if "pedestrian" in class_name else "vehicle"
    raw_mask = np.asarray(
        Image.open(scene_root / "dynamic_masks" / mask_kind / f"{frame:03d}_{camera}.png").convert("L")
    )
    component, metrics = _component_mask(raw_mask, projected["bbox_xyxy"])
    ys, xs = np.nonzero(component)
    if not len(xs):
        raise RuntimeError("reference mask is empty")
    margin = max(8, int(0.12 * max(xs.max() - xs.min(), ys.max() - ys.min())))
    left, right = max(0, int(xs.min()) - margin), min(image.shape[1], int(xs.max()) + margin + 1)
    top, bottom = max(0, int(ys.min()) - margin), min(image.shape[0], int(ys.max()) + margin + 1)
    cropped_image = image[top:bottom, left:right]
    cropped_mask = component[top:bottom, left:right]
    side = max(cropped_image.shape[:2])
    canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    mask_canvas = np.zeros((side, side), dtype=np.uint8)
    y0 = (side - cropped_image.shape[0]) // 2
    x0 = (side - cropped_image.shape[1]) // 2
    region = canvas[y0 : y0 + cropped_image.shape[0], x0 : x0 + cropped_image.shape[1]]
    region[cropped_mask > 0] = cropped_image[cropped_mask > 0]
    mask_canvas[y0 : y0 + cropped_mask.shape[0], x0 : x0 + cropped_mask.shape[1]] = cropped_mask * 255
    metrics["projected_box_area_px2"] = float(projected["area_px2"])
    metrics["reference_frame"] = int(frame)
    return canvas, mask_canvas, metrics


def _operation(family: str) -> str:
    return {
        "actor_speed_change": "Repositioning",
        "actor_lateral_relocation": "Repositioning",
        "actor_removal": "Deletion",
        "actor_insertion": "Insertion",
    }[family]


def compile_case(case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    scene_root = Path(case["dataset"]["root"]) / case["dataset"]["scene_id"]
    target = case["target"]
    actor_key = str(target.get("actor_key") or target.get("source_asset_actor_key"))
    poses, sizes, actor = _track(scene_root, actor_key)
    camera = _choose_camera(case, scene_root, poses, sizes)
    event = int(case["anchor"]["event_frame"])
    class_name = str(target["class_name"])
    token = str(target.get("source_asset_actor_id") or target["entity_id"])
    reference_frame = max(
        range(event, event + 10),
        key=lambda frame: float(
            project_box(scene_root, frame, camera, poses[frame], sizes[frame])["area_px2"]
        ),
    )
    reference_image, reference_mask, reference_metrics = _reference_crop(
        scene_root,
        reference_frame,
        camera,
        poses[reference_frame],
        sizes[reference_frame],
        class_name,
    )
    frames = []
    factual_paths = []
    for frame in range(event, event + 10):
        image_path = scene_root / "images" / f"{frame:03d}_{camera}.jpg"
        factual_paths.append(str(image_path))
        image = np.asarray(Image.open(image_path).convert("RGB"))
        factual_box = _box(scene_root, frame, camera, poses[frame], sizes[frame], class_name, token)
        cf_pose, cf_size = _counterfactual(case, poses, sizes, frame)
        edited_box = (
            _box(scene_root, frame, camera, cf_pose, cf_size, class_name, token)
            if cf_pose is not None and cf_size is not None
            else None
        )
        frames.append(
            {
                "im": image,
                "box": factual_box,
                "box_Repositioning": edited_box or factual_box,
                "box_Insertion": edited_box or factual_box,
            }
        )
    operation = _operation(case["intervention"]["family"])
    item = {
        "name": case["case_id"],
        "camera_intrinsic": _intrinsic(scene_root, camera),
        "category_name": class_name,
        "data": frames,
        "im": reference_image,
        "mask": reference_mask,
        "im_Insertion": [reference_image],
        "mask_Insertion": [reference_mask],
        "im_Replacement": [reference_image],
        "mask_Replacement": [reference_mask],
    }
    index = {
        "case_id": case["case_id"],
        "operation": operation,
        "scene_id": case["dataset"]["scene_id"],
        "camera": camera,
        "frame_start": event,
        "frame_count": 10,
        "factual_frames": factual_paths,
        "target_actor_key": actor_key,
        "source_actor_id": str(actor.get("id", token)),
        "reference_mask": reference_metrics,
    }
    return item, index


def _write_contact_sheet(rows: list[tuple[str, np.ndarray]], output: Path) -> None:
    tile_width, tile_height = 320, 230
    canvas = Image.new("RGB", (tile_width * 3, tile_height * 6), color=(30, 41, 59))
    for index, (case_id, array) in enumerate(rows):
        tile = Image.new("RGB", (tile_width, tile_height), color=(255, 255, 255))
        image = Image.fromarray(array).convert("RGB")
        image.thumbnail((280, 180))
        tile.paste(image, ((tile_width - image.width) // 2, 36 + (180 - image.height) // 2))
        ImageDraw.Draw(tile).text((10, 10), case_id, fill=(15, 23, 42))
        canvas.paste(tile, ((index % 3) * tile_width, (index // 3) * tile_height))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="JPEG", quality=88, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    cases = [case for case in manifest["cases"] if case["target"]["role"] == "non_ego"]
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    references = []
    for case in cases:
        item, row = compile_case(case)
        case_path = args.output_root / f"{case['case_id']}.pkl"
        with case_path.open("wb") as handle:
            pickle.dump([item], handle, protocol=pickle.HIGHEST_PROTOCOL)
        row["pickle"] = str(case_path.resolve())
        rows.append(row)
        references.append((case["case_id"], item["im"]))
    artifact = {
        "schema_version": "worldsim_v75_driveeditor_inputs_v1",
        "gpu_operations_run": False,
        "pickle_layout": "one official-format single-item list per case",
        "root": str(args.output_root.resolve()),
        "case_count": len(rows),
        "cases": rows,
    }
    args.index.parent.mkdir(parents=True, exist_ok=True)
    args.index.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    if args.contact_sheet is not None:
        _write_contact_sheet(references, args.contact_sheet)
    print(json.dumps({"cases": len(rows), "root": str(args.output_root), "index": str(args.index)}, indent=2))


if __name__ == "__main__":
    main()
