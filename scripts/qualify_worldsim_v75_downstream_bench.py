#!/usr/bin/env python3
"""Run CPU geometry gates and render review sheets for paired-edit cases."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion_proj.cfbench.geometry import (  # noqa: E402
    footprint,
    footprints_overlap,
    pose_at,
    project_box,
    shifted_pose,
)
from motion_proj.cfbench.schema import validate_manifest  # noqa: E402


VISIBLE_AREA_PX2 = 400.0
EGO_SIZE = [4.8, 2.0, 1.7]


@dataclass
class Track:
    poses: dict[int, np.ndarray]
    sizes: dict[int, list[float]]

    def pose(self, frame: float) -> np.ndarray | None:
        return pose_at(self.poses, frame)

    def size(self, frame: float) -> list[float] | None:
        nearest = int(round(frame))
        return self.sizes.get(nearest)


class Scene:
    def __init__(self, root: Path) -> None:
        self.root = root
        info = json.loads((root / "instances" / "instances_info.json").read_text())
        self.frame_instances = json.loads(
            (root / "instances" / "frame_instances.json").read_text()
        )
        self.tracks: dict[str, Track] = {}
        for actor_key, row in info.items():
            annotations = row["frame_annotations"]
            frames = [int(value) for value in annotations["frame_idx"]]
            self.tracks[str(actor_key)] = Track(
                poses={
                    frame: np.asarray(value, dtype=np.float64)
                    for frame, value in zip(frames, annotations["obj_to_world"])
                },
                sizes={
                    frame: list(map(float, value))
                    for frame, value in zip(frames, annotations["box_size"])
                },
            )
        self.ego_poses = {
            int(path.stem): np.loadtxt(path)
            for path in sorted((root / "lidar_pose").glob("*.txt"))
        }

    def actors_at(self, frame: int) -> list[str]:
        return [str(value) for value in self.frame_instances.get(str(frame), [])]


def _track_for_case(scene: Scene, case: dict[str, Any]) -> tuple[Track, str] | None:
    target = case["target"]
    actor_key = target.get("actor_key") or target.get("source_asset_actor_key")
    if actor_key is not None:
        return scene.tracks[str(actor_key)], str(actor_key)
    if target["role"] == "ego":
        return Track(scene.ego_poses, {frame: EGO_SIZE for frame in scene.ego_poses}), "ego"
    return None


def _counterfactual_pose(
    case: dict[str, Any], track: Track, frame: int
) -> tuple[np.ndarray | None, list[float] | None]:
    event = int(case["anchor"]["event_frame"])
    family = case["intervention"]["family"]
    control = case["intervention"]["counterfactual"]
    source_frame = float(frame)
    if family == "actor_speed_change":
        source_frame = event + (frame - event) * float(control["speed_scale"])
    pose = track.pose(source_frame)
    size = track.size(source_frame)
    if pose is None or size is None:
        return None, None
    if family == "actor_lateral_relocation":
        horizon = max(1, int(case["anchor"]["rollout_frames"]) - 1)
        progress = float(np.clip((frame - event) / horizon, 0.0, 1.0))
        smooth_progress = progress * progress * (3.0 - 2.0 * progress)
        pose = shifted_pose(
            pose,
            [0.0, float(control["lateral_offset_m"]) * smooth_progress, 0.0],
        )
    elif family == "actor_insertion":
        pose = shifted_pose(pose, case["target"]["proposal_offset_actor_frame_m"])
    elif family == "actor_removal":
        return None, size
    return pose, size


def _collisions(
    scene: Scene,
    case: dict[str, Any],
    track: Track,
    source_key: str,
    rollout_frames: list[int],
) -> list[dict[str, Any]]:
    if case["intervention"]["family"] == "actor_removal":
        return []
    collisions: list[dict[str, Any]] = []
    insertion = case["intervention"]["family"] == "actor_insertion"
    for frame in rollout_frames:
        target_pose, target_size = _counterfactual_pose(case, track, frame)
        if target_pose is None or target_size is None:
            collisions.append({"frame": frame, "other": "missing_target_pose"})
            continue
        target_footprint = footprint(target_pose, target_size)
        for other_key in scene.actors_at(frame):
            if not insertion and other_key == source_key:
                continue
            other = scene.tracks.get(other_key)
            if other is None:
                continue
            other_pose = other.pose(float(frame))
            other_size = other.size(float(frame))
            if other_pose is None or other_size is None:
                continue
            if footprints_overlap(target_footprint, footprint(other_pose, other_size)):
                collisions.append({"frame": frame, "other": other_key})
    return collisions


def _visibility(
    scene: Scene,
    case: dict[str, Any],
    track: Track,
    clip_frames: list[int],
) -> list[dict[str, Any]]:
    role = case["target"]["role"]
    if role == "ego":
        return []
    family = case["intervention"]["family"]
    rows: list[dict[str, Any]] = []
    for frame in clip_frames:
        if family == "actor_insertion":
            pose, size = _counterfactual_pose(case, track, frame)
        else:
            pose = track.pose(float(frame))
            size = track.size(float(frame))
        if pose is None or size is None:
            continue
        for camera in range(6):
            projection = project_box(scene.root, frame, camera, pose, size)
            if projection is not None:
                rows.append(projection)
    return sorted(rows, key=lambda row: -row["area_px2"])


def _render_sheet(
    scene: Scene,
    case: dict[str, Any],
    track: Track,
    visibility: list[dict[str, Any]],
    output: Path,
) -> None:
    selected = visibility[:4]
    if not selected:
        event = int(case["anchor"]["event_frame"])
        selected = [
            {"frame": event + offset, "camera": 0, "bbox_xyxy": None, "area_px2": 0.0}
            for offset in (0, 6, 12, 18)
        ]
    tiles: list[Image.Image] = []
    family = case["intervention"]["family"]
    for row in selected:
        frame, camera = int(row["frame"]), int(row["camera"])
        image_path = scene.root / "images" / f"{frame:03d}_{camera}.jpg"
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        factual_pose = track.pose(float(frame))
        factual_size = track.size(float(frame))
        if factual_pose is not None and factual_size is not None and family != "actor_insertion":
            factual = project_box(scene.root, frame, camera, factual_pose, factual_size)
            if factual is not None:
                draw.rectangle(factual["bbox_xyxy"], outline=(239, 68, 68), width=5)
        counterfactual_pose, counterfactual_size = _counterfactual_pose(case, track, frame)
        if counterfactual_pose is not None and counterfactual_size is not None:
            counterfactual = project_box(
                scene.root, frame, camera, counterfactual_pose, counterfactual_size
            )
            if counterfactual is not None:
                draw.rectangle(counterfactual["bbox_xyxy"], outline=(34, 197, 94), width=5)
        draw.rectangle((0, 0, 590, 42), fill=(15, 23, 42))
        draw.text(
            (12, 11),
            f"{case['case_id']}  f={frame} cam={camera}  red=factual green=counterfactual",
            fill=(255, 255, 255),
        )
        image.thumbnail((800, 450))
        tiles.append(image)
    canvas = Image.new("RGB", (1600, 900), color=(30, 41, 59))
    for index, tile in enumerate(tiles):
        canvas.paste(tile, ((index % 2) * 800, (index // 2) * 450))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="JPEG", quality=86, optimize=True)


def qualify_case(case: dict[str, Any], scene: Scene, sheets_root: Path) -> dict[str, Any]:
    event = int(case["anchor"]["event_frame"])
    pre = int(case["anchor"]["pre_frames"])
    rollout = int(case["anchor"]["rollout_frames"])
    clip_frames = list(range(event - pre, event + rollout))
    rollout_frames = list(range(event, event + rollout))
    resolved = _track_for_case(scene, case)
    if resolved is None:
        return {
            "case_id": case["case_id"],
            "automatic_status": "failed",
            "failure": "target track cannot be resolved",
            "human_verdict": None,
        }
    track, source_key = resolved
    required_source_frames = set(clip_frames)
    if case["intervention"]["family"] == "actor_speed_change":
        scale = float(case["intervention"]["counterfactual"]["speed_scale"])
        required_source_frames.update(
            int(round(event + (frame - event) * scale)) for frame in rollout_frames
        )
    coverage_missing = sorted(frame for frame in required_source_frames if track.pose(float(frame)) is None)
    visibility = _visibility(scene, case, track, clip_frames)
    visible_views = [row for row in visibility if row["area_px2"] >= VISIBLE_AREA_PX2]
    visibility_passed = case["target"]["role"] == "ego" or bool(visible_views)
    collisions = _collisions(scene, case, track, source_key, rollout_frames)
    sheet_path = sheets_root / f"{case['case_id']}.jpg"
    _render_sheet(scene, case, track, visibility, sheet_path)
    automatic_passed = not coverage_missing and visibility_passed and not collisions
    return {
        "case_id": case["case_id"],
        "automatic_status": "geometry_pass_manual_pending" if automatic_passed else "failed",
        "gates": {
            "source_track_coverage": {
                "passed": not coverage_missing,
                "missing_frames": coverage_missing,
            },
            "target_visibility": {
                "passed": visibility_passed,
                "not_applicable": case["target"]["role"] == "ego",
                "threshold_area_px2": VISIBLE_AREA_PX2,
                "visible_view_count": len(visible_views),
                "best_views": visibility[:4],
            },
            "counterfactual_initial_collision": {
                "passed": not collisions,
                "collisions": collisions[:20],
                "collision_count": len(collisions),
            },
            "road_validity": {
                "passed": None,
                "reason": "processed pilot scenes contain no map/lane layer; review sheet required",
            },
            "human_review": {"passed": None},
        },
        "review_sheet": str(sheet_path),
        "human_verdict": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sheets-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    scenes: dict[Path, Scene] = {}
    results = []
    for case in manifest["cases"]:
        scene_root = Path(case["dataset"]["root"]) / case["dataset"]["scene_id"]
        scene = scenes.setdefault(scene_root, Scene(scene_root))
        results.append(qualify_case(case, scene, args.sheets_root.resolve()))
    counts: dict[str, int] = {}
    for row in results:
        counts[row["automatic_status"]] = counts.get(row["automatic_status"], 0) + 1
    artifact = {
        "schema_version": "worldsim_v75_cfbench_qualification_v1",
        "manifest": str(args.manifest.resolve()),
        "automatic_only": True,
        "human_verdicts_written": False,
        "summary": counts,
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
