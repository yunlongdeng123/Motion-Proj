#!/usr/bin/env python3
"""Build a CPU-screened 24-case paired-edit pilot from DriveStudio metadata."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion_proj.cfbench.schema import DIMENSIONS, MANIFEST_SCHEMA, validate_manifest


@dataclass(frozen=True)
class ActorCandidate:
    scene_id: str
    actor_key: str
    actor_id: str
    class_name: str
    run_start: int
    run_end: int
    event_frame: int
    displacement_m: float
    score: float


def _longest_consecutive(frames: list[int]) -> tuple[int, int]:
    ordered = sorted(set(int(frame) for frame in frames))
    if not ordered:
        raise ValueError("empty frame sequence")
    best_start = best_end = run_start = previous = ordered[0]
    for frame in ordered[1:]:
        if frame != previous + 1:
            if previous - run_start > best_end - best_start:
                best_start, best_end = run_start, previous
            run_start = frame
        previous = frame
    if previous - run_start > best_end - best_start:
        best_start, best_end = run_start, previous
    return best_start, best_end


def _distance(a: list[list[float]], b: list[list[float]]) -> float:
    return math.sqrt(sum((float(a[index][3]) - float(b[index][3])) ** 2 for index in range(3)))


def _load_candidates(data_root: Path) -> tuple[list[str], dict[str, int], list[ActorCandidate]]:
    scenes: list[str] = []
    frame_counts: dict[str, int] = {}
    candidates: list[ActorCandidate] = []
    for scene_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        info_path = scene_dir / "instances" / "instances_info.json"
        frames_path = scene_dir / "instances" / "frame_instances.json"
        if not info_path.is_file() or not frames_path.is_file():
            continue
        info = json.loads(info_path.read_text(encoding="utf-8"))
        frame_instances = json.loads(frames_path.read_text(encoding="utf-8"))
        scene_id = scene_dir.name
        scenes.append(scene_id)
        frame_counts[scene_id] = len(frame_instances)
        for actor_key, row in info.items():
            if not str(row.get("class_name", "")).startswith("vehicle."):
                continue
            annotations = row.get("frame_annotations", {})
            frames = [int(value) for value in annotations.get("frame_idx", [])]
            transforms = annotations.get("obj_to_world", [])
            if len(frames) != len(transforms) or len(frames) < 24:
                continue
            run_start, run_end = _longest_consecutive(frames)
            if run_end - run_start + 1 < 24:
                continue
            transform_by_frame = dict(zip(frames, transforms))
            event_frame = run_start + 5
            if event_frame + 18 > run_end:
                continue
            displacement = _distance(transform_by_frame[run_start], transform_by_frame[run_end])
            score = float(run_end - run_start + 1) + min(displacement, 50.0)
            candidates.append(
                ActorCandidate(
                    scene_id=scene_id,
                    actor_key=str(actor_key),
                    actor_id=str(row.get("id", actor_key)),
                    class_name=str(row["class_name"]),
                    run_start=run_start,
                    run_end=run_end,
                    event_frame=event_frame,
                    displacement_m=displacement,
                    score=score,
                )
            )
    if len(scenes) < 3:
        raise RuntimeError("pilot construction requires at least three processed scenes")
    candidates.sort(key=lambda row: (-row.score, row.scene_id, row.actor_key))
    if len(candidates) < 18:
        raise RuntimeError(f"insufficient sustained vehicle tracks: {len(candidates)}")
    return scenes, frame_counts, candidates


def _base_case(
    *,
    case_id: str,
    data_root: Path,
    scene_id: str,
    event_frame: int,
    entity_id: str,
    role: str,
    family: str,
    variable: str,
    factual: dict[str, Any],
    counterfactual: dict[str, Any],
    direction: str,
    target_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target = {"role": role, "entity_id": entity_id}
    if target_extra:
        target.update(target_extra)
    return {
        "case_id": case_id,
        "status": "proposed_cpu_only",
        "dataset": {
            "name": "nuscenes_drivestudio_10hz",
            "root": str(data_root),
            "scene_id": scene_id,
            "role": "pilot_development_only",
        },
        "anchor": {
            "event_frame": int(event_frame),
            "pre_frames": 5,
            "rollout_frames": 24,
            "state_fixation": "same scene, prefix, camera, seed and non-target state",
        },
        "target": target,
        "intervention": {
            "family": family,
            "variable": variable,
            "factual": factual,
            "counterfactual": counterfactual,
        },
        "expected_outcome": {
            "direction": direction,
            "affected_entities": [entity_id],
            "unchanged_scope": "all_except_affected_entities",
        },
        "qualification": {
            "state_fixation": True,
            "single_variable_change": True,
            "requires_visible_effect": True,
            "requires_collision_and_road_gate": True,
            "human_verdict": None,
        },
    }


def build_manifest(data_root: Path) -> dict[str, Any]:
    scenes, frame_counts, candidates = _load_candidates(data_root)
    cases: list[dict[str, Any]] = []

    # Three ego and three non-ego speed probes. The role split prevents ReSim's
    # ego trajectory interface from being misrepresented as non-ego editing.
    for index, scene_id in enumerate(scenes[:3]):
        factor = 0.5 if index % 2 == 0 else 1.5
        cases.append(
            _base_case(
                case_id=f"CFB-SPEED-EGO-{index + 1:02d}",
                data_root=data_root,
                scene_id=scene_id,
                event_frame=min(65, frame_counts[scene_id] - 19),
                entity_id="ego",
                role="ego",
                family="actor_speed_change",
                variable="speed_scale",
                factual={"speed_scale": 1.0},
                counterfactual={"speed_scale": factor},
                direction="shorter progress" if factor < 1.0 else "longer progress",
            )
        )
    for index, actor in enumerate(candidates[:3]):
        factor = 0.5 if index % 2 == 0 else 1.5
        cases.append(
            _base_case(
                case_id=f"CFB-SPEED-ACTOR-{index + 1:02d}",
                data_root=data_root,
                scene_id=actor.scene_id,
                event_frame=actor.event_frame,
                entity_id=actor.actor_id,
                role="non_ego",
                family="actor_speed_change",
                variable="speed_scale",
                factual={"speed_scale": 1.0},
                counterfactual={"speed_scale": factor},
                direction="shorter progress" if factor < 1.0 else "longer progress",
                target_extra={"actor_key": actor.actor_key, "class_name": actor.class_name},
            )
        )

    for index, scene_id in enumerate(scenes[:3]):
        offset = 3.5 if index % 2 == 0 else -3.5
        cases.append(
            _base_case(
                case_id=f"CFB-LATERAL-EGO-{index + 1:02d}",
                data_root=data_root,
                scene_id=scene_id,
                event_frame=min(100, frame_counts[scene_id] - 19),
                entity_id="ego",
                role="ego",
                family="actor_lateral_relocation",
                variable="lateral_offset_m",
                factual={"lateral_offset_m": 0.0},
                counterfactual={"lateral_offset_m": offset},
                direction="leftward displacement" if offset > 0 else "rightward displacement",
            )
        )
    for index, actor in enumerate(candidates[3:6]):
        offset = 3.5 if index % 2 == 0 else -3.5
        cases.append(
            _base_case(
                case_id=f"CFB-LATERAL-ACTOR-{index + 1:02d}",
                data_root=data_root,
                scene_id=actor.scene_id,
                event_frame=actor.event_frame,
                entity_id=actor.actor_id,
                role="non_ego",
                family="actor_lateral_relocation",
                variable="lateral_offset_m",
                factual={"lateral_offset_m": 0.0},
                counterfactual={"lateral_offset_m": offset},
                direction="leftward displacement" if offset > 0 else "rightward displacement",
                target_extra={"actor_key": actor.actor_key, "class_name": actor.class_name},
            )
        )

    for index, actor in enumerate(candidates[6:12]):
        cases.append(
            _base_case(
                case_id=f"CFB-REMOVE-{index + 1:02d}",
                data_root=data_root,
                scene_id=actor.scene_id,
                event_frame=actor.event_frame,
                entity_id=actor.actor_id,
                role="non_ego",
                family="actor_removal",
                variable="presence",
                factual={"presence": True},
                counterfactual={"presence": False},
                direction="target absent with disoccluded content revealed",
                target_extra={"actor_key": actor.actor_key, "class_name": actor.class_name},
            )
        )

    for index, actor in enumerate(candidates[12:18]):
        offset = 3.5 if index % 2 == 0 else -3.5
        entity_id = f"inserted:{actor.actor_id}"
        cases.append(
            _base_case(
                case_id=f"CFB-INSERT-{index + 1:02d}",
                data_root=data_root,
                scene_id=actor.scene_id,
                event_frame=actor.event_frame,
                entity_id=entity_id,
                role="non_ego",
                family="actor_insertion",
                variable="presence",
                factual={"presence": False},
                counterfactual={"presence": True},
                direction="new actor present at a road-valid non-colliding pose",
                target_extra={
                    "source_asset_actor_key": actor.actor_key,
                    "source_asset_actor_id": actor.actor_id,
                    "class_name": actor.class_name,
                    "proposal_offset_actor_frame_m": [8.0, offset, 0.0],
                },
            )
        )

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "benchmark": {
            "task_id": "WS-V75-DOWNSTREAM-CFBENCH-01",
            "purpose": "problem discovery, not prevalence or leaderboard claims",
            "paired_design": True,
            "no_composite_score": True,
            "dimensions": list(DIMENSIONS),
            "construction": "CPU metadata screen; every case still requires visual/road/collision qualification",
        },
        "cases": cases,
    }
    validate_manifest(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.data_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for case in manifest["cases"]:
        family = case["intervention"]["family"]
        counts[family] = counts.get(family, 0) + 1
    print(json.dumps({"cases": len(manifest["cases"]), "families": counts}, indent=2))


if __name__ == "__main__":
    main()
