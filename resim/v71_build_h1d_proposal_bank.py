#!/usr/bin/env python
"""只读 source observation，冻结 H1-11D eligibility、proposal 与 scenario tags。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.resim.canonical_hash import canonical_sha256
from motion_proj.resim.coordinates import drivestudio_intrinsics
from motion_proj.resim.matched_pilot import (
    apply_lateral_proposal,
    clipped_bbox_area_fraction,
    frames_to_boxes,
    trajectory_hash,
)
from motion_proj.resim.safety_geometry import OrientedBox
from motion_proj.resim.scenario_effect import (
    ScenarioThresholds,
    build_counterfactual_pair,
    evaluate_scenario_effect,
)
from motion_proj.runtime.atomic import atomic_write_json
from motion_proj.runtime.fingerprint import file_fingerprint


def _load_actor_frames(raw_actor: dict, start: int, end: int) -> list[dict]:
    values = raw_actor["frame_annotations"]
    frames = []
    for frame, transform, dimensions in zip(
        values["frame_idx"], values["obj_to_world"], values["box_size"]
    ):
        frame = int(frame)
        if start <= frame <= end:
            frames.append(
                {
                    "frame_index": frame,
                    "T_world_actor": transform,
                    "dimensions_lwh": dimensions,
                }
            )
    return frames


def _ego_trajectory(scene_root: Path, start: int, end: int) -> dict[int, np.ndarray]:
    return {
        frame: np.loadtxt(scene_root / "lidar_pose" / f"{frame:03d}.txt").reshape(4, 4)
        for frame in range(start, end + 1)
    }


def _visible_frames(frames, scene_root, cameras, image_size_wh, threshold) -> tuple[int, dict]:
    by_camera = {camera["camera_id"]: 0 for camera in cameras}
    visible = 0
    for value in frames:
        areas = {}
        frame = int(value["frame_index"])
        for camera in cameras:
            index = int(camera["dataset_index"])
            area = clipped_bbox_area_fraction(
                np.asarray(value["T_world_actor"], dtype=float),
                value["dimensions_lwh"],
                T_world_camera=np.loadtxt(
                    scene_root / "extrinsics" / f"{frame:03d}_{index}.txt"
                ).reshape(4, 4),
                intrinsics=drivestudio_intrinsics(
                    np.loadtxt(scene_root / "intrinsics" / f"{index}.txt")
                ),
                image_size_wh=image_size_wh,
            )
            areas[camera["camera_id"]] = area
            by_camera[camera["camera_id"]] += int(area >= threshold)
        visible += int(max(areas.values(), default=0.0) >= threshold)
    return visible, by_camera


def _local_boxes(frames, ego_by_frame, actor_id) -> list[OrientedBox]:
    output = []
    for value, world_box in zip(frames, frames_to_boxes(frames, actor_id=actor_id)):
        inverse = np.linalg.inv(ego_by_frame[int(value["frame_index"])])
        center = inverse @ np.asarray([*world_box.center, 1.0])
        heading = inverse @ np.asarray([np.cos(world_box.yaw), np.sin(world_box.yaw), 0.0, 0.0])
        output.append(
            OrientedBox(
                tuple(center[:3]),
                world_box.dimensions_lwh,
                float(np.arctan2(heading[1], heading[0])),
                actor_id,
            )
        )
    return output


def _ego_speed(ego_by_frame: dict[int, np.ndarray], frame_indices: list[int], dt_s: float) -> float:
    centers = np.asarray([ego_by_frame[frame][:3, 3] for frame in frame_indices])
    if len(centers) < 2:
        return 0.0
    return float(np.median(np.linalg.norm(np.diff(centers, axis=0), axis=1) / dt_s))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/resim/v71/matched_pilot_v1.yaml"))
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = args.output_root or Path(config["proposal_output"])
    if output_root.exists():
        raise FileExistsError(f"proposal bank 已存在，拒绝覆盖: {output_root}")
    output_root.mkdir(parents=True)
    start, end = (int(value) for value in config["frame_range"])
    processed_root = Path(config["processed_root"])
    registry_root = Path(config["registry_root"])
    support = json.loads(Path(config["support_summary"]).read_text(encoding="utf-8"))
    scenario_cfg = yaml.safe_load(Path(config["scenario_config"]).read_text(encoding="utf-8"))
    thresholds = ScenarioThresholds(
        scenario_cfg["event_min_consecutive_frames"],
        scenario_cfg["min_lateral_gap_change_m"],
        scenario_cfg["min_boundary_crossings"],
        tuple(scenario_cfg["ttc_valid_range_s"]),
        tuple(scenario_cfg["time_headway_valid_range_s"]),
    )
    eligibility_cfg = config["eligibility"]
    bank = {
        "schema_version": "h1-proposal-bank-v1",
        "task_id": "V7-H1-11D",
        "split": config["split"],
        "seed": int(config["seed"]),
        "config_sha256": file_fingerprint(str(args.config)),
        "scenario_config_sha256": file_fingerprint(config["scenario_config"]),
        "selection_inputs": "source_observation_only",
        "scenes": {},
        "proposals": [],
        "counterfactual_pairs": [],
    }
    for scene in config["scenes"]:
        scene_id = scene["scene_id"]
        scene_root = processed_root / scene_id
        raw_path = scene_root / "instances" / "instances_info.json"
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        registry_path = registry_root / scene_id / "actor_registry.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        support_counts = support["scenes"][scene_id]["actor_gaussian_count_by_model_index"]
        ego_by_frame = _ego_trajectory(scene_root, start, end)
        audits = []
        registry_by_id = {int(value["true_instance_id"]): value for value in registry["actors"]}
        for actor_id, actor in sorted(registry_by_id.items()):
            frames = _load_actor_frames(raw[str(actor_id)], start, end)
            visible_count, visible_by_camera = _visible_frames(
                frames,
                scene_root,
                config["cameras"],
                config["image_size_wh"],
                float(eligibility_cfg["min_bbox_area_fraction"]),
            )
            gaussian_count = int(support_counts.get(str(actor["rigid_model_index"]), 0))
            checks = {
                "vehicle_class": str(actor["class_name"]).startswith("vehicle"),
                "registry_mapping": True,
                "track_length": len(frames) >= int(eligibility_cfg["min_track_frames"]),
                "visible_area": visible_count >= int(eligibility_cfg["min_visible_frames"]),
                "source_lidar": int(actor["lidar_point_count"]) >= int(eligibility_cfg["min_lidar_points"]),
                "render_support": gaussian_count >= int(eligibility_cfg["min_actor_gaussians"]),
            }
            audits.append(
                {
                    "actor_id": actor_id,
                    "class_name": actor["class_name"],
                    "track_frame_count": len(frames),
                    "visible_frame_count": visible_count,
                    "visible_frames_by_camera": visible_by_camera,
                    "lidar_point_count": int(actor["lidar_point_count"]),
                    "actor_gaussian_count": gaussian_count,
                    "checks": checks,
                    "eligible": all(checks.values()),
                }
            )
        eligible = [value for value in audits if value["eligible"]]
        eligible.sort(
            key=lambda value: (
                -value["visible_frame_count"],
                -value["track_frame_count"],
                -value["lidar_point_count"],
                value["actor_id"],
            )
        )
        selected = eligible[: int(eligibility_cfg["actors_per_scene"])]
        if len(selected) < int(eligibility_cfg["actors_per_scene"]):
            raise RuntimeError(f"{scene_id}: insufficient_actor_coverage")
        scene_records = {}
        for selection in selected:
            actor_id = int(selection["actor_id"])
            source_frames = _load_actor_frames(raw[str(actor_id)], start, end)
            source_hash = trajectory_hash(source_frames)
            scene_records[str(actor_id)] = {
                "source_trajectory_hash": source_hash,
                "source_frames": source_frames,
            }
            effects = {}
            for proposal in config["proposals"]:
                requested, edit = apply_lateral_proposal(
                    source_frames,
                    ego_by_frame,
                    peak_offset_m=float(proposal["peak_lateral_offset_m"]),
                    timing=proposal["timing"],
                )
                local_source = _local_boxes(source_frames, ego_by_frame, actor_id)
                local_requested = _local_boxes(requested, ego_by_frame, actor_id)
                effect = evaluate_scenario_effect(
                    local_source,
                    local_requested,
                    ego_speed_mps=_ego_speed(
                        ego_by_frame,
                        [int(value["frame_index"]) for value in source_frames],
                        float(config["frame_period_s"]),
                    ),
                    corridor_half_width_m=float(config["corridor"]["half_width_m"]),
                    dt_s=float(config["frame_period_s"]),
                    thresholds=thresholds,
                    corridor_source=config["corridor"]["source"],
                )
                record = {
                    "scene_id": scene_id,
                    "cohort_id": scene["cohort_id"],
                    "actor_id": actor_id,
                    "proposal_id": proposal["proposal_id"],
                    "proposal_key": f"{scene_id}:{actor_id}:{proposal['proposal_id']}",
                    "source_trajectory_hash": source_hash,
                    "requested_trajectory_hash": trajectory_hash(requested),
                    "requested_frames": requested,
                    "edit": edit,
                    "requested_scenario_effect": effect,
                }
                record["proposal_record_hash"] = canonical_sha256(record)
                bank["proposals"].append(record)
                effects[proposal["proposal_id"]] = effect
            positives = sorted(key for key, value in effects.items() if value["positive"])
            negatives = sorted(key for key, value in effects.items() if value["negative"])
            if positives and negatives:
                bank["counterfactual_pairs"].append(
                    build_counterfactual_pair(
                        scene_id=scene_id,
                        source_actor_id=actor_id,
                        positive_proposal_id=positives[0],
                        negative_proposal_id=negatives[0],
                        positive_effect=effects[positives[0]],
                        negative_effect=effects[negatives[0]],
                    )
                )
        bank["scenes"][scene_id] = {
            "cohort_id": scene["cohort_id"],
            "raw_sha256": file_fingerprint(str(raw_path)),
            "registry_sha256": registry["actor_registry_sha256"],
            "support_sha256": support["scenes"][scene_id]["scene_support_sha256"],
            "eligibility_audit": audits,
            "selected_actor_ids": [int(value["actor_id"]) for value in selected],
            "selected_actor_records": scene_records,
        }
    bank["proposal_count"] = len(bank["proposals"])
    bank["actor_count_by_scene"] = {
        scene_id: len(value["selected_actor_ids"]) for scene_id, value in bank["scenes"].items()
    }
    bank["effect_distribution"] = {
        "positive": sum(value["requested_scenario_effect"]["positive"] for value in bank["proposals"]),
        "negative": sum(value["requested_scenario_effect"]["negative"] for value in bank["proposals"]),
        "non_event_or_source_positive": sum(
            not value["requested_scenario_effect"]["positive"]
            and not value["requested_scenario_effect"]["negative"]
            for value in bank["proposals"]
        ),
    }
    bank["proposal_bank_sha256"] = canonical_sha256(bank)
    atomic_write_json(str(output_root / "proposal_bank.json"), bank)
    atomic_write_json(
        str(output_root / "summary.json"),
        {
            "task_id": bank["task_id"],
            "proposal_count": bank["proposal_count"],
            "actor_count_by_scene": bank["actor_count_by_scene"],
            "effect_distribution": bank["effect_distribution"],
            "counterfactual_pair_count": len(bank["counterfactual_pairs"]),
            "proposal_bank_sha256": bank["proposal_bank_sha256"],
            "source_only_selection": True,
        },
    )
    print(json.dumps(json.loads((output_root / "summary.json").read_text()), ensure_ascii=False))


if __name__ == "__main__":
    main()

