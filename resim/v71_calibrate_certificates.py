#!/usr/bin/env python
"""在真实 source controls 与机制明确的 synthetic negatives 上冻结 H1-11B certificate。"""
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
from motion_proj.resim.certificates import (
    ComponentResult,
    Verdict,
    aggregate_components,
    dynamic_collision_certificate,
    kinematic_certificate,
    unavailable_component,
)
from motion_proj.resim.safety_geometry import OrientedBox
from motion_proj.resim.scenario_effect import (
    ScenarioThresholds,
    build_counterfactual_pair,
    evaluate_scenario_effect,
)
from motion_proj.runtime.atomic import atomic_write_json
from motion_proj.runtime.fingerprint import file_fingerprint


def _box(object_to_world: np.ndarray, dimensions, actor_id: int) -> OrientedBox:
    return OrientedBox(
        tuple(object_to_world[:3, 3]),
        tuple(float(value) for value in dimensions),
        float(np.arctan2(object_to_world[1, 0], object_to_world[0, 0])),
        actor_id,
    )


def _load_trajectories(path: Path, allowed_ids: set[int], start: int, end: int) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    trajectories = {}
    for true_id, actor in raw.items():
        actor_id = int(true_id)
        if actor_id not in allowed_ids:
            continue
        frames, boxes = [], []
        values = actor["frame_annotations"]
        for frame, transform, dimensions in zip(
            values["frame_idx"], values["obj_to_world"], values["box_size"]
        ):
            frame = int(frame)
            if start <= frame <= end:
                frames.append(frame)
                boxes.append(_box(np.asarray(transform, dtype=float), dimensions, actor_id))
        if boxes:
            trajectories[actor_id] = {"frames": frames, "boxes": boxes}
    return trajectories


def _aligned_other_trajectories(actor_id: int, trajectories: dict) -> list[list[OrientedBox]]:
    own = trajectories[actor_id]
    result = []
    for other_id, other in trajectories.items():
        if other_id == actor_id:
            continue
        by_frame = dict(zip(other["frames"], other["boxes"]))
        if all(frame in by_frame for frame in own["frames"]):
            result.append([by_frame[frame] for frame in own["frames"]])
    return result


def _component_dict(value: ComponentResult) -> dict:
    return {
        "name": value.name,
        "verdict": value.verdict.value,
        "metrics": value.metrics,
        "reason": value.reason,
    }


def _grid_frame_boxes(
    scene_root: Path, trajectory: dict
) -> list[OrientedBox]:
    values = []
    for frame, world_box in zip(trajectory["frames"], trajectory["boxes"]):
        lidar_to_world = np.loadtxt(
            scene_root / "lidar_pose" / f"{frame:03d}.txt"
        ).reshape(4, 4)
        world_to_lidar = np.linalg.inv(lidar_to_world)
        center = world_to_lidar @ np.asarray([*world_box.center, 1.0])
        heading_world = np.asarray(
            [np.cos(world_box.yaw), np.sin(world_box.yaw), 0.0, 0.0]
        )
        heading_lidar = world_to_lidar @ heading_world
        values.append(
            OrientedBox(
                tuple(center[:3]), world_box.dimensions_lwh,
                float(np.arctan2(heading_lidar[1], heading_lidar[0])),
                world_box.actor_id,
            )
        )
    return values


def _scenario_fixture(thresholds: ScenarioThresholds) -> tuple[dict, dict, dict]:
    source = [
        OrientedBox((12 + index * 0.1, 4.0, 0), (4, 1.8, 1.5), 0)
        for index in range(10)
    ]
    positive_y = [4.0, 3.5, 2.8, 2.0, 1.2, 0.8, 0.4, 0.2, 0.0, 0.0]
    negative_y = [4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.4, 3.3, 3.2, 3.1]
    positive = [
        OrientedBox((12 + index * 0.1, y, 0), (4, 1.8, 1.5), 0)
        for index, y in enumerate(positive_y)
    ]
    negative = [
        OrientedBox((12 + index * 0.1, y, 0), (4, 1.8, 1.5), 0)
        for index, y in enumerate(negative_y)
    ]
    kwargs = dict(
        ego_speed_mps=6.0, corridor_half_width_m=1.75, dt_s=0.1,
        thresholds=thresholds, corridor_source="proxy",
    )
    positive_effect = evaluate_scenario_effect(source, positive, **kwargs)
    negative_effect = evaluate_scenario_effect(source, negative, **kwargs)
    pair = build_counterfactual_pair(
        scene_id="synthetic", source_actor_id=1,
        positive_proposal_id="fixture_positive",
        negative_proposal_id="fixture_negative",
        positive_effect=positive_effect, negative_effect=negative_effect,
    )
    return positive_effect, negative_effect, pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/resim/v71/certificate_calibration.yaml"),
    )
    parser.add_argument(
        "--scenario-config", type=Path,
        default=Path("configs/resim/v71/scenario_effect_v1.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    scenario_config = yaml.safe_load(args.scenario_config.read_text(encoding="utf-8"))
    output_root = args.output_root or Path(config["output_root"])
    if output_root.exists():
        raise FileExistsError(f"calibration output 已存在，拒绝覆盖: {output_root}")
    output_root.mkdir(parents=True)
    start, end = (int(value) for value in config["frame_range"])
    dt = float(config["frame_period_s"])
    threshold = config["thresholds"]
    real_controls = []
    processed_root = Path(config["processed_root"])
    evidence_summary = json.loads(
        (Path(config["evidence_root"]) / "summary.json").read_text(encoding="utf-8")
    )
    for scene_id in config["scenes"]:
        registry = json.loads(
            (Path(config["registry_root"]) / scene_id / "actor_registry.json").read_text(
                encoding="utf-8"
            )
        )
        allowed = {int(actor["true_instance_id"]) for actor in registry["actors"]}
        registry_by_id = {
            int(actor["true_instance_id"]): actor for actor in registry["actors"]
        }
        raw_path = processed_root / scene_id / "instances" / "instances_info.json"
        trajectories = _load_trajectories(raw_path, allowed, start, end)
        for actor_id, trajectory in sorted(trajectories.items()):
            centers = np.asarray([box.center for box in trajectory["boxes"]])
            times = np.asarray(trajectory["frames"], dtype=float) * dt
            kinematic = kinematic_certificate(
                centers, times,
                max_speed_mps=float(threshold["max_speed_mps"]),
                max_acceleration_mps2=float(threshold["max_acceleration_mps2"]),
                max_step_m=float(threshold["max_step_m"]),
            )
            collision = dynamic_collision_certificate(
                trajectory["boxes"],
                _aligned_other_trajectories(actor_id, trajectories),
                clearance_m=float(threshold["collision_clearance_m"]),
            )
            occupancy_id = str(actor_id + 1)
            actor_evidence = evidence_summary["scenes"][scene_id]["actors"].get(
                occupancy_id
            )
            direct_lidar_points = int(
                registry_by_id[actor_id].get("lidar_point_count", 0)
            )
            source_observation = (
                ComponentResult(
                    "source_observation", Verdict.PASS,
                    {
                        "frames": actor_evidence["frames"],
                        "base_unknown_ratio": actor_evidence["base_unknown_ratio"],
                        "direct_lidar_point_count": direct_lidar_points,
                    },
                    "direct_source_lidar_points_present",
                )
                if actor_evidence
                and actor_evidence["frames"] > 0
                and direct_lidar_points > 0
                else unavailable_component(
                    "source_observation", "no_direct_source_lidar_points"
                )
            )
            road = unavailable_component(
                "road_support", "nuscenes_map_expansion_unavailable"
            )
            components = [kinematic, collision, source_observation, road]
            measurable_verdict = aggregate_components(
                components,
                required=["kinematic", "dynamic_safety_geometry", "source_observation"],
            )
            full_verdict = aggregate_components(components)
            # real control 只用于 evaluator 可判定性，不用于挑选 event。
            grid_boxes = _grid_frame_boxes(processed_root / scene_id, trajectory)
            scenario = evaluate_scenario_effect(
                grid_boxes, grid_boxes, ego_speed_mps=5.0,
                corridor_half_width_m=1.75, dt_s=0.5,
                thresholds=ScenarioThresholds(
                    scenario_config["event_min_consecutive_frames"],
                    scenario_config["min_lateral_gap_change_m"],
                    scenario_config["min_boundary_crossings"],
                    tuple(scenario_config["ttc_valid_range_s"]),
                    tuple(scenario_config["time_headway_valid_range_s"]),
                ),
                corridor_source="proxy",
            )
            real_controls.append(
                {
                    "scene_id": scene_id,
                    "actor_id": actor_id,
                    "frame_count": len(trajectory["boxes"]),
                    "components": [_component_dict(value) for value in components],
                    "measurable_verdict": measurable_verdict.value,
                    "full_verdict": full_verdict.value,
                    "scenario_effect": scenario,
                }
            )

    # 机制明确的 controlled negatives，不从 edit outcome 调阈值。
    first = next(
        control for control in real_controls
        if control["frame_count"] >= 3
    )
    scene_trajectories = _load_trajectories(
        processed_root / first["scene_id"] / "instances" / "instances_info.json",
        {first["actor_id"]}, start, end,
    )
    seed = scene_trajectories[first["actor_id"]]
    seed_boxes = seed["boxes"][:3]
    seed_times = np.asarray(seed["frames"][:3], dtype=float) * dt
    teleport_centers = np.asarray([box.center for box in seed_boxes], dtype=float)
    teleport_centers[1:, 0] += 20.0
    teleport = kinematic_certificate(
        teleport_centers, seed_times,
        max_speed_mps=float(threshold["max_speed_mps"]),
        max_acceleration_mps2=float(threshold["max_acceleration_mps2"]),
        max_step_m=float(threshold["max_step_m"]),
    )
    collision_other = [
        OrientedBox(box.center, box.dimensions_lwh, box.yaw, 9999)
        for box in seed_boxes
    ]
    collision = dynamic_collision_certificate(seed_boxes, [collision_other])
    offroad = unavailable_component(
        "road_support", "nuscenes_map_expansion_unavailable"
    )
    controlled = [
        {
            "negative_type": "teleport",
            "expected": "FAIL",
            "result": _component_dict(teleport),
            "detectable": True,
        },
        {
            "negative_type": "collision",
            "expected": "FAIL",
            "result": _component_dict(collision),
            "detectable": True,
        },
        {
            "negative_type": "offroad_without_map",
            "expected": "UNKNOWN",
            "result": _component_dict(offroad),
            "detectable": False,
        },
    ]
    measurable = [
        value for value in real_controls
        if value["measurable_verdict"] != Verdict.UNKNOWN.value
    ]
    retention = (
        sum(value["measurable_verdict"] == Verdict.PASS.value for value in measurable)
        / len(measurable) if measurable else 0.0
    )
    detectable = [value for value in controlled if value["detectable"]]
    negative_recall = sum(
        value["result"]["verdict"] == value["expected"] for value in detectable
    ) / len(detectable)
    unknown_policy_pass = all(
        value["full_verdict"] != Verdict.PASS.value for value in real_controls
    ) and controlled[-1]["result"]["verdict"] == Verdict.UNKNOWN.value
    scenario_thresholds = ScenarioThresholds(
        scenario_config["event_min_consecutive_frames"],
        scenario_config["min_lateral_gap_change_m"],
        scenario_config["min_boundary_crossings"],
        tuple(scenario_config["ttc_valid_range_s"]),
        tuple(scenario_config["time_headway_valid_range_s"]),
    )
    fixture_positive, fixture_negative, fixture_pair = _scenario_fixture(
        scenario_thresholds
    )
    summary = {
        "schema_version": "certificate-calibration-v1",
        "task_id": "V7-H1-11B",
        "config_sha256": file_fingerprint(str(args.config)),
        "scenario_config_sha256": file_fingerprint(str(args.scenario_config)),
        "real_control_count": len(real_controls),
        "measurable_real_control_count": len(measurable),
        "measurable_real_control_retention": retention,
        "full_verdict_distribution": {
            verdict.value: sum(
                value["full_verdict"] == verdict.value for value in real_controls
            )
            for verdict in Verdict
        },
        "detectable_controlled_negative_recall": negative_recall,
        "controlled_negative_unknown_count": sum(
            value["result"]["verdict"] == Verdict.UNKNOWN.value for value in controlled
        ),
        "unknown_never_coerced_to_pass": unknown_policy_pass,
        "scenario_fixture_positive": fixture_positive["positive"],
        "scenario_fixture_negative": fixture_negative["negative"],
        "scenario_real_controls_determinate": all(
            len(value["scenario_effect"]["scenario_effect_hash"]) == 64
            for value in real_controls
        ),
        "gate": {
            "real_retention": retention
            >= float(config["targets"]["measurable_real_control_retention"]),
            "controlled_negative_detection": negative_recall
            >= float(config["targets"]["detectable_controlled_negative_recall"]),
            "offroad_without_map_is_unknown": offroad.verdict is Verdict.UNKNOWN,
            "unknown_never_pass": unknown_policy_pass,
            "scenario_effect_machine_determinate": (
                fixture_positive["positive"]
                and fixture_negative["negative"]
                and all(
                    len(value["scenario_effect"]["scenario_effect_hash"]) == 64
                    for value in real_controls
                )
            ),
        },
        "map_status": "nuscenes_map_expansion_unavailable",
        "road_support_policy": "UNKNOWN",
    }
    summary["gate"]["pass"] = all(summary["gate"].values())
    summary["calibration_sha256"] = canonical_sha256(summary)
    atomic_write_json(str(output_root / "real_controls.json"), real_controls)
    atomic_write_json(str(output_root / "controlled_negatives.json"), controlled)
    atomic_write_json(str(output_root / "fixture_pair.json"), fixture_pair)
    atomic_write_json(str(output_root / "summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
