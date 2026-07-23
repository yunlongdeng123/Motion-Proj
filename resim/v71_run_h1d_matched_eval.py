#!/usr/bin/env python
"""在冻结 proposal bank 上运行 A/B/C/D1/D2，并保存 proposal-level 原始结果。"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.resim.canonical_hash import canonical_sha256
from motion_proj.resim.certificates import Verdict
from motion_proj.resim.matched_pilot import (
    apply_lateral_proposal,
    continuous_safety_against_scene,
    occupancy_certificate,
    raw_lidar_external_violation,
    relative_kinematic_check,
    select_geometry_audit_frames,
    trajectory_bytes,
    trajectory_hash,
    visibility_certificate,
)
from motion_proj.resim.safety_geometry import GridSpec, OrientedBox
from motion_proj.resim.scenario_effect import ScenarioThresholds, evaluate_scenario_effect
from motion_proj.runtime.atomic import atomic_write_json, atomic_write_text
from motion_proj.runtime.fingerprint import file_fingerprint


def _load_raw_actor_frames(raw: dict, start: int, end: int) -> dict[int, list[dict]]:
    output = {}
    for actor_id, actor in raw.items():
        if not str(actor.get("class_name", "")).startswith("vehicle"):
            continue
        frames = []
        values = actor["frame_annotations"]
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
        if frames:
            output[int(actor_id)] = frames
    return output


def _ego(scene_root: Path, start: int, end: int) -> dict[int, np.ndarray]:
    return {
        frame: np.loadtxt(scene_root / "lidar_pose" / f"{frame:03d}.txt").reshape(4, 4)
        for frame in range(start, end + 1)
    }


def _local_boxes(frames, ego_by_frame, actor_id) -> list[OrientedBox]:
    output = []
    for value in frames:
        matrix = np.asarray(value["T_world_actor"], dtype=float)
        inverse = np.linalg.inv(ego_by_frame[int(value["frame_index"])])
        local = inverse @ matrix
        output.append(
            OrientedBox(
                tuple(local[:3, 3]),
                tuple(float(v) for v in value["dimensions_lwh"]),
                float(np.arctan2(local[1, 0], local[0, 0])),
                actor_id,
            )
        )
    return output


def _ego_speed(ego_by_frame, indices, dt_s):
    centers = np.asarray([ego_by_frame[index][:3, 3] for index in indices])
    return (
        float(np.median(np.linalg.norm(np.diff(centers, axis=0), axis=1) / dt_s))
        if len(centers) > 1
        else 0.0
    )


def _effect(frames, source, ego_by_frame, actor_id, config, scenario_cfg):
    indices = [int(value["frame_index"]) for value in frames]
    return evaluate_scenario_effect(
        _local_boxes(source, ego_by_frame, actor_id),
        _local_boxes(frames, ego_by_frame, actor_id),
        ego_speed_mps=_ego_speed(ego_by_frame, indices, float(config["frame_period_s"])),
        corridor_half_width_m=float(config["corridor"]["half_width_m"]),
        dt_s=float(config["frame_period_s"]),
        thresholds=ScenarioThresholds(
            scenario_cfg["event_min_consecutive_frames"],
            scenario_cfg["min_lateral_gap_change_m"],
            scenario_cfg["min_boundary_crossings"],
            tuple(scenario_cfg["ttc_valid_range_s"]),
            tuple(scenario_cfg["time_headway_valid_range_s"]),
        ),
        corridor_source=config["corridor"]["source"],
    )


def _candidate(
    source,
    proposal,
    scale,
    *,
    config,
    scene_root,
    evidence_root,
    ego_by_frame,
    raw_actor_frames,
    actor_gaussian_count,
    evidence_cache,
):
    frames, edit = apply_lateral_proposal(
        source,
        ego_by_frame,
        peak_offset_m=float(proposal["edit"]["peak_requested_offset_m"]),
        timing=proposal["edit"]["timing"],
        scale=float(scale),
    )
    kinematic = relative_kinematic_check(
        source,
        frames,
        ego_by_frame,
        dt_s=float(config["frame_period_s"]),
        max_delta_speed_mps=float(config["kinematic"]["max_delta_lateral_speed_mps"]),
        max_delta_acceleration_mps2=float(
            config["kinematic"]["max_delta_lateral_acceleration_mps2"]
        ),
    )
    safety = continuous_safety_against_scene(
        frames,
        actor_id=int(proposal["actor_id"]),
        raw_actor_frames=raw_actor_frames,
        ego_by_frame=ego_by_frame,
        ego_dimensions_lwh=config["safety"]["ego_dimensions_lwh"],
        clearance_m=float(config["safety"]["collision_clearance_m"]),
    )
    occupancy_cfg = config["occupancy_certificate"]
    grid_cfg = occupancy_cfg["grid"]
    occupancy = occupancy_certificate(
        frames,
        actor_id=int(proposal["actor_id"]),
        ego_by_frame=ego_by_frame,
        evidence_scene_root=evidence_root,
        grid=GridSpec(
            tuple(grid_cfg["minimum"]),
            tuple(grid_cfg["maximum"]),
            float(grid_cfg["voxel_size"]),
        ),
        lower_vertical_margin_m=float(occupancy_cfg["lower_vertical_margin_m"]),
        static_overlap_fail_voxels=int(occupancy_cfg["static_overlap_fail_voxels"]),
        minimum_known_fraction_for_pass=float(
            occupancy_cfg["minimum_known_fraction_for_pass"]
        ),
        evidence_cache=evidence_cache,
    )
    visibility_cfg = config["visibility_certificate"]
    visibility = visibility_certificate(
        frames,
        processed_scene_root=scene_root,
        cameras=config["cameras"],
        image_size_wh=config["image_size_wh"],
        min_bbox_area_fraction=float(visibility_cfg["min_bbox_area_fraction"]),
        min_visible_frames=int(visibility_cfg["min_visible_frames"]),
        actor_gaussian_count=int(actor_gaussian_count),
    )
    return {
        "frames": frames,
        "trajectory_hash": trajectory_hash(frames),
        "edit": edit,
        "kinematic": kinematic,
        "pairwise_safety": safety,
        "occupancy": occupancy,
        "visibility": visibility,
    }


def _first_candidate(candidates, required):
    for candidate in candidates:
        if all(candidate[name]["verdict"] == Verdict.PASS.value for name in required):
            return candidate
    return None


def _external(
    source,
    group,
    *,
    config,
    scene_root,
    ego_by_frame,
    raw_actor_frames,
    raw_points_cache,
    actor_id,
):
    if not group["accepted"]:
        return {
            "measurable": False,
            "hard_violation": None,
            "reason": "group_rejected_no_export",
            "components": {},
        }
    frames = group["realized_frames"]
    kinematic = relative_kinematic_check(
        source,
        frames,
        ego_by_frame,
        dt_s=float(config["frame_period_s"]),
        max_delta_speed_mps=float(config["kinematic"]["max_delta_lateral_speed_mps"]),
        max_delta_acceleration_mps2=float(
            config["kinematic"]["max_delta_lateral_acceleration_mps2"]
        ),
    )
    evaluator_cfg = config["external_evaluator"]
    safety = continuous_safety_against_scene(
        frames,
        actor_id=actor_id,
        raw_actor_frames=raw_actor_frames,
        ego_by_frame=ego_by_frame,
        ego_dimensions_lwh=config["safety"]["ego_dimensions_lwh"],
        clearance_m=float(config["safety"]["collision_clearance_m"]),
        max_translation_step_m=float(evaluator_cfg["continuous_obb_translation_step_m"]),
        max_yaw_step_rad=float(evaluator_cfg["continuous_obb_yaw_step_rad"]),
    )
    lidar = raw_lidar_external_violation(
        source,
        frames,
        processed_scene_root=scene_root,
        ego_by_frame=ego_by_frame,
        actor_id=actor_id,
        lower_vertical_margin_m=float(evaluator_cfg["lower_vertical_margin_m"]),
        minimum_points=int(evaluator_cfg["raw_lidar_min_non_source_points"]),
        raw_points_cache=raw_points_cache,
    )
    hash_match = trajectory_hash(frames) == group["realized_trajectory_hash"]
    components = {
        "relative_kinematic": kinematic,
        "continuous_obb_high_resolution": safety,
        "raw_lidar": lidar,
        "label_trajectory_hash_match": hash_match,
        "road_support": {
            "verdict": Verdict.UNKNOWN.value,
            "reason": "nuscenes_map_expansion_unavailable",
        },
    }
    hard = (
        kinematic["verdict"] == Verdict.FAIL.value
        or safety["verdict"] == Verdict.FAIL.value
        or bool(lidar["violation"])
        or not hash_match
    )
    return {
        "measurable": True,
        "hard_violation": hard,
        "reason": "external_hard_violation" if hard else "no_external_hard_violation",
        "components": components,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/resim/v71/matched_pilot_v1.yaml"))
    parser.add_argument("--proposal-bank", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    proposal_path = args.proposal_bank or Path(config["proposal_output"]) / "proposal_bank.json"
    bank = json.loads(proposal_path.read_text(encoding="utf-8"))
    bank_payload = dict(bank)
    stored_bank_hash = bank_payload.pop("proposal_bank_sha256")
    if canonical_sha256(bank_payload) != stored_bank_hash:
        raise RuntimeError("proposal bank hash mismatch")
    output_root = args.output_root or Path(config["evaluation_output"])
    if output_root.exists():
        raise FileExistsError(f"matched evaluation 已存在，拒绝覆盖: {output_root}")
    output_root.mkdir(parents=True)
    start, end = (int(v) for v in config["frame_range"])
    processed_root = Path(config["processed_root"])
    evidence_root = Path(config["evidence_root"])
    support = json.loads(Path(config["support_summary"]).read_text(encoding="utf-8"))
    scenario_cfg = yaml.safe_load(Path(config["scenario_config"]).read_text(encoding="utf-8"))
    registry_root = Path(config["registry_root"])
    scene_context = {}
    evidence_cache, raw_points_cache = {}, {}
    for scene in config["scenes"]:
        scene_id = scene["scene_id"]
        scene_root = processed_root / scene_id
        raw = json.loads((scene_root / "instances" / "instances_info.json").read_text())
        registry = json.loads((registry_root / scene_id / "actor_registry.json").read_text())
        registry_by_id = {int(v["true_instance_id"]): v for v in registry["actors"]}
        scene_context[scene_id] = {
            "root": scene_root,
            "raw": _load_raw_actor_frames(raw, start, end),
            "ego": _ego(scene_root, start, end),
            "registry": registry_by_id,
            "support_counts": support["scenes"][scene_id]["actor_gaussian_count_by_model_index"],
        }
    records = []
    trajectory_root = output_root / "trajectories"
    trajectory_root.mkdir()
    for proposal in bank["proposals"]:
        scene_id = proposal["scene_id"]
        actor_id = int(proposal["actor_id"])
        context = scene_context[scene_id]
        source = bank["scenes"][scene_id]["selected_actor_records"][str(actor_id)]["source_frames"]
        registry_actor = context["registry"][actor_id]
        gaussian_count = int(
            context["support_counts"].get(str(registry_actor["rigid_model_index"]), 0)
        )
        candidates = [
            _candidate(
                source,
                proposal,
                scale,
                config=config,
                scene_root=context["root"],
                evidence_root=evidence_root / scene_id,
                ego_by_frame=context["ego"],
                raw_actor_frames=context["raw"],
                actor_gaussian_count=gaussian_count,
                evidence_cache=evidence_cache,
            )
            for scale in config["projection"]["candidate_scales"]
        ]
        if candidates[0]["trajectory_hash"] != proposal["requested_trajectory_hash"]:
            raise RuntimeError(f"{proposal['proposal_key']}: requested trajectory drift")
        selected = {
            "A_raw_rigid": candidates[0],
            "B_kinematic": _first_candidate(candidates, ["kinematic"]),
            "C_pairwise": _first_candidate(candidates, ["kinematic", "pairwise_safety"]),
            "D2_occgs_project": _first_candidate(
                candidates, ["kinematic", "pairwise_safety", "occupancy", "visibility"]
            ),
        }
        groups = {}
        for name in ("A_raw_rigid", "B_kinematic", "C_pairwise", "D2_occgs_project"):
            candidate = selected[name]
            if candidate is None:
                groups[name] = {
                    "accepted": False,
                    "reason": "no_candidate_satisfies_required_constraints",
                    "realized_frames": None,
                    "realized_trajectory_hash": None,
                    "projection_scale": None,
                    "components": {},
                }
                continue
            effect = _effect(
                candidate["frames"], source, context["ego"], actor_id, config, scenario_cfg
            )
            groups[name] = {
                "accepted": True,
                "reason": "accepted",
                "realized_frames": candidate["frames"],
                "realized_trajectory_hash": candidate["trajectory_hash"],
                "projection_scale": candidate["edit"]["scale"],
                "projection_delta_peak_m": float(
                    proposal["edit"]["peak_requested_offset_m"]
                    * (1.0 - candidate["edit"]["scale"])
                ),
                "components": {
                    key: candidate[key]
                    for key in ("kinematic", "pairwise_safety", "occupancy", "visibility")
                },
                "scenario_effect": effect,
                "scenario_effect_retained": (
                    effect["label_transition"]
                    == proposal["requested_scenario_effect"]["label_transition"]
                ),
            }
        c_group = groups["C_pairwise"]
        if c_group["accepted"]:
            d1_frames = copy.deepcopy(c_group["realized_frames"])
            if trajectory_bytes(d1_frames) != trajectory_bytes(c_group["realized_frames"]):
                raise AssertionError("D1 failed byte reuse")
            occupancy = c_group["components"]["occupancy"]
            visibility = c_group["components"]["visibility"]
            if occupancy["verdict"] == Verdict.FAIL.value or visibility["verdict"] == Verdict.FAIL.value:
                d1_verdict = Verdict.FAIL.value
            elif occupancy["verdict"] == Verdict.UNKNOWN.value or visibility["verdict"] == Verdict.UNKNOWN.value:
                d1_verdict = Verdict.UNKNOWN.value
            else:
                d1_verdict = Verdict.PASS.value
            groups["D1_occgs_certify_only"] = {
                "accepted": True,
                "reason": "byte_reuse_C_realized_trajectory",
                "realized_frames": d1_frames,
                "realized_trajectory_hash": trajectory_hash(d1_frames),
                "projection_scale": c_group["projection_scale"],
                "projection_delta_peak_m": c_group["projection_delta_peak_m"],
                "components": {
                    "pairwise_safety": c_group["components"]["pairwise_safety"],
                    "occupancy": occupancy,
                    "visibility": visibility,
                    "road_support": {
                        "verdict": Verdict.UNKNOWN.value,
                        "reason": "nuscenes_map_expansion_unavailable",
                    },
                },
                "certificate_verdict": d1_verdict,
                "full_certificate_verdict": (
                    Verdict.FAIL.value if d1_verdict == Verdict.FAIL.value else Verdict.UNKNOWN.value
                ),
                "scenario_effect": copy.deepcopy(c_group["scenario_effect"]),
                "scenario_effect_retained": c_group["scenario_effect_retained"],
            }
        else:
            groups["D1_occgs_certify_only"] = {
                "accepted": False,
                "reason": "C_has_no_realized_trajectory",
                "realized_frames": None,
                "realized_trajectory_hash": None,
                "projection_scale": None,
                "components": {},
                "certificate_verdict": Verdict.UNKNOWN.value,
                "full_certificate_verdict": Verdict.UNKNOWN.value,
            }
        key_dir = trajectory_root / proposal["proposal_key"].replace(":", "_")
        key_dir.mkdir()
        for group_name, group in groups.items():
            if group["accepted"]:
                canonical = trajectory_bytes(group["realized_frames"]).decode("utf-8")
                atomic_write_text(str(key_dir / f"{group_name}.trajectory.canonical.json"), canonical)
        if c_group["accepted"]:
            c_path = key_dir / "C_pairwise.trajectory.canonical.json"
            d1_path = key_dir / "D1_occgs_certify_only.trajectory.canonical.json"
            if c_path.read_bytes() != d1_path.read_bytes():
                raise RuntimeError("C/D1 trajectory artifact bytes differ")
        for group_name, group in groups.items():
            group["external_evaluator"] = _external(
                source,
                group,
                config=config,
                scene_root=context["root"],
                ego_by_frame=context["ego"],
                raw_actor_frames=context["raw"],
                raw_points_cache=raw_points_cache,
                actor_id=actor_id,
            )
            if group["accepted"]:
                visibility = group["components"].get("visibility") or c_group["components"]["visibility"]
                group["render_audit_frames"] = select_geometry_audit_frames(
                    source,
                    group["realized_frames"],
                    visibility,
                    count=int(config["render_audit"]["frames_per_trajectory"]),
                )
                group["world_state_hash"] = canonical_sha256(
                    {
                        "scene_id": scene_id,
                        "actor_id": actor_id,
                        "group": group_name,
                        "realized_trajectory_hash": group["realized_trajectory_hash"],
                        "proposal_bank_sha256": stored_bank_hash,
                    }
                )
                group["label_sync"] = {
                    "trajectory_hash_recomputed": (
                        trajectory_hash(group["realized_frames"])
                        == group["realized_trajectory_hash"]
                    ),
                    "box_label_source": "same_realized_frames",
                    "pass": True,
                }
                group["adherence"] = {
                    "not_V0": float(group["projection_scale"]) > 0.0,
                    "scenario_effect_retained": bool(group["scenario_effect_retained"]),
                }
                group["adherence"]["pass"] = all(group["adherence"].values())
        record = {
            "proposal_key": proposal["proposal_key"],
            "scene_id": scene_id,
            "cohort_id": proposal["cohort_id"],
            "actor_id": actor_id,
            "proposal_id": proposal["proposal_id"],
            "requested_trajectory_hash": proposal["requested_trajectory_hash"],
            "requested_scenario_effect": proposal["requested_scenario_effect"],
            "groups": groups,
        }
        record["record_hash"] = canonical_sha256(record)
        atomic_write_json(str(output_root / f"{proposal['proposal_key'].replace(':', '_')}.json"), record)
        records.append(record)
    summary = {
        "schema_version": "h1-matched-eval-raw-v1",
        "task_id": "V7-H1-11D",
        "config_sha256": file_fingerprint(str(args.config)),
        "proposal_bank_sha256": stored_bank_hash,
        "proposal_count": len(records),
        "record_hashes": {value["proposal_key"]: value["record_hash"] for value in records},
    }
    summary["matched_eval_sha256"] = canonical_sha256(summary)
    atomic_write_json(str(output_root / "summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

