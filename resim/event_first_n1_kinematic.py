#!/usr/bin/env python
"""Kinematics-first 第三版 N1 natural-event 海选。

第二版 full-domain N1 把 target 有多个 incoming lane 等同于 subject 正在 merge，
导致大量“主路正常续接”误报。本版先用原始 2 Hz annotation keyframe 建立
subject-specific 物理事件，再检查跨关键帧持续的 target-corridor front/rear。
正式 evaluation 使用 official train，和第二次 val 人审及 mini calibration scene-disjoint。
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from nuscenes.map_expansion.map_api import NuScenesMap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
RESIM_DIR = Path(__file__).resolve().parent
if str(RESIM_DIR) not in sys.path:
    sys.path.insert(0, str(RESIM_DIR))

from event_first_n1_event import (  # noqa: E402
    LaneIndex,
    _eligibility,
    _stable_runs,
    _transition_type,
)
from motion_proj.resim.canonical_hash import canonical_sha256
from motion_proj.resim.event_interaction import temporal_relation
from motion_proj.resim.event_kinematics import (
    lane_keeping_features,
    motion_features,
)
from motion_proj.resim.nuscenes_trainval_tracks import (
    TrainvalAnnotationSource,
    build_scene_instances_info,
)
from motion_proj.runtime.atomic import atomic_write_json, atomic_write_text
from motion_proj.runtime.fingerprint import file_fingerprint, git_state
from motion_proj.runtime.v71_contract import generate_run_id, utc_now


def _load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"配置必须是 YAML object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} 不是 JSON object")
        rows.append(value)
    return rows


def _resolve_split(
    spec: dict,
    calibration_scenes: set[str],
) -> list[str]:
    from nuscenes.utils import splits as nusplits

    if spec.get("scene_names"):
        names = list(spec["scene_names"])
    elif spec.get("split_name"):
        names = list(getattr(nusplits, str(spec["split_name"])))
    else:
        raise ValueError("evaluation 需 scene_names 或 split_name")
    exclude = set(spec.get("exclude_scenes", [])) | calibration_scenes
    names = sorted(set(names) - exclude)
    if spec.get("max_scenes") is not None:
        names = names[: int(spec["max_scenes"])]
    return names


def _frame_times_s(
    source: TrainvalAnnotationSource,
    scene: dict,
    interpolate_n: int,
) -> dict[int, float]:
    stride = interpolate_n + 1
    return {
        int(index * stride): float(source.sample_by_token[token]["timestamp"]) / 1e6
        for token, index in source.keyframe_order(scene).items()
    }


def _track_and_matches(
    scene_id: str,
    instances_info: dict,
    lane_index: LaneIndex,
    config: dict,
) -> tuple[dict, dict, dict, list]:
    tracks: dict[int, list[dict]] = {}
    matches: dict[int, list[dict]] = {}
    actor_tokens: dict[int, str] = {}
    eligibility_rows = []
    for actor_id, (instance_token, actor) in enumerate(sorted(instances_info.items())):
        audit, rows = _eligibility(actor_id, actor, config["eligibility"])
        audit["scene_id"] = scene_id
        eligibility_rows.append(audit)
        actor_tokens[actor_id] = instance_token
        if not audit["eligible"]:
            continue
        tracks[actor_id] = rows
        matches[actor_id] = [lane_index.match(row) for row in rows]
    return tracks, matches, actor_tokens, eligibility_rows


def _candidate_events(
    scene_id: str,
    lane_index: LaneIndex,
    tracks: dict[int, list[dict]],
    matches: dict[int, list[dict]],
    actor_tokens: dict[int, str],
    frame_times: dict[int, float],
    config: dict,
) -> tuple[list[dict], list[dict], list[dict]]:
    transition_config = config["transition"]
    minimum = max(
        int(transition_config["min_stable_source_frames"]),
        int(transition_config["min_stable_target_frames"]),
    )
    runs_by_actor = {
        actor_id: _stable_runs(rows, minimum) for actor_id, rows in matches.items()
    }
    match_by_actor = {
        actor_id: {int(row["frame_index"]): row for row in rows}
        for actor_id, rows in matches.items()
    }
    track_by_actor = {
        actor_id: {int(row["frame_index"]): row for row in rows}
        for actor_id, rows in tracks.items()
    }
    positives = []
    transitions = []
    for actor_id, runs in sorted(runs_by_actor.items()):
        for source_run, target_run in zip(runs, runs[1:]):
            gap = int(target_run["start_frame"]) - int(source_run["end_frame"]) - 1
            if gap > int(transition_config["max_transition_gap_frames"]):
                continue
            nominal_crossing = (
                int(source_run["end_frame"]) + int(target_run["start_frame"])
            ) // 2
            available = [
                row
                for row in tracks[actor_id]
                if abs(int(row["frame_index"]) - nominal_crossing)
                <= int(transition_config["max_transition_gap_frames"])
            ]
            if not available:
                continue
            crossing = min(
                available,
                key=lambda row: (
                    abs(int(row["frame_index"]) - nominal_crossing),
                    int(row["frame_index"]),
                ),
            )
            topology = _transition_type(
                lane_index,
                source_run["token"],
                target_run["token"],
                crossing["xy"],
                transition_config,
            )
            if topology["topology_pass"]:
                motion = motion_features(
                    tracks[actor_id],
                    source_run,
                    target_run,
                    lane_index,
                    topology,
                    config["kinematics"],
                    frame_times,
                )
            else:
                motion = {
                    "status": "NOT_EVALUATED",
                    "reason": "topology_not_candidate",
                    "physical_motion_pass": False,
                }
            relation_frame = int(target_run["start_frame"]) + int(
                transition_config["min_stable_target_frames"]
            ) - 1
            if motion.get("physical_motion_pass"):
                interaction = temporal_relation(
                    actor_id,
                    relation_frame,
                    target_run["token"],
                    match_by_actor,
                    lane_index,
                    frame_times,
                    config["interaction"],
                )
            else:
                interaction = {
                    "status": "NOT_EVALUATED",
                    "reason": "subject_physical_motion_failed",
                }
            positive = bool(
                topology["topology_pass"]
                and motion.get("physical_motion_pass")
                and interaction.get("status") == "PASS"
            )
            record = {
                "event_id": (
                    f"{scene_id}:{actor_id}:K3:"
                    f"{source_run['end_frame']}:{target_run['start_frame']}"
                ),
                "scene_id": scene_id,
                "actor_id": actor_id,
                "subject_instance_token": actor_tokens[actor_id],
                "source_run": source_run,
                "target_run": target_run,
                "transition_gap_frames": gap,
                "crossing_frame": int(crossing["frame_index"]),
                "relation_frame": relation_frame,
                "topology": topology,
                "motion": motion,
                "interaction": interaction,
                "label": (
                    "machine_positive_candidate"
                    if positive
                    else "rejected_or_unresolved_transition"
                ),
                "machine_positive": positive,
            }
            if positive:
                front_id = int(interaction["front_actor_id"])
                rear_id = int(interaction["rear_actor_id"])
                record["front_instance_token"] = actor_tokens[front_id]
                record["rear_instance_token"] = actor_tokens[rear_id]
            record["event_record_sha256"] = canonical_sha256(record)
            transitions.append(record)
            if positive:
                positives.append(record)

    negatives = []
    window = int(transition_config["negative_window_frames"])
    stride = int(config["kinematics"]["annotation_keyframe_stride"])
    positive_actors = {int(row["actor_id"]) for row in positives}
    for actor_id in sorted(positive_actors):
        actor_positives = [
            row for row in positives if int(row["actor_id"]) == actor_id
        ]
        found = False
        for run in runs_by_actor[actor_id]:
            if int(run["frame_count"]) < window:
                continue
            last_start = int(run["end_frame"]) - window + 1
            for start in range(int(run["start_frame"]), last_start + 1, stride):
                end = start + window - 1
                overlaps = any(
                    not (
                        end < int(row["source_run"]["start_frame"])
                        or start > int(row["target_run"]["end_frame"])
                    )
                    for row in actor_positives
                )
                if overlaps:
                    continue
                lane_keep = lane_keeping_features(
                    tracks[actor_id],
                    start,
                    end,
                    lane_index,
                    run["token"],
                    config["kinematics"],
                    frame_times,
                )
                if lane_keep["status"] != "PASS":
                    continue
                midpoint = (start + end) // 2
                interaction = temporal_relation(
                    actor_id,
                    midpoint,
                    run["token"],
                    match_by_actor,
                    lane_index,
                    frame_times,
                    config["interaction"],
                )
                if interaction["status"] != "PASS":
                    continue
                record = {
                    "event_id": f"{scene_id}:{actor_id}:K3N:{start}:{end}",
                    "scene_id": scene_id,
                    "actor_id": actor_id,
                    "subject_instance_token": actor_tokens[actor_id],
                    "lane_token": run["token"],
                    "start_frame": start,
                    "end_frame": end,
                    "relation_frame": midpoint,
                    "lane_keeping": lane_keep,
                    "interaction": interaction,
                    "label": "machine_negative_lane_keeping",
                    "machine_negative": True,
                }
                record["event_record_sha256"] = canonical_sha256(record)
                negatives.append(record)
                found = True
                break
            if found:
                break
    return positives, negatives, transitions


def _process_scene(
    scene_id: str,
    instances_info: dict,
    lane_index: LaneIndex,
    frame_times: dict[int, float],
    config: dict,
) -> dict:
    tracks, matches, actor_tokens, eligibility_rows = _track_and_matches(
        scene_id, instances_info, lane_index, config
    )
    positives, negatives, transitions = _candidate_events(
        scene_id,
        lane_index,
        tracks,
        matches,
        actor_tokens,
        frame_times,
        config,
    )
    pose_count = sum(len(rows) for rows in matches.values())
    matched_count = sum(
        row["match_status"] == "MATCHED"
        for rows in matches.values()
        for row in rows
    )
    return {
        "eligible_actor_count": len(tracks),
        "pose_count": pose_count,
        "matched_pose_count": matched_count,
        "eligibility_rows": eligibility_rows,
        "positives": positives,
        "negatives": negatives,
        "transitions": transitions,
    }


def _aggregate(scene_results: dict[str, dict]) -> tuple[list, list, list, list, dict]:
    positives, negatives, transitions = [], [], []
    for result in scene_results.values():
        positives.extend(result["positives"])
        negatives.extend(result["negatives"])
        transitions.extend(result["transitions"])
    positives_by_actor = defaultdict(list)
    negatives_by_actor = defaultdict(list)
    for row in positives:
        positives_by_actor[(row["scene_id"], int(row["actor_id"]))].append(row)
    for row in negatives:
        negatives_by_actor[(row["scene_id"], int(row["actor_id"]))].append(row)
    pairs = []
    for key in sorted(set(positives_by_actor) & set(negatives_by_actor)):
        positive = sorted(
            positives_by_actor[key], key=lambda row: row["event_id"]
        )[0]
        negative = sorted(
            negatives_by_actor[key], key=lambda row: row["event_id"]
        )[0]
        pair = {
            "pair_id": f"{key[0]}:{key[1]}:kinematic-positive-vs-lane-keeping",
            "scene_id": key[0],
            "actor_id": key[1],
            "positive_event_id": positive["event_id"],
            "negative_event_id": negative["event_id"],
        }
        pair["pair_sha256"] = canonical_sha256(pair)
        pairs.append(pair)
    summary = {
        "positive_candidate_count": len(positives),
        "negative_window_count": len(negatives),
        "same_actor_pair_count": len(pairs),
        "candidate_scene_count": len({row["scene_id"] for row in positives}),
        "transition_candidate_count": len(transitions),
        "topology_pass_count": sum(
            bool(row["topology"]["topology_pass"]) for row in transitions
        ),
        "physical_motion_pass_count": sum(
            bool(row["motion"].get("physical_motion_pass")) for row in transitions
        ),
        "interaction_pass_count": sum(
            row["interaction"].get("status") == "PASS" for row in transitions
        ),
    }
    return positives, negatives, transitions, pairs, summary


def _machine_gate(summary: dict, config: dict) -> tuple[bool, dict]:
    checks = {
        "positive_candidates": summary["positive_candidate_count"]
        >= int(config["min_positive_candidates"]),
        "negative_windows": summary["negative_window_count"]
        >= int(config["min_negative_windows"]),
        "same_actor_pairs": summary["same_actor_pair_count"]
        >= int(config["min_same_actor_pairs"]),
        "candidate_scenes": summary["candidate_scene_count"]
        >= int(config["min_candidate_scenes"]),
    }
    return all(checks.values()), checks


def _calibration_audit(
    config: dict,
    source: TrainvalAnnotationSource,
    lane_indices: dict[str, LaneIndex],
) -> dict:
    calibration = config["calibration"]
    review_path = Path(calibration["completed_review_file"])
    if file_fingerprint(str(review_path)).lower() != str(
        calibration["completed_review_sha256"]
    ).lower():
        raise RuntimeError("calibration review SHA256 不匹配")
    audit_reject = Path(calibration["audit_reject_run"])
    if not (audit_reject / "REJECTED").is_file():
        raise RuntimeError("第二次 N1 人审 reject run 不完整")
    parent = Path(calibration["parent_run"])
    pool = json.loads((parent / "event_pool.json").read_text(encoding="utf-8"))
    labels = {row["event_id"]: row for row in _read_jsonl(review_path)}
    events = pool["evaluation"]["positives"]
    if {row["event_id"] for row in events} != set(labels):
        raise RuntimeError("calibration label/event 集合不一致")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        grouped[event["scene_id"]].append(event)

    rows = []
    cache_dir = Path(
        "/root/autodl-tmp/data/occgs/processed_10Hz/trainval_annots"
    )
    for scene_name, scene_events in sorted(grouped.items()):
        scene = source.resolve_scene(scene_name)
        map_name = source.map_name_by_scene[scene["token"]]
        lane_index = lane_indices[map_name]
        instances_info = json.loads(
            (
                cache_dir / scene_name / "instances" / "instances_info.json"
            ).read_text(encoding="utf-8")
        )
        tracks, matches, _, _ = _track_and_matches(
            scene_name, instances_info, lane_index, config
        )
        match_by_actor = {
            actor_id: {int(row["frame_index"]): row for row in values}
            for actor_id, values in matches.items()
        }
        frame_times = _frame_times_s(source, scene, int(config["interpolate_n"]))
        for event in scene_events:
            actor_id = int(event["actor_id"])
            motion = motion_features(
                tracks[actor_id],
                event["source_run"],
                event["target_run"],
                lane_index,
                event["topology"],
                config["kinematics"],
                frame_times,
            )
            if motion["physical_motion_pass"]:
                interaction = temporal_relation(
                    actor_id,
                    int(event["relation_frame"]),
                    event["target_run"]["token"],
                    match_by_actor,
                    lane_index,
                    frame_times,
                    config["interaction"],
                )
            else:
                interaction = {
                    "status": "NOT_EVALUATED",
                    "reason": "subject_physical_motion_failed",
                }
            retained = bool(
                motion["physical_motion_pass"]
                and interaction.get("status") == "PASS"
            )
            rows.append(
                {
                    "event_id": event["event_id"],
                    "scene_id": scene_name,
                    "human_verdict": labels[event["event_id"]]["verdict"],
                    "retained_by_v3": retained,
                    "motion": motion,
                    "interaction": interaction,
                }
            )
    counts = defaultdict(int)
    for row in rows:
        counts[(row["human_verdict"], row["retained_by_v3"])] += 1
    true_total = sum(row["human_verdict"] == "TRUE_POSITIVE" for row in rows)
    false_total = sum(row["human_verdict"] == "FALSE_POSITIVE" for row in rows)
    retained_true = counts[("TRUE_POSITIVE", True)]
    rejected_false = counts[("FALSE_POSITIVE", False)]
    return {
        "source": "second N1 human audit; calibration only",
        "scene_disjoint_from_evaluation": True,
        "completed_review_sha256": calibration["completed_review_sha256"],
        "event_count": len(rows),
        "human_true_positive_count": true_total,
        "human_false_positive_count": false_total,
        "retained_human_true_positive_count": retained_true,
        "rejected_human_false_positive_count": rejected_false,
        "calibration_true_positive_recall": (
            retained_true / true_total if true_total else None
        ),
        "calibration_false_positive_rejection_rate": (
            rejected_false / false_total if false_total else None
        ),
        "rows": rows,
    }


def run(
    config_path: Path,
    output_root: Path | None = None,
    allow_dirty_development: bool = False,
    skip_audit_panels: bool = False,
    max_evaluation_scenes_development: int | None = None,
    force_audit_development: bool = False,
) -> Path:
    started_at = utc_now()
    config = _load_yaml(config_path)
    repo_root = Path(config["repo_root"])
    code = git_state(str(repo_root))
    formal = not allow_dirty_development
    if formal and bool(config.get("require_clean_git", True)) and code["dirty"]:
        raise RuntimeError("正式 N1-K3 必须在 clean git worktree 上运行")
    if formal and skip_audit_panels:
        raise RuntimeError("正式 N1-K3 不允许跳过人工审计材料")
    if formal and force_audit_development:
        raise RuntimeError("正式 N1-K3 不允许强制越过 machine gate")

    n0_run = Path(config["n0_run"])
    if not (n0_run / "COMPLETE").is_file():
        raise RuntimeError("N0 未 COMPLETE")
    n0_summary = json.loads((n0_run / "summary.json").read_text(encoding="utf-8"))
    if n0_summary["research_verdict"] != "n0_asset_pass":
        raise RuntimeError("N0 verdict 非 n0_asset_pass")

    source = TrainvalAnnotationSource(Path(config["dataset_root"]))
    calibration_scenes = {
        row["scene"]
        for row in _read_jsonl(Path(config["calibration"]["completed_review_file"]))
    }
    evaluation_names = _resolve_split(config["evaluation"], calibration_scenes)
    if max_evaluation_scenes_development is not None:
        if formal:
            raise RuntimeError("正式 run 不允许 development scene 截断")
        evaluation_names = evaluation_names[:max_evaluation_scenes_development]
    if calibration_scenes & set(evaluation_names):
        raise RuntimeError("calibration/evaluation scene 未分离")
    config_sha = file_fingerprint(str(config_path))
    tag = "kinematic-v1" if formal else "kinematic-v1-dev"
    run_id = generate_run_id(config["task_id"], tag, int(config["seed"]), config_sha)
    run_dir = (output_root or Path(config["run_root"])) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_text(str(run_dir / "RUNNING"), "n1_kinematic_screen\n")
    atomic_write_text(str(run_dir / "resolved.yaml"), config_path.read_text(encoding="utf-8"))

    lane_indices = {
        map_name: LaneIndex(
            NuScenesMap(str(config["dataset_root"]), map_name),
            config["map_matching"],
        )
        for map_name in (
            "boston-seaport",
            "singapore-hollandvillage",
            "singapore-onenorth",
            "singapore-queenstown",
        )
    }
    calibration_audit = _calibration_audit(config, source, lane_indices)
    atomic_write_json(str(run_dir / "calibration_audit.json"), calibration_audit)

    scene_results = {}
    cache_dir = Path(config["cache_dir"])
    batch_size = int(config["scene_batch_size"])
    for batch_start in range(0, len(evaluation_names), batch_size):
        batch = evaluation_names[batch_start : batch_start + batch_size]
        meta = build_scene_instances_info(
            Path(config["dataset_root"]),
            batch,
            int(config["interpolate_n"]),
            cache_dir,
            retain=False,
        )
        for scene_name in batch:
            entry = meta[scene_name]
            instances_info = json.loads(
                Path(entry["cache_path"]).read_text(encoding="utf-8")
            )
            scene = source.resolve_scene(scene_name)
            result = _process_scene(
                scene_name,
                instances_info,
                lane_indices[entry["map_name"]],
                _frame_times_s(source, scene, int(config["interpolate_n"])),
                config,
            )
            result["map_name"] = entry["map_name"]
            result["scene_token"] = entry["scene_token"]
            scene_results[scene_name] = result
        print(
            json.dumps(
                {
                    "processed_scenes": min(
                        batch_start + len(batch), len(evaluation_names)
                    ),
                    "evaluation_scenes": len(evaluation_names),
                }
            ),
            flush=True,
        )

    positives, negatives, transitions, pairs, base_summary = _aggregate(scene_results)
    machine_passed, machine_checks = _machine_gate(
        base_summary, config["machine_gates"]
    )
    event_pool = {
        "schema_version": config["schema_version"],
        "task_id": config["task_id"],
        "seed": int(config["seed"]),
        "selection_inputs": (
            "nuscenes_2hz_annotation_keyframe_kinematics_plus_frozen_vector_map"
        ),
        "calibration_used_for_threshold_design_only": True,
        "evaluation_split": str(config["evaluation"]["split_name"]),
        "evaluation_scene_count": len(evaluation_names),
        "positives": positives,
        "negatives": negatives,
        "transition_candidates": transitions,
        "same_actor_pairs": pairs,
    }
    event_pool["event_pool_sha256"] = canonical_sha256(event_pool)
    atomic_write_json(str(run_dir / "event_pool.json"), event_pool)
    evaluation_audit = {
        scene_name: {
            "scene_token": result["scene_token"],
            "map_name": result["map_name"],
            "eligible_actor_count": result["eligible_actor_count"],
            "pose_count": result["pose_count"],
            "matched_pose_count": result["matched_pose_count"],
            "matched_pose_fraction": (
                result["matched_pose_count"] / result["pose_count"]
                if result["pose_count"]
                else None
            ),
            "transition_count": len(result["transitions"]),
            "topology_pass_count": sum(
                row["topology"]["topology_pass"]
                for row in result["transitions"]
            ),
            "physical_motion_pass_count": sum(
                row["motion"].get("physical_motion_pass", False)
                for row in result["transitions"]
            ),
            "positive_candidate_count": len(result["positives"]),
            "negative_window_count": len(result["negatives"]),
        }
        for scene_name, result in sorted(scene_results.items())
    }
    atomic_write_json(str(run_dir / "evaluation_scene_audit.json"), evaluation_audit)

    audit_ready = (
        base_summary["positive_candidate_count"]
        >= int(config["audit_readiness"]["min_positive_candidates"])
    )
    audit_manifest = None
    if (audit_ready or force_audit_development) and not skip_audit_panels:
        from motion_proj.resim.n1_kinematic_audit import build_audit_pack

        audit_manifest = build_audit_pack(
            run_dir=run_dir,
            positives=positives,
            config=config,
            source=source,
            lane_indices=lane_indices,
            machine_summary=base_summary,
            machine_checks=machine_checks,
        )
    if audit_ready:
        terminal = "AWAITING_HUMAN_REVIEW"
        verdict = "await_human_review_n1_kinematic"
    else:
        terminal = "REJECTED"
        verdict = config["stop_rule"]["on_no_audit_candidate"]
    summary = {
        "task_id": config["task_id"],
        "run_id": run_id,
        "seed": int(config["seed"]),
        "formal": formal,
        "calibration_scene_count": len(calibration_scenes),
        "evaluation_scene_count": len(evaluation_names),
        "evaluation_split": config["evaluation"]["split_name"],
        "calibration": {
            key: value
            for key, value in calibration_audit.items()
            if key != "rows"
        },
        "evaluation": base_summary,
        "machine_gate_checks": machine_checks,
        "machine_gate_passed": machine_passed,
        "audit_ready": audit_ready,
        "human_audit_item_count": (
            audit_manifest["audit_item_count"] if audit_manifest else 0
        ),
        "human_verdict_filled": False,
        "n2_authorized": False,
        "terminal_status": terminal,
        "research_verdict": verdict,
        "event_pool_sha256": event_pool["event_pool_sha256"],
    }
    data_fingerprint = canonical_sha256(
        {
            "n0_asset_manifest_sha256": n0_summary["asset_manifest_sha256"],
            "trainval_scene_json_sha256": file_fingerprint(
                str(Path(config["dataset_root"]) / "v1.0-trainval" / "scene.json")
            ),
            "trainval_sample_json_sha256": file_fingerprint(
                str(Path(config["dataset_root"]) / "v1.0-trainval" / "sample.json")
            ),
            "trainval_sample_annotation_json_sha256": file_fingerprint(
                str(
                    Path(config["dataset_root"])
                    / "v1.0-trainval"
                    / "sample_annotation.json"
                )
            ),
            "trainval_instance_json_sha256": file_fingerprint(
                str(Path(config["dataset_root"]) / "v1.0-trainval" / "instance.json")
            ),
            "trainval_category_json_sha256": file_fingerprint(
                str(Path(config["dataset_root"]) / "v1.0-trainval" / "category.json")
            ),
            "trainval_log_json_sha256": file_fingerprint(
                str(Path(config["dataset_root"]) / "v1.0-trainval" / "log.json")
            ),
            "calibration_review_sha256": config["calibration"][
                "completed_review_sha256"
            ],
            "evaluation_scenes": evaluation_names,
        }
    )
    artifact_hashes = {
        "calibration_audit": file_fingerprint(str(run_dir / "calibration_audit.json")),
        "evaluation_scene_audit": file_fingerprint(
            str(run_dir / "evaluation_scene_audit.json")
        ),
        "event_pool": file_fingerprint(str(run_dir / "event_pool.json")),
    }
    if audit_manifest:
        artifact_hashes["audit_manifest"] = file_fingerprint(
            str(run_dir / "audit" / "audit_manifest.json")
        )
    manifest = {
        "schema_version": 1,
        "task_id": config["task_id"],
        "run_id": run_id,
        "command": list(sys.argv),
        "formal": formal,
        "code_commit": code["commit"],
        "code_dirty": code["dirty"],
        "dirty_diff_hash": code["dirty_diff_hash"],
        "config_fingerprint": config_sha,
        "data_fingerprint": data_fingerprint,
        "artifact_hashes": artifact_hashes,
        "artifact_set_sha256": canonical_sha256(artifact_hashes),
        "n0_run": str(n0_run),
        "calibration_audit_reject_run": config["calibration"]["audit_reject_run"],
        "calibration_scenes": sorted(calibration_scenes),
        "evaluation_scenes": evaluation_names,
        "seed": int(config["seed"]),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "started_at": started_at,
        "ended_at": utc_now(),
        "terminal_status": terminal,
        "exit_reason": verdict,
        "n2_authorized": False,
    }
    atomic_write_json(str(run_dir / "summary.json"), summary)
    atomic_write_json(str(run_dir / "manifest.json"), manifest)
    atomic_write_text(
        str(run_dir / "metrics.jsonl"),
        json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n",
    )
    (run_dir / "RUNNING").unlink()
    atomic_write_text(str(run_dir / terminal), verdict + "\n")
    print(json.dumps({"run_dir": str(run_dir), **summary}, ensure_ascii=False))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/resim/event_first_n1_kinematic_v1.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--allow-dirty-development", action="store_true")
    parser.add_argument("--skip-audit-panels", action="store_true")
    parser.add_argument("--max-evaluation-scenes-development", type=int)
    parser.add_argument("--force-audit-development", action="store_true")
    args = parser.parse_args()
    run(
        args.config,
        args.output_root,
        allow_dirty_development=args.allow_dirty_development,
        skip_audit_panels=args.skip_audit_panels,
        max_evaluation_scenes_development=args.max_evaluation_scenes_development,
        force_audit_development=args.force_audit_development,
    )


if __name__ == "__main__":
    main()
