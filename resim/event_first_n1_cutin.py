#!/usr/bin/env python
"""第四版 N1：以目标车道接收车为参照挖掘真实 cut-in。

本版把第三次审核的 12 个假阳性和第二次审核标签只作为 calibration。
正式 evaluation 使用与所有已人工审核 scene 分离的 official train。任何正式
run 都只生成 N1 候选与第四次人工审核包，绝不授权或启动 N2。
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
RESIM_DIR = Path(__file__).resolve().parent
if str(RESIM_DIR) not in sys.path:
    sys.path.insert(0, str(RESIM_DIR))

from event_first_n1_event import LaneIndex, _stable_runs, _transition_type  # noqa: E402
from event_first_n1_kinematic import (  # noqa: E402
    _frame_times_s,
    _load_yaml,
    _read_jsonl,
    _track_and_matches,
)
from motion_proj.resim.canonical_hash import canonical_sha256  # noqa: E402
from motion_proj.resim.cutin_receiver import (  # noqa: E402
    lane_keeping_receiver,
    receiver_centric_cutin,
    receiver_centric_cutin_v2,
)
from motion_proj.resim.event_kinematics import lane_keeping_features  # noqa: E402
from motion_proj.resim.io_memory import (  # noqa: E402
    advise_sequential,
    drop_handle_page_cache,
    memory_snapshot,
    page_cache_control_available,
    trim_process_heap,
)
from motion_proj.resim.lightweight_nuscenes_map import (  # noqa: E402
    LightweightNuScenesMap,
)
from motion_proj.resim.nuscenes_trainval_tracks import (  # noqa: E402
    TrainvalAnnotationSource,
    build_scene_instances_info,
)
from motion_proj.runtime.atomic import atomic_write_json, atomic_write_text  # noqa: E402
from motion_proj.runtime.fingerprint import file_fingerprint, git_state  # noqa: E402
from motion_proj.runtime.v71_contract import generate_run_id, utc_now  # noqa: E402
from scripts.validate_n1_kinematic_review import validate as validate_k3_review  # noqa: E402


MAP_NAMES = (
    "boston-seaport",
    "singapore-hollandvillage",
    "singapore-onenorth",
    "singapore-queenstown",
)

_ACTIVE_RUN_DIR: Path | None = None

_REQUIRED_CONFIG_PATHS = (
    ("task_id",),
    ("seed",),
    ("repo_root",),
    ("dataset_root",),
    ("cache_dir",),
    ("run_root",),
    ("n0_run",),
    ("interpolate_n",),
    ("scene_batch_size",),
    ("calibration",),
    ("evaluation",),
    ("eligibility",),
    ("map_matching",),
    ("transition", "negative_window_frames"),
    ("transition", "positive_exclusion_guard_frames"),
    ("cutin", "annotation_keyframe_stride"),
    ("cutin", "min_median_speed_mps"),
    ("kinematics_control", "annotation_keyframe_stride"),
    ("kinematics_control", "dense_frame_period_s"),
    ("kinematics_control", "min_median_speed_mps"),
    ("kinematics_control", "max_acceleration_mps2"),
    ("kinematics_control", "negative_min_keyframes"),
    ("kinematics_control", "negative_max_centerline_distance_m"),
    ("kinematics_control", "negative_max_lateral_span_m"),
    ("kinematics_control", "negative_max_heading_error_deg"),
    ("machine_gates",),
    ("audit_readiness",),
    ("human_audit", "review_schema"),
    ("stop_rule", "never_start_n2_from_this_run"),
)


def _validate_config_contract(config: dict) -> None:
    missing = []
    for path in _REQUIRED_CONFIG_PATHS:
        value = config
        for key in path:
            if not isinstance(value, dict) or key not in value:
                missing.append(".".join(path))
                break
            value = value[key]
    if missing:
        raise ValueError(
            "K4 config contract missing required keys: " + ", ".join(missing)
        )
    if config["human_audit"]["review_schema"] != "receiver_cutin":
        raise ValueError("K4 human_audit.review_schema must be receiver_cutin")
    if config["stop_rule"]["never_start_n2_from_this_run"] is not True:
        raise ValueError("K4 stop_rule must keep N2 fail-closed")


def _mark_active_run_failed(exc: BaseException) -> None:
    run_dir = _ACTIVE_RUN_DIR
    if run_dir is None or not run_dir.is_dir():
        return
    failure = {
        "schema_version": "engineering-failure-v1",
        "task_id": "N1-EVENT-CUTIN-01",
        "terminal_status": "FAILED",
        "exit_reason": "uncaught_engineering_exception",
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": traceback.format_exc(),
        "written_research_outputs_before_failure": (run_dir / "event_pool.json").is_file(),
        "n2_authorized": False,
        "recovery": "preserve this directory, fix the defect, and use a new run ID",
    }
    atomic_write_json(str(run_dir / "failure.json"), failure)
    atomic_write_text(str(run_dir / "FAILED"), "uncaught_engineering_exception\n")
    running = run_dir / "RUNNING"
    if running.is_file():
        running.unlink()


class LaneIndexCache:
    """一次只常驻一个 location 的地图索引，避免四图叠加触发内存上限。"""

    def __init__(self, dataset_root: Path, map_matching: dict):
        self.dataset_root = Path(dataset_root)
        self.map_matching = map_matching
        self._map_name: str | None = None
        self._index: LaneIndex | None = None

    def __getitem__(self, map_name: str) -> LaneIndex:
        if map_name not in MAP_NAMES:
            raise KeyError(f"未知 nuScenes map location: {map_name}")
        if map_name != self._map_name:
            self.clear()
            print(
                json.dumps({"phase": "load_map", "map_name": map_name}),
                flush=True,
            )
            self._index = LaneIndex(
                LightweightNuScenesMap(self.dataset_root, map_name),
                self.map_matching,
            )
            self._map_name = map_name
        assert self._index is not None
        return self._index

    def clear(self) -> None:
        self._index = None
        self._map_name = None
        gc.collect()


def _resolve_evaluation_scenes(spec: dict, calibration_scenes: set[str]) -> list[str]:
    from nuscenes.utils import splits as nusplits

    if spec.get("scene_names"):
        names = list(spec["scene_names"])
    elif spec.get("split_name"):
        names = list(getattr(nusplits, str(spec["split_name"])))
    else:
        raise ValueError("evaluation 需 scene_names 或 split_name")
    excluded = set(spec.get("exclude_scenes", [])) | calibration_scenes
    names = sorted(set(names) - excluded)
    if spec.get("max_scenes") is not None:
        names = names[: int(spec["max_scenes"])]
    return names


def _read_json_bounded(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        advise_sequential(handle)
        try:
            return json.load(handle)
        finally:
            drop_handle_page_cache(handle)


def _transition_records(
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
    stride = int(config["cutin"]["annotation_keyframe_stride"])
    runs_by_actor = {
        actor_id: _stable_runs(rows, minimum) for actor_id, rows in matches.items()
    }
    matches_by_actor = {
        actor_id: {int(row["frame_index"]): row for row in rows}
        for actor_id, rows in matches.items()
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
                cutin = receiver_centric_cutin(
                    actor_id,
                    source_run,
                    target_run,
                    topology,
                    tracks,
                    matches_by_actor,
                    lane_index,
                    frame_times,
                    config["cutin"],
                )
            else:
                cutin = {
                    "schema_version": "receiver-centric-cutin-v1",
                    "status": "NOT_EVALUATED",
                    "reason": "topology_not_candidate",
                    "event_pass": False,
                    "uses_interpolated_physics": False,
                }
            positive = bool(topology["topology_pass"] and cutin.get("event_pass"))
            event_start = (
                int(cutin["pre_keyframes"][0]["frame"])
                if positive
                else int(round(nominal_crossing / stride) * stride - 2 * stride)
            )
            event_end = (
                int(cutin["post_keyframes"][-1]["frame"])
                if positive
                else int(round(nominal_crossing / stride) * stride + 2 * stride)
            )
            record = {
                "event_id": (
                    f"{scene_id}:{actor_id}:K4:"
                    f"{source_run['end_frame']}:{target_run['start_frame']}"
                ),
                "scene_id": scene_id,
                "actor_id": actor_id,
                "subject_instance_token": actor_tokens[actor_id],
                "source_run": source_run,
                "target_run": target_run,
                "transition_gap_frames": gap,
                "crossing_frame": int(crossing["frame_index"]),
                "event_start_frame": event_start,
                "event_end_frame": event_end,
                "topology": topology,
                "cutin": cutin,
                "label": (
                    "machine_positive_receiver_cutin"
                    if positive
                    else "rejected_or_unresolved_transition"
                ),
                "machine_positive": positive,
            }
            if positive:
                receiver_id = int(cutin["receiver_actor_id"])
                record["relation_frame"] = int(cutin["relation_frame"])
                record["receiver_actor_id"] = receiver_id
                record["receiver_instance_token"] = actor_tokens[receiver_id]
                front_id = cutin.get("front_actor_id")
                if front_id is not None:
                    record["front_actor_id"] = int(front_id)
                    record["front_instance_token"] = actor_tokens[int(front_id)]
                record["maneuver_mode"] = (
                    "parallel_lane_change"
                    if topology["type"] == "lane_change"
                    else "receiver_branch_merge"
                )
            record["event_record_sha256"] = canonical_sha256(record)
            transitions.append(record)
            if positive:
                positives.append(record)

    # 同一 actor 的重叠 transition 只保留最早、接收车支持最强的一条。
    deduplicated = []
    for actor_id in sorted({int(row["actor_id"]) for row in positives}):
        actor_rows = sorted(
            (row for row in positives if int(row["actor_id"]) == actor_id),
            key=lambda row: (
                int(row["event_start_frame"]),
                -int(row["cutin"]["receiver_pre_support_keyframes"]),
                -int(row["cutin"]["receiver_post_support_keyframes"]),
                row["event_id"],
            ),
        )
        for row in actor_rows:
            if any(
                not (
                    int(row["event_end_frame"]) < int(kept["event_start_frame"])
                    or int(row["event_start_frame"]) > int(kept["event_end_frame"])
                )
                for kept in deduplicated
                if int(kept["actor_id"]) == actor_id
            ):
                row["deduplicated_positive"] = False
                row["machine_positive"] = False
                row["label"] = "deduplicated_overlapping_receiver_cutin"
                row.pop("event_record_sha256", None)
                row["event_record_sha256"] = canonical_sha256(row)
                continue
            row["deduplicated_positive"] = True
            row.pop("event_record_sha256", None)
            row["event_record_sha256"] = canonical_sha256(row)
            deduplicated.append(row)
    positives = deduplicated

    negatives = []
    window = int(transition_config["negative_window_frames"])
    guard = int(transition_config["positive_exclusion_guard_frames"])
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
                        end < int(row["event_start_frame"]) - guard
                        or start > int(row["event_end_frame"]) + guard
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
                    config["kinematics_control"],
                    frame_times,
                )
                if lane_keep["status"] != "PASS":
                    continue
                midpoint = int(round(((start + end) / 2.0) / stride) * stride)
                receiver = lane_keeping_receiver(
                    actor_id,
                    midpoint,
                    run["token"],
                    matches_by_actor,
                    lane_index,
                    frame_times,
                    config["cutin"],
                )
                if receiver["status"] != "PASS":
                    continue
                receiver_id = int(receiver["receiver_actor_id"])
                record = {
                    "event_id": f"{scene_id}:{actor_id}:K4N:{start}:{end}",
                    "scene_id": scene_id,
                    "actor_id": actor_id,
                    "subject_instance_token": actor_tokens[actor_id],
                    "receiver_actor_id": receiver_id,
                    "receiver_instance_token": actor_tokens[receiver_id],
                    "lane_token": run["token"],
                    "start_frame": start,
                    "end_frame": end,
                    "relation_frame": midpoint,
                    "lane_keeping": lane_keep,
                    "receiver_control": receiver,
                    "label": "machine_negative_lane_keep_with_receiver",
                    "machine_negative": True,
                    "positive_overlap": False,
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
    tracks, matches, actor_tokens, _ = _track_and_matches(
        scene_id, instances_info, lane_index, config
    )
    positives, negatives, transitions = _transition_records(
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
    positive_by_actor = defaultdict(list)
    negative_by_actor = defaultdict(list)
    for row in positives:
        positive_by_actor[(row["scene_id"], int(row["actor_id"]))].append(row)
    for row in negatives:
        negative_by_actor[(row["scene_id"], int(row["actor_id"]))].append(row)
    pairs = []
    for key in sorted(set(positive_by_actor) & set(negative_by_actor)):
        positive = sorted(positive_by_actor[key], key=lambda row: row["event_id"])[0]
        negative = sorted(negative_by_actor[key], key=lambda row: row["event_id"])[0]
        pair = {
            "pair_id": f"{key[0]}:{key[1]}:receiver-cutin-vs-lane-keep",
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
        "subject_entry_pass_count": sum(
            bool(row["cutin"].get("subject_entry_pass")) for row in transitions
        ),
        "receiver_interaction_pass_count": sum(
            bool(row["cutin"].get("receiver_interaction_pass"))
            for row in transitions
        ),
        "mode_counts": dict(
            sorted(Counter(row["maneuver_mode"] for row in positives).items())
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


def _assert_event_record_hashes(rows: list[dict]) -> None:
    for row in rows:
        expected = row.get("event_record_sha256")
        payload = {
            key: value
            for key, value in row.items()
            if key != "event_record_sha256"
        }
        if expected != canonical_sha256(payload):
            raise RuntimeError(f"event_record_sha256 不匹配: {row.get('event_id')}")


def _load_calibration_events(config: dict) -> tuple[list[dict], set[str], dict]:
    calibration = config["calibration"]
    sources = []
    scenes: set[str] = set()

    second = calibration["second_audit"]
    second_review = Path(second["completed_review_file"])
    if file_fingerprint(str(second_review)) != second["completed_review_sha256"]:
        raise RuntimeError("第二次人审 SHA256 不匹配")
    if not (Path(second["audit_reject_run"]) / "REJECTED").is_file():
        raise RuntimeError("第二次人审 adjudication 未 REJECTED")
    second_pool = json.loads(
        (Path(second["parent_run"]) / "event_pool.json").read_text(encoding="utf-8")
    )
    second_labels = {
        row["event_id"]: row["verdict"] for row in _read_jsonl(second_review)
    }
    for event in second_pool["evaluation"]["positives"]:
        if event["event_id"] not in second_labels:
            raise RuntimeError("第二次 calibration event/label 不完整")
        scenes.add(event["scene_id"])
        sources.append(
            {
                "audit_source": "second_n1",
                "human_verdict": second_labels[event["event_id"]],
                "event": event,
            }
        )

    third = calibration["third_audit"]
    third_parent = Path(third["parent_run"])
    third_review = Path(third["completed_review_file"])
    if file_fingerprint(str(third_review)) != third["completed_review_sha256"]:
        raise RuntimeError("第三次人审 SHA256 不匹配")
    if not (Path(third["audit_reject_run"]) / "REJECTED").is_file():
        raise RuntimeError("第三次人审 adjudication 未 REJECTED")
    validation = validate_k3_review(third_parent, third_review)
    if validation["all_human_gates_passed"]:
        raise RuntimeError("第三次人审意外通过")
    third_pool = json.loads(
        (third_parent / "event_pool.json").read_text(encoding="utf-8")
    )
    third_events = {row["event_id"]: row for row in third_pool["positives"]}
    third_labels = {
        row["audit_id"]: row["overall_verdict"]
        for row in _read_jsonl(third_review)
    }
    evidence_dir = third_parent / "audit" / "evidence"
    for audit_id, verdict in sorted(third_labels.items()):
        evidence = json.loads(
            (evidence_dir / f"{audit_id}.json").read_text(encoding="utf-8")
        )
        event = third_events[evidence["event_id"]]
        scenes.add(event["scene_id"])
        sources.append(
            {
                "audit_source": "third_n1",
                "audit_id": audit_id,
                "human_verdict": verdict,
                "event": event,
            }
        )
    provenance = {
        "second_review_sha256": second["completed_review_sha256"],
        "third_review_sha256": third["completed_review_sha256"],
        "second_event_count": len(second_labels),
        "third_event_count": len(third_labels),
    }
    return sources, scenes, provenance


def _calibration_audit(
    config: dict,
    source: TrainvalAnnotationSource,
    lane_indices: LaneIndexCache,
) -> dict:
    labeled, scenes, provenance = _load_calibration_events(config)
    grouped = defaultdict(list)
    for row in labeled:
        grouped[row["event"]["scene_id"]].append(row)
    rows = []
    cache_dir = Path(config["cache_dir"]) / "calibration"
    ordered_scenes = sorted(
        grouped,
        key=lambda name: (
            source.map_name_by_scene[source.resolve_scene(name)["token"]],
            name,
        ),
    )
    for scene_names in [ordered_scenes]:
        cache_meta = build_scene_instances_info(
            Path(config["dataset_root"]),
            scene_names,
            int(config["interpolate_n"]),
            cache_dir,
            retain=False,
            source=source,
        )
        for scene_name in scene_names:
            scene = source.resolve_scene(scene_name)
            entry = cache_meta[scene_name]
            instances_info = json.loads(
                Path(entry["cache_path"]).read_text(encoding="utf-8")
            )
            tracks, matches, _, _ = _track_and_matches(
                scene_name,
                instances_info,
                lane_indices[entry["map_name"]],
                config,
            )
            matches_by_actor = {
                actor_id: {int(item["frame_index"]): item for item in values}
                for actor_id, values in matches.items()
            }
            frame_times = _frame_times_s(
                source, scene, int(config["interpolate_n"])
            )
            for labeled_row in grouped[scene_name]:
                event = labeled_row["event"]
                actor_id = int(event["actor_id"])
                if actor_id not in tracks:
                    result = {
                        "status": "UNKNOWN",
                        "reason": "calibration_actor_not_eligible",
                        "event_pass": False,
                    }
                else:
                    result = receiver_centric_cutin(
                        actor_id,
                        event["source_run"],
                        event["target_run"],
                        event["topology"],
                        tracks,
                        matches_by_actor,
                        lane_indices[entry["map_name"]],
                        frame_times,
                        config["cutin"],
                    )
                rows.append(
                    {
                        "audit_source": labeled_row["audit_source"],
                        "audit_id": labeled_row.get("audit_id"),
                        "event_id": event["event_id"],
                        "scene_id": scene_name,
                        "human_verdict": labeled_row["human_verdict"],
                        "retained_by_v4": bool(result.get("event_pass")),
                        "result": result,
                    }
                )
    counts = Counter(
        (
            row["audit_source"],
            row["human_verdict"],
            bool(row["retained_by_v4"]),
        )
        for row in rows
    )
    gates = config["calibration"]["gates"]
    checks = {
        "third_false_positive_rejection": counts[
            ("third_n1", "FALSE_POSITIVE", False)
        ]
        >= int(gates["min_third_false_positive_rejections"]),
        "second_false_positive_rejection": counts[
            ("second_n1", "FALSE_POSITIVE", False)
        ]
        >= int(gates["min_second_false_positive_rejections"]),
        "second_true_positive_retention": counts[
            ("second_n1", "TRUE_POSITIVE", True)
        ]
        >= int(gates["min_second_true_positive_retentions"]),
    }
    return {
        "schema_version": "receiver-cutin-calibration-v1",
        "human_reviewed_scenes": sorted(scenes),
        "scene_count": len(scenes),
        "provenance": provenance,
        "counts": {
            "|".join((source_name, verdict, str(retained).lower())): count
            for (source_name, verdict, retained), count in sorted(counts.items())
        },
        "gate_checks": checks,
        "calibration_gate_passed": all(checks.values()),
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
    global _ACTIVE_RUN_DIR
    started_at = utc_now()
    config = _load_yaml(config_path)
    _validate_config_contract(config)
    print(json.dumps({"phase": "config_loaded"}), flush=True)
    code = git_state(str(Path(config["repo_root"])))
    print(json.dumps({"phase": "git_state_loaded"}), flush=True)
    formal = not allow_dirty_development
    if formal and not page_cache_control_available():
        raise RuntimeError(
            "正式 N1-K4 在 2 GiB cgroup 下需要 POSIX page-cache control"
        )
    if formal and bool(config.get("require_clean_git", True)) and code["dirty"]:
        raise RuntimeError("正式 N1-K4 必须在 clean git worktree 上运行")
    if formal and skip_audit_panels:
        raise RuntimeError("正式 N1-K4 不允许跳过人工审计材料")
    if formal and force_audit_development:
        raise RuntimeError("正式 N1-K4 不允许强制生成空审核包")
    n0_run = Path(config["n0_run"])
    if not (n0_run / "COMPLETE").is_file():
        raise RuntimeError("N0 未 COMPLETE")
    n0_summary = json.loads((n0_run / "summary.json").read_text(encoding="utf-8"))

    print(json.dumps({"phase": "load_trainval_metadata"}), flush=True)
    source = TrainvalAnnotationSource(Path(config["dataset_root"]))
    print(json.dumps({"phase": "trainval_metadata_loaded"}), flush=True)
    lane_indices = LaneIndexCache(
        Path(config["dataset_root"]),
        config["map_matching"],
    )
    calibration = _calibration_audit(config, source, lane_indices)
    if not calibration["calibration_gate_passed"]:
        print(
            json.dumps(
                {
                    "phase": "calibration_gate_failed",
                    "counts": calibration["counts"],
                    "gate_checks": calibration["gate_checks"],
                    "rows": [
                        {
                            "audit_source": row["audit_source"],
                            "audit_id": row["audit_id"],
                            "event_id": row["event_id"],
                            "human_verdict": row["human_verdict"],
                            "retained_by_v4": row["retained_by_v4"],
                            "status": row["result"].get("status"),
                            "reason": row["result"].get("reason"),
                            "result": (
                                row["result"]
                                if row["human_verdict"] == "TRUE_POSITIVE"
                                or row["retained_by_v4"]
                                else None
                            ),
                        }
                        for row in calibration["rows"]
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        raise RuntimeError("receiver-centric calibration gate 未通过")
    calibration_scenes = set(calibration["human_reviewed_scenes"])
    evaluation_names = _resolve_evaluation_scenes(
        config["evaluation"], calibration_scenes
    )
    if max_evaluation_scenes_development is not None:
        if formal:
            raise RuntimeError("正式 run 不允许 development scene 截断")
        evaluation_names = evaluation_names[:max_evaluation_scenes_development]
    if calibration_scenes & set(evaluation_names):
        raise RuntimeError("calibration/evaluation scene 未分离")

    config_sha = file_fingerprint(str(config_path))
    tag = "receiver-cutin-v1" if formal else "receiver-cutin-v1-dev"
    run_id = generate_run_id(config["task_id"], tag, int(config["seed"]), config_sha)
    run_dir = (output_root or Path(config["run_root"])) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _ACTIVE_RUN_DIR = run_dir
    atomic_write_text(str(run_dir / "RUNNING"), "n1_receiver_cutin_screen\n")
    atomic_write_text(
        str(run_dir / "resolved.yaml"), config_path.read_text(encoding="utf-8")
    )
    atomic_write_json(str(run_dir / "calibration_audit.json"), calibration)

    scene_results = {}
    cache_dir = Path(config["cache_dir"]) / "evaluation"
    batch_size = int(config["scene_batch_size"])
    evaluation_order = sorted(
        evaluation_names,
        key=lambda name: (
            source.map_name_by_scene[source.resolve_scene(name)["token"]],
            name,
        ),
    )
    for batch_start in range(0, len(evaluation_order), batch_size):
        batch = evaluation_order[batch_start : batch_start + batch_size]
        meta = build_scene_instances_info(
            Path(config["dataset_root"]),
            batch,
            int(config["interpolate_n"]),
            cache_dir,
            retain=False,
            source=source,
        )
        for scene_name in batch:
            entry = meta[scene_name]
            instances_info = _read_json_bounded(Path(entry["cache_path"]))
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
            del instances_info
        del meta
        trim_process_heap()
        print(
            json.dumps(
                {
                    "processed_scenes": min(
                        batch_start + len(batch), len(evaluation_order)
                    ),
                    "evaluation_scenes": len(evaluation_order),
                    "memory": memory_snapshot(),
                }
            ),
            flush=True,
        )

    positives, negatives, transitions, pairs, evaluation_summary = _aggregate(
        scene_results
    )
    _assert_event_record_hashes(transitions)
    _assert_event_record_hashes(negatives)
    machine_passed, machine_checks = _machine_gate(
        evaluation_summary, config["machine_gates"]
    )
    event_pool = {
        "schema_version": config["schema_version"],
        "task_id": config["task_id"],
        "seed": int(config["seed"]),
        "selection_inputs": (
            "nuscenes_2hz_subject_center_crossing_plus_independent_receiver"
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
            "transition_count": len(result["transitions"]),
            "topology_pass_count": sum(
                row["topology"]["topology_pass"] for row in result["transitions"]
            ),
            "subject_entry_pass_count": sum(
                bool(row["cutin"].get("subject_entry_pass"))
                for row in result["transitions"]
            ),
            "positive_candidate_count": len(result["positives"]),
            "negative_window_count": len(result["negatives"]),
        }
        for scene_name, result in sorted(scene_results.items())
    }
    atomic_write_json(
        str(run_dir / "evaluation_scene_audit.json"), evaluation_audit
    )

    audit_ready = evaluation_summary["positive_candidate_count"] >= int(
        config["audit_readiness"]["min_positive_candidates"]
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
            machine_summary=evaluation_summary,
            machine_checks=machine_checks,
        )
    if audit_ready:
        terminal = "AWAITING_HUMAN_REVIEW"
        verdict = "await_human_review_n1_receiver_cutin"
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
            key: value for key, value in calibration.items() if key != "rows"
        },
        "evaluation": evaluation_summary,
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
            "trainval_sample_annotation_json_sha256": file_fingerprint(
                str(
                    Path(config["dataset_root"])
                    / "v1.0-trainval"
                    / "sample_annotation.json"
                )
            ),
            "second_review_sha256": calibration["provenance"][
                "second_review_sha256"
            ],
            "third_review_sha256": calibration["provenance"][
                "third_review_sha256"
            ],
            "evaluation_scenes": evaluation_names,
        }
    )
    artifact_hashes = {
        "calibration_audit": file_fingerprint(
            str(run_dir / "calibration_audit.json")
        ),
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
    _ACTIVE_RUN_DIR = None
    print(json.dumps({"run_dir": str(run_dir), **summary}, ensure_ascii=False))
    return run_dir


_FINAL_REQUIRED_CONFIG_PATHS = (
    ("task_id",),
    ("seed",),
    ("repo_root",),
    ("dataset_root",),
    ("cache_dir",),
    ("run_root",),
    ("n0_run",),
    ("interpolate_n",),
    ("calibration", "fourth_review", "fixture_dir"),
    ("evaluation", "expected_scene_count"),
    ("eligibility",),
    ("map_matching",),
    ("transition",),
    ("strict", "subject"),
    ("strict", "receiver"),
    ("strict", "corridor"),
    ("runtime",),
    ("machine_readiness",),
    ("audit",),
    ("human_gates",),
    ("stop_rule", "never_start_n2_from_this_run"),
)


def _validate_final_config_contract(config: dict) -> None:
    missing = []
    for path in _FINAL_REQUIRED_CONFIG_PATHS:
        current = config
        for key in path:
            if not isinstance(current, dict) or key not in current:
                missing.append(".".join(path))
                break
            current = current[key]
    if missing:
        raise ValueError("final cut-in config 缺少字段: " + ", ".join(missing))
    if config["schema_version"] != "receiver-centric-cutin-final-v1":
        raise ValueError("final cut-in schema_version 不匹配")
    if config["stop_rule"]["never_start_n2_from_this_run"] is not True:
        raise ValueError("final cut-in 必须保持 N2 fail-closed")
    if config["strict"].get("hard_evidence_source") != "raw_2hz_annotations":
        raise ValueError("strict hard evidence 必须是 raw_2hz_annotations")
    if config["strict"].get("interpolation_can_pass_hard_gate") is not False:
        raise ValueError("插值不得通过 strict hard gate")


def _load_final_calibration(config: dict) -> tuple[set[str], dict]:
    """冻结 49 条历史标签加第四轮 K4 18 条，并返回唯一 scene 集。"""
    historical, scenes, provenance = _load_calibration_events(config)
    fourth = config["calibration"]["fourth_review"]
    # fixture 必须来自执行中的 code worktree，以保证 formal clean worktree
    # 使用同一已提交的冻结 evidence，而不依赖 dirty shared checkout。
    fixture = PROJECT_ROOT / fourth["fixture_dir"]
    review_path = fixture / "review_working.jsonl"
    if file_fingerprint(str(review_path)) != fourth["completed_review_sha256"]:
        raise RuntimeError("第四轮 K4 fixture review SHA256 不匹配")
    review_rows = _read_jsonl(review_path)
    evidence_dir = fixture / "evidence"
    evidence_files = sorted(evidence_dir.glob("K4-*.json"))
    if len(review_rows) != int(fourth["expected_label_count"]) or len(evidence_files) != len(review_rows):
        raise RuntimeError("第四轮 K4 fixture 条数不完整")
    fourth_scenes = {
        json.loads(path.read_text(encoding="utf-8"))["scene_id"]
        for path in evidence_files
    }
    if len(fourth_scenes) != int(fourth["expected_scene_count"]):
        raise RuntimeError("第四轮 K4 fixture scene 数不匹配")
    all_scenes = set(scenes) | fourth_scenes
    expected_labels = int(config["calibration"]["expected_label_count"])
    expected_scenes = int(config["calibration"]["expected_scene_count"])
    if len(historical) + len(review_rows) != expected_labels:
        raise RuntimeError("冻结 calibration label 数不匹配")
    if len(all_scenes) != expected_scenes:
        raise RuntimeError("冻结 calibration scene 数不匹配")
    return all_scenes, {
        **provenance,
        "fourth_review_sha256": fourth["completed_review_sha256"],
        "historical_label_count": len(historical),
        "fourth_label_count": len(review_rows),
        "total_label_count": expected_labels,
        "historical_scene_count": len(scenes),
        "fourth_scene_count": len(fourth_scenes),
        "total_scene_count": expected_scenes,
        "fourth_scenes": sorted(fourth_scenes),
    }


def _final_transition_records(
    scene_id: str,
    lane_index: LaneIndex,
    tracks: dict[int, list[dict]],
    matches: dict[int, list[dict]],
    actor_tokens: dict[int, str],
    frame_times: dict[int, float],
    config: dict,
    map_name: str,
) -> tuple[list[dict], dict]:
    """仅为当前 scene 流式生成 strict-v2 transition records。"""
    transition_config = config["transition"]
    minimum = max(
        int(transition_config["min_stable_source_frames"]),
        int(transition_config["min_stable_target_frames"]),
    )
    runs_by_actor = {
        actor_id: _stable_runs(rows, minimum) for actor_id, rows in matches.items()
    }
    matches_by_actor = {
        actor_id: {int(row["frame_index"]): row for row in rows}
        for actor_id, rows in matches.items()
    }
    records = []
    topology_count = 0
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
            if not topology["topology_pass"]:
                continue
            topology_count += 1
            base = {
                "event_id": (
                    f"{scene_id}:{actor_id}:F1:"
                    f"{source_run['end_frame']}:{target_run['start_frame']}"
                ),
                "scene_id": scene_id,
                "map_name": map_name,
                "actor_id": int(actor_id),
                "subject_instance_token": actor_tokens[actor_id],
                "source_run": source_run,
                "target_run": target_run,
                "transition_gap_frames": gap,
                "crossing_frame": int(crossing["frame_index"]),
                "topology": topology,
            }
            strict = receiver_centric_cutin_v2(
                actor_id,
                source_run,
                target_run,
                topology,
                tracks,
                matches_by_actor,
                lane_index,
                frame_times,
                config,
                source_event_record_sha256=canonical_sha256(base),
                map_version=map_name,
            )
            record = {
                **base,
                "maneuver_mode": strict["maneuver_mode"],
                "strict": strict,
                "label": (
                    "machine_positive_strict_v2"
                    if strict["machine_positive"]
                    else "strict_v2_nonpositive"
                ),
                "machine_positive": bool(strict["machine_positive"]),
            }
            receiver_id = strict["receiver"].get("selected_actor_id")
            if receiver_id is not None:
                record["receiver_actor_id"] = int(receiver_id)
                if int(receiver_id) in actor_tokens:
                    record["receiver_instance_token"] = actor_tokens[int(receiver_id)]
            pre_frames = strict["subject"].get("pre_frames", [])
            post_frames = strict["subject"].get("post_frames", [])
            if pre_frames:
                record["event_start_frame"] = int(pre_frames[0])
            if post_frames:
                record["event_end_frame"] = int(post_frames[-1])
            record["event_record_sha256"] = canonical_sha256(record)
            records.append(record)
    return records, {"topology_pass_count": topology_count}


def _strict_transition_diagnostic(record: dict) -> dict:
    strict = record["strict"]
    return {
        "event_id": record["event_id"],
        "event_record_sha256": record["event_record_sha256"],
        "scene_id": record["scene_id"],
        "map_name": record["map_name"],
        "actor_id": record["actor_id"],
        "crossing_frame": record["crossing_frame"],
        "maneuver_mode": strict["maneuver_mode"],
        "status": strict["status"],
        "primary_reason": strict["primary_reason"],
        "machine_positive": strict["machine_positive"],
        "hard_evidence_source": strict["hard_evidence_source"],
        "uses_interpolated_physics": strict["uses_interpolated_physics"],
        "subject_frames": {
            "pre": strict["subject"].get("pre_frames", []),
            "post": strict["subject"].get("post_frames", []),
        },
        "receiver_actor_id": strict["receiver"].get("selected_actor_id"),
        "receiver_identity_switch_frames": strict["receiver"].get(
            "identity_switch_frames", []
        ),
    }


def _final_memory_check(config: dict) -> dict:
    snapshot = memory_snapshot()
    runtime = config["runtime"]
    rss = snapshot.get("process_rss_bytes")
    cgroup = snapshot.get("cgroup_memory_current_bytes")
    if rss is not None and rss > int(runtime["stop_process_rss_bytes"]):
        raise RuntimeError(f"process RSS 超过 stop 阈值: {rss}")
    if cgroup is not None and cgroup > int(runtime["stop_cgroup_current_bytes"]):
        raise RuntimeError(f"cgroup current 超过 stop 阈值: {cgroup}")
    return snapshot


def run_final_mining(
    config_path: Path,
    run_dir: Path,
    *,
    max_evaluation_scenes: int | None = None,
) -> dict:
    """final worker A：只挖掘并流式聚合，绝不导入 audit/render 依赖。"""
    config = _load_yaml(config_path)
    _validate_final_config_contract(config)
    if config["runtime"].get("require_posix_page_cache_control") and not page_cache_control_available():
        raise RuntimeError("final mining 需要 POSIX page-cache control")
    config["config_fingerprint"] = file_fingerprint(str(config_path))
    start_snapshot = memory_snapshot()
    max_start = int(config["runtime"]["max_start_cgroup_current_bytes"])
    if (
        start_snapshot.get("cgroup_memory_current_bytes") is not None
        and start_snapshot["cgroup_memory_current_bytes"] > max_start
    ):
        raise RuntimeError("启动 cgroup current 超过 final runtime 合同")
    n0_run = Path(config["n0_run"])
    if not (n0_run / "COMPLETE").is_file():
        raise RuntimeError("N0 未 COMPLETE")

    source = TrainvalAnnotationSource(Path(config["dataset_root"]))
    lane_indices = LaneIndexCache(Path(config["dataset_root"]), config["map_matching"])
    calibration_scenes, calibration_provenance = _load_final_calibration(config)
    evaluation_names = _resolve_evaluation_scenes(config["evaluation"], calibration_scenes)
    if max_evaluation_scenes is not None:
        evaluation_names = evaluation_names[: int(max_evaluation_scenes)]
    elif len(evaluation_names) != int(config["evaluation"]["expected_scene_count"]):
        raise RuntimeError(
            "final evaluation scene 数不匹配: "
            f"{len(evaluation_names)} != {config['evaluation']['expected_scene_count']}"
        )
    if calibration_scenes & set(evaluation_names):
        raise RuntimeError("calibration/evaluation scene 未分离")
    atomic_write_json(
        str(run_dir / "calibration_audit.json"),
        {
            "schema_version": "n1-cutin-final-calibration-freeze-v1",
            "calibration_scenes": sorted(calibration_scenes),
            "scene_count": len(calibration_scenes),
            "provenance": calibration_provenance,
            "evaluation_scene_count": len(evaluation_names),
            "intersection_count": len(calibration_scenes & set(evaluation_names)),
        },
    )

    cache_dir = Path(config["cache_dir"]) / "evaluation"
    evaluation_order = sorted(
        evaluation_names,
        key=lambda name: (
            source.map_name_by_scene[source.resolve_scene(name)["token"]],
            name,
        ),
    )
    aggregate = Counter()
    pass_scenes: set[str] = set()
    pass_records: list[dict] = []
    abstain_by_reason: dict[str, dict] = {}
    resource_observations = []
    scene_batch_size = int(config["runtime"]["scene_batch_size"])
    with (
        (run_dir / "scene_metrics.jsonl").open("w", encoding="utf-8") as scene_handle,
        (run_dir / "transition_diagnostics.jsonl").open("w", encoding="utf-8") as transition_handle,
        (run_dir / "strict_candidates.jsonl").open("w", encoding="utf-8") as candidate_handle,
        (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as metrics_handle,
    ):
        for batch_start in range(0, len(evaluation_order), scene_batch_size):
            batch = evaluation_order[batch_start : batch_start + scene_batch_size]
            meta = build_scene_instances_info(
                Path(config["dataset_root"]),
                batch,
                int(config["interpolate_n"]),
                cache_dir,
                retain=False,
                source=source,
            )
            for scene_name in batch:
                entry = meta[scene_name]
                instances_info = _read_json_bounded(Path(entry["cache_path"]))
                scene = source.resolve_scene(scene_name)
                tracks, matches, actor_tokens, _ = _track_and_matches(
                    scene_name,
                    instances_info,
                    lane_indices[entry["map_name"]],
                    config,
                )
                records, topology_summary = _final_transition_records(
                    scene_name,
                    lane_indices[entry["map_name"]],
                    tracks,
                    matches,
                    actor_tokens,
                    _frame_times_s(source, scene, int(config["interpolate_n"])),
                    config,
                    entry["map_name"],
                )
                status_counts = Counter(record["strict"]["status"] for record in records)
                reason_counts = Counter(
                    record["strict"]["primary_reason"]
                    for record in records
                    if record["strict"]["primary_reason"] is not None
                )
                for record in records:
                    diagnostic = _strict_transition_diagnostic(record)
                    transition_handle.write(
                        json.dumps(diagnostic, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    aggregate[f"status:{diagnostic['status']}"] += 1
                    aggregate[f"mode:{diagnostic['maneuver_mode']}"] += 1
                    if diagnostic["primary_reason"] is not None:
                        aggregate[f"reason:{diagnostic['primary_reason']}"] += 1
                    if record["strict"]["machine_positive"]:
                        pass_records.append(record)
                        pass_scenes.add(scene_name)
                        candidate_handle.write(
                            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                        )
                    elif diagnostic["status"] == "ABSTAIN":
                        reason = str(diagnostic["primary_reason"])
                        existing = abstain_by_reason.get(reason)
                        if (
                            existing is None
                            or record["event_record_sha256"]
                            < existing["event_record_sha256"]
                        ):
                            abstain_by_reason[reason] = record
                scene_metric = {
                    "scene_id": scene_name,
                    "scene_token": entry["scene_token"],
                    "map_name": entry["map_name"],
                    "eligible_actor_count": len(tracks),
                    "transition_count": len(records),
                    "topology_pass_count": topology_summary["topology_pass_count"],
                    "strict_status_counts": dict(sorted(status_counts.items())),
                    "strict_primary_reason_counts": dict(sorted(reason_counts.items())),
                    "strict_pass_count": status_counts["PASS"],
                }
                scene_handle.write(
                    json.dumps(scene_metric, ensure_ascii=False, sort_keys=True) + "\n"
                )
                del instances_info, tracks, matches, actor_tokens, records
            del meta
            trim_process_heap()
            snapshot = _final_memory_check(config)
            observation = {
                "processed_scenes": min(batch_start + len(batch), len(evaluation_order)),
                "evaluation_scenes": len(evaluation_order),
                "memory": snapshot,
            }
            resource_observations.append(observation)
            metrics_handle.write(
                json.dumps(observation, ensure_ascii=False, sort_keys=True) + "\n"
            )

    _assert_event_record_hashes(pass_records)
    strict_event_pool = {
        "schema_version": "receiver-centric-cutin-strict-v2-event-pool-v1",
        "task_id": config["task_id"],
        "seed": int(config["seed"]),
        "evaluation_scene_count": len(evaluation_names),
        "strict_passes": pass_records,
    }
    strict_event_pool["strict_event_pool_sha256"] = canonical_sha256(strict_event_pool)
    atomic_write_json(str(run_dir / "strict_event_pool.json"), strict_event_pool)
    limit = int(config["audit"]["abstain_diagnostic_max_count"])
    abstain_records = sorted(
        abstain_by_reason.values(),
        key=lambda row: (
            str(row["strict"]["primary_reason"]),
            row["event_record_sha256"],
        ),
    )[:limit]
    with (run_dir / "diagnostic_abstain_candidates.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for record in abstain_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    abstain_index = [_strict_transition_diagnostic(record) for record in abstain_records]
    summary = {
        "schema_version": "n1-cutin-final-mining-summary-v1",
        "task_id": config["task_id"],
        "evaluation_scene_count": len(evaluation_names),
        "calibration_scene_count": len(calibration_scenes),
        "strict_pass_candidate_count": len(pass_records),
        "strict_pass_scene_count": len(pass_scenes),
        "strict_status_counts": {
            key.removeprefix("status:"): value
            for key, value in sorted(aggregate.items())
            if key.startswith("status:")
        },
        "strict_mode_counts": {
            key.removeprefix("mode:"): value
            for key, value in sorted(aggregate.items())
            if key.startswith("mode:")
        },
        "strict_primary_reason_counts": {
            key.removeprefix("reason:"): value
            for key, value in sorted(aggregate.items())
            if key.startswith("reason:")
        },
        "diagnostic_abstain_index": abstain_index[:limit],
        "resource_observations": resource_observations,
        "uses_interpolated_physics": False,
        "n2_authorized": False,
    }
    readiness = config["machine_readiness"]
    summary["machine_readiness_checks"] = {
        "strict_pass_candidates": len(pass_records)
        >= int(readiness["min_strict_pass_candidates"]),
        "strict_pass_scenes": len(pass_scenes)
        >= int(readiness["min_strict_pass_scenes"]),
        "raw_evidence_only": True,
        "resource_contract_violations": 0
        <= int(readiness["max_resource_contract_violations"]),
    }
    summary["machine_readiness_passed_without_k4"] = all(
        summary["machine_readiness_checks"].values()
    )
    atomic_write_json(str(run_dir / "mining_summary.json"), summary)
    atomic_write_text(str(run_dir / "stages" / "MINING_COMPLETE"), "mining_complete\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/resim/event_first_n1_cutin_v1.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--allow-dirty-development", action="store_true")
    parser.add_argument("--skip-audit-panels", action="store_true")
    parser.add_argument("--max-evaluation-scenes-development", type=int)
    parser.add_argument("--force-audit-development", action="store_true")
    parser.add_argument("--internal-final-mining", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--max-evaluation-scenes", type=int)
    args = parser.parse_args()
    if args.internal_final_mining:
        if args.run_dir is None:
            parser.error("--internal-final-mining requires --run-dir")
        summary = run_final_mining(
            args.config,
            args.run_dir,
            max_evaluation_scenes=args.max_evaluation_scenes,
        )
        print(json.dumps({"run_dir": str(args.run_dir), **summary}, ensure_ascii=False))
        return
    try:
        run(
            args.config,
            args.output_root,
            allow_dirty_development=args.allow_dirty_development,
            skip_audit_panels=args.skip_audit_panels,
            max_evaluation_scenes_development=args.max_evaluation_scenes_development,
            force_audit_development=args.force_audit_development,
        )
    except BaseException as exc:
        _mark_active_run_failed(exc)
        raise


if __name__ == "__main__":
    main()
