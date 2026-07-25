#!/usr/bin/env python
"""Full-domain N1 natural-event pool（annotation + frozen vector map only）。

相对 mini N1（``event_first_n1_event.py``）的差异，全部预注册、可审计：

1. 轨迹来源：不再依赖 DriveStudio 预处理，改由 ``nuscenes_trainval_tracks`` 从
   v1.0-trainval 标注**流式**重建 10Hz interpolated tracks（已与 mini 产物逐位对拍）；
   因此 mini 的帧级冻结阈值原样沿用。
2. interaction relation：把 mini 的 **exact-target-token** front/rear 升级为
   **graph-corridor curvilinear coordinate**——沿 subject 所在 target lane 的有向
   lane/connector 链（贪心、min heading discontinuity，K=graph_hops 跳）展开，用累积
   弧长为邻车定位，解决 exact-token 把同一 longitudinal corridor 邻车切到相邻 token 的
   fragmentation。exact-token 结果同时记录作对照，不作为 gate。
3. calibration / evaluation 严格 scene-disjoint：calibration 为 mini 三场景（含其 22 个
   topology-pass candidate 的 corridor 复核），evaluation 为官方 split，二者不重叠。
   formal verdict 只由 evaluation gate 决定。

复用 mini N1 的冻结函数（LaneIndex/_eligibility/_stable_runs/_transition_type），保证
资格、地图匹配、topology 判定与 mini 完全一致。
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from nuscenes.map_expansion.map_api import NuScenesMap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
RESIM_DIR = Path(__file__).resolve().parent
if str(RESIM_DIR) not in sys.path:
    sys.path.insert(0, str(RESIM_DIR))

from motion_proj.resim.canonical_hash import canonical_sha256
from motion_proj.resim.nuscenes_trainval_tracks import build_scene_instances_info
from motion_proj.runtime.atomic import atomic_write_json, atomic_write_text
from motion_proj.runtime.fingerprint import file_fingerprint, git_state
from motion_proj.runtime.v71_contract import generate_run_id, utc_now

# 复用 mini N1 的冻结逻辑，保证资格/匹配/topology 定义完全一致
from event_first_n1_event import (  # noqa: E402
    LaneIndex,
    _angle_error,
    _eligibility,
    _stable_runs,
    _transition_type,
)


def _load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"配置必须是 YAML object: {path}")
    return value


def _resolve_split(dataset_root: Path, spec: dict, calibration_names: set[str]) -> list[dict]:
    """解析 evaluation scene 列表；支持官方 split 名与显式名单。"""
    from nuscenes.utils import splits as nusplits

    names: list[str] = []
    if spec.get("scene_names"):
        names = list(spec["scene_names"])
    elif spec.get("split_name"):
        names = list(getattr(nusplits, spec["split_name"]))
    else:
        raise ValueError("evaluation 需 scene_names 或 split_name")
    exclude = set(spec.get("exclude_scenes", [])) | calibration_names
    names = [name for name in names if name not in exclude]
    if spec.get("max_scenes"):
        names = sorted(names)[: int(spec["max_scenes"])]
    return sorted(set(names))


def _corridor_chain(
    lane_index: LaneIndex, token: str, direction: str, hops: int
) -> list[str]:
    """从 token 沿 direction 贪心展开有向 lane 链（min heading discontinuity）。"""
    connectivity = lane_index.nmap.connectivity
    chain: list[str] = []
    current = token
    visited = {token}
    for _ in range(hops):
        neighbors = [
            n
            for n in connectivity.get(current, {}).get(direction, [])
            if n not in visited and n in lane_index.centerlines
        ]
        if not neighbors:
            break
        cur_line = lane_index.centerlines[current]
        if direction == "outgoing":
            junction_heading = float(cur_line[-1, 2])
        else:
            junction_heading = float(cur_line[0, 2])

        def discontinuity(neighbor: str) -> tuple[float, str]:
            line = lane_index.centerlines[neighbor]
            heading = float(line[0, 2] if direction == "outgoing" else line[-1, 2])
            return (_angle_error(junction_heading, heading), neighbor)

        best = min(neighbors, key=discontinuity)
        chain.append(best)
        visited.add(best)
        current = best
    return chain


def _corridor_offsets(lane_index: LaneIndex, target_token: str, hops: int) -> dict[str, float]:
    """构建以 target_token 起点为 0 的 corridor s 轴 token->start_offset。"""
    offsets: dict[str, float] = {target_token: 0.0}
    down = _corridor_chain(lane_index, target_token, "outgoing", hops)
    cursor = float(lane_index.arc_lengths[target_token][-1])
    for tok in down:
        offsets[tok] = cursor
        cursor += float(lane_index.arc_lengths[tok][-1])
    up = _corridor_chain(lane_index, target_token, "incoming", hops)
    cursor = 0.0
    for tok in up:
        length = float(lane_index.arc_lengths[tok][-1])
        cursor -= length
        offsets[tok] = cursor
    return offsets


def _relation_corridor(
    actor_id: int,
    frame: int,
    target_token: str,
    matches_by_actor: dict[int, dict[int, dict]],
    lane_index: LaneIndex,
    config: dict,
) -> dict:
    """沿 graph-corridor curvilinear 坐标计算 front/rear。"""
    subject = matches_by_actor[actor_id].get(frame)
    if subject is None or subject.get("lane_token") != target_token:
        return {"status": "UNKNOWN", "reason": "subject_not_matched_to_target"}
    hops = int(config["graph_hops_for_shared_corridor"])
    offsets = _corridor_offsets(lane_index, target_token, hops)
    subject_pos = offsets[target_token] + float(subject["centerline_s_m"])
    neighbors = []
    for other_id, by_frame in sorted(matches_by_actor.items()):
        if other_id == actor_id:
            continue
        other = by_frame.get(frame)
        if other is None:
            continue
        token = other.get("lane_token")
        if token not in offsets:
            continue
        other_pos = offsets[token] + float(other["centerline_s_m"])
        neighbors.append(
            {
                "actor_id": other_id,
                "lane_token": token,
                "delta_s_m": other_pos - subject_pos,
                "same_exact_token": token == target_token,
            }
        )
    fronts = [row for row in neighbors if row["delta_s_m"] > 0]
    rears = [row for row in neighbors if row["delta_s_m"] < 0]
    front = min(fronts, key=lambda row: row["delta_s_m"], default=None)
    rear = max(rears, key=lambda row: row["delta_s_m"], default=None)
    min_gap = float(config["interaction_min_gap_m"])
    max_gap = float(config["interaction_max_gap_m"])
    front_gap = front["delta_s_m"] if front else None
    rear_gap = -rear["delta_s_m"] if rear else None
    pass_relation = (
        front_gap is not None
        and rear_gap is not None
        and min_gap <= front_gap <= max_gap
        and min_gap <= rear_gap <= max_gap
    )
    return {
        "status": "PASS" if pass_relation else "FAIL",
        "front": front,
        "rear": rear,
        "front_gap_m": front_gap,
        "rear_gap_m": rear_gap,
        "neighbor_count_on_corridor": len(neighbors),
        "corridor_token_count": len(offsets),
        "relation_mode": "graph_corridor",
    }


def _candidate_events_corridor(
    scene_id: str,
    lane_index: LaneIndex,
    tracks: dict[int, list[dict]],
    matches: dict[int, list[dict]],
    config: dict,
) -> tuple[list[dict], list[dict], list[dict]]:
    """与 mini _candidate_events 同结构，但 interaction 用 graph-corridor relation。"""
    minimum = max(
        int(config["min_stable_source_frames"]),
        int(config["min_stable_target_frames"]),
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
    positives: list[dict] = []
    transitions: list[dict] = []
    for actor_id, runs in sorted(runs_by_actor.items()):
        for source, target in zip(runs, runs[1:]):
            gap = int(target["start_frame"]) - int(source["end_frame"]) - 1
            if gap > int(config["max_transition_gap_frames"]):
                continue
            crossing_frame = (int(source["end_frame"]) + int(target["start_frame"])) // 2
            available = [
                row
                for frame, row in track_by_actor[actor_id].items()
                if abs(frame - crossing_frame) <= int(config["max_transition_gap_frames"])
            ]
            if not available:
                continue
            crossing = min(
                available,
                key=lambda row: (
                    abs(int(row["frame_index"]) - crossing_frame),
                    row["frame_index"],
                ),
            )
            topology = _transition_type(
                lane_index, source["token"], target["token"], crossing["xy"], config
            )
            relation_frame = (
                int(target["start_frame"]) + int(config["min_stable_target_frames"]) - 1
            )
            relation = _relation_corridor(
                actor_id, relation_frame, target["token"], match_by_actor, lane_index, config
            )
            positive = bool(topology["topology_pass"] and relation["status"] == "PASS")
            record = {
                "event_id": f"{scene_id}:{actor_id}:T:{source['end_frame']}:{target['start_frame']}",
                "scene_id": scene_id,
                "actor_id": actor_id,
                "source_run": source,
                "target_run": target,
                "transition_gap_frames": gap,
                "crossing_frame": int(crossing["frame_index"]),
                "relation_frame": relation_frame,
                "topology": topology,
                "interaction": relation,
                "label": "positive" if positive else "noninteractive_or_unresolved_transition",
                "positive": positive,
            }
            record["event_record_sha256"] = canonical_sha256(record)
            transitions.append(record)
            if positive:
                positives.append(record)

    negatives: list[dict] = []
    window = int(config["negative_window_frames"])
    positive_actors = {int(row["actor_id"]) for row in positives}
    for actor_id in sorted(positive_actors):
        for run in runs_by_actor[actor_id]:
            if int(run["frame_count"]) < window:
                continue
            start = int(run["start_frame"])
            end = start + window - 1
            midpoint = (start + end) // 2
            relation = _relation_corridor(
                actor_id, midpoint, run["token"], match_by_actor, lane_index, config
            )
            if relation["status"] != "PASS":
                continue
            overlaps_positive = any(
                row["actor_id"] == actor_id
                and not (
                    end < row["source_run"]["start_frame"]
                    or start > row["target_run"]["end_frame"]
                )
                for row in positives
            )
            if overlaps_positive:
                continue
            record = {
                "event_id": f"{scene_id}:{actor_id}:N:{start}:{end}",
                "scene_id": scene_id,
                "actor_id": actor_id,
                "lane_token": run["token"],
                "start_frame": start,
                "end_frame": end,
                "relation_frame": midpoint,
                "interaction": relation,
                "label": "negative",
                "negative": True,
            }
            record["event_record_sha256"] = canonical_sha256(record)
            negatives.append(record)
            break
    return positives, negatives, transitions


def _gate_decision(summary: dict, gates: dict) -> tuple[bool, dict]:
    checks = {
        "positive_events": summary["positive_event_count"] >= int(gates["min_positive_events"]),
        "negative_events": summary["negative_event_count"] >= int(gates["min_negative_events"]),
        "same_actor_pairs": summary["same_actor_pair_count"] >= int(gates["min_same_actor_pairs"]),
        "positive_scenes": summary["positive_scene_count"] >= int(gates["min_positive_scenes"]),
        "unknown_not_positive": not bool(gates["unknown_is_positive"]),
        "noninteractive_not_positive": not bool(gates["noninteractive_transition_is_positive"]),
    }
    return all(checks.values()), checks


def _process_scene(
    scene_id: str,
    instances_info: dict,
    lane_index: LaneIndex,
    config: dict,
) -> dict:
    tracks: dict[int, list[dict]] = {}
    matches: dict[int, list[dict]] = {}
    eligibility_rows = []
    # instances_info 以 instance_token 为 key；按 token 稳定排序后重编号为整数 actor_id
    for actor_id, (_, actor) in enumerate(sorted(instances_info.items())):
        audit, rows = _eligibility(actor_id, actor, config["eligibility"])
        audit["scene_id"] = scene_id
        eligibility_rows.append(audit)
        if not audit["eligible"]:
            continue
        tracks[actor_id] = rows
        matches[actor_id] = [lane_index.match(row) for row in rows]
    pos, neg, trans = _candidate_events_corridor(
        scene_id, lane_index, tracks, matches, config["event_definition"]
    )
    total = sum(len(rows) for rows in matches.values())
    matched = sum(
        row["match_status"] == "MATCHED" for rows in matches.values() for row in rows
    )
    return {
        "eligibility_rows": eligibility_rows,
        "eligible_actor_count": len(tracks),
        "pose_count": total,
        "matched_pose_count": matched,
        "positives": pos,
        "negatives": neg,
        "transitions": trans,
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
        positive = sorted(positives_by_actor[key], key=lambda row: row["event_id"])[0]
        negative = sorted(negatives_by_actor[key], key=lambda row: row["event_id"])[0]
        pair = {
            "pair_id": f"{key[0]}:{key[1]}:positive-vs-negative",
            "scene_id": key[0],
            "actor_id": key[1],
            "positive_event_id": positive["event_id"],
            "negative_event_id": negative["event_id"],
        }
        pair["pair_sha256"] = canonical_sha256(pair)
        pairs.append(pair)
    summary = {
        "positive_event_count": len(positives),
        "negative_event_count": len(negatives),
        "transition_candidate_count": len(transitions),
        "topology_pass_count": sum(
            1 for row in transitions if row["topology"]["topology_pass"]
        ),
        "same_actor_pair_count": len(pairs),
        "positive_scene_count": len({row["scene_id"] for row in positives}),
    }
    return positives, negatives, transitions, pairs, summary


def run(config_path: Path, output_root: Path | None) -> Path:
    config = _load_yaml(config_path)
    n0_run = Path(config["n0_run"])
    if not (n0_run / "COMPLETE").is_file():
        raise RuntimeError(f"N0 未 COMPLETE: {n0_run}")
    n0_summary = json.loads((n0_run / "summary.json").read_text(encoding="utf-8"))
    if n0_summary["research_verdict"] != "n0_asset_pass":
        raise RuntimeError(f"N0 verdict 非 pass: {n0_summary}")

    dataset_root = Path(config["dataset_root"])
    cache_dir = Path(config["cache_dir"])
    interpolate_n = int(config.get("interpolate_n", 4))
    calibration_names = list(config.get("calibration_scenes", []))
    evaluation_names = _resolve_split(
        dataset_root, config["evaluation"], set(calibration_names)
    )
    print(
        json.dumps(
            {
                "calibration_scene_count": len(calibration_names),
                "evaluation_scene_count": len(evaluation_names),
            }
        ),
        flush=True,
    )

    # 构建 10Hz 轨迹缓存（annotation-only）
    calib_meta = build_scene_instances_info(
        dataset_root, calibration_names, interpolate_n, cache_dir, retain=False
    )
    eval_meta = build_scene_instances_info(
        dataset_root, evaluation_names, interpolate_n, cache_dir, retain=False
    )

    lane_indices: dict[str, LaneIndex] = {}

    def get_lane_index(map_name: str) -> LaneIndex:
        if map_name not in lane_indices:
            lane_indices[map_name] = LaneIndex(
                NuScenesMap(str(dataset_root), map_name), config["map_matching"]
            )
        return lane_indices[map_name]

    def process_set(meta: dict[str, dict]) -> dict[str, dict]:
        results: dict[str, dict] = {}
        for name, entry in sorted(meta.items()):
            instances_info = json.loads(
                Path(entry["cache_path"]).read_text(encoding="utf-8")
            )
            lane_index = get_lane_index(entry["map_name"])
            results[name] = _process_scene(
                name, instances_info, lane_index, config
            )
            results[name]["map_name"] = entry["map_name"]
            results[name]["scene_token"] = entry["scene_token"]
        return results

    calib_results = process_set(calib_meta)
    eval_results = process_set(eval_meta)

    _, _, _, calib_pairs, calib_summary = _aggregate(calib_results)
    eval_pos, eval_neg, eval_trans, eval_pairs, eval_summary = _aggregate(eval_results)

    passed, checks = _gate_decision(eval_summary, config["gates"])
    terminal = "COMPLETE" if passed else "REJECTED"
    verdict = "n1_fulldomain_event_pool_pass" if passed else config["stop_rule"]["on_gate_failure"]

    config_sha = file_fingerprint(str(config_path))
    code = git_state(str(Path(config["repo_root"])))
    run_id = generate_run_id(config["task_id"], "fulldomain-v1", int(config["seed"]), config_sha)
    run_root = output_root or Path(config["run_root"])
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    event_pool = {
        "schema_version": config["schema_version"],
        "task_id": config["task_id"],
        "seed": int(config["seed"]),
        "selection_inputs": "trainval_annotation_tracks_10hz_and_frozen_vector_map_only",
        "relation_mode": "graph_corridor",
        "evaluation": {
            "positives": eval_pos,
            "negatives": eval_neg,
            "transition_candidates": eval_trans,
            "same_actor_pairs": eval_pairs,
        },
    }
    event_pool["event_pool_sha256"] = canonical_sha256(event_pool)

    calib_audit = {
        "scenes": calibration_names,
        "summary": calib_summary,
        "same_actor_pairs": calib_pairs,
        "note": "calibration/audit only; mini verdict 不被本 run 回写",
        "per_scene": {
            name: {
                "map_name": result["map_name"],
                "eligible_actor_count": result["eligible_actor_count"],
                "matched_pose_fraction": (
                    result["matched_pose_count"] / result["pose_count"]
                    if result["pose_count"]
                    else None
                ),
                "topology_pass_count": sum(
                    1 for row in result["transitions"] if row["topology"]["topology_pass"]
                ),
                "positive_count": len(result["positives"]),
            }
            for name, result in calib_results.items()
        },
    }

    eval_map_audit = {
        name: {
            "map_name": result["map_name"],
            "scene_token": result["scene_token"],
            "eligible_actor_count": result["eligible_actor_count"],
            "pose_count": result["pose_count"],
            "matched_pose_count": result["matched_pose_count"],
            "matched_pose_fraction": (
                result["matched_pose_count"] / result["pose_count"]
                if result["pose_count"]
                else None
            ),
            "topology_pass_count": sum(
                1 for row in result["transitions"] if row["topology"]["topology_pass"]
            ),
            "positive_count": len(result["positives"]),
        }
        for name, result in eval_results.items()
    }

    summary = {
        "task_id": config["task_id"],
        "run_id": run_id,
        "seed": int(config["seed"]),
        "relation_mode": "graph_corridor",
        "calibration_scene_count": len(calibration_names),
        "evaluation_scene_count": len(evaluation_names),
        "evaluation": eval_summary,
        "calibration": calib_summary,
        "gate_checks": checks,
        "terminal_status": terminal,
        "research_verdict": verdict,
        "event_pool_sha256": event_pool["event_pool_sha256"],
    }
    manifest = {
        "schema_version": 1,
        "task_id": config["task_id"],
        "run_id": run_id,
        "command": list(sys.argv),
        "code_commit": code["commit"],
        "code_dirty": code["dirty"],
        "dirty_diff_hash": code["dirty_diff_hash"],
        "config_fingerprint": config_sha,
        "n0_run": str(n0_run),
        "n0_asset_manifest_sha256": n0_summary["asset_manifest_sha256"],
        "calibration_scenes": calibration_names,
        "evaluation_scenes": evaluation_names,
        "interpolate_n": interpolate_n,
        "seed": int(config["seed"]),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "started_at": utc_now(),
        "ended_at": utc_now(),
        "terminal_status": terminal,
        "exit_reason": verdict,
    }
    atomic_write_text(str(run_dir / "resolved.yaml"), config_path.read_text(encoding="utf-8"))
    atomic_write_json(str(run_dir / "manifest.json"), manifest)
    atomic_write_json(str(run_dir / "event_pool.json"), event_pool)
    atomic_write_json(str(run_dir / "calibration_audit.json"), calib_audit)
    atomic_write_json(str(run_dir / "evaluation_map_audit.json"), eval_map_audit)
    atomic_write_text(
        str(run_dir / "metrics.jsonl"),
        json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n",
    )
    atomic_write_json(str(run_dir / "summary.json"), summary)
    atomic_write_text(str(run_dir / terminal), verdict + "\n")
    print(json.dumps({"run_dir": str(run_dir), **summary}, ensure_ascii=False))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/resim/event_first_n1_fulldomain_v1.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    run(args.config, args.output_root)


if __name__ == "__main__":
    main()
