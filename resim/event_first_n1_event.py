#!/usr/bin/env python
"""按冻结 map/track/interaction 定义构建 N1 natural-event pool。"""
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
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.resim.canonical_hash import canonical_sha256
from motion_proj.runtime.atomic import atomic_write_json, atomic_write_text
from motion_proj.runtime.fingerprint import file_fingerprint, git_state
from motion_proj.runtime.v71_contract import generate_run_id, utc_now


def _load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"配置必须是 YAML object: {path}")
    return value


def _angle_error(left: float, right: float) -> float:
    return abs((left - right + math.pi) % (2.0 * math.pi) - math.pi)


def _track_rows(actor_id: int, actor: dict) -> list[dict]:
    values = actor["frame_annotations"]
    output = []
    for frame, transform, dimensions in zip(
        values["frame_idx"], values["obj_to_world"], values["box_size"]
    ):
        matrix = np.asarray(transform, dtype=float)
        output.append(
            {
                "actor_id": actor_id,
                "frame_index": int(frame),
                "xy": [float(matrix[0, 3]), float(matrix[1, 3])],
                "yaw": float(math.atan2(matrix[1, 0], matrix[0, 0])),
                "dimensions_lwh": [float(value) for value in dimensions],
            }
        )
    return output


def _eligibility(actor_id: int, actor: dict, config: dict) -> tuple[dict, list[dict]]:
    rows = _track_rows(actor_id, actor)
    frames = [row["frame_index"] for row in rows]
    gaps = sum(right != left + 1 for left, right in zip(frames, frames[1:]))
    displacement = (
        float(np.linalg.norm(np.asarray(rows[-1]["xy"]) - np.asarray(rows[0]["xy"])))
        if len(rows) >= 2
        else 0.0
    )
    checks = {
        "class": str(actor["class_name"]).startswith(config["class_prefix"]),
        "track_frames": len(rows) >= int(config["min_track_frames"]),
        "frame_gaps": gaps <= int(config["max_frame_gaps"]),
        "displacement": displacement >= float(config["min_displacement_m"]),
    }
    return (
        {
            "actor_id": actor_id,
            "instance_token": actor["id"],
            "class_name": actor["class_name"],
            "track_frame_count": len(rows),
            "frame_gap_count": gaps,
            "displacement_m": displacement,
            "checks": checks,
            "eligible": all(checks.values()),
        },
        rows,
    )


class LaneIndex:
    def __init__(self, nmap: NuScenesMap, config: dict):
        self.nmap = nmap
        self.tokens = [row["token"] for row in nmap.lane + nmap.lane_connector]
        self.layer_by_token = {
            row["token"]: layer
            for layer in ("lane", "lane_connector")
            for row in getattr(nmap, layer)
        }
        discrete = nmap.discretize_lanes(
            self.tokens, float(config["centerline_resolution_m"])
        )
        self.centerlines = {
            token: np.asarray(discrete[token], dtype=float) for token in self.tokens
        }
        points = []
        point_tokens = []
        point_indices = []
        for token in self.tokens:
            for index, point in enumerate(self.centerlines[token]):
                points.append(point[:2])
                point_tokens.append(token)
                point_indices.append(index)
        self.tree = cKDTree(np.asarray(points, dtype=float))
        self.point_tokens = point_tokens
        self.point_indices = point_indices
        self.config = config
        self.arc_lengths = {}
        for token, line in self.centerlines.items():
            distances = np.linalg.norm(np.diff(line[:, :2], axis=0), axis=1)
            self.arc_lengths[token] = np.concatenate(([0.0], np.cumsum(distances)))

    def match(self, row: dict) -> dict:
        k = int(self.config["nearest_points_k"])
        distances, indices = self.tree.query(np.asarray(row["xy"]), k=k)
        candidates = {}
        for distance, global_index in zip(
            np.atleast_1d(distances), np.atleast_1d(indices)
        ):
            token = self.point_tokens[int(global_index)]
            point_index = self.point_indices[int(global_index)]
            heading_error = _angle_error(
                row["yaw"], float(self.centerlines[token][point_index, 2])
            )
            score = float(distance) + float(
                self.config["heading_score_weight_m_per_rad"]
            ) * heading_error
            previous = candidates.get(token)
            if previous is None or score < previous["score"]:
                candidates[token] = {
                    "token": token,
                    "layer": self.layer_by_token[token],
                    "distance_m": float(distance),
                    "heading_error_rad": heading_error,
                    "centerline_index": point_index,
                    "centerline_s_m": float(self.arc_lengths[token][point_index]),
                    "score": score,
                }
        accepted = [
            value
            for value in candidates.values()
            if value["distance_m"] <= float(self.config["max_centerline_distance_m"])
            and math.degrees(value["heading_error_rad"])
            <= float(self.config["max_heading_error_deg"])
        ]
        if not accepted:
            return {**row, "match_status": "UNKNOWN", "lane_token": None}
        best = min(accepted, key=lambda value: (value["score"], value["token"]))
        return {
            **row,
            "match_status": "MATCHED",
            "lane_token": best.pop("token"),
            **best,
        }

    def point_on_token(self, token: str, xy: list[float]) -> tuple[float, float, float]:
        line = self.centerlines[token]
        distances = np.linalg.norm(line[:, :2] - np.asarray(xy), axis=1)
        index = int(np.argmin(distances))
        return (
            float(distances[index]),
            float(line[index, 2]),
            float(self.arc_lengths[token][index]),
        )


def _stable_runs(matches: list[dict], minimum_frames: int) -> list[dict]:
    runs = []
    current = None
    for row in matches:
        token = row.get("lane_token")
        frame = int(row["frame_index"])
        if (
            current is None
            or token is None
            or current["token"] != token
            or frame != current["end_frame"] + 1
        ):
            if current is not None and current["frame_count"] >= minimum_frames:
                runs.append(current)
            current = (
                None
                if token is None
                else {
                    "token": token,
                    "start_frame": frame,
                    "end_frame": frame,
                    "frame_count": 1,
                }
            )
        else:
            current["end_frame"] = frame
            current["frame_count"] += 1
    if current is not None and current["frame_count"] >= minimum_frames:
        runs.append(current)
    return runs


def _closure(connectivity: dict, token: str, direction: str, hops: int) -> set[str]:
    visited = {token}
    frontier = {token}
    for _ in range(hops):
        frontier = {
            neighbor
            for current in frontier
            for neighbor in connectivity.get(current, {}).get(direction, [])
            if neighbor not in visited
        }
        visited.update(frontier)
    return visited


def _transition_type(
    lane_index: LaneIndex,
    source_token: str,
    target_token: str,
    crossing_xy: list[float],
    config: dict,
) -> dict:
    connectivity = lane_index.nmap.connectivity
    outgoing = connectivity.get(source_token, {}).get("outgoing", [])
    incoming_target = connectivity.get(target_token, {}).get("incoming", [])
    hops = int(config["graph_hops_for_shared_corridor"])
    directed_reachable = target_token in _closure(
        connectivity, source_token, "outgoing", hops
    )
    if directed_reachable:
        is_merge = (
            len(incoming_target) >= int(config["merge_min_target_incoming_lanes"])
        )
        return {
            "type": "merge" if is_merge else "route_continuation",
            "topology_pass": is_merge,
            "directed_connected": target_token in outgoing,
            "directed_reachable_within_hops": True,
            "target_incoming_count": len(incoming_target),
            "lateral_shift_m": None,
            "parallel_heading_error_deg": None,
        }

    source_distance, source_heading, _ = lane_index.point_on_token(
        source_token, crossing_xy
    )
    target_distance, target_heading, _ = lane_index.point_on_token(
        target_token, crossing_xy
    )
    lateral_shift = source_distance + target_distance
    heading_error = math.degrees(_angle_error(source_heading, target_heading))
    shared_successor = bool(
        (
            _closure(connectivity, source_token, "outgoing", hops)
            & _closure(connectivity, target_token, "outgoing", hops)
        )
        - {source_token, target_token}
    )
    shared_predecessor = bool(
        (
            _closure(connectivity, source_token, "incoming", hops)
            & _closure(connectivity, target_token, "incoming", hops)
        )
        - {source_token, target_token}
    )
    parallel_corridor = (
        float(config["min_lateral_centerline_shift_m"])
        <= lateral_shift
        <= float(config["max_lateral_centerline_shift_m"])
        and heading_error <= float(config["max_parallel_heading_error_deg"])
    )
    topology_pass = parallel_corridor and (
        shared_successor or shared_predecessor
    )
    return {
        "type": "lane_change" if topology_pass else "unresolved_transition",
        "topology_pass": topology_pass,
        "directed_connected": False,
        "shared_successor": shared_successor,
        "shared_predecessor": shared_predecessor,
        "target_incoming_count": len(incoming_target),
        "source_centerline_distance_m": source_distance,
        "target_centerline_distance_m": target_distance,
        "lateral_shift_m": lateral_shift,
        "parallel_heading_error_deg": heading_error,
    }


def _relation(
    actor_id: int,
    frame: int,
    target_token: str,
    matches_by_actor: dict[int, dict[int, dict]],
    config: dict,
) -> dict:
    subject = matches_by_actor[actor_id].get(frame)
    if subject is None or subject.get("lane_token") != target_token:
        return {"status": "UNKNOWN", "reason": "subject_not_matched_to_target"}
    subject_s = float(subject["centerline_s_m"])
    neighbors = []
    for other_id, by_frame in sorted(matches_by_actor.items()):
        if other_id == actor_id:
            continue
        other = by_frame.get(frame)
        if other is None or other.get("lane_token") != target_token:
            continue
        neighbors.append(
            {
                "actor_id": other_id,
                "delta_s_m": float(other["centerline_s_m"]) - subject_s,
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
        "neighbor_count_on_exact_target_token": len(neighbors),
    }


def _candidate_events(
    scene_id: str,
    lane_index: LaneIndex,
    tracks: dict[int, list[dict]],
    matches: dict[int, list[dict]],
    config: dict,
) -> tuple[list[dict], list[dict], list[dict]]:
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
    positives = []
    transitions = []
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
                lane_index,
                source["token"],
                target["token"],
                crossing["xy"],
                config,
            )
            relation_frame = int(target["start_frame"]) + int(
                config["min_stable_target_frames"]
            ) - 1
            relation = _relation(
                actor_id,
                relation_frame,
                target["token"],
                match_by_actor,
                config,
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

    negatives = []
    window = int(config["negative_window_frames"])
    positive_actors = {int(row["actor_id"]) for row in positives}
    for actor_id in sorted(positive_actors):
        for run in runs_by_actor[actor_id]:
            if int(run["frame_count"]) < window:
                continue
            start = int(run["start_frame"])
            end = start + window - 1
            midpoint = (start + end) // 2
            relation = _relation(
                actor_id, midpoint, run["token"], match_by_actor, config
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
        "eligible_actors_per_scene": all(
            count >= int(gates["min_eligible_actors_per_scene"])
            for count in summary["eligible_actor_count_by_scene"].values()
        ),
        "positive_events": summary["positive_event_count"]
        >= int(gates["min_positive_events"]),
        "negative_events": summary["negative_event_count"]
        >= int(gates["min_negative_events"]),
        "same_actor_pairs": summary["same_actor_pair_count"]
        >= int(gates["min_same_actor_pairs"]),
        "positive_scenes": summary["positive_scene_count"]
        >= int(gates["min_positive_scenes"]),
        "unknown_not_positive": not bool(gates["unknown_is_positive"]),
        "noninteractive_not_positive": not bool(
            gates["noninteractive_transition_is_positive"]
        ),
    }
    return all(checks.values()), checks


def run(config_path: Path, output_root: Path | None) -> Path:
    config = _load_yaml(config_path)
    n0_run = Path(config["n0_run"])
    if not (n0_run / "COMPLETE").is_file():
        raise RuntimeError(f"N0 未 COMPLETE: {n0_run}")
    n0_summary = json.loads((n0_run / "summary.json").read_text(encoding="utf-8"))
    if n0_summary["research_verdict"] != "n0_asset_pass":
        raise RuntimeError(f"N0 verdict 非 pass: {n0_summary}")

    dataset_root = Path(config["dataset_root"])
    processed_root = Path(config["processed_root"])
    eligibility_rows = []
    tracks_by_scene = {}
    maps = {}
    matches_by_scene = {}
    map_audit = {}
    for scene in config["scenes"]:
        scene_id = scene["scene_id"]
        map_name = scene["map_name"]
        if map_name not in maps:
            maps[map_name] = LaneIndex(
                NuScenesMap(str(dataset_root), map_name), config["map_matching"]
            )
        raw_path = (
            processed_root / scene_id / "instances" / "instances_info.json"
        )
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        tracks = {}
        matches = {}
        for actor_key, actor in sorted(raw.items(), key=lambda item: int(item[0])):
            actor_id = int(actor_key)
            audit, rows = _eligibility(actor_id, actor, config["eligibility"])
            audit["scene_id"] = scene_id
            eligibility_rows.append(audit)
            if not audit["eligible"]:
                continue
            tracks[actor_id] = rows
            matches[actor_id] = [maps[map_name].match(row) for row in rows]
        tracks_by_scene[scene_id] = tracks
        matches_by_scene[scene_id] = matches
        total = sum(len(rows) for rows in matches.values())
        matched = sum(
            row["match_status"] == "MATCHED"
            for rows in matches.values()
            for row in rows
        )
        map_audit[scene_id] = {
            "eligible_actor_count": len(tracks),
            "pose_count": total,
            "matched_pose_count": matched,
            "matched_pose_fraction": matched / total if total else None,
        }

    positives = []
    negatives = []
    transitions = []
    for scene in config["scenes"]:
        scene_id = scene["scene_id"]
        scene_positive, scene_negative, scene_transitions = _candidate_events(
            scene_id,
            maps[scene["map_name"]],
            tracks_by_scene[scene_id],
            matches_by_scene[scene_id],
            config["event_definition"],
        )
        positives.extend(scene_positive)
        negatives.extend(scene_negative)
        transitions.extend(scene_transitions)

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

    eligible_count_by_scene = {
        scene["scene_id"]: sum(
            row["scene_id"] == scene["scene_id"] and row["eligible"]
            for row in eligibility_rows
        )
        for scene in config["scenes"]
    }
    base_summary = {
        "eligible_actor_count_by_scene": eligible_count_by_scene,
        "positive_event_count": len(positives),
        "negative_event_count": len(negatives),
        "transition_candidate_count": len(transitions),
        "same_actor_pair_count": len(pairs),
        "positive_scene_count": len({row["scene_id"] for row in positives}),
    }
    passed, checks = _gate_decision(base_summary, config["gates"])
    terminal = "COMPLETE" if passed else "REJECTED"
    verdict = "n1_event_pool_pass" if passed else config["stop_rule"]["on_gate_failure"]

    config_sha = file_fingerprint(str(config_path))
    code = git_state(str(Path(config["repo_root"])))
    run_id = generate_run_id(config["task_id"], "mini-event-v1", int(config["seed"]), config_sha)
    run_root = output_root or Path(config["run_root"])
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    event_pool = {
        "schema_version": config["schema_version"],
        "task_id": config["task_id"],
        "split": config["split"],
        "seed": int(config["seed"]),
        "selection_inputs": "annotation_tracks_and_frozen_vector_map_only",
        "positives": positives,
        "negatives": negatives,
        "transition_candidates": transitions,
        "same_actor_pairs": pairs,
    }
    event_pool["event_pool_sha256"] = canonical_sha256(event_pool)
    summary = {
        "task_id": config["task_id"],
        "run_id": run_id,
        "split": config["split"],
        "seed": int(config["seed"]),
        **base_summary,
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
        "data_fingerprint": canonical_sha256(
            {
                "n0_asset_manifest_sha256": n0_summary["asset_manifest_sha256"],
                "n0_scene_map_registry_sha256": n0_summary[
                    "scene_map_registry_sha256"
                ],
                "instance_files": {
                    scene["scene_id"]: file_fingerprint(
                        str(
                            processed_root
                            / scene["scene_id"]
                            / "instances"
                            / "instances_info.json"
                        )
                    )
                    for scene in config["scenes"]
                },
            }
        ),
        "n0_run": str(n0_run),
        "split": config["split"],
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
    atomic_write_json(
        str(run_dir / "actor_eligibility.json"),
        {"actors": eligibility_rows, "counts_by_scene": eligible_count_by_scene},
    )
    atomic_write_json(str(run_dir / "map_match_audit.json"), map_audit)
    atomic_write_json(str(run_dir / "event_pool.json"), event_pool)
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
        default=Path("configs/resim/event_first_n1_event_v1.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    run(args.config, args.output_root)


if __name__ == "__main__":
    main()
