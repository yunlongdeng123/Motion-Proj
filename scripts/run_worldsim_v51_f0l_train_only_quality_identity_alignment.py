#!/usr/bin/env python3
"""Evaluate faithful Gaussian Grouping masks on the frozen train-only weak alignment gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v51.protocol import ProtocolError, sha256_file
from scripts.run_worldsim_v51_f0b_three_view_association_parity import _load_yaml, _verify, repository_source_identity
from scripts.run_worldsim_v51_h_uplift import _inventory, _utc_now, _write_json, _write_jsonl, _write_text


SCHEMA = "worldsim_v51_stage_f_f0l_train_only_quality_identity_alignment_v1"
TASK_ID = "WS-V51-M1-F-IDENTITY-EMBEDDING-01"


def _validate_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_yaml(path)
    if config.get("schema_version") != SCHEMA or config.get("task_id") != TASK_ID:
        raise ProtocolError("F0l config drift")
    auth = config["authorization"]["f0k_freeze"]
    freeze = _load_yaml(_verify(PROJECT / auth["path"], auth["sha256"], "F0k freeze", int(auth["bytes"])))
    if freeze.get("status") != auth["required_status"] or freeze["governance"].get("next_phase") != auth["required_next_phase"]:
        raise ProtocolError("F0l authorization drift")
    spec = config["inputs"]["manifest"]
    manifest = json.loads(_verify(Path(spec["path"]), spec["sha256"], "F0k input manifest", int(spec["bytes"])).read_text())
    if manifest.get("view_count") != spec["view_count"] or manifest.get("projection_count") != spec["projection_count"]:
        raise ProtocolError("F0l input denominator drift")
    if manifest.get("quality_gate_preregistration", {}).get("per_scene_thresholds") != config["evaluation"]["per_scene_thresholds"]:
        raise ProtocolError("F0l threshold binding drift")
    if config["evaluation"].get("projected_box_rasterization") != "floor_xy0_ceil_xy1_half_open_clipped":
        raise ProtocolError("F0l rasterization drift")
    if config["locks"] != {
        "candidate_mask_pixel_reads": 45, "dynamic_mask_pixel_reads": 45, "image_pixels_read": False,
        "threshold_search": False, "identity_training": False, "h_quality_read": False,
        "screening_quality_read": False, "confirmation_quality_read": False, "validation_quality_read": False,
        "test_quality_read": False, "kitti_method_tuning": False, "f1_execution": False, "f2_execution": False,
        "m2_status": "pending", "m3_status": "pending",
    }:
        raise ProtocolError("F0l research lock drift")
    return config, manifest


def _assignment_metrics(actor_views: list[dict[str, Any]], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    by_track: dict[str, list[dict[str, Any]]] = {}
    for row in actor_views:
        by_track.setdefault(row["instance_token"], []).append(row)
    eligible = {token: rows for token, rows in by_track.items() if len(rows) >= 2}
    rows = [row for token in sorted(eligible) for row in eligible[token]]
    tracks = sorted(eligible)
    labels = sorted({int(label) for row in rows for label in row["label_counts"]})
    total_support = sum(int(row["support_pixels"]) for row in rows)
    positive = sum(int(row["positive_pixels"]) for row in rows)
    matrix = np.zeros((len(tracks), len(labels)), dtype=np.int64)
    track_index = {token: index for index, token in enumerate(tracks)}
    label_index = {label: index for index, label in enumerate(labels)}
    for row in rows:
        for label, count in row["label_counts"].items():
            matrix[track_index[row["instance_token"]], label_index[int(label)]] += int(count)
    independent_hits = int(sum(int(matrix[index].max()) if matrix.shape[1] else 0 for index in range(len(tracks))))
    assignments: dict[str, int] = {}
    assignment_hits = 0
    if matrix.size:
        track_indices, label_indices = linear_sum_assignment(-matrix)
        for track_i, label_i in zip(track_indices.tolist(), label_indices.tolist()):
            if int(matrix[track_i, label_i]) > 0:
                assignments[tracks[track_i]] = labels[label_i]
                assignment_hits += int(matrix[track_i, label_i])
    persistent = 0
    track_rows = []
    for token in tracks:
        assigned = assignments.get(token)
        present_views = sum(int(row["label_counts"].get(assigned, 0)) > 0 for row in eligible[token]) if assigned is not None else 0
        persistent += present_views >= 2
        track_rows.append({
            "instance_token": token, "eligible_view_count": len(eligible[token]), "assigned_short_id": assigned,
            "assigned_present_view_count": present_views, "support_pixels": sum(int(row["support_pixels"]) for row in eligible[token]),
        })
    metrics = {
        "eligible_tracks": len(tracks), "eligible_actor_views": len(rows), "support_pixels": total_support,
        "foreground_coverage": float(positive / total_support) if total_support else 0.0,
        "independent_best_identity_recall": float(independent_hits / total_support) if total_support else 0.0,
        "one_to_one_assignment_recall": float(assignment_hits / total_support) if total_support else 0.0,
        "assignment_efficiency": float(assignment_hits / independent_hits) if independent_hits else 0.0,
        "persistent_track_fraction": float(persistent / len(tracks)) if tracks else 0.0,
    }
    checks = {
        "eligible_tracks": metrics["eligible_tracks"] >= int(thresholds["minimum_eligible_tracks"]),
        "eligible_actor_views": metrics["eligible_actor_views"] >= int(thresholds["minimum_eligible_actor_views"]),
        "foreground_coverage": metrics["foreground_coverage"] >= float(thresholds["minimum_foreground_coverage"]),
        "one_to_one_assignment_recall": metrics["one_to_one_assignment_recall"] >= float(thresholds["minimum_one_to_one_assignment_recall"]),
        "assignment_efficiency": metrics["assignment_efficiency"] >= float(thresholds["minimum_assignment_efficiency"]),
        "persistent_track_fraction": metrics["persistent_track_fraction"] >= float(thresholds["minimum_persistent_track_fraction"]),
    }
    return {"metrics": metrics, "checks": checks, "assignments": assignments, "tracks": track_rows, "eligible_actor_views_detail": rows}


def _read_mask(path: Path, expected_sha: str, expected_bytes: int) -> np.ndarray:
    if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha:
        raise ProtocolError(f"mask identity drift: {path}")
    with Image.open(path) as image:
        value = np.asarray(image)
    if value.shape != (900, 1600) or value.dtype != np.uint8:
        raise ProtocolError(f"mask schema drift: {path}")
    return value


def run(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config, manifest = _validate_config(config_path)
    if run_dir.exists():
        raise ProtocolError(f"refusing overwrite: {run_dir}")
    run_dir.mkdir(parents=True)
    _write_text(run_dir / "resolved_config.yaml", config_path.read_text(encoding="utf-8"))
    identity = repository_source_identity()
    events = [{"event": "run_started", "at_utc": _utc_now()}]
    _write_jsonl(run_dir / "events.jsonl", events)
    _write_json(run_dir / "status.json", {"task_id": TASK_ID, "status": "running", "source_commit": identity["commit"]})
    started = time.perf_counter()
    scene_views: dict[str, list[dict[str, Any]]] = {}
    for view in manifest["views"]:
        scene_views.setdefault(view["scene"], []).append(view)
    scene_reports = []
    for scene in ("scene-0471", "scene-1087", "scene-0379"):
        actor_views = []
        view_reports = []
        for view in scene_views[scene]:
            candidate_spec = view["candidate_mask"]
            dynamic_spec = view["dynamic_mask"]
            candidate = _read_mask(Path(candidate_spec["path"]), candidate_spec["sha256"], int(candidate_spec["bytes"]))
            dynamic = _read_mask(Path(dynamic_spec["path"]), dynamic_spec["sha256"], int(dynamic_spec["bytes"])) > 0
            boxes = []
            overlap = np.zeros(candidate.shape, dtype=np.uint16)
            for projection in view["projections"]:
                x0 = max(0, int(math.floor(projection["box_xyxy"][0])))
                y0 = max(0, int(math.floor(projection["box_xyxy"][1])))
                x1 = min(candidate.shape[1], int(math.ceil(projection["box_xyxy"][2])))
                y1 = min(candidate.shape[0], int(math.ceil(projection["box_xyxy"][3])))
                if x1 <= x0 or y1 <= y0:
                    continue
                overlap[y0:y1, x0:x1] += 1
                boxes.append((projection, x0, y0, x1, y1))
            view_actor_rows = []
            for projection, x0, y0, x1, y1 in boxes:
                support = dynamic[y0:y1, x0:x1] & (overlap[y0:y1, x0:x1] == 1)
                support_pixels = int(support.sum())
                if support_pixels < int(config["evaluation"]["eligible_actor_view_minimum_support_pixels"]):
                    continue
                values = candidate[y0:y1, x0:x1][support]
                positive_values = values[values > 0]
                labels, counts = np.unique(positive_values, return_counts=True)
                row = {
                    "scene": scene, "frame": int(view["frame"]), "camera": int(view["camera"]),
                    "instance_id": int(projection["instance_id"]), "instance_token": projection["instance_token"],
                    "class_name": projection["class_name"], "support_pixels": support_pixels,
                    "positive_pixels": int(positive_values.size),
                    "label_counts": {int(label): int(count) for label, count in zip(labels.tolist(), counts.tolist())},
                }
                actor_views.append(row)
                view_actor_rows.append(row)
            view_reports.append({"frame": int(view["frame"]), "camera": int(view["camera"]), "projected_actor_count": len(view["projections"]), "eligible_actor_view_count": len(view_actor_rows)})
        result = _assignment_metrics(actor_views, config["evaluation"]["per_scene_thresholds"])
        result.update({"scene": scene, "view_count": len(scene_views[scene]), "view_reports": view_reports, "all_checks_pass": all(result["checks"].values())})
        scene_reports.append(result)
    all_scenes_pass = all(row["all_checks_pass"] for row in scene_reports)
    outcome = config["decision"]["pass_outcome"] if all_scenes_pass else config["decision"]["reject_outcome"]
    conclusion = config["decision"]["pass_conclusion"] if all_scenes_pass else config["decision"]["reject_conclusion"]
    next_action = config["decision"]["pass_next_action"] if all_scenes_pass else config["decision"]["reject_next_action"]
    resources = {"wall_seconds": time.perf_counter() - started, "cgroup_memory_current_bytes": int(Path("/sys/fs/cgroup/memory.current").read_text().strip()), "disk_free_after_bytes": shutil.disk_usage(run_dir).free}
    limits = config["resources"]
    checks = {"wall": resources["wall_seconds"] <= float(limits["maximum_wall_seconds"]), "cgroup": resources["cgroup_memory_current_bytes"] <= int(limits["maximum_cgroup_memory_bytes"]), "disk": resources["disk_free_after_bytes"] >= int(limits["minimum_disk_free_bytes_after"])}
    if not all(checks.values()):
        raise ProtocolError(f"F0l resource gate: {checks}")
    _write_json(run_dir / "artifacts/quality_alignment_report.json", {"scene_reports": scene_reports, "all_scenes_pass": all_scenes_pass, "outcome": outcome, "conclusion": conclusion})
    _write_json(run_dir / "artifacts/resources.json", resources)
    summary = {
        "schema_version": "worldsim_v51_f0l_summary_v1", "task_id": TASK_ID, "status": "done", "outcome": outcome,
        "conclusion": conclusion, "source_commit": identity["commit"], "source_tree": identity["tree"],
        "scene_reports": scene_reports, "all_scenes_pass": all_scenes_pass, "resources": resources, "resource_checks": checks,
        "candidate_mask_pixel_reads": 45, "dynamic_mask_pixel_reads": 45, "image_pixels_read": False,
        "threshold_search": False, "identity_training_authorized": False, "next_action": next_action,
        "h_quality_read": False, "screening_quality_read": False, "confirmation_quality_read": False,
        "validation_quality_read": False, "test_quality_read": False, "kitti_method_tuning": False,
        "f1_execution": False, "f2_execution": False, "m2_status": "pending", "m3_status": "pending",
    }
    _write_json(run_dir / "summary.json", summary)
    events.append({"event": "run_completed", "at_utc": _utc_now(), "outcome": outcome})
    _write_jsonl(run_dir / "events.jsonl", events)
    _write_json(run_dir / "manifest.json", {"task_id": TASK_ID, "status": "done", "inventory": _inventory(run_dir)})
    _write_json(run_dir / "status.json", {"task_id": TASK_ID, "status": "done", "outcome": outcome, "conclusion": conclusion, "source_commit": identity["commit"]})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/worldsim_v51/stage_f_f0l_train_only_quality_identity_alignment_v1.yaml")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.run_dir.resolve()), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
