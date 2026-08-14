#!/usr/bin/env python3
"""从 nuScenes annotation metadata 结果盲冻结 V5 M3 development clips。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

import ijson
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.worldsim_v5_forensics_common import (
    atomic_json,
    copy_source_snapshot,
    finalize_formal_run,
    prepare_formal_run,
    sha256_file,
    utc_now,
    verify_file,
    write_events,
    write_resolved_config,
)


TASK_ID = "WS-V5-M3-CONSTRAINT-PROJECTED-TEMPORAL-01"
SCHEMA_VERSION = "worldsim_v5_m3_development_clip_inventory_v1"


class M3ClipInventoryError(RuntimeError):
    """M3 result-blind clip inventory 合约失败。"""


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise M3ClipInventoryError("M3 clip inventory config schema 漂移")
    if payload.get("task_id") != TASK_ID or payload.get("status") != "running":
        raise M3ClipInventoryError("M3 clip inventory task/status 漂移")
    scope = payload["scope"]
    for name in (
        "image_read",
        "lidar_blob_read",
        "reconstruction_quality_read",
        "edit_quality_read",
        "development_quality_read",
        "validation_quality_read",
        "test_quality_read",
        "kitti_quality_read",
        "model_inference",
        "gpu_required",
        "parameter_search_performed",
        "method_arm_selection_performed",
    ):
        if scope.get(name) is not False:
            raise M3ClipInventoryError(f"result-blind scope 漂移: {name}")
    if payload["clip_policy"].get("keyframe_count") != 7:
        raise M3ClipInventoryError("M3 clip denominator 必须为七 keyframes")
    if payload["protocol_audit"].get("conclusion") != (
        "m3_result_blind_protocol_frozen_development_implementation_unlocked"
    ):
        raise M3ClipInventoryError("M3 protocol audit conclusion 合约漂移")
    return payload


def _load_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise M3ClipInventoryError(f"metadata table 非 object list: {path}")
    return payload


def enumerate_windows(
    annotations: Iterable[Mapping[str, Any]],
    *,
    sample_index: Mapping[str, int],
    sample_timestamp: Mapping[str, int],
    keyframe_count: int,
    maximum_gap_seconds: float,
    minimum_lidar_points: int,
    minimum_visibility: int,
) -> list[list[dict[str, Any]]]:
    rows = sorted(
        (dict(row) for row in annotations),
        key=lambda row: sample_index[str(row["sample_token"])],
    )
    windows: list[list[dict[str, Any]]] = []
    for start in range(max(0, len(rows) - keyframe_count + 1)):
        window = rows[start : start + keyframe_count]
        indices = [sample_index[str(row["sample_token"])] for row in window]
        timestamps = [sample_timestamp[str(row["sample_token"])] for row in window]
        if any(right - left != 1 for left, right in zip(indices[:-1], indices[1:])):
            continue
        if any(
            (right - left) / 1_000_000.0 > maximum_gap_seconds
            for left, right in zip(timestamps[:-1], timestamps[1:])
        ):
            continue
        if any(int(row.get("num_lidar_pts", 0)) < minimum_lidar_points for row in window):
            continue
        if any(int(row.get("visibility_token", 0)) < minimum_visibility for row in window):
            continue
        windows.append(window)
    return windows


def select_scene_clip(
    candidates: Iterable[Mapping[str, Any]], *, category_priority: Mapping[str, int]
) -> dict[str, Any] | None:
    rows = [dict(candidate) for candidate in candidates]
    if not rows:
        return None
    return min(
        rows,
        key=lambda row: (
            int(category_priority[row["category_name"]]),
            -int(row["minimum_lidar_points"]),
            -int(row["total_lidar_points"]),
            str(row["instance_token"]),
            int(row["start_sample_index"]),
        ),
    )


def build_inventory(config: Mapping[str, Any]) -> dict[str, Any]:
    metadata_root = Path(config["metadata"]["root"])
    scene_rows = _load_json(metadata_root / "scene.json")
    sample_rows = _load_json(metadata_root / "sample.json")
    category_rows = _load_json(metadata_root / "category.json")
    instance_rows = _load_json(metadata_root / "instance.json")
    development = list(config["cohort"]["development_scenes"])
    scene_name_by_token = {str(row["token"]): str(row["name"]) for row in scene_rows}
    development_tokens = {
        token for token, name in scene_name_by_token.items() if name in development
    }
    if len(development_tokens) != len(development):
        raise M3ClipInventoryError("fresh development scene token 映射不完整")
    samples_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        token = str(row["scene_token"])
        if token in development_tokens:
            samples_by_scene[token].append(row)
    sample_scene: dict[str, str] = {}
    sample_index: dict[str, int] = {}
    sample_timestamp: dict[str, int] = {}
    for scene_token, rows in samples_by_scene.items():
        ordered = sorted(rows, key=lambda row: int(row["timestamp"]))
        for index, row in enumerate(ordered):
            token = str(row["token"])
            sample_scene[token] = scene_name_by_token[scene_token]
            sample_index[token] = index
            sample_timestamp[token] = int(row["timestamp"])
    category_by_token = {str(row["token"]): str(row["name"]) for row in category_rows}
    allowed = set(config["clip_policy"]["allowed_categories"])
    instance_category = {
        str(row["token"]): category_by_token[str(row["category_token"])]
        for row in instance_rows
        if category_by_token.get(str(row["category_token"])) in allowed
    }
    annotations_by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    annotation_path = metadata_root / "sample_annotation.json"
    with annotation_path.open("rb") as handle:
        for row in ijson.items(handle, "item"):
            sample_token = str(row["sample_token"])
            instance_token = str(row["instance_token"])
            if sample_token in sample_scene and instance_token in instance_category:
                annotations_by_instance[instance_token].append(row)
    candidates_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    policy = config["clip_policy"]
    for instance_token, rows in annotations_by_instance.items():
        windows = enumerate_windows(
            rows,
            sample_index=sample_index,
            sample_timestamp=sample_timestamp,
            keyframe_count=int(policy["keyframe_count"]),
            maximum_gap_seconds=float(policy["maximum_timestamp_gap_seconds"]),
            minimum_lidar_points=int(policy["minimum_lidar_points_each_frame"]),
            minimum_visibility=int(policy["minimum_visibility_token_each_frame"]),
        )
        for window in windows:
            scene = sample_scene[str(window[0]["sample_token"])]
            lidar = [int(row["num_lidar_pts"]) for row in window]
            candidates_by_scene[scene].append(
                {
                    "scene": scene,
                    "instance_token": instance_token,
                    "category_name": instance_category[instance_token],
                    "start_sample_index": sample_index[str(window[0]["sample_token"])],
                    "minimum_lidar_points": min(lidar),
                    "total_lidar_points": sum(lidar),
                    "frames": [
                        {
                            "sample_token": str(row["sample_token"]),
                            "sample_index": sample_index[str(row["sample_token"])],
                            "timestamp": sample_timestamp[str(row["sample_token"])],
                            "translation": [float(value) for value in row["translation"]],
                            "size": [float(value) for value in row["size"]],
                            "rotation": [float(value) for value in row["rotation"]],
                            "bottom_z": float(row["translation"][2])
                            - 0.5 * float(row["size"][2]),
                            "lidar_points": int(row["num_lidar_pts"]),
                            "visibility_token": int(row["visibility_token"]),
                        }
                        for row in window
                    ],
                }
            )
    scenes = []
    for scene in development:
        selected = select_scene_clip(
            candidates_by_scene.get(scene, []),
            category_priority=policy["category_priority"],
        )
        scenes.append(
            {
                "scene": scene,
                "status": "ready" if selected is not None else "abstain",
                "reason": None if selected is not None else policy["absent_policy"].removeprefix("retain_as_"),
                "eligible_window_count": len(candidates_by_scene.get(scene, [])),
                "selected": selected,
                "retained_in_denominator": True,
            }
        )
    return {
        "schema_version": "worldsim_v5_m3_development_clip_inventory_artifact_v1",
        "task_id": TASK_ID,
        "selection_quality_read": False,
        "scene_denominator": len(development),
        "ready_scene_count": sum(row["status"] == "ready" for row in scenes),
        "abstain_scene_count": sum(row["status"] == "abstain" for row in scenes),
        "scenes": scenes,
    }


def _run_impl(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = load_config(config_path)
    source_head = prepare_formal_run(run_dir, TASK_ID, PROJECT)
    resolved = write_resolved_config(run_dir, config)
    events = [{"event": "run_started", "at_utc": utc_now(), "source_commit": source_head}]
    write_events(run_dir, events)
    started = time.perf_counter()
    inputs: dict[str, Any] = {}
    for name in ("summary", "status", "lock"):
        binding = config["protocol_audit"][name]
        inputs[f"protocol_{name}"] = verify_file(binding["path"], binding["sha256"])
    inputs["cohort"] = verify_file(config["cohort"]["path"], config["cohort"]["sha256"])
    root = Path(config["metadata"]["root"])
    inputs["metadata"] = {
        name: verify_file(root / name, digest)
        for name, digest in config["metadata"]["files"].items()
    }
    protocol_summary = json.loads(Path(inputs["protocol_summary"]["path"]).read_text())
    if protocol_summary.get("conclusion") != config["protocol_audit"]["conclusion"]:
        raise M3ClipInventoryError("M3 protocol audit conclusion 漂移")
    inventory = build_inventory(config)
    artifact = run_dir / "artifacts/development_clip_inventory.json"
    atomic_json(artifact, inventory)
    snapshot = copy_source_snapshot(
        run_dir,
        [
            config_path,
            PROJECT / "scripts/run_worldsim_v5_m3_clip_inventory.py",
            PROJECT / "tests/test_worldsim_v5_m3_clip_inventory.py",
        ],
        PROJECT,
    )
    conclusion = (
        "m3_result_blind_development_clips_frozen"
        if inventory["ready_scene_count"] > 0
        else "m3_development_clips_all_abstain"
    )
    summary = {
        "schema_version": "worldsim_v5_m3_development_clip_inventory_summary_v1",
        "task_id": TASK_ID,
        "task_status": "running",
        "status": "done",
        "phase": config["phase"],
        "source_commit": source_head,
        "conclusion": conclusion,
        "scene_denominator": inventory["scene_denominator"],
        "ready_scene_count": inventory["ready_scene_count"],
        "abstain_scene_count": inventory["abstain_scene_count"],
        "selected_scenes": [
            row["scene"] for row in inventory["scenes"] if row["status"] == "ready"
        ],
        "inventory_sha256": sha256_file(artifact),
        "source_snapshot_count": len(snapshot),
        "duration_seconds": time.perf_counter() - started,
        "gpu_started": False,
        "image_read": False,
        "lidar_blob_read": False,
        "development_quality_read": False,
        "validation_quality_read": False,
        "test_quality_read": False,
        "kitti_quality_read": False,
        "parameter_search_performed": False,
        "method_arm_selected": False,
    }
    events.append({"event": "run_done", "at_utc": utc_now(), **summary})
    events_record = write_events(run_dir, events)
    status = finalize_formal_run(
        run_dir=run_dir,
        task_id=TASK_ID,
        task_status="running",
        conclusion=conclusion,
        project_head=source_head,
        input_bindings=inputs,
        summary=summary,
        resolved_config_record=resolved,
        events_record=events_record,
    )
    return {**summary, "formal_status": status}


def run(config_path: Path, run_dir: Path) -> dict[str, Any]:
    try:
        return _run_impl(config_path, run_dir)
    except Exception as error:
        if run_dir.is_dir() and not (run_dir / "status.json").exists():
            source_head = subprocess.check_output(
                ["git", "-C", str(PROJECT), "rev-parse", "HEAD"], text=True
            ).strip()
            atomic_json(
                run_dir / "status.json",
                {
                    "schema_version": "worldsim_v5_m3_development_clip_inventory_status_v1",
                    "task_id": TASK_ID,
                    "task_status": "running",
                    "status": "blocked",
                    "source_commit": source_head,
                    "summary_sha256": None,
                    "manifest_sha256": None,
                    "reason": f"{type(error).__name__}: {error}",
                    "development_quality_read": False,
                    "validation_quality_read": False,
                    "test_quality_read": False,
                    "kitti_quality_read": False,
                    "gpu_started": False,
                    "finished_at_utc": utc_now(),
                },
            )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.run_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
