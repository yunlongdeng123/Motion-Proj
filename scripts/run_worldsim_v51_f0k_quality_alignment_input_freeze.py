#!/usr/bin/env python3
"""Freeze F0k quality/alignment inputs and 3D-box projection denominator without mask pixel reads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v32.semantic_schema import project_box_prompt
from motion_proj.worldsim_v51.protocol import ProtocolError, sha256_file
from scripts.run_worldsim_v51_f0b_three_view_association_parity import _load_yaml, _verify, repository_source_identity
from scripts.run_worldsim_v51_h_uplift import _inventory, _utc_now, _write_json, _write_jsonl, _write_text


SCHEMA = "worldsim_v51_stage_f_f0k_quality_alignment_input_freeze_v1"
TASK_ID = "WS-V51-M1-F-IDENTITY-EMBEDDING-01"


def _annotation(actor: dict[str, Any], frame: int) -> tuple[np.ndarray, np.ndarray] | None:
    values = actor["frame_annotations"]
    try:
        index = [int(value) for value in values["frame_idx"]].index(frame)
    except ValueError:
        return None
    return np.asarray(values["obj_to_world"][index], dtype=np.float64), np.asarray(values["box_size"][index], dtype=np.float64)


def _intrinsics(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if values.size < 4:
        raise ProtocolError(f"invalid intrinsics: {path}")
    return np.asarray([[values[0], 0.0, values[2]], [0.0, values[1], values[3]], [0.0, 0.0, 1.0]])


def _validate_config(path: Path) -> dict[str, Any]:
    config = _load_yaml(path)
    if config.get("schema_version") != SCHEMA or config.get("task_id") != TASK_ID:
        raise ProtocolError("F0k config drift")
    auth = config["authorization"]["f0j_freeze"]
    freeze = _load_yaml(_verify(PROJECT / auth["path"], auth["sha256"], "F0j freeze", int(auth["bytes"])))
    if freeze.get("status") != auth["required_status"] or freeze["governance"].get("next_phase") != auth["required_next_phase"]:
        raise ProtocolError("F0k authorization drift")
    materialization = config["materialization"]["manifest"]
    manifest_path = Path(config["materialization"]["run_dir"]) / materialization["path"]
    payload = json.loads(_verify(manifest_path, materialization["sha256"], "F0j materialization", int(materialization["bytes"])).read_text())
    if payload.get("output_record_chain_sha256") != materialization["output_record_chain_sha256"]:
        raise ProtocolError("F0k materialization chain drift")
    source = config["source_image_manifest"]
    image_manifest = json.loads(_verify(Path(source["path"]), source["sha256"], "source images", int(source["bytes"])).read_text())
    if image_manifest.get("record_count") != source["record_count"] or image_manifest.get("record_chain_sha256") != source["record_chain_sha256"]:
        raise ProtocolError("F0k source image denominator drift")
    thresholds = config["quality_gate_preregistration"]["per_scene_thresholds"]
    if thresholds != {
        "minimum_eligible_tracks": 1,
        "minimum_eligible_actor_views": 2,
        "minimum_foreground_coverage": 0.70,
        "minimum_one_to_one_assignment_recall": 0.35,
        "minimum_assignment_efficiency": 0.75,
        "minimum_persistent_track_fraction": 0.50,
    }:
        raise ProtocolError("F0k threshold drift")
    if any(config["locks"][name] is not False for name in ("candidate_mask_pixels_read", "dynamic_mask_pixels_read", "image_pixels_read", "quality_metrics_read", "threshold_search")):
        raise ProtocolError("F0k no-quality lock drift")
    return config


def run(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = _validate_config(config_path)
    if run_dir.exists():
        raise ProtocolError(f"refusing overwrite: {run_dir}")
    run_dir.mkdir(parents=True)
    _write_text(run_dir / "resolved_config.yaml", config_path.read_text(encoding="utf-8"))
    identity = repository_source_identity()
    events = [{"event": "run_started", "at_utc": _utc_now()}]
    _write_jsonl(run_dir / "events.jsonl", events)
    _write_json(run_dir / "status.json", {"task_id": TASK_ID, "status": "running", "source_commit": identity["commit"]})
    started = time.perf_counter()
    image_manifest = json.loads(Path(config["source_image_manifest"]["path"]).read_text(encoding="utf-8"))
    image_rows = {(row["scene"], int(row["frame"]), int(row["camera"])): row for row in image_manifest["records"]}
    f0j_manifest = json.loads((Path(config["materialization"]["run_dir"]) / config["materialization"]["manifest"]["path"]).read_text(encoding="utf-8"))
    candidate_rows = {(row["scene"], row["filename"]): row for row in f0j_manifest["records"]}
    views = []
    projection_count = 0
    for scene in config["scenes"]:
        scene_name = scene["scene"]
        root = Path(scene["processed_scene"])
        instances_path = _verify(root / "instances/instances_info.json", scene["instances_info_sha256"], f"{scene_name} instances")
        frame_instances_path = _verify(root / "instances/frame_instances.json", scene["frame_instances_sha256"], f"{scene_name} frame instances")
        instances = json.loads(instances_path.read_text(encoding="utf-8"))
        frame_instances = json.loads(frame_instances_path.read_text(encoding="utf-8"))
        attempt_name = config["materialization"]["attempts"][scene_name]
        for frame in config["views"]["frames"]:
            for camera in config["views"]["cameras"]:
                key = (scene_name, int(frame), int(camera))
                source = image_rows[key]
                stem = f"{int(frame):03d}_{int(camera)}"
                candidate_path = Path(config["materialization"]["run_dir"]) / "artifacts/attempts" / attempt_name / "output/Annotations" / f"{stem}.png"
                candidate = candidate_rows[(scene_name, f"{stem}.png")]
                if candidate_path.stat().st_size != int(candidate["bytes"]) or sha256_file(candidate_path) != candidate["sha256"]:
                    raise ProtocolError(f"F0k candidate identity drift: {scene_name}/{stem}")
                dynamic_path = root / "dynamic_masks/all" / f"{stem}.png"
                if not dynamic_path.is_file():
                    raise ProtocolError(f"F0k dynamic mask missing: {scene_name}/{stem}")
                intrinsic_path = root / "intrinsics" / f"{camera}.txt"
                extrinsic_path = root / "extrinsics" / f"{stem}.txt"
                intrinsic = _intrinsics(intrinsic_path)
                c2w = np.loadtxt(extrinsic_path, dtype=np.float64).reshape(4, 4)
                projections = []
                for instance_id in sorted({int(value) for value in frame_instances.get(str(frame), [])}):
                    actor = instances[str(instance_id)]
                    annotation = _annotation(actor, int(frame))
                    if annotation is None:
                        continue
                    box = project_box_prompt(
                        obj_to_world=annotation[0], box_size=annotation[1], camera_to_world=c2w, intrinsics=intrinsic,
                        image_width=int(config["views"]["width"]), image_height=int(config["views"]["height"]),
                        minimum_depth_m=float(config["projection"]["minimum_depth_m"]),
                        padding_fraction=float(config["projection"]["padding_fraction"]),
                        minimum_side_pixels=float(config["projection"]["minimum_side_pixels"]),
                    )
                    if box is None:
                        continue
                    area = float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))
                    if area < int(config["projection"]["minimum_box_area_pixels"]):
                        continue
                    projections.append({"instance_id": instance_id, "instance_token": actor["id"], "class_name": actor["class_name"], "box_xyxy": [float(value) for value in box], "box_area_pixels": area})
                projection_count += len(projections)
                views.append({
                    "scene": scene_name, "scene_index": int(scene["index"]), "frame": int(frame), "camera": int(camera),
                    "width": int(source["width"]), "height": int(source["height"]),
                    "source_image": {"path": source["path"], "bytes": int(source["bytes"]), "sha256": source["sha256"]},
                    "candidate_mask": {"path": str(candidate_path), "bytes": candidate_path.stat().st_size, "sha256": sha256_file(candidate_path)},
                    "dynamic_mask": {"path": str(dynamic_path), "bytes": dynamic_path.stat().st_size, "sha256": sha256_file(dynamic_path)},
                    "intrinsics": {"path": str(intrinsic_path), "bytes": intrinsic_path.stat().st_size, "sha256": sha256_file(intrinsic_path)},
                    "extrinsics": {"path": str(extrinsic_path), "bytes": extrinsic_path.stat().st_size, "sha256": sha256_file(extrinsic_path)},
                    "projections": projections,
                })
    if len(views) != 45 or projection_count == 0:
        raise ProtocolError("F0k projection denominator drift")
    input_manifest = {
        "schema_version": "worldsim_v51_f0k_quality_alignment_input_manifest_v1", "task_id": TASK_ID,
        "view_count": len(views), "projection_count": projection_count,
        "candidate_mask_pixels_read": False, "dynamic_mask_pixels_read": False, "image_pixels_read": False,
        "quality_gate_preregistration": config["quality_gate_preregistration"], "views": views,
    }
    _write_json(run_dir / "artifacts/quality_alignment_input_manifest.json", input_manifest)
    resources = {
        "wall_seconds": time.perf_counter() - started,
        "cgroup_memory_current_bytes": int(Path("/sys/fs/cgroup/memory.current").read_text().strip()),
        "disk_free_after_bytes": shutil.disk_usage(run_dir).free,
    }
    limits = config["resources"]
    checks = {"wall": resources["wall_seconds"] <= float(limits["maximum_wall_seconds"]), "cgroup": resources["cgroup_memory_current_bytes"] <= int(limits["maximum_cgroup_memory_bytes"]), "disk": resources["disk_free_after_bytes"] >= int(limits["minimum_disk_free_bytes_after"])}
    if not all(checks.values()):
        raise ProtocolError(f"F0k resource gate: {checks}")
    _write_json(run_dir / "artifacts/resources.json", resources)
    summary = {
        "schema_version": "worldsim_v51_f0k_summary_v1", "task_id": TASK_ID, "status": "done",
        "conclusion": config["decision"]["expected_conclusion"], "source_commit": identity["commit"], "source_tree": identity["tree"],
        "view_count": len(views), "projection_count": projection_count, "input_manifest": input_manifest,
        "resources": resources, "resource_checks": checks, "candidate_mask_pixels_read": False,
        "dynamic_mask_pixels_read": False, "image_pixels_read": False, "quality_metrics_read": False,
        "identity_training_authorized": False, "next_action": config["decision"]["next_action"], "m2_status": "pending", "m3_status": "pending",
    }
    _write_json(run_dir / "summary.json", summary)
    events.append({"event": "run_completed", "at_utc": _utc_now()})
    _write_jsonl(run_dir / "events.jsonl", events)
    _write_json(run_dir / "manifest.json", {"task_id": TASK_ID, "status": "done", "inventory": _inventory(run_dir)})
    _write_json(run_dir / "status.json", {"task_id": TASK_ID, "status": "done", "conclusion": summary["conclusion"], "source_commit": identity["commit"]})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/worldsim_v51/stage_f_f0k_quality_alignment_input_freeze_v1.yaml")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.run_dir.resolve()), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
