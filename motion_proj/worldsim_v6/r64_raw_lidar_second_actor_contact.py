"""WorldSim V6 R64：用 box-filtered 360° raw LiDAR 验证 actor2 contact。"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial import cKDTree

from motion_proj.worldsim_v6.r40_actor_lidar_contact_factor import (
    _content_sha256, _evaluate, _git, _resolve_runs_uri, _sha256,
    _verify, _write_json, _write_jsonl,
)
from motion_proj.worldsim_v6.r46_detached_transform_package_logsim import _verify_package
from motion_proj.worldsim_v6.r63_temporal_second_actor_lidar_contact import _voxel_first


TASK_ID = "WS-V6-R64-RAW-LIDAR-SECOND-ACTOR-CONTACT-01"


class R64ExperimentError(RuntimeError):
    """R64 正式实验合同失败。"""


def _source_index(root: Path, frames: list[int]) -> list[dict[str, Any]]:
    relative = ["extrinsics/000_0.txt", "instances/instances_info.json", "instances/frame_instances.json"]
    for frame in frames:
        relative.extend([f"lidar/{frame:03d}.bin", f"lidar_pose/{frame:03d}.txt"])
    rows = []
    for name in relative:
        path = root / name
        rows.append({"path": name, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    return rows


def _annotation_at(info: dict[str, Any], frame: int) -> tuple[np.ndarray, np.ndarray]:
    annotations = info["frame_annotations"]
    try:
        index = [int(value) for value in annotations["frame_idx"]].index(frame)
    except ValueError as error:
        raise R64ExperimentError(f"instance annotation 缺少 frame {frame}") from error
    pose = np.asarray(annotations["obj_to_world"][index], dtype=np.float64).reshape(4, 4)
    size = np.asarray(annotations["box_size"][index], dtype=np.float64)
    return pose, size


def _inside_box(points_world: np.ndarray, object_world: np.ndarray, size_lwh: np.ndarray, margin: float) -> np.ndarray:
    local = (points_world - object_world[:3, 3]) @ object_world[:3, :3]
    return np.all(np.abs(local) <= size_lwh[None, :] / 2.0 + margin, axis=1)


def _load_box_filtered_raw_lidar(root: Path, frames: list[int], config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    camera_front_start = np.loadtxt(root / "extrinsics/000_0.txt").astype(np.float64)
    align = np.linalg.inv(camera_front_start)
    instances = json.loads((root / "instances/instances_info.json").read_text(encoding="utf-8"))
    frame_instances = json.loads((root / "instances/frame_instances.json").read_text(encoding="utf-8"))
    width = int(config["raw_lidar"]["point_record_float32_width"])
    margin = float(config["raw_lidar"]["dynamic_box_margin_m"])
    kept_sets = []
    per_frame = []
    for frame in frames:
        raw = np.fromfile(root / f"lidar/{frame:03d}.bin", dtype=np.float32).reshape(-1, width)
        points_sensor = raw[:, :3].astype(np.float64)
        lidar_global = np.loadtxt(root / f"lidar_pose/{frame:03d}.txt").astype(np.float64)
        lidar_world = align @ lidar_global
        points_world = points_sensor @ lidar_world[:3, :3].T + lidar_world[:3, 3]
        dynamic = np.zeros(points_world.shape[0], dtype=bool)
        ids = [int(value) for value in frame_instances[str(frame)]]
        for instance_id in ids:
            object_global, box_size = _annotation_at(instances[str(instance_id)], frame)
            dynamic |= _inside_box(points_world, align @ object_global, box_size, margin)
        kept = points_world[~dynamic]
        kept_sets.append(kept)
        per_frame.append({"frame_index": frame, "raw_point_count": int(points_world.shape[0]), "dynamic_removed_count": int(dynamic.sum()), "static_point_count": int(kept.shape[0]), "annotated_box_count": len(ids)})
    points = np.concatenate(kept_sets, axis=0)
    return points, {"per_frame": per_frame, "raw_point_count": sum(row["raw_point_count"] for row in per_frame), "dynamic_removed_count": sum(row["dynamic_removed_count"] for row in per_frame), "static_point_count": int(points.shape[0])}


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R64ExperimentError("正式 R64 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R64ExperimentError("R64 task_id 漂移")

    sources = config["sources"]
    r63_run = _resolve_runs_uri(sources["r63_run"])
    r56_run = _resolve_runs_uri(sources["r56_run"])
    r40_run = _resolve_runs_uri(sources["r40_run"])
    package = r56_run / "package"
    geometry_path = package / "TRAJECTORY_GEOMETRY.json"
    streetgs_config = Path(sources["streetgs_config"])
    scene_root = Path(sources["processed_scene_root"])
    frozen_files = {
        r63_run / "MANIFEST.json": sources["r63_manifest_sha256"],
        r63_run / "R63_GATE.json": sources["r63_gate_sha256"],
        r63_run / "SUMMARY.json": sources["r63_summary_sha256"],
        r56_run / "MANIFEST.json": sources["r56_manifest_sha256"],
        r56_run / "R56_GATE.json": sources["r56_gate_sha256"],
        package / "PACKAGE_MANIFEST.json": sources["r56_package_manifest_sha256"],
        geometry_path: sources["r56_trajectory_geometry_sha256"],
        r40_run / "MANIFEST.json": sources["r40_manifest_sha256"],
        r40_run / "R40_GATE.json": sources["r40_gate_sha256"],
        streetgs_config: sources["streetgs_config_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    cohort = config["cohort"]
    frames = [int(value) for value in cohort["support_frame_indices"]]
    source_index = _source_index(scene_root, frames)
    if len(source_index) != int(sources["raw_source_file_count"]) or _content_sha256(source_index) != sources["raw_source_index_sha256"]:
        raise R64ExperimentError("raw LiDAR source index 漂移")
    r63_gate = json.loads((r63_run / "R63_GATE.json").read_text(encoding="utf-8"))
    r56_gate = json.loads((r56_run / "R56_GATE.json").read_text(encoding="utf-8"))
    r40_gate = json.loads((r40_run / "R40_GATE.json").read_text(encoding="utf-8"))
    r63_rejection_preserved = bool(not r63_gate["checks"]["passed"] and not r63_gate["checks"]["all_local_lidar_denominators_sufficient"])
    package_manifest = _verify_package(package)
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R64ExperimentError("R64 磁盘资源不足")

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__raw-lidar-contact-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        raw_static_points, raw_audit = _load_box_filtered_raw_lidar(scene_root, frames, config)
        fused_points = _voxel_first(raw_static_points, float(config["fusion"]["voxel_size_m"]))
        np.save(run_dir / "RAW_STATIC_LIDAR_VOXEL_UNION.npy", fused_points.astype(np.float32), allow_pickle=False)
        _write_json(run_dir / "RAW_SOURCE_INDEX.json", {"schema_version": "worldsim_v6.r64_raw_source_index.v1", "files": source_index, "aggregate_sha256": _content_sha256(source_index)})
        _write_json(run_dir / "RAW_LIDAR_AUDIT.json", {"schema_version": "worldsim_v6.r64_raw_lidar_audit.v1", **raw_audit, "voxel_point_count": int(fused_points.shape[0]), "voxel_size_m": float(config["fusion"]["voxel_size_m"]), "dynamic_box_margin_m": float(config["raw_lidar"]["dynamic_box_margin_m"])})

        geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
        timestamps = [int(row["timestamp_us"]) for row in geometry["trajectory"]]
        trajectory_index = timestamps.index(int(cohort["target_timestamp_us"]))
        actor_means = np.load(package / geometry["arrays"]["means_world_m"]["path"], allow_pickle=False)[trajectory_index]
        validity = np.load(package / geometry["arrays"]["actor_frame_validity"]["path"], allow_pickle=False)
        horizontal = [int(value) for value in config["coordinate_contract"]["horizontal_axis_indices"]]
        tree = cKDTree(fused_points[:, horizontal])
        rows_1 = [_evaluate(row, actor_means, fused_points, tree, config) for row in config["interventions"]]
        rows_2 = [_evaluate(row, actor_means, fused_points, tree, config) for row in config["interventions"]]
        _write_jsonl(run_dir / "RAW_LIDAR_CONTACT_DECISIONS.jsonl", rows_1)
        wall_seconds = time.monotonic() - started
        checks: dict[str, Any] = {
            "r63_projected_support_rejection_preserved": r63_rejection_preserved,
            "r56_and_r40_authorities_accepted": bool(r56_gate["checks"]["passed"] and r40_gate["checks"]["passed"]),
            "raw_source_index_exact": len(source_index) == int(sources["raw_source_file_count"]) and _content_sha256(source_index) == sources["raw_source_index_sha256"],
            "symmetric_window_exact": len(frames) == int(cohort["expected_support_frame_count"]) and frames == list(range(88, 109)) and frames[10] == int(cohort["target_frame_index"]),
            "coordinate_box_margin_and_voxel_contract_exact": config["coordinate_contract"]["world_up_axis"] == "y" and horizontal == [0, 2] and float(config["raw_lidar"]["dynamic_box_margin_m"]) == 0.10 and float(config["fusion"]["voxel_size_m"]) == 0.05,
            "target_actor_and_lifecycle_exact": geometry["frontend_model_index"] == int(cohort["expected_actor_model_index"]) and bool(validity[trajectory_index]) == bool(cohort["expected_frame_validity"]),
            "raw_and_static_denominators_sufficient": raw_audit["raw_point_count"] >= int(config["raw_lidar"]["minimum_raw_point_count"]) and raw_audit["static_point_count"] >= int(config["raw_lidar"]["minimum_static_point_count"]),
            "dynamic_points_removed": raw_audit["dynamic_removed_count"] > 0,
            "voxel_union_deduplicates_without_emptying": 0 < fused_points.shape[0] <= raw_static_points.shape[0],
            "actor_primitive_denominator_exact": actor_means.shape[0] == int(cohort["expected_actor_primitives"]),
            "all_local_lidar_denominators_sufficient": all(row["local_lidar_candidate_count"] >= int(config["contact_query"]["minimum_lidar_candidates"]) for row in rows_1),
            "pre_registered_baseline_and_selected_contact_accept": all(row["q_lidar_contact"] == row["expected_decision"] for row in rows_1),
            "decision_repeat_exact": _content_sha256(rows_1) == _content_sha256(rows_2),
            "semantic_road_and_physical_validity_abstain": all(row["q_semantic_road"] == "ABSTAIN" and row["physical_trajectory_validity"] == "ABSTAIN" for row in rows_1),
            "source_immutable": all(_sha256(path) == expected_sha for path, expected_sha in frozen_files.items()) and _content_sha256(_source_index(scene_root, frames)) == sources["raw_source_index_sha256"] and package_manifest == _verify_package(package),
            "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True, "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        status = "done" if checks["passed"] else "rejected"
        _write_json(run_dir / "R64_GATE.json", {"schema_version": "worldsim_v6.r64_gate.v1", "checks": checks, "decision": "accept_raw_lidar_second_actor_contact" if checks["passed"] else "reject_or_repair_raw_lidar_second_actor_contact"})
        _write_json(run_dir / "RESOURCE_AUDIT.json", {"schema_version": "worldsim_v6.r64_resource_audit.v1", "gpu_used": False, "wall_seconds": wall_seconds, "disk_free_gib_at_start": free_gib, "training_started": False, "confirmation_content_read": False})
        _write_json(run_dir / "SUMMARY.json", {
            "schema_version": "worldsim_v6.r64_summary.v1", "task_id": TASK_ID, "hypothesis_id": config["hypothesis_id"], "status": status,
            "hypothesis_outcome": "accepted_development_raw_lidar_second_actor_contact" if checks["passed"] else "rejected", "source_commit": source_commit,
            "support_frame_count": len(frames), "raw_lidar_point_count": raw_audit["raw_point_count"], "dynamic_removed_point_count": raw_audit["dynamic_removed_count"], "static_lidar_point_count": raw_audit["static_point_count"], "voxel_static_lidar_point_count": int(fused_points.shape[0]),
            "logged_local_candidate_count": rows_1[0]["local_lidar_candidate_count"], "selected_local_candidate_count": rows_1[1]["local_lidar_candidate_count"],
            "logged_contact_error_m": rows_1[0]["contact_absolute_error_m"], "selected_contact_error_m": rows_1[1]["contact_absolute_error_m"],
            "logged_contact_decision": rows_1[0]["q_lidar_contact"], "selected_contact_decision": rows_1[1]["q_lidar_contact"],
            "semantic_road": "ABSTAIN", "physical_trajectory_validity": "ABSTAIN", "claim_boundary": config["claim_boundary"],
        })
        tracked = ["R64_GATE.json", "SUMMARY.json", "RAW_LIDAR_CONTACT_DECISIONS.jsonl", "RAW_STATIC_LIDAR_VOXEL_UNION.npy", "RAW_SOURCE_INDEX.json", "RAW_LIDAR_AUDIT.json", "RESOURCE_AUDIT.json"]
        _write_json(run_dir / "MANIFEST.json", {"schema_version": "worldsim_v6.r64_manifest.v1", "files": {name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)} for name in tracked}})
        _write_json(run_dir / "TERMINAL.json", {"schema_version": "worldsim_v6.terminal.v1", "status": status, "manifest_sha256": _sha256(run_dir / "MANIFEST.json"), "summary_sha256": _sha256(run_dir / "SUMMARY.json")})
        print(str(run_dir), flush=True)
        return run_dir
    except Exception as error:
        _write_json(run_dir / "TERMINAL.json", {"schema_version": "worldsim_v6.terminal.v1", "status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/r64_raw_lidar_second_actor_contact_v1.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
