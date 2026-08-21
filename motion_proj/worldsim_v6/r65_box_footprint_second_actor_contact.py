"""WorldSim V6 R65：用 native box bottom 与 footprint ring 验证 actor2 contact。"""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from motion_proj.worldsim_v6.r40_actor_lidar_contact_factor import _content_sha256, _git, _resolve_runs_uri, _sha256, _verify, _write_json, _write_jsonl
from motion_proj.worldsim_v6.r46_detached_transform_package_logsim import _verify_package


TASK_ID = "WS-V6-R65-BOX-FOOTPRINT-SECOND-ACTOR-CONTACT-01"


class R65ExperimentError(RuntimeError):
    """R65 正式实验合同失败。"""


def _annotation_pose(info: dict[str, Any], frame: int) -> tuple[np.ndarray, np.ndarray]:
    annotations = info["frame_annotations"]
    index = [int(value) for value in annotations["frame_idx"]].index(frame)
    return np.asarray(annotations["obj_to_world"][index], dtype=np.float64).reshape(4, 4), np.asarray(annotations["box_size"][index], dtype=np.float64)


def _bind_instance(infos: dict[str, Any], frame: int, align: np.ndarray, actor_center: np.ndarray) -> tuple[str, dict[str, Any], np.ndarray, np.ndarray, float, float]:
    rows = []
    for instance_id, info in infos.items():
        if frame not in [int(value) for value in info["frame_annotations"]["frame_idx"]]:
            continue
        pose_global, size = _annotation_pose(info, frame)
        pose_world = align @ pose_global
        rows.append((float(np.linalg.norm(pose_world[:3, 3] - actor_center)), instance_id, info, pose_world, size))
    rows.sort(key=lambda row: (row[0], int(row[1])))
    best, second = rows[0], rows[1]
    return best[1], best[2], best[3], best[4], best[0], second[0] - best[0]


def _evaluate(intervention: dict[str, Any], pose_world: np.ndarray, size: np.ndarray, points: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    delta = np.asarray(intervention["translation_delta_m"], dtype=np.float64)
    edited_pose = pose_world.copy()
    edited_pose[:3, 3] += delta
    local = (points - edited_pose[:3, 3]) @ edited_pose[:3, :3]
    dx = np.maximum(np.abs(local[:, 0]) - size[0] / 2.0, 0.0)
    dy = np.maximum(np.abs(local[:, 1]) - size[1] / 2.0, 0.0)
    outside = (np.abs(local[:, 0]) > size[0] / 2.0) | (np.abs(local[:, 1]) > size[1] / 2.0)
    ring = outside & (np.hypot(dx, dy) <= float(config["footprint_contact"]["outside_boundary_ring_width_m"]))
    local_y = points[ring, int(config["vertical_contract"]["world_vertical_axis_index"])]
    bottom_world = edited_pose @ np.asarray([0.0, 0.0, -size[2] / 2.0, 1.0])
    ground = float(np.quantile(local_y, float(config["footprint_contact"]["ground_height_quantile"]))) if local_y.size else None
    error = abs(float(bottom_world[1]) - ground) if ground is not None else None
    accepted = local_y.size >= int(config["footprint_contact"]["minimum_ring_lidar_candidates"]) and error is not None and error <= float(config["footprint_contact"]["maximum_contact_error_m"])
    return {"intervention_id": intervention["id"], "translation_delta_m": delta.tolist(), "box_bottom_world_y_m": float(bottom_world[1]), "ring_lidar_candidate_count": int(local_y.size), "ring_ground_proxy_world_y_m": ground, "contact_absolute_error_m": error, "q_box_footprint_lidar_contact": "ACCEPT" if accepted else "REJECT", "expected_decision": intervention["expected_contact_decision"], "semantic_road": "ABSTAIN", "physical_trajectory_validity": "ABSTAIN"}


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R65ExperimentError("正式 R65 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R65ExperimentError("R65 task_id 漂移")
    sources = config["sources"]
    r64_run = _resolve_runs_uri(sources["r64_run"])
    r56_run = _resolve_runs_uri(sources["r56_run"])
    package = r56_run / "package"
    geometry_path = package / "TRAJECTORY_GEOMETRY.json"
    scene_root = Path(sources["processed_scene_root"])
    infos_path = scene_root / "instances/instances_info.json"
    camera_path = scene_root / "extrinsics/000_0.txt"
    raw_union_path = r64_run / "RAW_STATIC_LIDAR_VOXEL_UNION.npy"
    frozen = {r64_run / "MANIFEST.json": sources["r64_manifest_sha256"], r64_run / "R64_GATE.json": sources["r64_gate_sha256"], r64_run / "SUMMARY.json": sources["r64_summary_sha256"], raw_union_path: sources["r64_raw_union_sha256"], r56_run / "R56_GATE.json": sources["r56_gate_sha256"], package / "PACKAGE_MANIFEST.json": sources["r56_package_manifest_sha256"], geometry_path: sources["r56_trajectory_geometry_sha256"], infos_path: sources["instances_info_sha256"], camera_path: sources["camera_front_start_sha256"]}
    for path, digest in frozen.items():
        _verify(path, digest)
    r64_gate = json.loads((r64_run / "R64_GATE.json").read_text(encoding="utf-8"))
    r56_gate = json.loads((r56_run / "R56_GATE.json").read_text(encoding="utf-8"))
    package_manifest = _verify_package(package)
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R65ExperimentError("R65 磁盘资源不足")
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__box-footprint-contact-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        cohort = config["cohort"]
        geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
        timestamps = [int(row["timestamp_us"]) for row in geometry["trajectory"]]
        index = timestamps.index(int(cohort["target_timestamp_us"]))
        actor_means = np.load(package / geometry["arrays"]["means_world_m"]["path"], allow_pickle=False)[index]
        validity = np.load(package / geometry["arrays"]["actor_frame_validity"]["path"], allow_pickle=False)
        infos = json.loads(infos_path.read_text(encoding="utf-8"))
        align = np.linalg.inv(np.loadtxt(camera_path).astype(np.float64))
        instance_id, info, pose, size, mapping_distance, mapping_margin = _bind_instance(infos, int(cohort["target_frame_index"]), align, actor_means.mean(axis=0))
        points = np.load(raw_union_path, allow_pickle=False).astype(np.float64)
        rows_1 = [_evaluate(row, pose, size, points, config) for row in config["interventions"]]
        rows_2 = [_evaluate(row, pose, size, points, config) for row in config["interventions"]]
        _write_jsonl(run_dir / "BOX_FOOTPRINT_CONTACT_DECISIONS.jsonl", rows_1)
        wall_seconds = time.monotonic() - started
        checks = {
            "r64_rejection_and_r56_acceptance_preserved": bool(not r64_gate["checks"]["passed"] and r56_gate["checks"]["passed"]),
            "actor_package_denominator_and_lifecycle_exact": geometry["frontend_model_index"] == int(cohort["actor_model_index"]) and actor_means.shape[0] == int(cohort["expected_primitive_count"]) and bool(validity[index]) == bool(cohort["expected_frame_validity"]),
            "nearest_native_instance_binding_exact": int(instance_id) == int(config["mapping"]["expected_instance_id"]) and info["class_name"] == config["mapping"]["expected_class_name"] and mapping_distance <= float(config["mapping"]["maximum_centroid_distance_m"]) and mapping_margin >= float(config["mapping"]["minimum_next_nearest_margin_m"]),
            "native_box_size_exact": float(np.max(np.abs(size - np.asarray(config["mapping"]["expected_size_lwh_m"])))) <= float(config["mapping"]["maximum_size_error_m"]),
            "vertical_polarity_and_bottom_owner_exact": float(pose[1, 2]) <= float(config["vertical_contract"]["maximum_local_up_world_y"]) and int(config["vertical_contract"]["object_bottom_local_z_sign"]) == -1,
            "ring_denominators_sufficient": all(row["ring_lidar_candidate_count"] >= int(config["footprint_contact"]["minimum_ring_lidar_candidates"]) for row in rows_1),
            "pre_registered_contact_decisions_exact": all(row["q_box_footprint_lidar_contact"] == row["expected_decision"] for row in rows_1),
            "repeat_exact": _content_sha256(rows_1) == _content_sha256(rows_2),
            "semantic_road_and_physical_validity_abstain": all(row["semantic_road"] == "ABSTAIN" and row["physical_trajectory_validity"] == "ABSTAIN" for row in rows_1),
            "source_immutable": all(_sha256(path) == digest for path, digest in frozen.items()) and package_manifest == _verify_package(package),
            "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]), "training_not_started": True, "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        status = "done" if checks["passed"] else "rejected"
        _write_json(run_dir / "R65_GATE.json", {"schema_version": "worldsim_v6.r65_gate.v1", "checks": checks, "decision": "accept_box_footprint_second_actor_contact" if checks["passed"] else "reject_or_repair_box_footprint_second_actor_contact"})
        _write_json(run_dir / "MAPPING_AUDIT.json", {"schema_version": "worldsim_v6.r65_mapping_audit.v1", "actor_model_index": int(cohort["actor_model_index"]), "processed_instance_id": int(instance_id), "class_name": info["class_name"], "centroid_distance_m": mapping_distance, "next_nearest_margin_m": mapping_margin, "size_lwh_m": size.tolist(), "object_local_up_world_vector": pose[:3, 2].tolist()})
        _write_json(run_dir / "RESOURCE_AUDIT.json", {"schema_version": "worldsim_v6.r65_resource_audit.v1", "gpu_used": False, "wall_seconds": wall_seconds, "disk_free_gib_at_start": free_gib, "training_started": False, "confirmation_content_read": False})
        _write_json(run_dir / "SUMMARY.json", {"schema_version": "worldsim_v6.r65_summary.v1", "task_id": TASK_ID, "hypothesis_id": config["hypothesis_id"], "status": status, "hypothesis_outcome": "accepted_development_box_footprint_second_actor_contact" if checks["passed"] else "rejected", "source_commit": source_commit, "processed_instance_id": int(instance_id), "class_name": info["class_name"], "logged_ring_candidates": rows_1[0]["ring_lidar_candidate_count"], "selected_ring_candidates": rows_1[1]["ring_lidar_candidate_count"], "logged_contact_error_m": rows_1[0]["contact_absolute_error_m"], "selected_contact_error_m": rows_1[1]["contact_absolute_error_m"], "logged_contact_decision": rows_1[0]["q_box_footprint_lidar_contact"], "selected_contact_decision": rows_1[1]["q_box_footprint_lidar_contact"], "claim_boundary": config["claim_boundary"]})
        tracked = ["R65_GATE.json", "SUMMARY.json", "BOX_FOOTPRINT_CONTACT_DECISIONS.jsonl", "MAPPING_AUDIT.json", "RESOURCE_AUDIT.json"]
        _write_json(run_dir / "MANIFEST.json", {"schema_version": "worldsim_v6.r65_manifest.v1", "files": {name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)} for name in tracked}})
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
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/r65_box_footprint_second_actor_contact_v1.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
