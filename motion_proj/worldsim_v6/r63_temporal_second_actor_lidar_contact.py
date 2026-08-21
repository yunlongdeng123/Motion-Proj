"""WorldSim V6 R63：用对称时间窗 world-map 验证 actor2 LiDAR contact。"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial import cKDTree

from motion_proj.worldsim_v6.r40_actor_lidar_contact_factor import (
    _content_sha256, _evaluate, _git, _lift_static_lidar, _resolve_runs_uri,
    _sha256, _verify, _write_json, _write_jsonl,
)
from motion_proj.worldsim_v6.r46_detached_transform_package_logsim import _verify_package
from motion_proj.worldsim_v6.r62_selected_second_actor_lidar_contact import _array_bundle_sha256


TASK_ID = "WS-V6-R63-TEMPORAL-SECOND-ACTOR-LIDAR-CONTACT-01"


class R63ExperimentError(RuntimeError):
    """R63 正式实验合同失败。"""


def _load_bundle(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {name: np.asarray(values[name]) for name in values.files}


def _frame_support(bundle: dict[str, np.ndarray], frame: int, camera_ids: list[int]) -> dict[str, np.ndarray]:
    support = {}
    for camera_id in camera_ids:
        for suffix in ("lidar_depth", "dynamic_mask", "intrinsics", "camera_to_world"):
            support[f"cam{camera_id}_{suffix}"] = bundle[f"frame{frame:03d}_cam{camera_id}_{suffix}"]
    return support


def _voxel_first(points: np.ndarray, voxel_size: float) -> np.ndarray:
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(indices)]


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R63ExperimentError("正式 R63 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R63ExperimentError("R63 task_id 漂移")

    sources = config["sources"]
    r62_run = _resolve_runs_uri(sources["r62_run"])
    r56_run = _resolve_runs_uri(sources["r56_run"])
    r40_run = _resolve_runs_uri(sources["r40_run"])
    package = r56_run / "package"
    geometry_path = package / "TRAJECTORY_GEOMETRY.json"
    streetgs_config = Path(sources["streetgs_config"])
    upstream = Path(sources["streetgs_upstream_root"])
    frozen_files = {
        r62_run / "MANIFEST.json": sources["r62_manifest_sha256"],
        r62_run / "R62_GATE.json": sources["r62_gate_sha256"],
        r62_run / "SUMMARY.json": sources["r62_summary_sha256"],
        r62_run / "LIDAR_CONTACT_DECISIONS.jsonl": sources["r62_decisions_sha256"],
        r62_run / "SUPPORT_AUDIT.json": sources["r62_support_audit_sha256"],
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
    if _git(upstream, "rev-parse", "HEAD") != sources["streetgs_upstream_commit"]:
        raise R63ExperimentError("StreetGS upstream commit 漂移")
    r62_gate = json.loads((r62_run / "R62_GATE.json").read_text(encoding="utf-8"))
    r40_gate = json.loads((r40_run / "R40_GATE.json").read_text(encoding="utf-8"))
    r56_gate = json.loads((r56_run / "R56_GATE.json").read_text(encoding="utf-8"))
    r62_rejection_preserved = bool(not r62_gate["checks"]["passed"] and not r62_gate["checks"]["all_local_lidar_denominators_sufficient"])
    package_manifest = _verify_package(package)
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R63ExperimentError("R63 磁盘资源不足")

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__temporal-lidar-contact-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        cohort = config["cohort"]
        frames = [int(value) for value in cohort["support_frame_indices"]]
        camera_ids = [int(value) for value in cohort["camera_ids"]]
        support_path = run_dir / "TEMPORAL_LIDAR_SUPPORT.npz"
        worker_audit_path = run_dir / "SUPPORT_WORKER_AUDIT.json"
        command = [
            sources["drivestudio_python"], str(repo_root / "scripts/worldsim_v6/r63_temporal_lidar_support_worker.py"),
            "--repo-root", str(repo_root), "--streetgs-config", str(streetgs_config), "--upstream-root", str(upstream),
            "--frames", ",".join(str(value) for value in frames), "--camera-ids", ",".join(str(value) for value in camera_ids),
            "--camera-downscale", str(cohort["camera_downscale"]), "--output", str(support_path), "--audit", str(worker_audit_path),
        ]
        completed = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, timeout=float(config["resources"]["maximum_support_worker_seconds"]))
        (run_dir / "support_worker.log").write_text(completed.stdout + "\n--- STDERR ---\n" + completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise R63ExperimentError(f"temporal support worker 失败：rc={completed.returncode}")
        bundle = _load_bundle(support_path)
        worker_audit = json.loads(worker_audit_path.read_text(encoding="utf-8"))
        frame98_support = _frame_support(bundle, int(cohort["target_frame_index"]), camera_ids)
        frame98_exact = _array_bundle_sha256(frame98_support) == sources["r62_frame98_support_array_sha256"]
        point_sets = [_lift_static_lidar(_frame_support(bundle, frame, camera_ids), int(cohort["camera_count"])) for frame in frames]
        raw_points = np.concatenate(point_sets, axis=0).astype(np.float64)
        fused_points = _voxel_first(raw_points, float(config["fusion"]["voxel_size_m"]))

        geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
        timestamps = [int(row["timestamp_us"]) for row in geometry["trajectory"]]
        trajectory_index = timestamps.index(int(cohort["target_timestamp_us"]))
        means_by_time = np.load(package / geometry["arrays"]["means_world_m"]["path"], allow_pickle=False)
        actor_means = means_by_time[trajectory_index]
        validity = np.load(package / geometry["arrays"]["actor_frame_validity"]["path"], allow_pickle=False)
        horizontal = [int(value) for value in config["coordinate_contract"]["horizontal_axis_indices"]]
        tree = cKDTree(fused_points[:, horizontal])
        rows_1 = [_evaluate(row, actor_means, fused_points, tree, config) for row in config["interventions"]]
        rows_2 = [_evaluate(row, actor_means, fused_points, tree, config) for row in config["interventions"]]
        _write_jsonl(run_dir / "TEMPORAL_LIDAR_CONTACT_DECISIONS.jsonl", rows_1)
        wall_seconds = time.monotonic() - started
        checks: dict[str, Any] = {
            "r62_support_rejection_preserved": r62_rejection_preserved,
            "r56_and_r40_authorities_accepted": bool(r56_gate["checks"]["passed"] and r40_gate["checks"]["passed"]),
            "symmetric_window_exact": len(frames) == int(cohort["expected_support_frame_count"]) and frames == list(range(88, 109)) and frames[10] == int(cohort["target_frame_index"]),
            "frame98_support_matches_r62_exact": frame98_exact,
            "support_worker_repeat_exact": bool(worker_audit["repeat_exact"]),
            "streetgs_coordinate_and_voxel_contract_exact": config["coordinate_contract"]["world_up_axis"] == "y" and horizontal == [0, 2] and float(config["fusion"]["voxel_size_m"]) == 0.05,
            "target_frame_actor_and_lifecycle_exact": geometry["frontend_model_index"] == int(cohort["expected_actor_model_index"]) and bool(validity[trajectory_index]) == bool(cohort["expected_frame_validity"]),
            "raw_static_lidar_denominator_sufficient": raw_points.shape[0] >= int(cohort["minimum_raw_static_lidar_points"]),
            "voxel_union_deduplicates_without_emptying": 0 < fused_points.shape[0] <= raw_points.shape[0],
            "actor_primitive_denominator_exact": actor_means.shape[0] == int(cohort["expected_actor_primitives"]),
            "all_local_lidar_denominators_sufficient": all(row["local_lidar_candidate_count"] >= int(config["contact_query"]["minimum_lidar_candidates"]) for row in rows_1),
            "pre_registered_baseline_and_selected_contact_accept": all(row["q_lidar_contact"] == row["expected_decision"] for row in rows_1),
            "decision_repeat_exact": _content_sha256(rows_1) == _content_sha256(rows_2),
            "semantic_road_and_physical_validity_abstain": all(row["q_semantic_road"] == "ABSTAIN" and row["physical_trajectory_validity"] == "ABSTAIN" for row in rows_1),
            "source_immutable": all(_sha256(path) == expected_sha for path, expected_sha in frozen_files.items()) and package_manifest == _verify_package(package),
            "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True, "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        status = "done" if checks["passed"] else "rejected"
        _write_json(run_dir / "R63_GATE.json", {"schema_version": "worldsim_v6.r63_gate.v1", "checks": checks, "decision": "accept_temporal_second_actor_lidar_contact" if checks["passed"] else "reject_or_repair_temporal_second_actor_lidar_contact"})
        _write_json(run_dir / "RESOURCE_AUDIT.json", {"schema_version": "worldsim_v6.r63_resource_audit.v1", "gpu_used": False, "wall_seconds": wall_seconds, "disk_free_gib_at_start": free_gib, "training_started": False, "confirmation_content_read": False})
        _write_json(run_dir / "SUMMARY.json", {
            "schema_version": "worldsim_v6.r63_summary.v1", "task_id": TASK_ID, "hypothesis_id": config["hypothesis_id"], "status": status,
            "hypothesis_outcome": "accepted_development_temporal_second_actor_lidar_contact" if checks["passed"] else "rejected", "source_commit": source_commit,
            "support_frame_count": len(frames), "raw_static_lidar_point_count": int(raw_points.shape[0]), "voxel_static_lidar_point_count": int(fused_points.shape[0]),
            "logged_local_candidate_count": rows_1[0]["local_lidar_candidate_count"], "selected_local_candidate_count": rows_1[1]["local_lidar_candidate_count"],
            "logged_contact_error_m": rows_1[0]["contact_absolute_error_m"], "selected_contact_error_m": rows_1[1]["contact_absolute_error_m"],
            "logged_contact_decision": rows_1[0]["q_lidar_contact"], "selected_contact_decision": rows_1[1]["q_lidar_contact"],
            "semantic_road": "ABSTAIN", "physical_trajectory_validity": "ABSTAIN", "claim_boundary": config["claim_boundary"],
        })
        tracked = ["R63_GATE.json", "SUMMARY.json", "TEMPORAL_LIDAR_CONTACT_DECISIONS.jsonl", "TEMPORAL_LIDAR_SUPPORT.npz", "SUPPORT_WORKER_AUDIT.json", "RESOURCE_AUDIT.json", "support_worker.log"]
        _write_json(run_dir / "MANIFEST.json", {"schema_version": "worldsim_v6.r63_manifest.v1", "files": {name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)} for name in tracked}})
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
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/r63_temporal_second_actor_lidar_contact_v1.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
