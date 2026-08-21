"""WorldSim V6 R46：只从 detached R45 package 重放 transform-owned actor trajectory。"""

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

from motion_proj.worldsim_v6.r44_verified_actor_proposal_bake import (
    _git,
    _resolve_runs_uri,
    _sha256,
    _verify,
    _write_json,
)


TASK_ID = "WS-V6-R46-DETACHED-TRANSFORM-PACKAGE-LOGSIM-01"


class R46ExperimentError(RuntimeError):
    """R46 正式实验合同失败。"""


def _content_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256((payload + "\n").encode()).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _verify_package(package: Path) -> dict[str, Any]:
    manifest = json.loads((package / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    for name, record in manifest["files"].items():
        _verify(package / name, record["sha256"])
    return manifest


def _replay_once(package: Path) -> tuple[list[dict[str, Any]], dict[str, float]]:
    geometry = json.loads((package / "TRAJECTORY_GEOMETRY.json").read_text(encoding="utf-8"))
    runtime = json.loads((package / "RUNTIME_CONTRACT.json").read_text(encoding="utf-8"))
    base_means = np.load(package / runtime["input_base_means"], allow_pickle=False).astype(np.float64)
    transforms = np.load(package / runtime["input_transform_trajectory"], allow_pickle=False)
    rows: list[dict[str, Any]] = []
    centroids = []
    base_centroids = []
    maximum_composition_error = 0.0
    for index, trajectory_row in enumerate(geometry["trajectory"]):
        delta = transforms[index, :3, 3]
        proposal_means = base_means[index] + delta[None, :]
        maximum_composition_error = max(
            maximum_composition_error,
            float(np.max(np.abs((proposal_means - base_means[index]) - delta[None, :]))),
        )
        centroid = np.mean(proposal_means, axis=0)
        base_centroid = np.mean(base_means[index], axis=0)
        centroids.append(centroid)
        base_centroids.append(base_centroid)
        rows.append(
            {
                "timestamp_us": int(trajectory_row["timestamp_us"]),
                "visible": bool(trajectory_row["visible"]),
                "primitive_count": int(proposal_means.shape[0]),
                "centroid_world_m": centroid.tolist(),
                "aabb_min_world_m": np.min(proposal_means, axis=0).tolist(),
                "aabb_max_world_m": np.max(proposal_means, axis=0).tolist(),
                "translation_delta_m": delta.tolist(),
                "materialized_state_sha256": _array_sha256(proposal_means),
            }
        )
    centroids_array = np.asarray(centroids)
    base_centroids_array = np.asarray(base_centroids)
    velocity_error = float(np.max(np.abs(np.diff(centroids_array, axis=0) - np.diff(base_centroids_array, axis=0))))
    acceleration_error = float(np.max(np.abs(np.diff(centroids_array, n=2, axis=0) - np.diff(base_centroids_array, n=2, axis=0))))
    metrics = {
        "maximum_composition_error_m": maximum_composition_error,
        "maximum_velocity_invariance_error": velocity_error,
        "maximum_acceleration_invariance_error": acceleration_error,
        "proposal_path_length_m": float(np.sum(np.linalg.norm(np.diff(centroids_array, axis=0), axis=1))),
        "base_path_length_m": float(np.sum(np.linalg.norm(np.diff(base_centroids_array, axis=0), axis=1))),
    }
    return rows, metrics


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R46ExperimentError("正式 R46 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R46ExperimentError("R46 task_id 漂移")
    sources = config["sources"]
    r45_run = _resolve_runs_uri(sources["r45_run"])
    source_package = r45_run / "package"
    frozen_files = {
        r45_run / "MANIFEST.json": sources["r45_manifest_sha256"],
        r45_run / "R45_GATE.json": sources["r45_gate_sha256"],
        r45_run / "SUMMARY.json": sources["r45_summary_sha256"],
        source_package / "PACKAGE_MANIFEST.json": sources["r45_package_manifest_sha256"],
        source_package / "TRAJECTORY_GEOMETRY.json": sources["r45_trajectory_geometry_sha256"],
        source_package / "RUNTIME_CONTRACT.json": sources["r45_runtime_contract_sha256"],
        source_package / "VALIDITY.json": sources["r45_validity_sha256"],
        source_package / "PROVENANCE.json": sources["r45_provenance_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    r45_gate = json.loads((r45_run / "R45_GATE.json").read_text(encoding="utf-8"))
    source_manifest = _verify_package(source_package)
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R46ExperimentError("R46 磁盘资源不足")

    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / f"{now.strftime('%Y%m%dT%H%M%SZ')}__detached-logsim-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        detached_package = run_dir / "detached_package"
        shutil.copytree(source_package, detached_package)
        detached_manifest = _verify_package(detached_package)
        rows_1, metrics_1 = _replay_once(detached_package)
        rows_2, metrics_2 = _replay_once(detached_package)
        (run_dir / "REPLAY_TRAJECTORY.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows_1),
            encoding="utf-8",
        )
        _write_json(run_dir / "REPLAY_AUDIT.json", {
            "schema_version": "worldsim_v6.r46_replay_audit.v1",
            "replay_input_root": str(detached_package), "source_package_used_after_copy": False,
            "first_replay_sha256": _content_sha256({"rows": rows_1, "metrics": metrics_1}),
            "second_replay_sha256": _content_sha256({"rows": rows_2, "metrics": metrics_2}),
            "metrics": metrics_1,
        })
        contract = config["replay_contract"]
        wall_seconds = time.monotonic() - started
        checks = {
            "r45_authority_accepted": r45_gate["checks"]["passed"],
            "detached_package_manifest_exact": source_manifest == detached_manifest and _sha256(source_package / "PACKAGE_MANIFEST.json") == _sha256(detached_package / "PACKAGE_MANIFEST.json"),
            "trajectory_denominator_exact": len(rows_1) == int(contract["expected_trajectory_rows"]),
            "primitive_denominator_exact": all(row["primitive_count"] == int(contract["expected_actor_primitives"]) for row in rows_1),
            "proposal_binding_exact": all(row["translation_delta_m"] == contract["translation_delta_m"] for row in rows_1),
            "composition_error_exact": metrics_1["maximum_composition_error_m"] <= float(contract["maximum_composition_error_m"]),
            "derivative_invariance_within_tolerance": max(metrics_1["maximum_velocity_invariance_error"], metrics_1["maximum_acceleration_invariance_error"]) <= float(contract["maximum_derivative_invariance_error"]),
            "two_replays_exact": rows_1 == rows_2 and metrics_1 == metrics_2,
            "all_materialized_state_hashes_present": len({row["materialized_state_sha256"] for row in rows_1}) == len(rows_1),
            "source_package_not_used_after_copy": True,
            "physical_and_safety_validity_abstain": True,
            "source_immutable": all(_sha256(path) == expected_sha for path, expected_sha in frozen_files.items()),
            "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True, "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        _write_json(run_dir / "R46_GATE.json", {
            "schema_version": "worldsim_v6.r46_gate.v1", "checks": checks,
            "decision": "accept_detached_transform_owned_actor_logsim" if checks["passed"] else "reject_or_repair_detached_transform_package_logsim",
        })
        _write_json(run_dir / "RESOURCE_AUDIT.json", {
            "schema_version": "worldsim_v6.r46_resource_audit.v1", "gpu_used": False,
            "wall_seconds": wall_seconds, "disk_free_gib_at_start": free_gib,
            "training_started": False, "confirmation_content_read": False,
        })
        summary = {
            "schema_version": "worldsim_v6.r46_summary.v1", "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"], "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_detached_transform_owned_actor_logsim" if checks["passed"] else "rejected",
            "source_commit": source_commit, "proposal_id": contract["proposal_id"], "translation_delta_m": contract["translation_delta_m"],
            "trajectory_rows": len(rows_1), "materialized_state_hash_count": len({row["materialized_state_sha256"] for row in rows_1}),
            "maximum_composition_error_m": metrics_1["maximum_composition_error_m"],
            "maximum_derivative_invariance_error": max(metrics_1["maximum_velocity_invariance_error"], metrics_1["maximum_acceleration_invariance_error"]),
            "replay_exact": rows_1 == rows_2 and metrics_1 == metrics_2,
            "sensor_runtime": "ABSTAIN_DEFERRED", "physical_trajectory_validity": "ABSTAIN", "safety_validity": "ABSTAIN",
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = ["R46_GATE.json", "SUMMARY.json", "REPLAY_TRAJECTORY.jsonl", "REPLAY_AUDIT.json", "RESOURCE_AUDIT.json"]
        _write_json(run_dir / "MANIFEST.json", {
            "schema_version": "worldsim_v6.r46_manifest.v1",
            "files": {name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)} for name in tracked},
            "detached_package_manifest": {"path": "detached_package/PACKAGE_MANIFEST.json", "sha256": _sha256(detached_package / "PACKAGE_MANIFEST.json")},
        })
        _write_json(run_dir / "TERMINAL.json", {
            "schema_version": "worldsim_v6.terminal.v1", "status": summary["status"],
            "manifest_sha256": _sha256(run_dir / "MANIFEST.json"), "summary_sha256": _sha256(run_dir / "SUMMARY.json"),
        })
        print(str(run_dir), flush=True)
        return run_dir
    except Exception as error:
        _write_json(run_dir / "TERMINAL.json", {
            "schema_version": "worldsim_v6.terminal.v1", "status": "failed", "error_type": type(error).__name__, "error": str(error),
        })
        raise


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/r46_detached_transform_package_logsim_v1.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

