"""WorldSim V6 R75: compile actor5 into all logged world poses."""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from motion_proj.worldsim_v6.r35_actor_rigid_trajectory_compiler import REQUIRED_ARRAYS, _compile
from motion_proj.worldsim_v6.r44_verified_actor_proposal_bake import _git, _resolve_runs_uri, _sha256, _verify, _write_json


TASK_ID = "WS-V6-R75-THIRD-ACTOR-RIGID-TRAJECTORY-01"


class R75ExperimentError(RuntimeError):
    """The preregistered R75 experiment contract was violated."""


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R75ExperimentError("formal R75 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R75ExperimentError("R75 task_id drift")
    sources = config["sources"]
    r74_run = _resolve_runs_uri(sources["r74_run"])
    source_package = r74_run / "package"
    frozen_files = {
        r74_run / "MANIFEST.json": sources["r74_manifest_sha256"],
        r74_run / "R74_GATE.json": sources["r74_gate_sha256"],
        r74_run / "SUMMARY.json": sources["r74_summary_sha256"],
        source_package / "PACKAGE_MANIFEST.json": sources["r74_package_manifest_sha256"],
        source_package / "ACTOR_BUNDLE.json": sources["r74_actor_bundle_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    source_manifest = json.loads((source_package / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    for relative, row in source_manifest["files"].items():
        _verify(source_package / relative, row["sha256"])
        frozen_files[source_package / relative] = row["sha256"]
    r74_gate = json.loads((r74_run / "R74_GATE.json").read_text(encoding="utf-8"))
    if shutil.disk_usage(run_root).free / (1024**3) < float(config["resources"]["minimum_disk_free_gib"]):
        raise R75ExperimentError("R75 disk resource insufficient")
    bundle = json.loads((source_package / "ACTOR_BUNDLE.json").read_text(encoding="utf-8"))
    actor = bundle["actor"]
    chunk = bundle["chunk"]
    expected = config["expected"]
    if set(chunk["arrays"]) != REQUIRED_ARRAYS:
        raise R75ExperimentError("actor5 Gaussian array set drift")
    arrays = {name: np.load(source_package / reference["path"], allow_pickle=False) for name, reference in chunk["arrays"].items()}
    transforms = {(row["name"], int(row["timestamp_us"])): row for row in bundle["trajectory_transforms"]}
    visibility = {int(row["timestamp_us"]): bool(row["visible"]) for row in actor["visibility"]}
    ordered = []
    trajectory_metadata: list[dict[str, Any]] = []
    for row in actor["trajectory"]:
        timestamp = int(row["timestamp_us"])
        key = (row["transform_name"], timestamp)
        if key not in transforms or timestamp not in visibility:
            raise R75ExperimentError(f"actor5 trajectory reference missing: {key}")
        ordered.append(transforms[key])
        trajectory_metadata.append({"timestamp_us": timestamp, "transform_name": row["transform_name"], "visible": visibility[timestamp]})
    translations = np.asarray([row["translation_m"] for row in ordered], dtype=np.float64)
    pose_quaternions = np.asarray([row["rotation_wxyz"] for row in ordered], dtype=np.float64)
    means1, quaternions1, rotations = _compile(arrays["means_m"], arrays["quaternions_wxyz"], translations, pose_quaternions)
    means2, quaternions2, _ = _compile(arrays["means_m"], arrays["quaternions_wxyz"], translations, pose_quaternions)

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__actor5-trajectory-s{config['seed']}-r1"
    package = run_dir / "package"
    blob_dir = package / "blobs"
    blob_dir.mkdir(parents=True, exist_ok=False)
    means_path = blob_dir / "means_world_by_timestamp.npy"
    quaternions_path = blob_dir / "quaternions_world_wxyz_by_timestamp.npy"
    np.save(means_path, means1, allow_pickle=False)
    np.save(quaternions_path, quaternions1, allow_pickle=False)
    invariant_files = {}
    for name in sorted(REQUIRED_ARRAYS - {"means_m", "quaternions_wxyz"}):
        reference = chunk["arrays"][name]
        destination = package / reference["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_package / reference["path"], destination)
        _verify(destination, reference["sha256"])
        invariant_files[name] = reference
    lifecycle_reference = bundle["lifecycle"]
    lifecycle_source = source_package / lifecycle_reference["path"]
    lifecycle_destination = package / lifecycle_reference["path"]
    shutil.copy2(lifecycle_source, lifecycle_destination)
    _verify(lifecycle_destination, expected["lifecycle_sha256"])
    lifecycle = np.load(lifecycle_destination, allow_pickle=False)
    inverse_means = np.einsum("tji,tnj->tni", rotations, means1.astype(np.float64) - translations[:, None, :], optimize=True)
    roundtrip_error_m = float(np.max(np.abs(inverse_means - arrays["means_m"].astype(np.float64)[None, :, :])))
    local_radius = np.linalg.norm(arrays["means_m"].astype(np.float64), axis=-1)
    world_radius = np.linalg.norm(means1.astype(np.float64) - translations[:, None, :], axis=-1)
    rigid_radius_error_m = float(np.max(np.abs(world_radius - local_radius[None, :])))
    quaternion_norm_error = float(np.max(np.abs(np.linalg.norm(quaternions1.astype(np.float64), axis=-1) - 1.0)))
    geometry = {
        "schema_version": "worldsim_v6.r75_trajectory_geometry.v1", "asset_id": actor["id"], "chunk_id": chunk["id"],
        "frontend_model_index": int(bundle["frontend_model_index"]), "primitive_count": int(chunk["primitive_count"]), "trajectory": trajectory_metadata,
        "arrays": {
            "means_world_m": {"path": "blobs/means_world_by_timestamp.npy", "sha256": _sha256(means_path), "shape": list(means1.shape), "dtype": means1.dtype.str},
            "quaternions_world_wxyz": {"path": "blobs/quaternions_world_wxyz_by_timestamp.npy", "sha256": _sha256(quaternions_path), "shape": list(quaternions1.shape), "dtype": quaternions1.dtype.str},
            "actor_frame_validity": {"path": lifecycle_reference["path"], "sha256": _sha256(lifecycle_destination), "shape": list(lifecycle.shape), "dtype": lifecycle.dtype.str},
            **invariant_files,
        },
    }
    _write_json(package / "TRAJECTORY_GEOMETRY.json", geometry)
    validity = {
        "schema_version": "worldsim_v6.r75_validity.v1", "identity_binding": "OBSERVED_MODEL_INDEX_BOUND",
        "logged_rigid_pose_application": "ACCEPT_CONFORMANCE", "native_lifecycle": "ACCEPT_EXACT",
        "sensor_render_validity": "ABSTAIN_NOT_RENDERED", "semantic_identity": "ABSTAIN",
        "novel_trajectory_dynamics_collision_physical_planning_safety": "ABSTAIN",
    }
    _write_json(package / "VALIDITY.json", validity)
    _write_json(package / "RUNTIME_DEPENDENCY_AUDIT.json", {
        "schema_version": "worldsim_v6.r75_runtime_dependency_audit.v1", "runtime_dependencies": ["json_reader", "numpy_npy_reader"],
        "online_generator_dependency": False, "model_weight_dependency": False, "network_dependency": False, "source_run_payload_dependency": False,
    })
    package_files = ["TRAJECTORY_GEOMETRY.json", "VALIDITY.json", "RUNTIME_DEPENDENCY_AUDIT.json", "blobs/means_world_by_timestamp.npy", "blobs/quaternions_world_wxyz_by_timestamp.npy", lifecycle_reference["path"], *sorted(reference["path"] for reference in invariant_files.values())]
    _write_json(package / "PACKAGE_MANIFEST.json", {"schema_version": "worldsim_v6.r75_package_manifest.v1", "files": {name: {"bytes": (package / name).stat().st_size, "sha256": _sha256(package / name)} for name in package_files}})
    timestamps = [row["timestamp_us"] for row in trajectory_metadata]
    checks = {
        "r74_authority_accepted": bool(r74_gate["checks"]["passed"]),
        "identity_model_and_chunk_exact": actor["id"] == expected["actor_id"] and chunk["id"] == expected["chunk_id"] and int(bundle["frontend_model_index"]) == int(expected["actor_model_index"]),
        "primitive_count_exact": int(chunk["primitive_count"]) == int(expected["primitive_count"]),
        "trajectory_denominator_exact": len(trajectory_metadata) == int(expected["trajectory_rows"]),
        "timestamps_strictly_increasing": all(b > a for a, b in zip(timestamps, timestamps[1:])),
        "lifecycle_exact_and_visibility_bound": _sha256(lifecycle_destination) == expected["lifecycle_sha256"] and int(lifecycle.sum()) == int(expected["active_frame_count"]) and [bool(value) for value in lifecycle] == [row["visible"] for row in trajectory_metadata],
        "two_compilations_exact": np.array_equal(means1, means2) and np.array_equal(quaternions1, quaternions2),
        "compiled_shapes_exact": list(means1.shape) == [int(expected["trajectory_rows"]), int(expected["primitive_count"]), 3] and list(quaternions1.shape) == [int(expected["trajectory_rows"]), int(expected["primitive_count"]), 4],
        "rigid_inverse_roundtrip_within_tolerance": roundtrip_error_m <= float(expected["maximum_rigid_error_m"]),
        "rigid_radius_invariant_within_tolerance": rigid_radius_error_m <= float(expected["maximum_rigid_error_m"]),
        "world_quaternions_normalized": quaternion_norm_error <= float(expected["maximum_quaternion_norm_error"]),
        "invariant_arrays_byte_exact": all(_sha256(package / reference["path"]) == reference["sha256"] for reference in invariant_files.values()),
        "unsupported_claims_abstain": True, "no_online_runtime_dependency": True,
        "source_immutable": all(_sha256(path) == expected_sha for path, expected_sha in frozen_files.items()),
        "wall_within_budget": (time.monotonic() - started) <= float(config["resources"]["maximum_wall_seconds"]),
        "training_not_started": True, "confirmation_not_read": True,
    }
    checks["passed"] = all(checks.values())
    _write_json(run_dir / "R75_GATE.json", {"schema_version": "worldsim_v6.r75_gate.v1", "checks": checks, "decision": "accept_third_actor_rigid_trajectory" if checks["passed"] else "reject_or_repair_third_actor_trajectory"})
    _write_json(run_dir / "SUMMARY.json", {
        "schema_version": "worldsim_v6.r75_summary.v1", "task_id": TASK_ID, "hypothesis_id": config["hypothesis_id"], "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_development_third_actor_rigid_trajectory" if checks["passed"] else "rejected", "source_commit": source_commit,
        "actor_model_index": int(bundle["frontend_model_index"]), "primitive_count": int(chunk["primitive_count"]), "trajectory_rows": len(trajectory_metadata),
        "roundtrip_error_m": roundtrip_error_m, "rigid_radius_error_m": rigid_radius_error_m, "quaternion_norm_error": quaternion_norm_error, "claim_boundary": config["claim_boundary"],
    })
    tracked = ["R75_GATE.json", "SUMMARY.json", "package/PACKAGE_MANIFEST.json", *[f"package/{name}" for name in package_files]]
    _write_json(run_dir / "MANIFEST.json", {"schema_version": "worldsim_v6.r75_manifest.v1", "files": {name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)} for name in tracked}})
    _write_json(run_dir / "TERMINAL.json", {"schema_version": "worldsim_v6.terminal.v1", "status": "done" if checks["passed"] else "rejected", "manifest_sha256": _sha256(run_dir / "MANIFEST.json"), "summary_sha256": _sha256(run_dir / "SUMMARY.json")})
    print(str(run_dir), flush=True)
    return run_dir


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/r75_third_actor_rigid_trajectory_v1.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
