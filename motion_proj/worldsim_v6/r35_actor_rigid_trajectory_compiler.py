"""WorldSim V6 R35：把身份绑定 Gaussian actor 编译到全部 logged world poses。"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


TASK_ID = "WS-V6-R35-ACTOR-RIGID-TRAJECTORY-COMPILER-01"
REQUIRED_ARRAYS = {
    "means_m",
    "scales_m",
    "quaternions_wxyz",
    "opacities",
    "features_dc",
    "features_rest",
    "source_indices",
}


class R35ExperimentError(RuntimeError):
    """R35 正式实验合同失败。"""


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _resolve_runs_uri(uri: str) -> Path:
    prefix = "runs://"
    relative = Path(uri[len(prefix) :]) if uri.startswith(prefix) else Path("..")
    if not uri.startswith(prefix) or relative.is_absolute() or ".." in relative.parts:
        raise R35ExperimentError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / relative).resolve()


def _verify(path: Path, expected: str) -> None:
    if not path.is_file() or _sha256(path) != expected:
        raise R35ExperimentError(f"冻结输入漂移：{path}")


def _rotation_matrices(quaternions_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternions_wxyz, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = (q[:, index] for index in range(4))
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(-1, 3, 3)


def _quaternion_multiply(pose: np.ndarray, local: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = (pose[:, None, index] for index in range(4))
    bw, bx, by, bz = (local[None, :, index] for index in range(4))
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def _compile(
    means_local: np.ndarray,
    quaternions_local: np.ndarray,
    translations: np.ndarray,
    pose_quaternions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotations = _rotation_matrices(pose_quaternions)
    means_world = (
        np.einsum("tij,nj->tni", rotations, means_local.astype(np.float64), optimize=True)
        + translations[:, None, :]
    ).astype(np.float32)
    local_q = quaternions_local.astype(np.float64)
    local_q = local_q / np.linalg.norm(local_q, axis=-1, keepdims=True)
    pose_q = pose_quaternions.astype(np.float64)
    pose_q = pose_q / np.linalg.norm(pose_q, axis=-1, keepdims=True)
    quaternions_world = _quaternion_multiply(pose_q, local_q).astype(np.float32)
    return means_world, quaternions_world, rotations


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R35ExperimentError("正式 R35 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R35ExperimentError("R35 task_id 漂移")
    sources = config["sources"]
    r34_run = _resolve_runs_uri(sources["r34_run"])
    source_package = r34_run / "runtime_package"
    frozen_files = {
        r34_run / "MANIFEST.json": sources["r34_manifest_sha256"],
        r34_run / "R34_GATE.json": sources["r34_gate_sha256"],
        r34_run / "SUMMARY.json": sources["r34_summary_sha256"],
        r34_run / "REPLAY_AUDIT.json": sources["r34_replay_audit_sha256"],
        source_package / "PACKAGE_MANIFEST.json": sources["package_manifest_sha256"],
        source_package / "ACTOR_BUNDLE.json": sources["actor_bundle_sha256"],
        source_package / "VALIDITY.json": sources["validity_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    r34_gate = json.loads((r34_run / "R34_GATE.json").read_text(encoding="utf-8"))
    if not r34_gate["checks"]["passed"]:
        raise R35ExperimentError("R34 identity-bound LogSim authority 未通过")
    source_manifest = json.loads(
        (source_package / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8")
    )
    for relative, record in source_manifest["files"].items():
        _verify(source_package / relative, record["sha256"])
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R35ExperimentError("R35 磁盘资源不足")

    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / (
        f"{now.strftime('%Y%m%dT%H%M%SZ')}__rigid-trajectory-s{config['seed']}-r1"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        package = run_dir / "package"
        blob_dir = package / "blobs"
        blob_dir.mkdir(parents=True)
        bundle = json.loads((source_package / "ACTOR_BUNDLE.json").read_text(encoding="utf-8"))
        validity_source = json.loads(
            (source_package / "VALIDITY.json").read_text(encoding="utf-8")
        )
        actor = bundle["actor"]
        chunk = bundle["chunk"]
        if set(chunk["arrays"]) != REQUIRED_ARRAYS:
            raise R35ExperimentError("actor array 集合漂移")
        arrays = {
            name: np.load(source_package / reference["path"], allow_pickle=False)
            for name, reference in chunk["arrays"].items()
        }
        transforms = {
            (row["name"], int(row["timestamp_us"])): row
            for row in bundle["trajectory_transforms"]
        }
        visibility = {int(row["timestamp_us"]): bool(row["visible"]) for row in actor["visibility"]}
        ordered_transforms: list[dict[str, Any]] = []
        trajectory_metadata: list[dict[str, Any]] = []
        for row in actor["trajectory"]:
            timestamp = int(row["timestamp_us"])
            key = (row["transform_name"], timestamp)
            if key not in transforms or timestamp not in visibility:
                raise R35ExperimentError(f"轨迹引用缺失：{key}")
            transform = transforms[key]
            ordered_transforms.append(transform)
            trajectory_metadata.append(
                {
                    "timestamp_us": timestamp,
                    "transform_name": transform["name"],
                    "visible": visibility[timestamp],
                }
            )
        translations = np.asarray(
            [row["translation_m"] for row in ordered_transforms], dtype=np.float64
        )
        pose_quaternions = np.asarray(
            [row["rotation_wxyz"] for row in ordered_transforms], dtype=np.float64
        )
        means_1, quaternions_1, rotations = _compile(
            arrays["means_m"], arrays["quaternions_wxyz"], translations, pose_quaternions
        )
        means_2, quaternions_2, _ = _compile(
            arrays["means_m"], arrays["quaternions_wxyz"], translations, pose_quaternions
        )
        compile_exact = np.array_equal(means_1, means_2) and np.array_equal(
            quaternions_1, quaternions_2
        )
        means_path = blob_dir / "means_world_by_timestamp.npy"
        quaternions_path = blob_dir / "quaternions_world_wxyz_by_timestamp.npy"
        np.save(means_path, means_1, allow_pickle=False)
        np.save(quaternions_path, quaternions_1, allow_pickle=False)
        invariant_files: dict[str, dict[str, Any]] = {}
        for name in sorted(REQUIRED_ARRAYS - {"means_m", "quaternions_wxyz"}):
            reference = chunk["arrays"][name]
            source_path = source_package / reference["path"]
            destination = package / reference["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)
            _verify(destination, reference["sha256"])
            invariant_files[name] = reference
        reloaded_means = np.load(means_path, allow_pickle=False)
        reloaded_quaternions = np.load(quaternions_path, allow_pickle=False)
        inverse_means = np.einsum(
            "tji,tnj->tni",
            rotations,
            reloaded_means.astype(np.float64) - translations[:, None, :],
            optimize=True,
        )
        roundtrip_error_m = float(
            np.max(np.abs(inverse_means - arrays["means_m"].astype(np.float64)[None, :, :]))
        )
        local_radius = np.linalg.norm(arrays["means_m"].astype(np.float64), axis=-1)
        world_radius = np.linalg.norm(
            reloaded_means.astype(np.float64) - translations[:, None, :], axis=-1
        )
        rigid_radius_error_m = float(np.max(np.abs(world_radius - local_radius[None, :])))
        quaternion_norm_error = float(
            np.max(np.abs(np.linalg.norm(reloaded_quaternions.astype(np.float64), axis=-1) - 1.0))
        )
        timestamps = [row["timestamp_us"] for row in trajectory_metadata]
        geometry = {
            "schema_version": "worldsim_v6.r35_trajectory_geometry.v1",
            "asset_id": actor["id"],
            "chunk_id": chunk["id"],
            "source_scope": "observed_reconstructed_support_logged_trajectory",
            "primitive_count": int(chunk["primitive_count"]),
            "trajectory": trajectory_metadata,
            "arrays": {
                "means_world_m": {
                    "path": "blobs/means_world_by_timestamp.npy",
                    "sha256": _sha256(means_path),
                    "shape": list(reloaded_means.shape),
                    "dtype": reloaded_means.dtype.str,
                },
                "quaternions_world_wxyz": {
                    "path": "blobs/quaternions_world_wxyz_by_timestamp.npy",
                    "sha256": _sha256(quaternions_path),
                    "shape": list(reloaded_quaternions.shape),
                    "dtype": reloaded_quaternions.dtype.str,
                },
                **invariant_files,
            },
        }
        _write_json(package / "TRAJECTORY_GEOMETRY.json", geometry)
        validity = {
            "schema_version": "worldsim_v6.r35_validity.v1",
            "identity_binding": validity_source["identity_binding"],
            "logged_rigid_pose_application": "ACCEPT_CONFORMANCE",
            "logged_visibility": "PRESENT_EXACT",
            "actor_class": "unknown",
            "semantic_accuracy": "ABSTAIN_UNKNOWN_CLASS",
            "novel_trajectory_validity": "ABSTAIN",
            "dynamics_validity": "ABSTAIN_RIGID_KINEMATICS_ONLY",
            "collision_validity": "ABSTAIN_NO_PHYSICS_GEOMETRY",
            "sensor_render_validity": "ABSTAIN_NOT_RENDERED",
            "planning_and_safety": "ABSTAIN",
        }
        _write_json(package / "VALIDITY.json", validity)
        dependency = {
            "schema_version": "worldsim_v6.r35_runtime_dependency_audit.v1",
            "runtime_dependencies": ["json_reader", "numpy_npy_reader"],
            "online_generator_dependency": False,
            "model_weight_dependency": False,
            "network_dependency": False,
            "source_run_dependency_for_payload_read": False,
        }
        _write_json(package / "RUNTIME_DEPENDENCY_AUDIT.json", dependency)
        package_files = [
            "TRAJECTORY_GEOMETRY.json",
            "VALIDITY.json",
            "RUNTIME_DEPENDENCY_AUDIT.json",
            "blobs/means_world_by_timestamp.npy",
            "blobs/quaternions_world_wxyz_by_timestamp.npy",
            *sorted(reference["path"] for reference in invariant_files.values()),
        ]
        _write_json(
            package / "PACKAGE_MANIFEST.json",
            {
                "schema_version": "worldsim_v6.r35_package_manifest.v1",
                "files": {
                    name: {
                        "bytes": (package / name).stat().st_size,
                        "sha256": _sha256(package / name),
                    }
                    for name in package_files
                },
            },
        )
        wall_seconds = time.monotonic() - started
        expected = config["expected"]
        checks = {
            "r34_authority_accepted": r34_gate["checks"]["passed"],
            "identity_and_chunk_exact": actor["id"] == expected["actor_id"]
            and chunk["id"] == expected["chunk_id"],
            "primitive_count_exact": int(chunk["primitive_count"])
            == int(expected["primitive_count"]),
            "trajectory_denominator_exact": len(trajectory_metadata)
            == int(expected["trajectory_rows"]),
            "timestamps_strictly_increasing": all(b > a for a, b in zip(timestamps, timestamps[1:])),
            "two_compilations_exact": compile_exact,
            "compiled_arrays_reload_exact": np.array_equal(reloaded_means, means_1)
            and np.array_equal(reloaded_quaternions, quaternions_1),
            "compiled_shapes_exact": list(reloaded_means.shape)
            == [int(expected["trajectory_rows"]), int(expected["primitive_count"]), 3]
            and list(reloaded_quaternions.shape)
            == [int(expected["trajectory_rows"]), int(expected["primitive_count"]), 4],
            "rigid_inverse_roundtrip_within_tolerance": roundtrip_error_m
            <= float(expected["maximum_rigid_error_m"]),
            "rigid_radius_invariant_within_tolerance": rigid_radius_error_m
            <= float(expected["maximum_rigid_error_m"]),
            "world_quaternions_normalized": quaternion_norm_error
            <= float(expected["maximum_quaternion_norm_error"]),
            "invariant_arrays_byte_exact": all(
                _sha256(package / reference["path"]) == reference["sha256"]
                for reference in invariant_files.values()
            ),
            "unsupported_claims_abstain": all(
                str(validity[key]).startswith("ABSTAIN")
                for key in (
                    "semantic_accuracy",
                    "novel_trajectory_validity",
                    "dynamics_validity",
                    "collision_validity",
                    "sensor_render_validity",
                    "planning_and_safety",
                )
            ),
            "no_online_runtime_dependency": not any(
                dependency[key]
                for key in (
                    "online_generator_dependency",
                    "model_weight_dependency",
                    "network_dependency",
                    "source_run_dependency_for_payload_read",
                )
            ),
            "source_immutable": all(
                _sha256(path) == expected_sha for path, expected_sha in frozen_files.items()
            ),
            "wall_within_budget": wall_seconds
            <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True,
            "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        _write_json(
            run_dir / "R35_GATE.json",
            {
                "schema_version": "worldsim_v6.r35_gate.v1",
                "checks": checks,
                "decision": "accept_logged_actor_rigid_trajectory_compiler"
                if checks["passed"]
                else "reject_or_repair_rigid_trajectory_compiler",
            },
        )
        audit = {
            "schema_version": "worldsim_v6.r35_geometry_audit.v1",
            "trajectory_rows": len(trajectory_metadata),
            "primitive_count": int(chunk["primitive_count"]),
            "compiled_world_primitive_states": int(means_1.shape[0] * means_1.shape[1]),
            "means_compile_1_sha256": _array_sha256(means_1),
            "means_compile_2_sha256": _array_sha256(means_2),
            "quaternion_compile_1_sha256": _array_sha256(quaternions_1),
            "quaternion_compile_2_sha256": _array_sha256(quaternions_2),
            "rigid_inverse_roundtrip_max_error_m": roundtrip_error_m,
            "rigid_radius_max_error_m": rigid_radius_error_m,
            "world_quaternion_norm_max_error": quaternion_norm_error,
        }
        _write_json(run_dir / "GEOMETRY_AUDIT.json", audit)
        _write_json(
            run_dir / "RESOURCE_AUDIT.json",
            {
                "schema_version": "worldsim_v6.r35_resource_audit.v1",
                "gpu_used": False,
                "wall_seconds": wall_seconds,
                "disk_free_gib_at_start": free_gib,
                "training_started": False,
                "confirmation_content_read": False,
            },
        )
        summary = {
            "schema_version": "worldsim_v6.r35_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_logged_actor_rigid_trajectory"
            if checks["passed"]
            else "rejected",
            "source_commit": source_commit,
            "actor_id": actor["id"],
            "primitive_count": int(chunk["primitive_count"]),
            "trajectory_rows": len(trajectory_metadata),
            "compiled_world_primitive_states": audit["compiled_world_primitive_states"],
            "two_compilations_exact": compile_exact,
            "full_worldsim_coverage": False,
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = [
            "R35_GATE.json",
            "SUMMARY.json",
            "GEOMETRY_AUDIT.json",
            "RESOURCE_AUDIT.json",
            "package/PACKAGE_MANIFEST.json",
            *[f"package/{name}" for name in package_files],
        ]
        _write_json(
            run_dir / "MANIFEST.json",
            {
                "schema_version": "worldsim_v6.r35_manifest.v1",
                "files": {
                    name: {
                        "bytes": (run_dir / name).stat().st_size,
                        "sha256": _sha256(run_dir / name),
                    }
                    for name in tracked
                },
            },
        )
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": summary["status"],
                "manifest_sha256": _sha256(run_dir / "MANIFEST.json"),
                "summary_sha256": _sha256(run_dir / "SUMMARY.json"),
            },
        )
        print(str(run_dir), flush=True)
        return run_dir
    except Exception as error:
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/worldsim_v6/r35_actor_rigid_trajectory_compiler_v1.yaml"),
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6")
    )
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0

