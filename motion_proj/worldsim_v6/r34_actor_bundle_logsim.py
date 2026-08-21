"""WorldSim V6 R34：对身份绑定 SceneIR actor bundle 做脱离源 run 的确定性重放。"""

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


TASK_ID = "WS-V6-R34-IDENTITY-BOUND-ACTOR-LOGSIM-01"
REQUIRED_ARRAYS = {
    "means_m",
    "scales_m",
    "quaternions_wxyz",
    "opacities",
    "features_dc",
    "features_rest",
    "source_indices",
}


class R34ExperimentError(RuntimeError):
    """R34 正式实验合同失败。"""


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


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _resolve_runs_uri(uri: str) -> Path:
    prefix = "runs://"
    relative = Path(uri[len(prefix) :]) if uri.startswith(prefix) else Path("..")
    if not uri.startswith(prefix) or ".." in relative.parts or relative.is_absolute():
        raise R34ExperimentError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / relative).resolve()


def _verify_file(path: Path, expected_sha256: str) -> None:
    if not path.is_file() or _sha256(path) != expected_sha256:
        raise R34ExperimentError(f"冻结输入漂移：{path}")


def _copy_detached_package(source: Path, destination: Path, manifest: dict[str, Any]) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for relative, record in sorted(manifest["files"].items()):
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise R34ExperimentError(f"非法 package 路径：{relative}")
        source_path = source / rel
        _verify_file(source_path, record["sha256"])
        destination_path = destination / rel
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        _verify_file(destination_path, record["sha256"])
    shutil.copy2(source / "PACKAGE_MANIFEST.json", destination / "PACKAGE_MANIFEST.json")


def _load_array_audit(package: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    chunk = bundle["chunk"]
    if set(chunk["arrays"]) != REQUIRED_ARRAYS:
        raise R34ExperimentError("Gaussian array 集合漂移")
    rows: dict[str, Any] = {}
    for name, reference in sorted(chunk["arrays"].items()):
        path = package / reference["path"]
        _verify_file(path, reference["sha256"])
        array = np.load(path, allow_pickle=False)
        numeric = np.issubdtype(array.dtype, np.number)
        rows[name] = {
            "path": reference["path"],
            "sha256": _sha256(path),
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "shape_exact": list(array.shape) == reference["shape"],
            "dtype_exact": array.dtype.str == reference["dtype"],
            "primitive_axis_exact": array.ndim >= 1
            and int(array.shape[0]) == int(chunk["primitive_count"]),
            "finite": bool(np.isfinite(array).all()) if numeric else True,
        }
    return rows


def _replay(package: Path) -> dict[str, Any]:
    bundle = json.loads((package / "ACTOR_BUNDLE.json").read_text(encoding="utf-8"))
    actor = bundle["actor"]
    chunk = bundle["chunk"]
    transforms = {
        (row["name"], int(row["timestamp_us"])): row
        for row in bundle["trajectory_transforms"]
    }
    visibility = {int(row["timestamp_us"]): bool(row["visible"]) for row in actor["visibility"]}
    states: list[dict[str, Any]] = []
    quaternion_errors: list[float] = []
    translations: list[list[float]] = []
    for trajectory_row in actor["trajectory"]:
        timestamp_us = int(trajectory_row["timestamp_us"])
        key = (trajectory_row["transform_name"], timestamp_us)
        if key not in transforms or timestamp_us not in visibility:
            raise R34ExperimentError(f"轨迹引用缺失：{key}")
        transform = transforms[key]
        translation = [float(value) for value in transform["translation_m"]]
        rotation = [float(value) for value in transform["rotation_wxyz"]]
        if not all(math.isfinite(value) for value in translation + rotation):
            raise R34ExperimentError(f"非有限 pose：{key}")
        quaternion_errors.append(abs(float(np.linalg.norm(np.asarray(rotation))) - 1.0))
        translations.append(translation)
        states.append(
            {
                "actor_id": actor["id"],
                "chunk_id": chunk["id"],
                "timestamp_us": timestamp_us,
                "transform_name": transform["name"],
                "src_frame": transform["src_frame"],
                "dst_frame": transform["dst_frame"],
                "translation_m": translation,
                "rotation_wxyz": rotation,
                "visible": visibility[timestamp_us],
            }
        )
    timestamps = [row["timestamp_us"] for row in states]
    path_length_m = float(
        sum(
            np.linalg.norm(np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64))
            for a, b in zip(translations, translations[1:])
        )
    )
    return {
        "schema_version": "worldsim_v6.r34_actor_replay.v1",
        "actor_id": actor["id"],
        "chunk_id": chunk["id"],
        "primitive_count": int(chunk["primitive_count"]),
        "array_audit": _load_array_audit(package, bundle),
        "states": states,
        "trajectory_rows": len(states),
        "timestamps_strictly_increasing": all(b > a for a, b in zip(timestamps, timestamps[1:])),
        "maximum_quaternion_norm_error": max(quaternion_errors, default=float("inf")),
        "logged_path_length_m": path_length_m,
    }


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R34ExperimentError("正式 R34 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R34ExperimentError("R34 task_id 漂移")
    sources = config["sources"]
    r33_run = _resolve_runs_uri(sources["r33_run"])
    source_package = r33_run / "package"
    frozen_files = {
        r33_run / "MANIFEST.json": sources["r33_manifest_sha256"],
        r33_run / "R33_GATE.json": sources["r33_gate_sha256"],
        r33_run / "SUMMARY.json": sources["r33_summary_sha256"],
        source_package / "PACKAGE_MANIFEST.json": sources["r33_package_manifest_sha256"],
        source_package / "ACTOR_BUNDLE.json": sources["r33_actor_bundle_sha256"],
        source_package / "ARRAY_AUDIT.json": sources["r33_array_audit_sha256"],
        source_package / "VALIDITY.json": sources["r33_validity_sha256"],
        source_package / "RUNTIME_DEPENDENCY_AUDIT.json": sources[
            "r33_runtime_dependency_audit_sha256"
        ],
    }
    for path, expected in frozen_files.items():
        _verify_file(path, expected)
    source_gate = json.loads((r33_run / "R33_GATE.json").read_text(encoding="utf-8"))
    if not source_gate["checks"]["passed"]:
        raise R34ExperimentError("R33 actor bundle authority 未通过")
    package_manifest = json.loads(
        (source_package / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8")
    )
    for relative, record in package_manifest["files"].items():
        _verify_file(source_package / relative, record["sha256"])
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R34ExperimentError("R34 磁盘资源不足")

    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / (
        f"{now.strftime('%Y%m%dT%H%M%SZ')}__actor-logsim-s{config['seed']}-r1"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        runtime_package = run_dir / "runtime_package"
        _copy_detached_package(source_package, runtime_package, package_manifest)
        replay_1 = _replay(runtime_package)
        replay_2 = _replay(runtime_package)
        _write_json(run_dir / "REPLAY_1.json", replay_1)
        _write_json(run_dir / "REPLAY_2.json", replay_2)
        replay_1_sha = _sha256(run_dir / "REPLAY_1.json")
        replay_2_sha = _sha256(run_dir / "REPLAY_2.json")
        bundle = json.loads((runtime_package / "ACTOR_BUNDLE.json").read_text(encoding="utf-8"))
        validity = json.loads((runtime_package / "VALIDITY.json").read_text(encoding="utf-8"))
        dependency = json.loads(
            (runtime_package / "RUNTIME_DEPENDENCY_AUDIT.json").read_text(encoding="utf-8")
        )
        replay_audit = {
            "schema_version": "worldsim_v6.r34_replay_audit.v1",
            "replay_count": 2,
            "replay_1_sha256": replay_1_sha,
            "replay_2_sha256": replay_2_sha,
            "replay_exact": replay_1_sha == replay_2_sha,
            "trajectory_rows": replay_1["trajectory_rows"],
            "logged_path_length_m": replay_1["logged_path_length_m"],
            "maximum_quaternion_norm_error": replay_1["maximum_quaternion_norm_error"],
            "runtime_input_root": "runtime_package",
        }
        _write_json(run_dir / "REPLAY_AUDIT.json", replay_audit)
        wall_seconds = time.monotonic() - started
        expected = config["expected"]
        array_rows = replay_1["array_audit"].values()
        checks = {
            "r33_authority_accepted": source_gate["checks"]["passed"],
            "detached_package_exact": all(
                _sha256(runtime_package / relative) == record["sha256"]
                for relative, record in package_manifest["files"].items()
            ),
            "identity_binding_exact": replay_1["actor_id"] == expected["actor_id"]
            and replay_1["chunk_id"] == expected["chunk_id"],
            "primitive_count_exact": replay_1["primitive_count"]
            == int(expected["primitive_count"]),
            "all_gaussian_arrays_exact": set(replay_1["array_audit"]) == REQUIRED_ARRAYS
            and all(
                row["shape_exact"]
                and row["dtype_exact"]
                and row["primitive_axis_exact"]
                and row["finite"]
                for row in array_rows
            ),
            "two_replays_exact": replay_audit["replay_exact"],
            "trajectory_denominator_exact": replay_1["trajectory_rows"]
            == int(expected["trajectory_rows"]),
            "timestamps_strictly_increasing": replay_1["timestamps_strictly_increasing"],
            "quaternions_normalized": replay_1["maximum_quaternion_norm_error"]
            <= float(expected["maximum_quaternion_norm_error"]),
            "actor_class_remains_unknown": validity["actor_class"] == "unknown",
            "generated_identity_route_remains_rejected": bundle["generated_identity_route"][
                "status"
            ]
            == "REJECTED",
            "unsupported_claims_abstain": all(
                str(validity[key]).startswith("ABSTAIN")
                for key in (
                    "semantic_accuracy",
                    "novel_trajectory_validity",
                    "dynamics_validity",
                    "planning_and_safety",
                )
            ),
            "runtime_has_no_online_dependency": not any(
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
        gate = {
            "schema_version": "worldsim_v6.r34_gate.v1",
            "checks": checks,
            "decision": "accept_identity_bound_logged_trajectory_replay"
            if checks["passed"]
            else "reject_or_repair_actor_bundle_logsim",
        }
        _write_json(run_dir / "R34_GATE.json", gate)
        _write_json(
            run_dir / "RESOURCE_AUDIT.json",
            {
                "schema_version": "worldsim_v6.r34_resource_audit.v1",
                "gpu_used": False,
                "wall_seconds": wall_seconds,
                "disk_free_gib_at_start": free_gib,
                "training_started": False,
                "confirmation_content_read": False,
            },
        )
        summary = {
            "schema_version": "worldsim_v6.r34_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_identity_bound_actor_logsim"
            if checks["passed"]
            else "rejected",
            "source_commit": source_commit,
            "actor_id": replay_1["actor_id"],
            "chunk_id": replay_1["chunk_id"],
            "primitive_count": replay_1["primitive_count"],
            "trajectory_rows": replay_1["trajectory_rows"],
            "replay_count": 2,
            "replay_exact": replay_audit["replay_exact"],
            "logged_path_length_m": replay_1["logged_path_length_m"],
            "full_logsim_coverage": False,
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = [
            "R34_GATE.json",
            "SUMMARY.json",
            "RESOURCE_AUDIT.json",
            "REPLAY_1.json",
            "REPLAY_2.json",
            "REPLAY_AUDIT.json",
        ]
        _write_json(
            run_dir / "MANIFEST.json",
            {
                "schema_version": "worldsim_v6.r34_manifest.v1",
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
        default=Path("configs/worldsim_v6/r34_actor_bundle_logsim_v1.yaml"),
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6")
    )
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0

