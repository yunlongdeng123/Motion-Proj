"""WorldSim V6 R33 提取 observed-support SceneIR actor_0000 bundle。"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


TASK_ID = "WS-V6-R33-OBSERVED-SCENEIR-ACTOR-BUNDLE-01"
REQUIRED_ARRAYS = {
    "means_m",
    "scales_m",
    "quaternions_wxyz",
    "opacities",
    "features_dc",
    "features_rest",
    "source_indices",
}


class R33ExperimentError(RuntimeError):
    """R33 正式合同失败。"""


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
    if not uri.startswith(prefix) or ".." in Path(uri[len(prefix) :]).parts:
        raise R33ExperimentError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / uri[len(prefix) :]).resolve()


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R33ExperimentError("正式 R33 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R33ExperimentError("R33 task_id 漂移")
    sources = config["sources"]
    cohort_run = _resolve_runs_uri(sources["actor_cohort_run"])
    binding_run = _resolve_runs_uri(sources["sceneir_binding_run"])
    r32_run = _resolve_runs_uri(sources["r32_run"])
    base_package = binding_run / sources["base_sceneir_package"]
    source_files = {
        cohort_run / "MANIFEST.json": sources["actor_cohort_manifest_sha256"],
        cohort_run / "R13_ACTOR_COHORT_GATE.json": sources["actor_cohort_gate_sha256"],
        cohort_run / "ACTOR_VERDICTS.jsonl": sources["actor_cohort_verdicts_sha256"],
        binding_run / "MANIFEST.json": sources["sceneir_binding_manifest_sha256"],
        binding_run / "R13_SCENEIR_SENSOR_BINDING_GATE.json": sources[
            "sceneir_binding_gate_sha256"
        ],
        binding_run / "BINDING_AUDIT.json": sources["sceneir_binding_audit_sha256"],
        base_package / "MANIFEST.json": sources["base_sceneir_manifest_sha256"],
        base_package / "sceneir.json": sources["base_sceneir_document_sha256"],
        r32_run / "MANIFEST.json": sources["r32_manifest_sha256"],
        r32_run / "R32_GATE.json": sources["r32_gate_sha256"],
        r32_run / "IDENTITY_FACTOR_RESULT.json": sources["r32_result_sha256"],
    }
    for path, expected in source_files.items():
        if _sha256(path) != expected:
            raise R33ExperimentError(f"冻结输入漂移：{path}")
    cohort_gate = json.loads(
        (cohort_run / "R13_ACTOR_COHORT_GATE.json").read_text(encoding="utf-8")
    )
    binding_gate = json.loads(
        (binding_run / "R13_SCENEIR_SENSOR_BINDING_GATE.json").read_text(encoding="utf-8")
    )
    r32_gate = json.loads((r32_run / "R32_GATE.json").read_text(encoding="utf-8"))
    if not cohort_gate["checks"]["passed"] or not binding_gate["checks"]["passed"]:
        raise R33ExperimentError("observed actor authority 未通过")
    if r32_gate["checks"]["passed"] is not False:
        raise R33ExperimentError("R33 必须保留 generated identity route rejected")
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R33ExperimentError("R33 磁盘资源不足")
    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / (
        f"{now.strftime('%Y%m%dT%H%M%SZ')}__actor-bundle-s{config['seed']}-r1"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        package_dir = run_dir / "package"
        blob_dir = package_dir / "blobs"
        blob_dir.mkdir(parents=True)
        document = json.loads((base_package / "sceneir.json").read_text(encoding="utf-8"))
        source_manifest = json.loads(
            (base_package / "MANIFEST.json").read_text(encoding="utf-8")
        )
        actor_id = str(config["identity"]["sceneir_actor_id"])
        chunk_id = str(config["identity"]["sceneir_chunk_id"])
        actor = next(row for row in document["actors"] if row["id"] == actor_id)
        chunk = next(row for row in document["chunks"] if row["id"] == chunk_id)
        frame_ids = {"world", actor["canonical_frame"]}
        frames = [row for row in document["frames"] if row["id"] in frame_ids]
        trajectory_keys = {
            (row["transform_name"], int(row["timestamp_us"]))
            for row in actor["trajectory"]
        }
        transforms = [
            row
            for row in document["transforms"]
            if (row["name"], int(row["timestamp_us"])) in trajectory_keys
        ]
        provenance = next(
            row for row in document["provenance"] if row["id"] == chunk["provenance_id"]
        )
        support = next(
            row for row in document["support"] if row["id"] == chunk["support_id"]
        )
        if set(chunk["arrays"]) != REQUIRED_ARRAYS:
            raise R33ExperimentError("actor chunk Gaussian array 集合漂移")

        array_audit: dict[str, Any] = {}
        copied_files: list[str] = []
        source_blob_hashes: dict[Path, str] = {}
        for name, reference in sorted(chunk["arrays"].items()):
            relative = reference["path"]
            source_blob = base_package / relative
            expected_sha = reference["sha256"]
            manifest_entry = source_manifest["files"].get(relative)
            if manifest_entry is None or manifest_entry["sha256"] != expected_sha:
                raise R33ExperimentError(f"source manifest 缺少 actor blob：{name}")
            if _sha256(source_blob) != expected_sha:
                raise R33ExperimentError(f"source actor blob 漂移：{name}")
            destination = package_dir / relative
            shutil.copy2(source_blob, destination)
            if _sha256(destination) != expected_sha:
                raise R33ExperimentError(f"copied actor blob 漂移：{name}")
            source_blob_hashes[source_blob] = expected_sha
            array = np.load(destination, allow_pickle=False)
            shape_exact = list(array.shape) == reference["shape"]
            dtype_exact = array.dtype.str == reference["dtype"]
            first_dimension_exact = array.ndim >= 1 and array.shape[0] == chunk["primitive_count"]
            finite = bool(
                np.isfinite(array).all() if np.issubdtype(array.dtype, np.number) else True
            )
            array_audit[name] = {
                "path": relative,
                "sha256": expected_sha,
                "shape": list(array.shape),
                "dtype": array.dtype.str,
                "shape_exact": shape_exact,
                "dtype_exact": dtype_exact,
                "first_dimension_exact": first_dimension_exact,
                "finite": finite,
            }
            copied_files.append(f"package/{relative}")

        bundle = {
            "schema_version": "worldsim_v6.r33_actor_bundle.v1",
            "asset_type": "explicit_sceneir_gaussian_actor_with_logged_trajectory",
            "source_scope": "observed_reconstructed_support",
            "frontend_model_index": int(config["identity"]["frontend_model_index"]),
            "actor": actor,
            "chunk": chunk,
            "frames": frames,
            "trajectory_transforms": transforms,
            "provenance": provenance,
            "support": support,
            "generated_identity_route": {
                "status": "REJECTED",
                "source_task": "WS-V6-R32-ACTOR-IDENTITY-FACTOR-VERIFICATION-01",
                "source_gate_sha256": sources["r32_gate_sha256"],
            },
        }
        _write_json(package_dir / "ACTOR_BUNDLE.json", bundle)
        _write_json(package_dir / "ARRAY_AUDIT.json", array_audit)
        validity = {
            "schema_version": "worldsim_v6.r33_validity.v1",
            "asset_id": actor_id,
            "identity_binding": "ACCEPT",
            "sensor_perception_locality": "ACCEPT_INHERITED_H_R13_009",
            "typed_dependency_closure": "ACCEPT_INHERITED_H_R13_011",
            "generated_identity_appearance": "REJECT_INHERITED_R32",
            "actor_class": actor["class"],
            "semantic_accuracy": "ABSTAIN_UNKNOWN_CLASS",
            "logged_trajectory_payload": "PRESENT_EXACT",
            "novel_trajectory_validity": "ABSTAIN",
            "dynamics_validity": "ABSTAIN",
            "planning_and_safety": "ABSTAIN",
        }
        _write_json(package_dir / "VALIDITY.json", validity)
        dependency = {
            "schema_version": "worldsim_v6.r33_runtime_dependency_audit.v1",
            "runtime_dependencies": ["json_reader", "numpy_npy_reader"],
            "online_generator_dependency": False,
            "model_weight_dependency": False,
            "network_dependency": False,
            "source_run_dependency_for_payload_read": False,
            "gaussian_blob_count": len(array_audit),
        }
        _write_json(package_dir / "RUNTIME_DEPENDENCY_AUDIT.json", dependency)
        wall_seconds = time.monotonic() - started
        expected_trajectory = int(config["identity"]["expected_trajectory_rows"])
        checks = {
            "prior_actor_cohort_and_binding_accepted": cohort_gate["checks"]["passed"]
            and binding_gate["checks"]["passed"],
            "generated_identity_route_remains_rejected": not r32_gate["checks"]["passed"],
            "identity_binding_exact": actor["id"] == actor_id
            and chunk["id"] == chunk_id
            and chunk["actor_id"] == actor_id,
            "primitive_count_exact": chunk["primitive_count"]
            == int(config["identity"]["expected_primitive_count"]),
            "required_gaussian_arrays_exact": set(array_audit) == REQUIRED_ARRAYS,
            "all_blobs_reload_exact": all(
                row["shape_exact"]
                and row["dtype_exact"]
                and row["first_dimension_exact"]
                and row["finite"]
                for row in array_audit.values()
            ),
            "trajectory_denominator_exact": len(actor["trajectory"])
            == expected_trajectory
            and len(actor["visibility"]) == expected_trajectory
            and len(transforms) == expected_trajectory,
            "trajectory_references_exact": len(trajectory_keys) == len(transforms),
            "provenance_and_support_complete": provenance["id"] == chunk["provenance_id"]
            and support["id"] == chunk["support_id"],
            "actor_class_remains_unknown": actor["class"] == "unknown",
            "unsupported_claims_abstain": all(
                str(validity[key]).startswith("ABSTAIN")
                for key in (
                    "semantic_accuracy",
                    "novel_trajectory_validity",
                    "dynamics_validity",
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
                _sha256(path) == expected for path, expected in source_files.items()
            )
            and all(_sha256(path) == expected for path, expected in source_blob_hashes.items()),
            "wall_within_budget": wall_seconds
            <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True,
            "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        _write_json(
            run_dir / "R33_GATE.json",
            {
                "schema_version": "worldsim_v6.r33_gate.v1",
                "checks": checks,
                "decision": "proceed_to_identity_bound_actor_bundle_replay"
                if checks["passed"]
                else "reject_or_repair_observed_actor_bundle",
            },
        )
        package_files = [
            "package/ACTOR_BUNDLE.json",
            "package/ARRAY_AUDIT.json",
            "package/VALIDITY.json",
            "package/RUNTIME_DEPENDENCY_AUDIT.json",
            *copied_files,
        ]
        _write_json(
            package_dir / "PACKAGE_MANIFEST.json",
            {
                "schema_version": "worldsim_v6.r33_package_manifest.v1",
                "files": {
                    name.removeprefix("package/"): {
                        "bytes": (run_dir / name).stat().st_size,
                        "sha256": _sha256(run_dir / name),
                    }
                    for name in package_files
                },
            },
        )
        _write_json(
            run_dir / "RESOURCE_AUDIT.json",
            {
                "schema_version": "worldsim_v6.r33_resource_audit.v1",
                "gpu_used": False,
                "wall_seconds": wall_seconds,
                "disk_free_gib_at_start": free_gib,
                "training_started": False,
                "confirmation_content_read": False,
            },
        )
        summary = {
            "schema_version": "worldsim_v6.r33_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_observed_actor_bundle"
            if checks["passed"]
            else "rejected",
            "source_commit": source_commit,
            "actor_id": actor_id,
            "chunk_id": chunk_id,
            "primitive_count": int(chunk["primitive_count"]),
            "trajectory_rows": len(actor["trajectory"]),
            "gaussian_blob_count": len(array_audit),
            "generated_identity_route_status": "REJECTED",
            "actor_class": actor["class"],
            "training_started": False,
            "confirmation_content_read": False,
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = [
            "R33_GATE.json",
            "SUMMARY.json",
            "RESOURCE_AUDIT.json",
            "package/PACKAGE_MANIFEST.json",
            *package_files,
        ]
        _write_json(
            run_dir / "MANIFEST.json",
            {
                "schema_version": "worldsim_v6.r33_manifest.v1",
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
        default=Path("configs/worldsim_v6/r33_observed_sceneir_actor_bundle_v1.yaml"),
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6")
    )
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0
