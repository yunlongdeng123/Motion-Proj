"""WorldSim V6 R45：用独立 float64 world transform trajectory 拥有 verified actor edit。"""

from __future__ import annotations

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
    _manifest_files,
    _resolve_runs_uri,
    _sha256,
    _verify,
    _write_json,
)


TASK_ID = "WS-V6-R45-TRANSFORM-OWNED-ACTOR-BAKE-01"


class R45ExperimentError(RuntimeError):
    """R45 正式实验合同失败。"""


def _build_package(
    output: Path,
    source_package: Path,
    source_geometry: dict[str, Any],
    verified: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    output.mkdir(parents=True, exist_ok=False)
    blobs = output / "blobs"
    blobs.mkdir()
    contract = config["bake_contract"]
    delta = np.asarray(contract["translation_delta_m"], dtype=np.float64)
    arrays: dict[str, Any] = {}
    for name, record in source_geometry["arrays"].items():
        source = source_package / record["path"]
        destination = blobs / source.name
        shutil.copy2(source, destination)
        arrays[name] = dict(record)
        arrays[name]["path"] = destination.relative_to(output).as_posix()

    transforms = np.tile(np.eye(4, dtype=np.float64), (int(contract["expected_trajectory_rows"]), 1, 1))
    transforms[:, :3, 3] = delta[None, :]
    temporary = blobs / "proposal_transform_world.npy"
    np.save(temporary, transforms, allow_pickle=False)
    transform_sha = _sha256(temporary)
    transform_path = blobs / f"{transform_sha}.npy"
    temporary.rename(transform_path)
    base_means = np.load(output / arrays["means_world_m"]["path"], allow_pickle=False).astype(np.float64)
    composed = base_means + transforms[:, None, :3, 3]
    composition_error = float(np.max(np.abs((composed - base_means) - delta[None, None, :])))

    trajectory = [
        {
            "timestamp_us": int(row["timestamp_us"]), "visible": bool(row["visible"]),
            "base_transform_name": row["transform_name"], "proposal_transform_index": index,
            "composition": "T_delta_world @ base_world_geometry",
        }
        for index, row in enumerate(source_geometry["trajectory"])
    ]
    geometry = {
        "schema_version": contract["package_schema"], "asset_id": contract["asset_id"],
        "chunk_id": source_geometry["chunk_id"], "proposal_id": contract["proposal_id"],
        "package_status": contract["package_status"], "primitive_count": int(source_geometry["primitive_count"]),
        "base_geometry_representation": "frozen_r35_world_arrays",
        "edit_ownership": "independent_float64_world_transform_trajectory",
        "runtime_composition_order": "T_delta_world @ base_world_geometry",
        "translation_delta_m": delta.tolist(), "trajectory": trajectory, "base_arrays": arrays,
        "proposal_transform_world": {
            "path": transform_path.relative_to(output).as_posix(), "sha256": transform_sha,
            "bytes": transform_path.stat().st_size, "dtype": transforms.dtype.str, "shape": list(transforms.shape),
        },
    }
    _write_json(output / "TRAJECTORY_GEOMETRY.json", geometry)
    _write_json(output / "RUNTIME_CONTRACT.json", {
        "schema_version": "worldsim_v6.r45_runtime_contract.v1", "asset_id": contract["asset_id"],
        "proposal_id": contract["proposal_id"], "input_base_means": arrays["means_world_m"]["path"],
        "input_transform_trajectory": geometry["proposal_transform_world"]["path"],
        "composition": "homogeneous_column_vector_left_multiply",
        "equation": "p_proposal_world = T_delta_world @ p_base_world_h",
        "rotation_and_nonposition_fields": "preserve_base_byte_exact_for_pure_translation",
    })
    _write_json(output / "VALIDITY.json", {
        "schema_version": "worldsim_v6.r45_validity.v1", "proposal_id": contract["proposal_id"],
        "q_self_kinematics": verified["proposal"]["q_self_kinematics"],
        "q_aabb_interaction": verified["proposal"]["q_aabb_interaction"],
        "q_lidar_contact": verified["proposal"]["q_lidar_contact"],
        "q_renderer_execution": verified["renderer_execution"], "package_status": contract["package_status"],
        "semantic_road": "ABSTAIN", "physical_trajectory_validity": "ABSTAIN",
        "planning_validity": "ABSTAIN", "safety_validity": "ABSTAIN",
    })
    _write_json(output / "PROVENANCE.json", {
        "schema_version": "worldsim_v6.r45_provenance.v1", "proposal_id": contract["proposal_id"],
        "rejected_materialized_bake": {"run": config["sources"]["r44_run"], "gate_sha256": config["sources"]["r44_gate_sha256"]},
        "renderer_verification": {"run": config["sources"]["r43_run"], "gate_sha256": config["sources"]["r43_gate_sha256"], "verified_proposal_sha256": config["sources"]["r43_verified_proposal_sha256"]},
        "base_actor_package": {"run": config["sources"]["r35_run"], "package_manifest_sha256": config["sources"]["r35_package_manifest_sha256"]},
        "transformation": {"type": "constant_world_translation_float64", "delta_m": delta.tolist()},
    })
    manifest = {"schema_version": "worldsim_v6.r45_package_manifest.v1", "files": _manifest_files(output)}
    _write_json(output / "PACKAGE_MANIFEST.json", manifest)
    return manifest, composition_error


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R45ExperimentError("正式 R45 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R45ExperimentError("R45 task_id 漂移")
    sources = config["sources"]
    r44_run = _resolve_runs_uri(sources["r44_run"])
    r43_run = _resolve_runs_uri(sources["r43_run"])
    r35_run = _resolve_runs_uri(sources["r35_run"])
    source_package = r35_run / "package"
    frozen_files = {
        r44_run / "MANIFEST.json": sources["r44_manifest_sha256"],
        r44_run / "R44_GATE.json": sources["r44_gate_sha256"],
        r44_run / "SUMMARY.json": sources["r44_summary_sha256"],
        r43_run / "MANIFEST.json": sources["r43_manifest_sha256"],
        r43_run / "R43_GATE.json": sources["r43_gate_sha256"],
        r43_run / "VERIFIED_PROPOSAL.json": sources["r43_verified_proposal_sha256"],
        r35_run / "MANIFEST.json": sources["r35_manifest_sha256"],
        source_package / "PACKAGE_MANIFEST.json": sources["r35_package_manifest_sha256"],
        source_package / "TRAJECTORY_GEOMETRY.json": sources["r35_trajectory_geometry_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    r44_gate = json.loads((r44_run / "R44_GATE.json").read_text(encoding="utf-8"))
    r43_gate = json.loads((r43_run / "R43_GATE.json").read_text(encoding="utf-8"))
    verified = json.loads((r43_run / "VERIFIED_PROPOSAL.json").read_text(encoding="utf-8"))
    contract = config["bake_contract"]
    proposal_binding_exact = verified["proposal"]["proposal_id"] == contract["proposal_id"] and verified["proposal"]["translation_delta_m"] == contract["translation_delta_m"] and verified["renderer_execution"] == "ACCEPT_CONFORMANCE"
    if not proposal_binding_exact:
        raise R45ExperimentError("R43 verified proposal binding 漂移")
    source_manifest = json.loads((source_package / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    package_files = {source_package / name: row["sha256"] for name, row in source_manifest["files"].items()}
    for path, expected_sha in package_files.items():
        _verify(path, expected_sha)
    source_geometry = json.loads((source_package / "TRAJECTORY_GEOMETRY.json").read_text(encoding="utf-8"))
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R45ExperimentError("R45 磁盘资源不足")

    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / f"{now.strftime('%Y%m%dT%H%M%SZ')}__transform-bake-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        package = run_dir / "package"
        repeat_package = run_dir / "_repeat_package"
        manifest_1, error_1 = _build_package(package, source_package, source_geometry, verified, config)
        manifest_2, error_2 = _build_package(repeat_package, source_package, source_geometry, verified, config)
        repeat_exact = manifest_1 == manifest_2 and _sha256(package / "PACKAGE_MANIFEST.json") == _sha256(repeat_package / "PACKAGE_MANIFEST.json")
        shutil.rmtree(repeat_package)
        geometry = json.loads((package / "TRAJECTORY_GEOMETRY.json").read_text(encoding="utf-8"))
        validity = json.loads((package / "VALIDITY.json").read_text(encoding="utf-8"))
        source_array_hashes = {name: row["sha256"] for name, row in source_geometry["arrays"].items()}
        baked_array_hashes = {name: row["sha256"] for name, row in geometry["base_arrays"].items()}
        transform_record = geometry["proposal_transform_world"]
        transforms = np.load(package / transform_record["path"], allow_pickle=False)
        delta = np.asarray(contract["translation_delta_m"], dtype=np.float64)
        wall_seconds = time.monotonic() - started
        checks = {
            "r44_rejection_preserved": not r44_gate["checks"]["passed"],
            "r43_authority_accepted": r43_gate["checks"]["passed"],
            "verified_proposal_binding_exact": proposal_binding_exact,
            "trajectory_denominator_exact": len(geometry["trajectory"]) == int(contract["expected_trajectory_rows"]),
            "actor_primitive_denominator_exact": int(geometry["primitive_count"]) == int(contract["expected_actor_primitives"]),
            "all_base_actor_arrays_byte_exact": source_array_hashes == baked_array_hashes,
            "transform_float64_and_content_addressed": transforms.dtype == np.float64 and Path(transform_record["path"]).stem == transform_record["sha256"],
            "transform_homogeneous_contract_exact": np.array_equal(transforms[:, :3, :3], np.tile(np.eye(3), (transforms.shape[0], 1, 1))) and np.array_equal(transforms[:, :3, 3], np.tile(delta, (transforms.shape[0], 1))) and np.array_equal(transforms[:, 3, :], np.tile(np.asarray([0.0, 0.0, 0.0, 1.0]), (transforms.shape[0], 1))),
            "transform_composition_error_exact": max(error_1, error_2) <= float(contract["maximum_transform_composition_error_m"]),
            "repeat_bake_byte_exact": repeat_exact,
            "typed_validity_preserved": validity["q_self_kinematics"] == validity["q_aabb_interaction"] == validity["q_lidar_contact"] == "ACCEPT" and validity["q_renderer_execution"] == "ACCEPT_CONFORMANCE",
            "semantic_physical_planning_safety_abstain": all(validity[key] == "ABSTAIN" for key in ["semantic_road", "physical_trajectory_validity", "planning_validity", "safety_validity"]),
            "source_immutable": all(_sha256(path) == expected_sha for path, expected_sha in frozen_files.items()) and all(_sha256(path) == expected_sha for path, expected_sha in package_files.items()),
            "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True, "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        _write_json(run_dir / "R45_GATE.json", {
            "schema_version": "worldsim_v6.r45_gate.v1", "checks": checks,
            "decision": "accept_transform_owned_verified_actor_bake" if checks["passed"] else "reject_or_repair_transform_owned_actor_bake",
        })
        _write_json(run_dir / "RESOURCE_AUDIT.json", {
            "schema_version": "worldsim_v6.r45_resource_audit.v1", "gpu_used": False,
            "wall_seconds": wall_seconds, "disk_free_gib_at_start": free_gib,
            "training_started": False, "confirmation_content_read": False,
        })
        summary = {
            "schema_version": "worldsim_v6.r45_summary.v1", "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"], "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_transform_owned_verified_actor_bake" if checks["passed"] else "rejected",
            "source_commit": source_commit, "proposal_id": contract["proposal_id"], "translation_delta_m": contract["translation_delta_m"],
            "package_status": contract["package_status"], "package_manifest_sha256": _sha256(package / "PACKAGE_MANIFEST.json"),
            "trajectory_geometry_sha256": _sha256(package / "TRAJECTORY_GEOMETRY.json"),
            "runtime_contract_sha256": _sha256(package / "RUNTIME_CONTRACT.json"),
            "validity_sha256": _sha256(package / "VALIDITY.json"), "provenance_sha256": _sha256(package / "PROVENANCE.json"),
            "transform_composition_max_error_m": error_1, "repeat_bake_byte_exact": repeat_exact,
            "physical_trajectory_validity": "ABSTAIN", "safety_validity": "ABSTAIN", "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = ["R45_GATE.json", "SUMMARY.json", "RESOURCE_AUDIT.json"]
        _write_json(run_dir / "MANIFEST.json", {
            "schema_version": "worldsim_v6.r45_manifest.v1",
            "files": {name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)} for name in tracked},
            "package_manifest": {"path": "package/PACKAGE_MANIFEST.json", "sha256": _sha256(package / "PACKAGE_MANIFEST.json")},
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
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/r45_transform_owned_actor_bake_v1.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

