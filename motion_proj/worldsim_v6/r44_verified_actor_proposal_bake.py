"""WorldSim V6 R44：把 factor/renderer verified actor proposal 确定性烘焙为类型化包。"""

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


TASK_ID = "WS-V6-R44-VERIFIED-ACTOR-PROPOSAL-BAKE-01"


class R44ExperimentError(RuntimeError):
    """R44 正式实验合同失败。"""


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _resolve_runs_uri(uri: str) -> Path:
    prefix = "runs://"
    relative = Path(uri[len(prefix) :]) if uri.startswith(prefix) else Path("..")
    if not uri.startswith(prefix) or relative.is_absolute() or ".." in relative.parts:
        raise R44ExperimentError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / relative).resolve()


def _verify(path: Path, expected: str) -> None:
    if not path.is_file() or _sha256(path) != expected:
        raise R44ExperimentError(f"冻结输入漂移：{path}")


def _manifest_files(package: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(package).as_posix(): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(package.rglob("*"))
        if path.is_file() and path.name != "PACKAGE_MANIFEST.json"
    }


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
    delta = np.asarray(contract["translation_delta_m"], dtype=np.float32)
    source_arrays = source_geometry["arrays"]
    means_source = source_package / source_arrays["means_world_m"]["path"]
    means = np.load(means_source, allow_pickle=False)
    shifted_means = (means.astype(np.float32) + delta[None, None, :]).astype(np.float32)
    temporary_means = blobs / "means_world_shifted.npy"
    np.save(temporary_means, shifted_means, allow_pickle=False)
    shifted_sha = _sha256(temporary_means)
    shifted_path = blobs / f"{shifted_sha}.npy"
    temporary_means.rename(shifted_path)
    translation_error = float(np.max(np.abs(shifted_means.astype(np.float64) - means.astype(np.float64) - delta.astype(np.float64))))

    arrays: dict[str, Any] = {}
    for name, record in source_arrays.items():
        if name == "means_world_m":
            arrays[name] = {
                "path": shifted_path.relative_to(output).as_posix(), "sha256": shifted_sha,
                "bytes": shifted_path.stat().st_size, "dtype": shifted_means.dtype.str, "shape": list(shifted_means.shape),
            }
            continue
        source = source_package / record["path"]
        destination = blobs / source.name
        shutil.copy2(source, destination)
        arrays[name] = dict(record)
        arrays[name]["path"] = destination.relative_to(output).as_posix()

    trajectory = []
    for row in source_geometry["trajectory"]:
        baked = dict(row)
        baked["transform_name"] = f"T_world_{contract['asset_id']}_{contract['proposal_id']}"
        baked["translation_delta_m"] = delta.tolist()
        trajectory.append(baked)
    geometry = {
        "schema_version": contract["package_schema"], "asset_id": contract["asset_id"],
        "chunk_id": source_geometry["chunk_id"], "proposal_id": contract["proposal_id"],
        "package_status": contract["package_status"], "source_scope": "verified_development_actor_translation",
        "primitive_count": int(source_geometry["primitive_count"]), "translation_delta_m": delta.tolist(),
        "trajectory": trajectory, "arrays": arrays,
    }
    _write_json(output / "TRAJECTORY_GEOMETRY.json", geometry)
    _write_json(output / "VALIDITY.json", {
        "schema_version": "worldsim_v6.r44_validity.v1", "proposal_id": contract["proposal_id"],
        "q_self_kinematics": verified["proposal"]["q_self_kinematics"],
        "q_aabb_interaction": verified["proposal"]["q_aabb_interaction"],
        "q_lidar_contact": verified["proposal"]["q_lidar_contact"],
        "q_renderer_execution": verified["renderer_execution"],
        "package_status": contract["package_status"], "semantic_road": "ABSTAIN",
        "physical_trajectory_validity": "ABSTAIN", "planning_validity": "ABSTAIN", "safety_validity": "ABSTAIN",
    })
    _write_json(output / "PROVENANCE.json", {
        "schema_version": "worldsim_v6.r44_provenance.v1", "proposal_id": contract["proposal_id"],
        "r43_run": config["sources"]["r43_run"], "r43_manifest_sha256": config["sources"]["r43_manifest_sha256"],
        "r43_gate_sha256": config["sources"]["r43_gate_sha256"],
        "r43_verified_proposal_sha256": config["sources"]["r43_verified_proposal_sha256"],
        "r35_run": config["sources"]["r35_run"], "r35_package_manifest_sha256": config["sources"]["r35_package_manifest_sha256"],
        "transformation": {"type": "constant_world_translation", "delta_m": delta.tolist()},
    })
    manifest = {"schema_version": "worldsim_v6.r44_package_manifest.v1", "files": _manifest_files(output)}
    _write_json(output / "PACKAGE_MANIFEST.json", manifest)
    return manifest, translation_error


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R44ExperimentError("正式 R44 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R44ExperimentError("R44 task_id 漂移")
    sources = config["sources"]
    r43_run = _resolve_runs_uri(sources["r43_run"])
    r35_run = _resolve_runs_uri(sources["r35_run"])
    source_package = r35_run / "package"
    frozen_files = {
        r43_run / "MANIFEST.json": sources["r43_manifest_sha256"],
        r43_run / "R43_GATE.json": sources["r43_gate_sha256"],
        r43_run / "SUMMARY.json": sources["r43_summary_sha256"],
        r43_run / "VERIFIED_PROPOSAL.json": sources["r43_verified_proposal_sha256"],
        r35_run / "MANIFEST.json": sources["r35_manifest_sha256"],
        source_package / "PACKAGE_MANIFEST.json": sources["r35_package_manifest_sha256"],
        source_package / "TRAJECTORY_GEOMETRY.json": sources["r35_trajectory_geometry_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    r43_gate = json.loads((r43_run / "R43_GATE.json").read_text(encoding="utf-8"))
    verified = json.loads((r43_run / "VERIFIED_PROPOSAL.json").read_text(encoding="utf-8"))
    contract = config["bake_contract"]
    proposal_binding_exact = verified["proposal"]["proposal_id"] == contract["proposal_id"] and verified["proposal"]["translation_delta_m"] == contract["translation_delta_m"] and verified["renderer_execution"] == "ACCEPT_CONFORMANCE"
    if not proposal_binding_exact:
        raise R44ExperimentError("R43 verified proposal binding 漂移")
    source_manifest = json.loads((source_package / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    package_files = {source_package / name: row["sha256"] for name, row in source_manifest["files"].items()}
    for path, expected_sha in package_files.items():
        _verify(path, expected_sha)
    source_geometry = json.loads((source_package / "TRAJECTORY_GEOMETRY.json").read_text(encoding="utf-8"))
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R44ExperimentError("R44 磁盘资源不足")

    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / f"{now.strftime('%Y%m%dT%H%M%SZ')}__verified-bake-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        package = run_dir / "package"
        repeat_package = run_dir / "_repeat_package"
        manifest_1, translation_error_1 = _build_package(package, source_package, source_geometry, verified, config)
        manifest_2, translation_error_2 = _build_package(repeat_package, source_package, source_geometry, verified, config)
        repeat_exact = manifest_1 == manifest_2 and _sha256(package / "PACKAGE_MANIFEST.json") == _sha256(repeat_package / "PACKAGE_MANIFEST.json")
        shutil.rmtree(repeat_package)
        geometry = json.loads((package / "TRAJECTORY_GEOMETRY.json").read_text(encoding="utf-8"))
        validity = json.loads((package / "VALIDITY.json").read_text(encoding="utf-8"))
        source_nonmeans = {name: row["sha256"] for name, row in source_geometry["arrays"].items() if name != "means_world_m"}
        baked_nonmeans = {name: row["sha256"] for name, row in geometry["arrays"].items() if name != "means_world_m"}
        wall_seconds = time.monotonic() - started
        checks = {
            "r43_authority_accepted": r43_gate["checks"]["passed"],
            "verified_proposal_binding_exact": proposal_binding_exact,
            "trajectory_denominator_exact": len(geometry["trajectory"]) == int(contract["expected_trajectory_rows"]),
            "actor_primitive_denominator_exact": int(geometry["primitive_count"]) == int(contract["expected_actor_primitives"]),
            "translation_bake_error_within_tolerance": max(translation_error_1, translation_error_2) <= float(contract["maximum_translation_bake_error_m"]),
            "nontranslation_actor_fields_byte_exact": source_nonmeans == baked_nonmeans,
            "shifted_means_content_addressed": Path(geometry["arrays"]["means_world_m"]["path"]).stem == geometry["arrays"]["means_world_m"]["sha256"],
            "package_manifest_complete": set(manifest_1["files"]) == {path.relative_to(package).as_posix() for path in package.rglob("*") if path.is_file() and path.name != "PACKAGE_MANIFEST.json"},
            "repeat_bake_byte_exact": repeat_exact,
            "typed_validity_preserved": validity["q_self_kinematics"] == validity["q_aabb_interaction"] == validity["q_lidar_contact"] == "ACCEPT" and validity["q_renderer_execution"] == "ACCEPT_CONFORMANCE",
            "semantic_physical_planning_safety_abstain": all(validity[key] == "ABSTAIN" for key in ["semantic_road", "physical_trajectory_validity", "planning_validity", "safety_validity"]),
            "source_immutable": all(_sha256(path) == expected_sha for path, expected_sha in frozen_files.items()) and all(_sha256(path) == expected_sha for path, expected_sha in package_files.items()),
            "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True, "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        _write_json(run_dir / "R44_GATE.json", {
            "schema_version": "worldsim_v6.r44_gate.v1", "checks": checks,
            "decision": "accept_typed_verified_actor_proposal_bake" if checks["passed"] else "reject_or_repair_verified_actor_proposal_bake",
        })
        _write_json(run_dir / "RESOURCE_AUDIT.json", {
            "schema_version": "worldsim_v6.r44_resource_audit.v1", "gpu_used": False,
            "wall_seconds": wall_seconds, "disk_free_gib_at_start": free_gib,
            "training_started": False, "confirmation_content_read": False,
        })
        summary = {
            "schema_version": "worldsim_v6.r44_summary.v1", "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"], "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_typed_verified_actor_proposal_bake" if checks["passed"] else "rejected",
            "source_commit": source_commit, "proposal_id": contract["proposal_id"], "translation_delta_m": contract["translation_delta_m"],
            "package_status": contract["package_status"], "package_manifest_sha256": _sha256(package / "PACKAGE_MANIFEST.json"),
            "trajectory_geometry_sha256": _sha256(package / "TRAJECTORY_GEOMETRY.json"),
            "validity_sha256": _sha256(package / "VALIDITY.json"), "provenance_sha256": _sha256(package / "PROVENANCE.json"),
            "translation_bake_max_error_m": translation_error_1, "repeat_bake_byte_exact": repeat_exact,
            "physical_trajectory_validity": "ABSTAIN", "safety_validity": "ABSTAIN", "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = ["R44_GATE.json", "SUMMARY.json", "RESOURCE_AUDIT.json"]
        _write_json(run_dir / "MANIFEST.json", {
            "schema_version": "worldsim_v6.r44_manifest.v1",
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
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/r44_verified_actor_proposal_bake_v1.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

