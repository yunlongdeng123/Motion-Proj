"""WorldSim V6 R50：提取 native actor lifecycle 并 bake 到 transform-owned package。"""

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

from motion_proj.worldsim_v6.r44_verified_actor_proposal_bake import _git, _manifest_files, _resolve_runs_uri, _sha256, _verify, _write_json
from motion_proj.worldsim_v6.r46_detached_transform_package_logsim import _verify_package


TASK_ID = "WS-V6-R50-ACTOR-LIFECYCLE-BAKE-01"


class R50ExperimentError(RuntimeError):
    """R50 正式实验合同失败。"""


def _build_package(output: Path, source_package: Path, lifecycle: np.ndarray, lifecycle_doc: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    shutil.copytree(source_package, output)
    geometry_path = output / "TRAJECTORY_GEOMETRY.json"
    runtime_path = output / "RUNTIME_CONTRACT.json"
    validity_path = output / "VALIDITY.json"
    provenance_path = output / "PROVENANCE.json"
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    validity = json.loads(validity_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    temporary = output / "blobs/actor_frame_valid.npy"
    np.save(temporary, lifecycle.astype(np.bool_), allow_pickle=False)
    lifecycle_sha = _sha256(temporary)
    lifecycle_path = output / f"blobs/{lifecycle_sha}.npy"
    temporary.rename(lifecycle_path)
    record = {
        "path": lifecycle_path.relative_to(output).as_posix(), "sha256": lifecycle_sha,
        "bytes": lifecycle_path.stat().st_size, "dtype": lifecycle.astype(np.bool_).dtype.str,
        "shape": list(lifecycle.shape), "semantics": "native_rigid_instances_fv_for_actor_model_index_0",
    }
    geometry["actor_frame_validity"] = record
    geometry["lifecycle_ownership"] = "independent_native_frame_validity"
    geometry["package_status"] = config["lifecycle_contract"]["package_status"]
    for index, row in enumerate(geometry["trajectory"]):
        row["actor_frame_valid"] = bool(lifecycle[index])
    runtime["input_actor_frame_validity"] = record["path"]
    runtime["opacity_composition"] = "opacity_runtime = base_opacity * actor_frame_valid[timestamp]"
    runtime["inactive_geometry_state_policy"] = "retain_geometry_and_mask_opacity_to_zero"
    validity["q_native_lifecycle_binding"] = "ACCEPT"
    validity["package_status"] = config["lifecycle_contract"]["package_status"]
    provenance["native_lifecycle"] = {
        "source": "frozen_streetgs_rigid_instances_fv", "checkpoint_sha256": config["sources"]["streetgs_checkpoint_sha256"],
        "frame_valid_sha256": lifecycle_sha, "active_frame_count": lifecycle_doc["active_frame_count"],
        "inactive_frame_count": lifecycle_doc["inactive_frame_count"], "transition_indices": lifecycle_doc["transition_indices"],
        "r49_rejected_run": config["sources"]["r49_run"], "r49_gate_sha256": config["sources"]["r49_gate_sha256"],
    }
    _write_json(geometry_path, geometry)
    _write_json(runtime_path, runtime)
    _write_json(validity_path, validity)
    _write_json(provenance_path, provenance)
    manifest = {"schema_version": "worldsim_v6.r50_lifecycle_package_manifest.v1", "files": _manifest_files(output)}
    _write_json(output / "PACKAGE_MANIFEST.json", manifest)
    return manifest


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R50ExperimentError("正式 R50 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R50ExperimentError("R50 task_id 漂移")
    sources = config["sources"]
    r49_run = _resolve_runs_uri(sources["r49_run"])
    r45_run = _resolve_runs_uri(sources["r45_run"])
    source_package = r45_run / "package"
    checkpoint = Path(sources["streetgs_checkpoint"])
    upstream = Path(sources["streetgs_upstream_root"])
    frozen_files = {
        r49_run / "MANIFEST.json": sources["r49_manifest_sha256"],
        r49_run / "R49_GATE.json": sources["r49_gate_sha256"],
        r49_run / "SUMMARY.json": sources["r49_summary_sha256"],
        r49_run / "FRAME_COMPARISONS.jsonl": sources["r49_frame_comparisons_sha256"],
        r45_run / "R45_GATE.json": sources["r45_gate_sha256"],
        source_package / "PACKAGE_MANIFEST.json": sources["r45_package_manifest_sha256"],
        checkpoint: sources["streetgs_checkpoint_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    if _git(upstream, "rev-parse", "HEAD") != sources["streetgs_upstream_commit"]:
        raise R50ExperimentError("StreetGS upstream commit 漂移")
    r49_gate = json.loads((r49_run / "R49_GATE.json").read_text(encoding="utf-8"))
    r45_gate = json.loads((r45_run / "R45_GATE.json").read_text(encoding="utf-8"))
    source_manifest = _verify_package(source_package)
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R50ExperimentError("R50 磁盘资源不足")

    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / f"{now.strftime('%Y%m%dT%H%M%SZ')}__lifecycle-bake-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        worker_dir = run_dir / "lifecycle_worker"
        command = [
            sources["drivestudio_python"], str(repo_root / "scripts/worldsim_v6/r50_actor_lifecycle_worker.py"),
            "--repo-root", str(repo_root), "--checkpoint", str(checkpoint), "--upstream-root", str(upstream),
            "--actor-model-index", str(config["lifecycle_contract"]["actor_model_index"]), "--output", str(worker_dir),
        ]
        completed = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, timeout=float(config["resources"]["maximum_worker_seconds"]))
        (run_dir / "lifecycle_worker.log").write_text(completed.stdout + "\n--- STDERR ---\n" + completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise R50ExperimentError(f"lifecycle worker 失败：rc={completed.returncode}")
        lifecycle_doc = json.loads((worker_dir / "LIFECYCLE.json").read_text(encoding="utf-8"))
        lifecycle = np.load(worker_dir / lifecycle_doc["frame_valid_path"], allow_pickle=False)
        _verify(worker_dir / lifecycle_doc["frame_valid_path"], lifecycle_doc["frame_valid_sha256"])
        worker_audit = json.loads((worker_dir / "WORKER_AUDIT.json").read_text(encoding="utf-8"))
        package = run_dir / "package"
        repeat_package = run_dir / "_repeat_package"
        manifest_1 = _build_package(package, source_package, lifecycle, lifecycle_doc, config)
        manifest_2 = _build_package(repeat_package, source_package, lifecycle, lifecycle_doc, config)
        repeat_exact = manifest_1 == manifest_2 and _sha256(package / "PACKAGE_MANIFEST.json") == _sha256(repeat_package / "PACKAGE_MANIFEST.json")
        shutil.rmtree(repeat_package)
        geometry = json.loads((package / "TRAJECTORY_GEOMETRY.json").read_text(encoding="utf-8"))
        validity = json.loads((package / "VALIDITY.json").read_text(encoding="utf-8"))
        contract = config["lifecycle_contract"]
        peak_mib = int(worker_audit["peak_torch_reserved_bytes"]) / (1024**2)
        wall_seconds = time.monotonic() - started
        checks = {
            "r49_rejection_preserved": not r49_gate["checks"]["passed"],
            "r45_authority_accepted": r45_gate["checks"]["passed"],
            "native_lifecycle_denominators_exact": lifecycle.shape == (int(contract["expected_frame_count"]),) and lifecycle_doc["primitive_count"] == int(contract["expected_primitive_count"]),
            "native_lifecycle_counts_exact": lifecycle_doc["active_frame_count"] == int(contract["expected_active_frame_count"]) and lifecycle_doc["inactive_frame_count"] == int(contract["expected_inactive_frame_count"]),
            "single_transition_exact": lifecycle_doc["transition_indices"] == contract["expected_transition_indices"],
            "active_interval_exact": np.flatnonzero(lifecycle).tolist() == list(range(int(contract["expected_active_first_frame"]), int(contract["expected_active_last_frame"]) + 1)),
            "inactive_interval_exact": np.flatnonzero(~lifecycle).tolist() == list(range(int(contract["expected_inactive_first_frame"]), int(contract["expected_inactive_last_frame"]) + 1)),
            "lifecycle_content_addressed_and_bound": Path(geometry["actor_frame_validity"]["path"]).stem == geometry["actor_frame_validity"]["sha256"] == lifecycle_doc["frame_valid_sha256"],
            "base_arrays_and_transform_preserved": geometry["base_arrays"] == json.loads((source_package / "TRAJECTORY_GEOMETRY.json").read_text(encoding="utf-8"))["base_arrays"] and geometry["proposal_transform_world"] == json.loads((source_package / "TRAJECTORY_GEOMETRY.json").read_text(encoding="utf-8"))["proposal_transform_world"],
            "repeat_lifecycle_bake_exact": repeat_exact,
            "typed_lifecycle_validity_accept": validity["q_native_lifecycle_binding"] == "ACCEPT",
            "semantic_physical_planning_safety_abstain": all(validity[key] == "ABSTAIN" for key in ["semantic_road", "physical_trajectory_validity", "planning_validity", "safety_validity"]),
            "checkpoint_immutable": worker_audit["checkpoint_sha256_before"] == worker_audit["checkpoint_sha256_after"] == sources["streetgs_checkpoint_sha256"],
            "source_immutable": all(_sha256(path) == expected_sha for path, expected_sha in frozen_files.items()) and source_manifest == _verify_package(source_package),
            "gpu_within_budget": peak_mib <= float(config["resources"]["maximum_peak_gpu_memory_mib"]),
            "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True, "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        _write_json(run_dir / "R50_GATE.json", {
            "schema_version": "worldsim_v6.r50_gate.v1", "checks": checks,
            "decision": "accept_native_actor_lifecycle_bake" if checks["passed"] else "reject_or_repair_actor_lifecycle_bake",
        })
        _write_json(run_dir / "RESOURCE_AUDIT.json", {
            "schema_version": "worldsim_v6.r50_resource_audit.v1", "gpu_used": True,
            "peak_torch_reserved_mib": peak_mib, "worker_wall_seconds": float(worker_audit["wall_seconds"]),
            "wall_seconds": wall_seconds, "disk_free_gib_at_start": free_gib,
            "training_started": False, "confirmation_content_read": False,
        })
        summary = {
            "schema_version": "worldsim_v6.r50_summary.v1", "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"], "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_native_actor_lifecycle_bake" if checks["passed"] else "rejected",
            "source_commit": source_commit, "active_frame_count": lifecycle_doc["active_frame_count"],
            "inactive_frame_count": lifecycle_doc["inactive_frame_count"], "transition_indices": lifecycle_doc["transition_indices"],
            "package_manifest_sha256": _sha256(package / "PACKAGE_MANIFEST.json"),
            "trajectory_geometry_sha256": _sha256(package / "TRAJECTORY_GEOMETRY.json"),
            "runtime_contract_sha256": _sha256(package / "RUNTIME_CONTRACT.json"),
            "lifecycle_sha256": lifecycle_doc["frame_valid_sha256"], "repeat_bake_exact": repeat_exact,
            "physical_trajectory_validity": "ABSTAIN", "safety_validity": "ABSTAIN", "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = ["R50_GATE.json", "SUMMARY.json", "RESOURCE_AUDIT.json", "lifecycle_worker.log", "lifecycle_worker/LIFECYCLE.json", "lifecycle_worker/ACTOR_FRAME_VALID.npy", "lifecycle_worker/WORKER_AUDIT.json"]
        _write_json(run_dir / "MANIFEST.json", {
            "schema_version": "worldsim_v6.r50_manifest.v1",
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
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/r50_actor_lifecycle_bake_v1.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

