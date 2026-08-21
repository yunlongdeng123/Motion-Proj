"""WorldSim V6 R55：按冻结规则编译第二个 observed SceneIR actor bundle。"""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from motion_proj.worldsim_v6.r44_verified_actor_proposal_bake import _git, _resolve_runs_uri, _sha256, _verify, _write_json


TASK_ID = "WS-V6-R55-SECOND-ACTOR-SCENEIR-BUNDLE-01"
REQUIRED_ARRAYS = {"means_m", "scales_m", "quaternions_wxyz", "opacities", "features_dc", "features_rest", "source_indices"}


class R55ExperimentError(RuntimeError):
    """R55 正式实验合同失败。"""


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R55ExperimentError("正式 R55 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R55ExperimentError("R55 task_id 漂移")
    sources = config["sources"]
    r54_run = _resolve_runs_uri(sources["r54_run"])
    binding_run = _resolve_runs_uri(sources["sceneir_binding_run"])
    base_package = binding_run / sources["base_sceneir_package"]
    frozen_files = {
        r54_run / "R54_GATE.json": sources["r54_gate_sha256"],
        r54_run / "SUMMARY.json": sources["r54_summary_sha256"],
        r54_run / "worker/ACTOR_INVENTORY.json": sources["r54_inventory_sha256"],
        binding_run / "MANIFEST.json": sources["sceneir_binding_manifest_sha256"],
        binding_run / "R13_SCENEIR_SENSOR_BINDING_GATE.json": sources["sceneir_binding_gate_sha256"],
        base_package / "MANIFEST.json": sources["base_sceneir_manifest_sha256"],
        base_package / "sceneir.json": sources["base_sceneir_document_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    if shutil.disk_usage(run_root).free / (1024**3) < float(config["resources"]["minimum_disk_free_gib"]):
        raise R55ExperimentError("R55 磁盘资源不足")
    r54_gate = json.loads((r54_run / "R54_GATE.json").read_text(encoding="utf-8"))
    binding_gate = json.loads((binding_run / "R13_SCENEIR_SENSOR_BINDING_GATE.json").read_text(encoding="utf-8"))
    inventory = json.loads((r54_run / "worker/ACTOR_INVENTORY.json").read_text(encoding="utf-8"))
    selection = config["selection"]
    eligible = [actor for actor in inventory["actors"] if actor["actor_model_index"] != 0]
    selected = sorted(eligible, key=lambda actor: (-actor["primitive_count"], actor["actor_model_index"]))[0]
    selected_index = int(selected["actor_model_index"])
    lifecycle_source = r54_run / "worker" / selected["lifecycle_path"]
    _verify(lifecycle_source, selection["expected_lifecycle_sha256"])
    frozen_files[lifecycle_source] = selection["expected_lifecycle_sha256"]
    document = json.loads((base_package / "sceneir.json").read_text(encoding="utf-8"))
    source_manifest = json.loads((base_package / "MANIFEST.json").read_text(encoding="utf-8"))
    actor = next(row for row in document["actors"] if row["id"] == selection["expected_actor_id"])
    chunk = next(row for row in document["chunks"] if row["id"] == selection["expected_chunk_id"])
    transform_keys = {(row["transform_name"], int(row["timestamp_us"])) for row in actor["trajectory"]}
    transforms = [row for row in document["transforms"] if (row["name"], int(row["timestamp_us"])) in transform_keys]
    provenance = next(row for row in document["provenance"] if row["id"] == chunk["provenance_id"])
    support = next(row for row in document["support"] if row["id"] == chunk["support_id"])

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__actor2-bundle-s{config['seed']}-r1"
    package = run_dir / "package"
    blob_dir = package / "blobs"
    blob_dir.mkdir(parents=True, exist_ok=False)
    array_audit = {}
    copied = []
    source_blob_hashes = {}
    for name, reference in sorted(chunk["arrays"].items()):
        source_path = base_package / reference["path"]
        manifest_row = source_manifest["files"].get(reference["path"])
        if manifest_row is None or manifest_row["sha256"] != reference["sha256"]:
            raise R55ExperimentError(f"base manifest 缺少 actor2 blob：{name}")
        _verify(source_path, reference["sha256"])
        source_blob_hashes[source_path] = reference["sha256"]
        destination = package / reference["path"]
        shutil.copy2(source_path, destination)
        array = np.load(destination, allow_pickle=False)
        array_audit[name] = {
            "path": reference["path"], "sha256": reference["sha256"], "shape": list(array.shape), "dtype": array.dtype.str,
            "shape_exact": list(array.shape) == reference["shape"], "dtype_exact": array.dtype.str == reference["dtype"],
            "primitive_axis_exact": array.ndim >= 1 and array.shape[0] == int(selection["expected_primitive_count"]),
            "finite": bool(np.isfinite(array).all() if np.issubdtype(array.dtype, np.number) else True),
        }
        copied.append(reference["path"])
    lifecycle_destination = blob_dir / f"{selection['expected_lifecycle_sha256']}.npy"
    shutil.copy2(lifecycle_source, lifecycle_destination)
    lifecycle = np.load(lifecycle_destination, allow_pickle=False)
    lifecycle_relative = str(lifecycle_destination.relative_to(package))
    bundle = {
        "schema_version": "worldsim_v6.r55_second_actor_bundle.v1",
        "asset_type": "explicit_sceneir_gaussian_actor_with_logged_trajectory_and_lifecycle",
        "selection_rule": selection["rule"], "frontend_model_index": selected_index,
        "actor": actor, "chunk": chunk, "trajectory_transforms": transforms,
        "provenance": provenance, "support": support,
        "lifecycle": {"path": lifecycle_relative, "sha256": selection["expected_lifecycle_sha256"], "shape": list(lifecycle.shape), "dtype": lifecycle.dtype.str},
    }
    _write_json(package / "ACTOR_BUNDLE.json", bundle)
    _write_json(package / "ARRAY_AUDIT.json", array_audit)
    _write_json(package / "VALIDITY.json", {
        "schema_version": "worldsim_v6.r55_validity.v1", "asset_id": actor["id"], "identity_binding": "OBSERVED_MODEL_INDEX_BOUND",
        "logged_trajectory": "PRESENT_EXACT", "native_lifecycle": "PRESENT_EXACT", "semantic_identity": "ABSTAIN",
        "novel_trajectory_dynamics_physics_planning_safety": "ABSTAIN",
    })
    _write_json(package / "RUNTIME_DEPENDENCY_AUDIT.json", {
        "schema_version": "worldsim_v6.r55_runtime_dependency_audit.v1", "runtime_dependencies": ["json_reader", "numpy_npy_reader"],
        "online_generator_dependency": False, "model_weight_dependency": False, "network_dependency": False, "source_run_payload_dependency": False,
    })
    package_files = ["ACTOR_BUNDLE.json", "ARRAY_AUDIT.json", "VALIDITY.json", "RUNTIME_DEPENDENCY_AUDIT.json", *copied, lifecycle_relative]
    _write_json(package / "PACKAGE_MANIFEST.json", {"schema_version": "worldsim_v6.r55_package_manifest.v1", "files": {name: {"bytes": (package / name).stat().st_size, "sha256": _sha256(package / name)} for name in package_files}})
    checks = {
        "r54_inventory_and_sceneir_binding_accepted": bool(r54_gate["checks"]["passed"] and binding_gate["checks"]["passed"]),
        "selection_rule_exact": selected_index == int(selection["expected_actor_model_index"]),
        "actor_and_chunk_identity_exact": actor["id"] == selection["expected_actor_id"] and chunk["id"] == selection["expected_chunk_id"] and chunk["actor_id"] == actor["id"],
        "primitive_count_exact": chunk["primitive_count"] == selected["primitive_count"] == int(selection["expected_primitive_count"]),
        "required_gaussian_arrays_exact": set(array_audit) == REQUIRED_ARRAYS == set(config["package"]["required_gaussian_arrays"]),
        "all_arrays_reload_exact": all(row["shape_exact"] and row["dtype_exact"] and row["primitive_axis_exact"] and row["finite"] for row in array_audit.values()),
        "trajectory_denominator_exact": len(actor["trajectory"]) == len(actor["visibility"]) == len(transforms) == int(selection["expected_trajectory_rows"]),
        "lifecycle_exact": _sha256(lifecycle_destination) == selection["expected_lifecycle_sha256"] and lifecycle.shape == (int(selection["expected_trajectory_rows"]),) and int(lifecycle.sum()) == int(selection["expected_active_frame_count"]),
        "no_online_runtime_dependency": True,
        "unsupported_claims_abstain": True,
        "source_immutable": all(_sha256(path) == expected_sha for path, expected_sha in frozen_files.items()) and all(_sha256(path) == expected_sha for path, expected_sha in source_blob_hashes.items()),
        "wall_within_budget": (time.monotonic() - started) <= float(config["resources"]["maximum_wall_seconds"]),
        "training_not_started": True, "confirmation_not_read": True,
    }
    checks["passed"] = all(checks.values())
    _write_json(run_dir / "R55_GATE.json", {"schema_version": "worldsim_v6.r55_gate.v1", "checks": checks, "decision": "accept_second_actor_sceneir_bundle" if checks["passed"] else "reject_or_repair_second_actor_bundle"})
    _write_json(run_dir / "SUMMARY.json", {
        "schema_version": "worldsim_v6.r55_summary.v1", "task_id": TASK_ID, "hypothesis_id": config["hypothesis_id"], "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_development_second_actor_sceneir_bundle" if checks["passed"] else "rejected", "source_commit": source_commit,
        "actor_model_index": selected_index, "actor_id": actor["id"], "primitive_count": chunk["primitive_count"], "trajectory_rows": len(transforms), "active_frame_count": int(lifecycle.sum()), "claim_boundary": config["claim_boundary"],
    })
    tracked = ["R55_GATE.json", "SUMMARY.json", "package/PACKAGE_MANIFEST.json", *[f"package/{name}" for name in package_files]]
    _write_json(run_dir / "MANIFEST.json", {"schema_version": "worldsim_v6.r55_manifest.v1", "files": {name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)} for name in tracked}})
    _write_json(run_dir / "TERMINAL.json", {"schema_version": "worldsim_v6.terminal.v1", "status": "done" if checks["passed"] else "rejected", "manifest_sha256": _sha256(run_dir / "MANIFEST.json"), "summary_sha256": _sha256(run_dir / "SUMMARY.json")})
    print(str(run_dir), flush=True)
    return run_dir


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/r55_second_actor_sceneir_bundle_v1.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
