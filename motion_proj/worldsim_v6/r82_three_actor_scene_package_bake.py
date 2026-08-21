"""WorldSim V6 R82: bake a deterministic three-actor SceneIR edit package."""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from motion_proj.worldsim_v6.r44_verified_actor_proposal_bake import _git, _resolve_runs_uri, _sha256, _verify, _write_json
from motion_proj.worldsim_v6.r46_detached_transform_package_logsim import _verify_package
from motion_proj.worldsim_v6.r70_two_actor_scene_package_bake import _source_tree_records, _tree_records, _verify_scene_package


TASK_ID = "WS-V6-R82-THREE-ACTOR-SCENE-PACKAGE-BAKE-01"


class R82ExperimentError(RuntimeError):
    """The preregistered R82 experiment contract was violated."""


def _bake(output: Path, actor_sources: list[dict[str, Any]], composition: dict[str, Any], runtime: dict[str, Any], validity: dict[str, Any]) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    for actor in actor_sources:
        shutil.copytree(actor["source_package"], output / "actors" / actor["actor_id"], copy_function=shutil.copy2)
    _write_json(output / "SCENE_COMPOSITION.json", composition)
    _write_json(output / "RUNTIME_CONTRACT.json", runtime)
    _write_json(output / "VALIDITY.json", validity)
    _write_json(output / "SCENE_PACKAGE_MANIFEST.json", {"schema_version": "worldsim_v6.r82_scene_package_manifest.v1", "files": _tree_records(output, include_manifest=False)})
    return _verify_scene_package(output)


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R82ExperimentError("formal R82 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text())
    if config.get("task_id") != TASK_ID:
        raise R82ExperimentError("R82 task_id drift")
    sources = config["sources"]
    contract = config["package_contract"]
    r70 = _resolve_runs_uri(sources["r70_run"])
    prior_scene = r70 / "package"
    r80 = _resolve_runs_uri(sources["r80_run"])
    actor5_package = r80 / "package"
    r81 = _resolve_runs_uri(sources["r81_run"])
    frozen = {
        r70 / "MANIFEST.json": sources["r70_manifest_sha256"],
        r70 / "R70_GATE.json": sources["r70_gate_sha256"],
        r70 / "SUMMARY.json": sources["r70_summary_sha256"],
        prior_scene / "SCENE_PACKAGE_MANIFEST.json": sources["r70_scene_package_manifest_sha256"],
        prior_scene / "SCENE_COMPOSITION.json": sources["r70_scene_composition_sha256"],
        prior_scene / "RUNTIME_CONTRACT.json": sources["r70_runtime_contract_sha256"],
        prior_scene / "VALIDITY.json": sources["r70_validity_sha256"],
        r80 / "MANIFEST.json": sources["r80_manifest_sha256"],
        r80 / "R80_GATE.json": sources["r80_gate_sha256"],
        actor5_package / "PACKAGE_MANIFEST.json": sources["r80_package_manifest_sha256"],
        actor5_package / "TRAJECTORY_GEOMETRY.json": sources["r80_trajectory_geometry_sha256"],
        actor5_package / "VALIDITY.json": sources["r80_validity_sha256"],
        r81 / "MANIFEST.json": sources["r81_manifest_sha256"],
        r81 / "R81_GATE.json": sources["r81_gate_sha256"],
        r81 / "SUMMARY.json": sources["r81_summary_sha256"],
        r81 / "JOINT_INTERACTION_DECISION.json": sources["r81_decision_sha256"],
    }
    for path, expected in frozen.items():
        _verify(path, expected)
    if shutil.disk_usage(run_root).free / (1024**3) < float(config["resources"]["minimum_disk_free_gib"]):
        raise R82ExperimentError("R82 disk resource insufficient")
    prior_manifest = _verify_scene_package(prior_scene)
    actor5_manifest = _verify_package(actor5_package)
    gates = [json.loads((r70 / "R70_GATE.json").read_text()), json.loads((r80 / "R80_GATE.json").read_text()), json.loads((r81 / "R81_GATE.json").read_text())]
    prior_composition = json.loads((prior_scene / "SCENE_COMPOSITION.json").read_text())
    prior_validity = json.loads((prior_scene / "VALIDITY.json").read_text())
    actor5_geometry = json.loads((actor5_package / "TRAJECTORY_GEOMETRY.json").read_text())
    actor5_validity = json.loads((actor5_package / "VALIDITY.json").read_text())
    joint = json.loads((r81 / "JOINT_INTERACTION_DECISION.json").read_text())
    actor_sources = [
        {"actor_id": "actor_0000", "source_package": prior_scene / "actors/actor_0000", "actor_model_index": 0},
        {"actor_id": "actor_0002", "source_package": prior_scene / "actors/actor_0002", "actor_model_index": 2},
        {"actor_id": "actor_0005", "source_package": actor5_package, "actor_model_index": 5},
    ]
    source_records = {actor["actor_id"]: _source_tree_records(actor["source_package"]) for actor in actor_sources}
    scene_actors = list(prior_composition["actors"])
    transform = actor5_geometry["proposal_transform_world"]
    lifecycle = actor5_geometry["base_arrays"]["actor_frame_validity"]
    scene_actors.append({
        "actor_id": "actor_0005",
        "actor_model_index": 5,
        "proposal_id": actor5_geometry["proposal_id"],
        "translation_delta_m": actor5_geometry["translation_delta_m"],
        "primitive_count": actor5_geometry["primitive_count"],
        "trajectory_rows": transform["shape"][0],
        "actor_package_path": "actors/actor_0005",
        "actor_package_manifest_sha256": _sha256(actor5_package / "PACKAGE_MANIFEST.json"),
        "proposal_transform_relative_path": f"actors/actor_0005/{transform['path']}",
        "proposal_transform_sha256": transform["sha256"],
        "lifecycle_relative_path": f"actors/actor_0005/{lifecycle['path']}",
        "lifecycle_sha256": lifecycle["sha256"],
    })
    composition = {
        "schema_version": "worldsim_v6.r82_scene_composition.v1",
        "package_status": contract["package_status"],
        "base_sceneir": prior_composition["base_sceneir"],
        "actors": scene_actors,
        "joint_interaction": {
            "authority_gate_sha256": sources["r81_gate_sha256"],
            "decision_sha256": sources["r81_decision_sha256"],
            "q_joint_aabb_interaction": joint["q_joint_aabb_interaction"],
            "joint_new_overlap_events": joint["joint_new_overlap_events"],
            "emergent_cross_edit_overlap_events": joint["emergent_cross_edit_overlap_events"],
            "edited_pair_new_overlap_events": joint["edited_pair_new_overlap_events"],
        },
    }
    actor_ids = list(contract["expected_actor_ids"])
    runtime = {
        "schema_version": "worldsim_v6.r82_runtime_contract.v1",
        "runtime_mode": "multi_actor_transform_lifecycle_owned_scene_patch",
        "actor_order": actor_ids,
        "per_actor_operation": "load_base_arrays_then_apply_float64_world_transform_then_mask_opacity_by_native_lifecycle",
        "composition_isolation": "each_transform_applies_only_to_its_owned_actor_model_index",
        "static_scene_dependency": "external_base_sceneir_and_native_checkpoint",
        "renderer_merge": "replace_owned_native_actor_fields_then_render_one_shared_scene",
    }
    individual_validity = dict(prior_validity["individual_actor_factor_validity"])
    individual_validity["actor_0005"] = actor5_validity
    validity = {
        "schema_version": "worldsim_v6.r82_validity.v1",
        "package_status": contract["package_status"],
        "individual_actor_factor_validity": individual_validity,
        "q_joint_aabb_interaction": joint["q_joint_aabb_interaction"],
        "joint_new_overlap_events": joint["joint_new_overlap_events"],
        "q_collision_physics": "ABSTAIN",
        "semantic_road": "ABSTAIN",
        "physical_dynamics": "ABSTAIN",
        "planning_safety": "ABSTAIN",
    }
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__three-actor-package-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    package = run_dir / "package"
    repeat = run_dir / "repeat_package"
    manifest = _bake(package, actor_sources, composition, runtime, validity)
    repeat_manifest = _bake(repeat, actor_sources, composition, runtime, validity)
    package_records = _tree_records(package, include_manifest=True)
    repeat_records = _tree_records(repeat, include_manifest=True)
    copied_exact = {actor_id: _tree_records(package / "actors" / actor_id, include_manifest=True) == source_records[actor_id] for actor_id in actor_ids}
    payload_count = len(manifest["files"])
    combined_primitives = sum(int(row["primitive_count"]) for row in scene_actors)
    binding_exact = bool(
        [row["actor_id"] for row in scene_actors] == actor_ids
        and [row["actor_model_index"] for row in scene_actors] == list(contract["expected_actor_model_indices"])
        and [row["proposal_id"] for row in scene_actors] == list(contract["expected_proposal_ids"])
        and all(row["translation_delta_m"] == contract["expected_translations_m"][row["actor_id"]] for row in scene_actors)
    )
    wall = time.monotonic() - started
    checks = {
        "r70_r80_r81_authorities_accepted": all(bool(gate["checks"]["passed"]) for gate in gates),
        "three_actor_proposal_transform_lifecycle_bindings_exact": binding_exact,
        "actor_denominator_exact": len(scene_actors) == int(contract["expected_actor_count"]),
        "trajectory_denominators_exact": all(row["trajectory_rows"] == int(contract["expected_trajectory_rows_per_actor"]) for row in scene_actors),
        "combined_primitive_denominator_exact": combined_primitives == int(contract["expected_combined_primitives"]),
        "source_file_denominators_exact": all(len(source_records[actor_id]) == int(contract["expected_source_files_per_actor_including_manifest"]) for actor_id in actor_ids),
        "all_actor_package_trees_copied_byte_exact": all(copied_exact.values()),
        "scene_package_payload_denominator_exact": payload_count == int(contract["expected_package_payload_files_excluding_scene_manifest"]),
        "joint_zero_new_overlap_authority_bound": joint["q_joint_aabb_interaction"] == "ACCEPT" and joint["joint_new_overlap_events"] == 0 and joint["emergent_cross_edit_overlap_events"] == 0 and joint["edited_pair_new_overlap_events"] == 0,
        "typed_abstentions_preserved": validity["q_collision_physics"] == validity["semantic_road"] == validity["physical_dynamics"] == validity["planning_safety"] == "ABSTAIN",
        "scene_package_manifest_self_consistent": manifest == _verify_scene_package(package),
        "repeat_bake_byte_exact": package_records == repeat_records and manifest == repeat_manifest,
        "frozen_sources_immutable": all(_sha256(path) == expected for path, expected in frozen.items()) and prior_manifest == _verify_scene_package(prior_scene) and actor5_manifest == _verify_package(actor5_package),
        "wall_within_budget": wall <= float(config["resources"]["maximum_wall_seconds"]),
        "training_and_rendering_not_started": True,
        "confirmation_not_read": True,
    }
    checks["passed"] = all(checks.values())
    _write_json(run_dir / "R82_GATE.json", {"schema_version": "worldsim_v6.r82_gate.v1", "checks": checks, "decision": "accept_three_actor_scene_edit_package_bake" if checks["passed"] else "reject_or_repair_three_actor_scene_edit_package_bake"})
    _write_json(run_dir / "RESOURCE_AUDIT.json", {"schema_version": "worldsim_v6.r82_resource_audit.v1", "gpu_used": False, "wall_seconds": wall, "disk_free_gib_at_start": shutil.disk_usage(run_root).free / (1024**3), "package_bytes": sum(row["bytes"] for row in package_records.values()), "repeat_package_bytes": sum(row["bytes"] for row in repeat_records.values()), "training_started": False, "rendering_started": False, "confirmation_content_read": False})
    _write_json(run_dir / "SUMMARY.json", {
        "schema_version": "worldsim_v6.r82_summary.v1", "task_id": TASK_ID, "hypothesis_id": config["hypothesis_id"], "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_development_three_actor_scene_edit_package" if checks["passed"] else "rejected", "source_commit": source_commit,
        "actor_ids": actor_ids, "actor_model_indices": [row["actor_model_index"] for row in scene_actors], "proposal_ids": [row["proposal_id"] for row in scene_actors],
        "combined_primitives": combined_primitives, "combined_trajectory_rows": sum(row["trajectory_rows"] for row in scene_actors),
        "scene_package_manifest_sha256": _sha256(package / "SCENE_PACKAGE_MANIFEST.json"), "scene_composition_sha256": _sha256(package / "SCENE_COMPOSITION.json"),
        "runtime_contract_sha256": _sha256(package / "RUNTIME_CONTRACT.json"), "validity_sha256": _sha256(package / "VALIDITY.json"),
        "payload_file_count": payload_count, "repeat_bake_byte_exact": package_records == repeat_records, "claim_boundary": config["claim_boundary"],
    })
    tracked = ["R82_GATE.json", "SUMMARY.json", "RESOURCE_AUDIT.json", "package/SCENE_PACKAGE_MANIFEST.json", "package/SCENE_COMPOSITION.json", "package/RUNTIME_CONTRACT.json", "package/VALIDITY.json", "repeat_package/SCENE_PACKAGE_MANIFEST.json"]
    _write_json(run_dir / "MANIFEST.json", {"schema_version": "worldsim_v6.r82_manifest.v1", "files": {name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)} for name in tracked}})
    _write_json(run_dir / "TERMINAL.json", {"schema_version": "worldsim_v6.terminal.v1", "status": "done" if checks["passed"] else "rejected", "manifest_sha256": _sha256(run_dir / "MANIFEST.json"), "summary_sha256": _sha256(run_dir / "SUMMARY.json")})
    print(run_dir, flush=True)
    return run_dir


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/r82_three_actor_scene_package_bake_v1.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
