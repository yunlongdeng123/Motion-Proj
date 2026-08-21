"""WorldSim V6 R70: bake a deterministic two-actor scene-edit package."""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from motion_proj.worldsim_v6.r44_verified_actor_proposal_bake import (
    _git,
    _resolve_runs_uri,
    _sha256,
    _verify,
    _write_json,
)
from motion_proj.worldsim_v6.r46_detached_transform_package_logsim import _verify_package


TASK_ID = "WS-V6-R70-TWO-ACTOR-SCENE-PACKAGE-BAKE-01"


class R70ExperimentError(RuntimeError):
    """The preregistered R70 experiment contract was violated."""


def _tree_records(root: Path, include_manifest: bool = True) -> dict[str, dict[str, Any]]:
    records = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if not include_manifest and relative == "SCENE_PACKAGE_MANIFEST.json":
            continue
        records[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return records


def _verify_scene_package(package: Path) -> dict[str, Any]:
    manifest_path = package / "SCENE_PACKAGE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_paths = set(manifest["files"])
    actual_paths = set(_tree_records(package, include_manifest=False))
    if expected_paths != actual_paths:
        raise R70ExperimentError("scene package manifest file denominator drift")
    for relative, record in manifest["files"].items():
        _verify(package / relative, record["sha256"])
        if (package / relative).stat().st_size != int(record["bytes"]):
            raise R70ExperimentError(f"scene package byte count drift: {relative}")
    return manifest


def _source_tree_records(package: Path) -> dict[str, dict[str, Any]]:
    return _tree_records(package, include_manifest=True)


def _bake_package(
    output: Path,
    actor_sources: list[dict[str, Any]],
    scene_composition: dict[str, Any],
    runtime_contract: dict[str, Any],
    validity: dict[str, Any],
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    actors_root = output / "actors"
    for actor in actor_sources:
        shutil.copytree(actor["source_package"], actors_root / actor["actor_id"], copy_function=shutil.copy2)
    _write_json(output / "SCENE_COMPOSITION.json", scene_composition)
    _write_json(output / "RUNTIME_CONTRACT.json", runtime_contract)
    _write_json(output / "VALIDITY.json", validity)
    _write_json(
        output / "SCENE_PACKAGE_MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r70_scene_package_manifest.v1",
            "files": _tree_records(output, include_manifest=False),
        },
    )
    return _verify_scene_package(output)


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R70ExperimentError("formal R70 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R70ExperimentError("R70 task_id drift")
    sources = config["sources"]
    contract = config["package_contract"]
    resources = config["resources"]

    r69_run = _resolve_runs_uri(sources["r69_run"])
    r50_run = _resolve_runs_uri(sources["r50_run"])
    r67_run = _resolve_runs_uri(sources["r67_run"])
    r50_package = r50_run / "package"
    r67_package = r67_run / "package"
    frozen_files = {
        r69_run / "MANIFEST.json": sources["r69_manifest_sha256"],
        r69_run / "R69_GATE.json": sources["r69_gate_sha256"],
        r69_run / "SUMMARY.json": sources["r69_summary_sha256"],
        r69_run / "JOINT_INTERACTION_DECISION.json": sources["r69_joint_interaction_decision_sha256"],
        r50_run / "MANIFEST.json": sources["r50_manifest_sha256"],
        r50_run / "R50_GATE.json": sources["r50_gate_sha256"],
        r50_package / "PACKAGE_MANIFEST.json": sources["r50_package_manifest_sha256"],
        r67_run / "MANIFEST.json": sources["r67_manifest_sha256"],
        r67_run / "R67_GATE.json": sources["r67_gate_sha256"],
        r67_package / "PACKAGE_MANIFEST.json": sources["r67_package_manifest_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    r50_manifest = _verify_package(r50_package)
    r67_manifest = _verify_package(r67_package)
    source_tree_records = {
        "actor_0000": _source_tree_records(r50_package),
        "actor_0002": _source_tree_records(r67_package),
    }
    r69_gate = json.loads((r69_run / "R69_GATE.json").read_text(encoding="utf-8"))
    r69_decision = json.loads(
        (r69_run / "JOINT_INTERACTION_DECISION.json").read_text(encoding="utf-8")
    )
    geometries = {
        "actor_0000": json.loads(
            (r50_package / "TRAJECTORY_GEOMETRY.json").read_text(encoding="utf-8")
        ),
        "actor_0002": json.loads(
            (r67_package / "TRAJECTORY_GEOMETRY.json").read_text(encoding="utf-8")
        ),
    }
    validities = {
        "actor_0000": json.loads((r50_package / "VALIDITY.json").read_text(encoding="utf-8")),
        "actor_0002": json.loads((r67_package / "VALIDITY.json").read_text(encoding="utf-8")),
    }
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(resources["minimum_disk_free_gib"]):
        raise R70ExperimentError("R70 disk resource insufficient")

    actor_ids = list(contract["expected_actor_ids"])
    actor_sources = [
        {"actor_id": "actor_0000", "source_package": r50_package, "actor_model_index": 0},
        {"actor_id": "actor_0002", "source_package": r67_package, "actor_model_index": 2},
    ]
    scene_actors = []
    for actor in actor_sources:
        actor_id = actor["actor_id"]
        geometry = geometries[actor_id]
        proposal_transform = geometry["proposal_transform_world"]
        lifecycle = geometry.get("actor_frame_validity") or geometry["base_arrays"]["actor_frame_validity"]
        scene_actors.append(
            {
                "actor_id": actor_id,
                "actor_model_index": int(actor["actor_model_index"]),
                "proposal_id": geometry["proposal_id"],
                "translation_delta_m": geometry["translation_delta_m"],
                "primitive_count": int(geometry["primitive_count"]),
                "trajectory_rows": int(proposal_transform["shape"][0]),
                "actor_package_path": f"actors/{actor_id}",
                "actor_package_manifest_sha256": _sha256(
                    actor["source_package"] / "PACKAGE_MANIFEST.json"
                ),
                "proposal_transform_relative_path": f"actors/{actor_id}/{proposal_transform['path']}",
                "proposal_transform_sha256": proposal_transform["sha256"],
                "lifecycle_relative_path": f"actors/{actor_id}/{lifecycle['path']}",
                "lifecycle_sha256": lifecycle["sha256"],
            }
        )
    scene_composition = {
        "schema_version": "worldsim_v6.r70_scene_composition.v1",
        "package_status": contract["package_status"],
        "base_sceneir": {
            "ownership": "frozen_external_dependency",
            "manifest_sha256": sources["base_sceneir_manifest_sha256"],
            "sceneir_document_sha256": sources["base_sceneir_document_sha256"],
        },
        "actors": scene_actors,
        "joint_interaction": {
            "authority_gate_sha256": sources["r69_gate_sha256"],
            "decision_sha256": sources["r69_joint_interaction_decision_sha256"],
            "q_joint_aabb_interaction": r69_decision["q_joint_aabb_interaction"],
            "joint_new_overlap_events": int(r69_decision["joint_new_overlap_events"]),
            "emergent_cross_edit_overlap_events": int(
                r69_decision["emergent_cross_edit_overlap_events"]
            ),
        },
    }
    runtime_contract = {
        "schema_version": "worldsim_v6.r70_runtime_contract.v1",
        "runtime_mode": "multi_actor_transform_lifecycle_owned_scene_patch",
        "actor_order": actor_ids,
        "per_actor_operation": "load_base_arrays_then_apply_float64_world_transform_then_mask_opacity_by_native_lifecycle",
        "composition_isolation": "each_transform_applies_only_to_its_owned_actor_model_index",
        "static_scene_dependency": "external_base_sceneir_and_native_checkpoint",
        "renderer_merge": "replace_owned_native_actor_fields_then_render_one_shared_scene",
    }
    validity = {
        "schema_version": "worldsim_v6.r70_validity.v1",
        "package_status": contract["package_status"],
        "individual_actor_factor_validity": {
            "actor_0000": validities["actor_0000"],
            "actor_0002": validities["actor_0002"],
        },
        "q_joint_aabb_interaction": r69_decision["q_joint_aabb_interaction"],
        "joint_new_overlap_events": int(r69_decision["joint_new_overlap_events"]),
        "q_collision_physics": "ABSTAIN",
        "semantic_road": "ABSTAIN",
        "physical_dynamics": "ABSTAIN",
        "planning_safety": "ABSTAIN",
    }

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__two-actor-package-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    package = run_dir / "package"
    repeat_package = run_dir / "repeat_package"
    package_manifest = _bake_package(
        package, actor_sources, scene_composition, runtime_contract, validity
    )
    repeat_manifest = _bake_package(
        repeat_package, actor_sources, scene_composition, runtime_contract, validity
    )
    package_records = _tree_records(package, include_manifest=True)
    repeat_records = _tree_records(repeat_package, include_manifest=True)
    copied_tree_exact = {
        actor_id: _tree_records(package / "actors" / actor_id, include_manifest=True)
        == source_tree_records[actor_id]
        for actor_id in actor_ids
    }
    payload_count = len(package_manifest["files"])
    combined_primitives = sum(int(row["primitive_count"]) for row in scene_actors)
    actor_binding_exact = bool(
        [row["actor_id"] for row in scene_actors] == actor_ids
        and [row["actor_model_index"] for row in scene_actors]
        == list(contract["expected_actor_model_indices"])
        and [row["proposal_id"] for row in scene_actors]
        == list(contract["expected_proposal_ids"])
        and all(
            row["translation_delta_m"] == contract["expected_translations_m"][row["actor_id"]]
            for row in scene_actors
        )
    )
    wall_seconds = time.monotonic() - started
    checks = {
        "r69_r50_r67_authorities_accepted": bool(
            r69_gate["checks"]["passed"]
            and json.loads((r50_run / "R50_GATE.json").read_text(encoding="utf-8"))["checks"]["passed"]
            and json.loads((r67_run / "R67_GATE.json").read_text(encoding="utf-8"))["checks"]["passed"]
        ),
        "two_actor_proposal_transform_lifecycle_bindings_exact": actor_binding_exact,
        "actor_denominator_exact": len(scene_actors) == int(contract["expected_actor_count"]),
        "trajectory_denominators_exact": all(
            row["trajectory_rows"] == int(contract["expected_trajectory_rows_per_actor"])
            for row in scene_actors
        ),
        "combined_primitive_denominator_exact": combined_primitives
        == int(contract["expected_combined_primitives"]),
        "source_file_denominators_exact": all(
            len(source_tree_records[actor_id])
            == int(contract["expected_source_files_per_actor_including_manifest"])
            for actor_id in actor_ids
        ),
        "both_actor_package_trees_copied_byte_exact": all(copied_tree_exact.values()),
        "scene_package_payload_denominator_exact": payload_count
        == int(contract["expected_package_payload_files_excluding_scene_manifest"]),
        "joint_zero_new_overlap_authority_bound": bool(
            r69_decision["q_joint_aabb_interaction"] == "ACCEPT"
            and int(r69_decision["joint_new_overlap_events"]) == 0
            and int(r69_decision["emergent_cross_edit_overlap_events"]) == 0
        ),
        "typed_abstentions_preserved": bool(
            validity["q_collision_physics"] == "ABSTAIN"
            and validity["semantic_road"] == "ABSTAIN"
            and validity["physical_dynamics"] == "ABSTAIN"
            and validity["planning_safety"] == "ABSTAIN"
        ),
        "scene_package_manifest_self_consistent": package_manifest
        == _verify_scene_package(package),
        "repeat_bake_byte_exact": package_records == repeat_records
        and package_manifest == repeat_manifest,
        "frozen_sources_immutable": bool(
            all(_sha256(path) == expected_sha for path, expected_sha in frozen_files.items())
            and r50_manifest == _verify_package(r50_package)
            and r67_manifest == _verify_package(r67_package)
        ),
        "wall_within_budget": wall_seconds <= float(resources["maximum_wall_seconds"]),
        "training_and_rendering_not_started": True,
        "confirmation_not_read": True,
    }
    checks["passed"] = all(checks.values())
    _write_json(
        run_dir / "R70_GATE.json",
        {
            "schema_version": "worldsim_v6.r70_gate.v1",
            "checks": checks,
            "decision": "accept_two_actor_scene_edit_package_bake"
            if checks["passed"]
            else "reject_or_repair_two_actor_scene_edit_package_bake",
        },
    )
    _write_json(
        run_dir / "RESOURCE_AUDIT.json",
        {
            "schema_version": "worldsim_v6.r70_resource_audit.v1",
            "gpu_used": False,
            "wall_seconds": wall_seconds,
            "disk_free_gib_at_start": free_gib,
            "package_bytes": sum(row["bytes"] for row in package_records.values()),
            "repeat_package_bytes": sum(row["bytes"] for row in repeat_records.values()),
            "training_started": False,
            "rendering_started": False,
            "confirmation_content_read": False,
        },
    )
    summary = {
        "schema_version": "worldsim_v6.r70_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_development_two_actor_scene_edit_package"
        if checks["passed"]
        else "rejected",
        "source_commit": source_commit,
        "actor_ids": actor_ids,
        "actor_model_indices": [row["actor_model_index"] for row in scene_actors],
        "proposal_ids": [row["proposal_id"] for row in scene_actors],
        "combined_primitives": combined_primitives,
        "combined_trajectory_rows": sum(row["trajectory_rows"] for row in scene_actors),
        "scene_package_manifest_sha256": _sha256(package / "SCENE_PACKAGE_MANIFEST.json"),
        "scene_composition_sha256": _sha256(package / "SCENE_COMPOSITION.json"),
        "runtime_contract_sha256": _sha256(package / "RUNTIME_CONTRACT.json"),
        "validity_sha256": _sha256(package / "VALIDITY.json"),
        "payload_file_count": payload_count,
        "repeat_bake_byte_exact": package_records == repeat_records,
        "base_sceneir_ownership": "frozen_external_dependency",
        "q_joint_aabb_interaction": "ACCEPT",
        "q_collision_physics": "ABSTAIN",
        "semantic_road": "ABSTAIN",
        "physical_dynamics": "ABSTAIN",
        "planning_safety": "ABSTAIN",
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    tracked = [
        "R70_GATE.json",
        "SUMMARY.json",
        "RESOURCE_AUDIT.json",
        "package/SCENE_PACKAGE_MANIFEST.json",
        "package/SCENE_COMPOSITION.json",
        "package/RUNTIME_CONTRACT.json",
        "package/VALIDITY.json",
        "repeat_package/SCENE_PACKAGE_MANIFEST.json",
    ]
    _write_json(
        run_dir / "MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r70_manifest.v1",
            "files": {
                name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)}
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


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/worldsim_v6/r70_two_actor_scene_package_bake_v1.yaml"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
