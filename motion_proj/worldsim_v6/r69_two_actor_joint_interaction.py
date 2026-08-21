"""WorldSim V6 R69: test joint AABB conformance of two accepted actor edits."""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from motion_proj.worldsim_v6.r12_dynamic_logsim import _replay_once
from motion_proj.worldsim_v6.r38_actor_interaction_factor import (
    _collision_rows,
    _compile_intervention,
    _content_sha256,
)
from motion_proj.worldsim_v6.r44_verified_actor_proposal_bake import (
    _git,
    _resolve_runs_uri,
    _sha256,
    _verify,
    _write_json,
)
from motion_proj.worldsim_v6.r46_detached_transform_package_logsim import _verify_package


TASK_ID = "WS-V6-R69-TWO-ACTOR-JOINT-INTERACTION-01"


class R69ExperimentError(RuntimeError):
    """The preregistered R69 experiment contract was violated."""


def _compile_joint(
    base_states: list[dict[str, Any]], edits: dict[str, np.ndarray]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    states = json.loads(json.dumps(base_states))
    for row in states:
        delta = edits.get(str(row["actor_id"]))
        if delta is None:
            continue
        for field in ("centroid_world_m", "aabb_min_world_m", "aabb_max_world_m"):
            row[field] = (np.asarray(row[field], dtype=np.float64) + delta).tolist()
    return states, _collision_rows(states)


def _overlap_keys(rows: list[dict[str, Any]]) -> set[tuple[int, tuple[str, str]]]:
    return {
        (int(row["timestamp_us"]), tuple(sorted(str(value) for value in row["actor_pair"])))
        for row in rows
        if row["aabb_overlap"]
    }


def _event_rows(keys: set[tuple[int, tuple[str, str]]]) -> list[dict[str, Any]]:
    return [
        {"timestamp_us": timestamp, "actor_pair": list(pair)}
        for timestamp, pair in sorted(keys)
    ]


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R69ExperimentError("formal R69 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R69ExperimentError("R69 task_id drift")

    sources = config["sources"]
    r50_run = _resolve_runs_uri(sources["r50_run"])
    r50_package = r50_run / "package"
    r51_run = _resolve_runs_uri(sources["r51_run"])
    r67_run = _resolve_runs_uri(sources["r67_run"])
    r67_package = r67_run / "package"
    r68_run = _resolve_runs_uri(sources["r68_run"])
    binding_run = _resolve_runs_uri(sources["sceneir_binding_run"])
    base_package = binding_run / sources["base_sceneir_package"]
    frozen_files = {
        r50_run / "MANIFEST.json": sources["r50_manifest_sha256"],
        r50_run / "R50_GATE.json": sources["r50_gate_sha256"],
        r50_run / "SUMMARY.json": sources["r50_summary_sha256"],
        r50_package / "PACKAGE_MANIFEST.json": sources["r50_package_manifest_sha256"],
        r50_package / "TRAJECTORY_GEOMETRY.json": sources["r50_trajectory_geometry_sha256"],
        r50_package / "VALIDITY.json": sources["r50_validity_sha256"],
        r51_run / "MANIFEST.json": sources["r51_manifest_sha256"],
        r51_run / "R51_GATE.json": sources["r51_gate_sha256"],
        r51_run / "SUMMARY.json": sources["r51_summary_sha256"],
        r67_run / "MANIFEST.json": sources["r67_manifest_sha256"],
        r67_run / "R67_GATE.json": sources["r67_gate_sha256"],
        r67_run / "SUMMARY.json": sources["r67_summary_sha256"],
        r67_package / "PACKAGE_MANIFEST.json": sources["r67_package_manifest_sha256"],
        r67_package / "TRAJECTORY_GEOMETRY.json": sources["r67_trajectory_geometry_sha256"],
        r67_package / "VALIDITY.json": sources["r67_validity_sha256"],
        r68_run / "MANIFEST.json": sources["r68_manifest_sha256"],
        r68_run / "R68_GATE.json": sources["r68_gate_sha256"],
        r68_run / "SUMMARY.json": sources["r68_summary_sha256"],
        binding_run / "MANIFEST.json": sources["sceneir_binding_manifest_sha256"],
        binding_run / "R13_SCENEIR_SENSOR_BINDING_GATE.json": sources["sceneir_binding_gate_sha256"],
        base_package / "MANIFEST.json": sources["base_sceneir_manifest_sha256"],
        base_package / "sceneir.json": sources["base_sceneir_document_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    base_manifest = json.loads((base_package / "MANIFEST.json").read_text(encoding="utf-8"))
    base_files = {
        base_package / relative: record["sha256"]
        for relative, record in base_manifest["files"].items()
    }
    for path, expected_sha in base_files.items():
        _verify(path, expected_sha)
    r50_package_manifest = _verify_package(r50_package)
    r67_package_manifest = _verify_package(r67_package)
    if shutil.disk_usage(run_root).free / (1024**3) < float(config["resources"]["minimum_disk_free_gib"]):
        raise R69ExperimentError("R69 disk resource insufficient")

    gates = [
        json.loads((r50_run / "R50_GATE.json").read_text(encoding="utf-8")),
        json.loads((r51_run / "R51_GATE.json").read_text(encoding="utf-8")),
        json.loads((r67_run / "R67_GATE.json").read_text(encoding="utf-8")),
        json.loads((r68_run / "R68_GATE.json").read_text(encoding="utf-8")),
        json.loads(
            (binding_run / "R13_SCENEIR_SENSOR_BINDING_GATE.json").read_text(encoding="utf-8")
        ),
    ]
    geometries = [
        json.loads((r50_package / "TRAJECTORY_GEOMETRY.json").read_text(encoding="utf-8")),
        json.loads((r67_package / "TRAJECTORY_GEOMETRY.json").read_text(encoding="utf-8")),
    ]
    validities = [
        json.loads((r50_package / "VALIDITY.json").read_text(encoding="utf-8")),
        json.loads((r67_package / "VALIDITY.json").read_text(encoding="utf-8")),
    ]
    actor_contracts = config["composition"]["actors"]
    if len(actor_contracts) != 2:
        raise R69ExperimentError("R69 requires exactly two actor edits")
    package_binding_exact = all(
        geometry["asset_id"] == contract["actor_id"]
        and int(geometry.get("actor_model_index", contract["actor_model_index"]))
        == int(contract["actor_model_index"])
        and geometry["proposal_id"] == contract["proposal_id"]
        and geometry["translation_delta_m"] == contract["translation_delta_m"]
        and validity["proposal_id"] == contract["proposal_id"]
        for geometry, validity, contract in zip(geometries, validities, actor_contracts)
    )
    individual_factor_validity = bool(
        validities[0]["q_self_kinematics"] == "ACCEPT"
        and validities[0]["q_aabb_interaction"] == "ACCEPT"
        and validities[0]["q_lidar_contact"] == "ACCEPT"
        and validities[0]["q_renderer_execution"] == "ACCEPT_CONFORMANCE"
        and validities[1]["joint_admissibility"] == "ACCEPT_CONFORMANCE"
        and all(value == "ACCEPT" for value in validities[1]["factor_decisions"].values())
    )

    base = _replay_once(base_package, 1)
    base_states = base["actor_states"]
    actor_ids = sorted({str(row["actor_id"]) for row in base_states})
    expected_rows = int(config["composition"]["expected_trajectory_rows_per_actor"])
    edits = {
        str(contract["actor_id"]): np.asarray(contract["translation_delta_m"], dtype=np.float64)
        for contract in actor_contracts
    }
    target_states = {
        actor_id: sorted(
            (row for row in base_states if str(row["actor_id"]) == actor_id),
            key=lambda row: int(row["timestamp_us"]),
        )
        for actor_id in edits
    }
    if any(len(rows) != expected_rows for rows in target_states.values()):
        raise R69ExperimentError("target trajectory denominator drift")

    base_keys = _overlap_keys(base["collision_labels"])
    individual_keys: dict[str, set[tuple[int, tuple[str, str]]]] = {}
    individual_new: dict[str, set[tuple[int, tuple[str, str]]]] = {}
    kinematic_rows = []
    dt = float(config["composition"]["timestamp_step_seconds"])
    tolerance = float(config["thresholds"]["maximum_kinematic_invariance_error"])
    for actor_id, delta in edits.items():
        states, collisions = _compile_intervention(base_states, actor_id, delta)
        keys = _overlap_keys(collisions)
        individual_keys[actor_id] = keys
        individual_new[actor_id] = keys - base_keys
        before = np.asarray([row["centroid_world_m"] for row in target_states[actor_id]], dtype=np.float64)
        after_rows = sorted(
            (row for row in states if str(row["actor_id"]) == actor_id),
            key=lambda row: int(row["timestamp_us"]),
        )
        after = np.asarray([row["centroid_world_m"] for row in after_rows], dtype=np.float64)
        velocity_error = float(np.max(np.abs(np.diff(before, axis=0) / dt - np.diff(after, axis=0) / dt)))
        acceleration_error = float(
            np.max(
                np.abs(
                    np.diff(np.diff(before, axis=0) / dt, axis=0) / dt
                    - np.diff(np.diff(after, axis=0) / dt, axis=0) / dt
                )
            )
        )
        kinematic_rows.append(
            {
                "actor_id": actor_id,
                "translation_delta_m": delta.tolist(),
                "maximum_velocity_invariance_error": velocity_error,
                "maximum_acceleration_invariance_error": acceleration_error,
                "q_self_kinematics": "ACCEPT"
                if velocity_error <= tolerance and acceleration_error <= tolerance
                else "REJECT",
                "individual_new_overlap_events": len(individual_new[actor_id]),
                "q_individual_aabb_interaction": "ACCEPT"
                if not individual_new[actor_id]
                else "REJECT",
            }
        )

    joint_states_1, joint_collisions_1 = _compile_joint(base_states, edits)
    joint_states_2, joint_collisions_2 = _compile_joint(base_states, edits)
    joint_keys = _overlap_keys(joint_collisions_1)
    joint_new = joint_keys - base_keys
    union_individual_new = set().union(*individual_new.values())
    emergent_new = joint_new - union_individual_new
    edited_pair = tuple(sorted(edits))
    baseline_pair = {key for key in base_keys if key[1] == edited_pair}
    joint_pair = {key for key in joint_keys if key[1] == edited_pair}
    new_pair = joint_pair - baseline_pair
    joint_repeat_exact = _content_sha256(
        {"states": joint_states_1, "collisions": joint_collisions_1}
    ) == _content_sha256({"states": joint_states_2, "collisions": joint_collisions_2})
    maximum_new = int(config["thresholds"]["maximum_new_overlap_events"])
    decision = {
        "schema_version": "worldsim_v6.r69_joint_interaction_decision.v1",
        "actors": kinematic_rows,
        "baseline_overlap_events": len(base_keys),
        "joint_overlap_events": len(joint_keys),
        "joint_new_overlap_events": len(joint_new),
        "joint_removed_overlap_events": len(base_keys - joint_keys),
        "emergent_cross_edit_overlap_events": len(emergent_new),
        "edited_pair_baseline_overlap_events": len(baseline_pair),
        "edited_pair_joint_overlap_events": len(joint_pair),
        "edited_pair_new_overlap_events": len(new_pair),
        "joint_new_overlap_examples": _event_rows(joint_new)[:40],
        "emergent_overlap_examples": _event_rows(emergent_new)[:40],
        "joint_repeat_exact": joint_repeat_exact,
        "q_joint_aabb_interaction": "ACCEPT" if len(joint_new) <= maximum_new else "REJECT",
        "q_emergent_cross_edit_interaction": "ACCEPT" if not emergent_new else "REJECT",
        "q_collision_physics": "ABSTAIN",
        "semantic_road": "ABSTAIN",
        "physical_dynamics": "ABSTAIN",
        "planning_safety": "ABSTAIN",
    }

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__two-actor-joint-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "JOINT_INTERACTION_DECISION.json", decision)
    wall_seconds = time.monotonic() - started
    checks = {
        "r50_r51_r67_r68_and_sceneir_authorities_accepted": all(
            bool(gate["checks"]["passed"]) for gate in gates
        ),
        "two_package_actor_proposal_bindings_exact": package_binding_exact,
        "individual_four_factor_validity_accepted": individual_factor_validity,
        "actor_and_trajectory_denominators_exact": len(actor_ids)
        == int(config["composition"]["expected_actor_count"])
        and all(len(rows) == expected_rows for rows in target_states.values()),
        "both_constant_translation_self_kinematics_accept": all(
            row["q_self_kinematics"] == "ACCEPT" for row in kinematic_rows
        ),
        "individual_aabb_acceptances_recomputed": all(
            row["q_individual_aabb_interaction"] == "ACCEPT" for row in kinematic_rows
        ),
        "joint_edit_creates_no_new_aabb_overlap": len(joint_new) <= maximum_new,
        "no_emergent_cross_edit_overlap": not emergent_new,
        "edited_actor_pair_creates_no_new_overlap": not new_pair,
        "joint_compile_repeat_exact": joint_repeat_exact,
        "collision_physics_semantic_dynamics_planning_safety_abstain": True,
        "packages_and_sceneir_sources_immutable": bool(
            all(_sha256(path) == expected_sha for path, expected_sha in frozen_files.items())
            and all(_sha256(path) == expected_sha for path, expected_sha in base_files.items())
            and r50_package_manifest == _verify_package(r50_package)
            and r67_package_manifest == _verify_package(r67_package)
        ),
        "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]),
        "training_not_started": True,
        "confirmation_not_read": True,
    }
    checks["passed"] = all(checks.values())
    _write_json(
        run_dir / "R69_GATE.json",
        {
            "schema_version": "worldsim_v6.r69_gate.v1",
            "checks": checks,
            "decision": "accept_two_actor_joint_interaction_conformance"
            if checks["passed"]
            else "reject_and_search_joint_actor_composition",
        },
    )
    _write_json(
        run_dir / "RESOURCE_AUDIT.json",
        {
            "schema_version": "worldsim_v6.r69_resource_audit.v1",
            "gpu_used": False,
            "wall_seconds": wall_seconds,
            "disk_free_gib_at_start": shutil.disk_usage(run_root).free / (1024**3),
            "training_started": False,
            "confirmation_content_read": False,
        },
    )
    summary = {
        "schema_version": "worldsim_v6.r69_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_development_two_actor_joint_interaction"
        if checks["passed"]
        else "rejected",
        "source_commit": source_commit,
        "actor_ids": list(edits),
        "translations_m": {key: value.tolist() for key, value in edits.items()},
        "individual_new_overlap_events": {
            key: len(value) for key, value in individual_new.items()
        },
        "joint_new_overlap_events": len(joint_new),
        "emergent_cross_edit_overlap_events": len(emergent_new),
        "edited_pair_new_overlap_events": len(new_pair),
        "q_joint_aabb_interaction": decision["q_joint_aabb_interaction"],
        "q_collision_physics": "ABSTAIN",
        "semantic_road": "ABSTAIN",
        "physical_dynamics": "ABSTAIN",
        "planning_safety": "ABSTAIN",
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    tracked = ["R69_GATE.json", "SUMMARY.json", "JOINT_INTERACTION_DECISION.json", "RESOURCE_AUDIT.json"]
    _write_json(
        run_dir / "MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r69_manifest.v1",
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
        default=Path("configs/worldsim_v6/r69_two_actor_joint_interaction_v1.yaml"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
