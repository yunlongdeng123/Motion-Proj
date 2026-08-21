"""WorldSim V6 R81: recompute joint AABB interaction for three accepted actor edits."""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from motion_proj.worldsim_v6.r12_dynamic_logsim import _replay_once
from motion_proj.worldsim_v6.r38_actor_interaction_factor import _compile_intervention, _content_sha256
from motion_proj.worldsim_v6.r44_verified_actor_proposal_bake import _git, _resolve_runs_uri, _sha256, _verify, _write_json
from motion_proj.worldsim_v6.r46_detached_transform_package_logsim import _verify_package
from motion_proj.worldsim_v6.r69_two_actor_joint_interaction import _compile_joint, _event_rows, _overlap_keys


TASK_ID = "WS-V6-R81-THREE-ACTOR-JOINT-INTERACTION-01"


class R81ExperimentError(RuntimeError):
    """The preregistered R81 experiment contract was violated."""


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R81ExperimentError("formal R81 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text())
    if config.get("task_id") != TASK_ID:
        raise R81ExperimentError("R81 task_id drift")
    sources = config["sources"]
    r69 = _resolve_runs_uri(sources["r69_run"])
    r79 = _resolve_runs_uri(sources["r79_run"])
    r80 = _resolve_runs_uri(sources["r80_run"])
    actor5_package = r80 / "package"
    binding = _resolve_runs_uri(sources["sceneir_binding_run"])
    base_package = binding / sources["base_sceneir_package"]
    frozen = {
        r69 / "MANIFEST.json": sources["r69_manifest_sha256"],
        r69 / "R69_GATE.json": sources["r69_gate_sha256"],
        r69 / "SUMMARY.json": sources["r69_summary_sha256"],
        r69 / "JOINT_INTERACTION_DECISION.json": sources["r69_decision_sha256"],
        r79 / "MANIFEST.json": sources["r79_manifest_sha256"],
        r79 / "R79_GATE.json": sources["r79_gate_sha256"],
        r79 / "SUMMARY.json": sources["r79_summary_sha256"],
        r79 / "FUSED_EDIT_DECISION.json": sources["r79_fused_decision_sha256"],
        r80 / "MANIFEST.json": sources["r80_manifest_sha256"],
        r80 / "R80_GATE.json": sources["r80_gate_sha256"],
        r80 / "SUMMARY.json": sources["r80_summary_sha256"],
        actor5_package / "PACKAGE_MANIFEST.json": sources["r80_package_manifest_sha256"],
        actor5_package / "TRAJECTORY_GEOMETRY.json": sources["r80_trajectory_geometry_sha256"],
        actor5_package / "VALIDITY.json": sources["r80_validity_sha256"],
        binding / "MANIFEST.json": sources["sceneir_binding_manifest_sha256"],
        binding / "R13_SCENEIR_SENSOR_BINDING_GATE.json": sources["sceneir_binding_gate_sha256"],
        base_package / "MANIFEST.json": sources["base_sceneir_manifest_sha256"],
        base_package / "sceneir.json": sources["base_sceneir_document_sha256"],
    }
    for path, expected in frozen.items():
        _verify(path, expected)
    if shutil.disk_usage(run_root).free / (1024**3) < float(config["resources"]["minimum_disk_free_gib"]):
        raise R81ExperimentError("R81 disk resource insufficient")
    base_manifest = json.loads((base_package / "MANIFEST.json").read_text())
    base_files = {base_package / name: row["sha256"] for name, row in base_manifest["files"].items()}
    for path, expected in base_files.items():
        _verify(path, expected)
    actor5_manifest = _verify_package(actor5_package)
    gates = [
        json.loads((r69 / "R69_GATE.json").read_text()),
        json.loads((r79 / "R79_GATE.json").read_text()),
        json.loads((r80 / "R80_GATE.json").read_text()),
        json.loads((binding / "R13_SCENEIR_SENSOR_BINDING_GATE.json").read_text()),
    ]
    prior_joint = json.loads((r69 / "JOINT_INTERACTION_DECISION.json").read_text())
    prior_rows = {row["actor_id"]: row for row in prior_joint["actors"]}
    actor5_geometry = json.loads((actor5_package / "TRAJECTORY_GEOMETRY.json").read_text())
    actor5_validity = json.loads((actor5_package / "VALIDITY.json").read_text())
    fused = json.loads((r79 / "FUSED_EDIT_DECISION.json").read_text())
    contracts = config["composition"]["actors"]
    if len(contracts) != 3:
        raise R81ExperimentError("R81 requires exactly three actor edits")
    first_two_exact = all(
        contract["actor_id"] in prior_rows
        and prior_rows[contract["actor_id"]]["translation_delta_m"] == contract["translation_delta_m"]
        and prior_rows[contract["actor_id"]]["q_self_kinematics"] == "ACCEPT"
        and prior_rows[contract["actor_id"]]["q_individual_aabb_interaction"] == "ACCEPT"
        for contract in contracts[:2]
    )
    third_exact = bool(
        actor5_geometry["asset_id"] == contracts[2]["actor_id"]
        and actor5_geometry["actor_model_index"] == contracts[2]["actor_model_index"]
        and actor5_geometry["proposal_id"] == contracts[2]["proposal_id"]
        and actor5_geometry["translation_delta_m"] == contracts[2]["translation_delta_m"]
        and actor5_validity["joint_admissibility"] == "ACCEPT_CONFORMANCE"
        and fused["joint_admissibility"] == "ACCEPT_CONFORMANCE"
    )
    base = _replay_once(base_package, 1)
    base_states = base["actor_states"]
    actor_ids = sorted({str(row["actor_id"]) for row in base_states})
    edits = {contract["actor_id"]: np.asarray(contract["translation_delta_m"], dtype=np.float64) for contract in contracts}
    expected_rows = int(config["composition"]["expected_trajectory_rows_per_actor"])
    target_states = {
        actor_id: sorted((row for row in base_states if str(row["actor_id"]) == actor_id), key=lambda row: int(row["timestamp_us"]))
        for actor_id in edits
    }
    if any(len(rows) != expected_rows for rows in target_states.values()):
        raise R81ExperimentError("target trajectory denominator drift")
    base_keys = _overlap_keys(base["collision_labels"])
    individual_new = {}
    kinematic_rows = []
    dt = float(config["composition"]["timestamp_step_seconds"])
    tolerance = float(config["thresholds"]["maximum_kinematic_invariance_error"])
    for actor_id, delta in edits.items():
        states, collisions = _compile_intervention(base_states, actor_id, delta)
        keys = _overlap_keys(collisions)
        individual_new[actor_id] = keys - base_keys
        before = np.asarray([row["centroid_world_m"] for row in target_states[actor_id]], dtype=np.float64)
        after_rows = sorted((row for row in states if str(row["actor_id"]) == actor_id), key=lambda row: int(row["timestamp_us"]))
        after = np.asarray([row["centroid_world_m"] for row in after_rows], dtype=np.float64)
        velocity_error = float(np.max(np.abs(np.diff(before, axis=0) / dt - np.diff(after, axis=0) / dt)))
        acceleration_error = float(np.max(np.abs(np.diff(np.diff(before, axis=0) / dt, axis=0) / dt - np.diff(np.diff(after, axis=0) / dt, axis=0) / dt)))
        kinematic_rows.append({
            "actor_id": actor_id,
            "translation_delta_m": delta.tolist(),
            "maximum_velocity_invariance_error": velocity_error,
            "maximum_acceleration_invariance_error": acceleration_error,
            "q_self_kinematics": "ACCEPT" if velocity_error <= tolerance and acceleration_error <= tolerance else "REJECT",
            "individual_new_overlap_events": len(individual_new[actor_id]),
            "q_individual_aabb_interaction": "ACCEPT" if not individual_new[actor_id] else "REJECT",
        })
    joint_states_1, joint_collisions_1 = _compile_joint(base_states, edits)
    joint_states_2, joint_collisions_2 = _compile_joint(base_states, edits)
    joint_keys = _overlap_keys(joint_collisions_1)
    joint_new = joint_keys - base_keys
    union_individual_new = set().union(*individual_new.values())
    emergent_new = joint_new - union_individual_new
    edited_ids = set(edits)
    edited_pair_new = {key for key in joint_new if set(key[1]).issubset(edited_ids)}
    repeat_exact = _content_sha256({"states": joint_states_1, "collisions": joint_collisions_1}) == _content_sha256({"states": joint_states_2, "collisions": joint_collisions_2})
    maximum_new = int(config["thresholds"]["maximum_new_overlap_events"])
    decision = {
        "schema_version": "worldsim_v6.r81_joint_interaction_decision.v1",
        "actors": kinematic_rows,
        "baseline_overlap_events": len(base_keys),
        "joint_overlap_events": len(joint_keys),
        "joint_new_overlap_events": len(joint_new),
        "joint_removed_overlap_events": len(base_keys - joint_keys),
        "emergent_cross_edit_overlap_events": len(emergent_new),
        "edited_pair_new_overlap_events": len(edited_pair_new),
        "joint_new_overlap_examples": _event_rows(joint_new)[:40],
        "emergent_overlap_examples": _event_rows(emergent_new)[:40],
        "joint_repeat_exact": repeat_exact,
        "q_joint_aabb_interaction": "ACCEPT" if len(joint_new) <= maximum_new else "REJECT",
        "q_emergent_cross_edit_interaction": "ACCEPT" if not emergent_new else "REJECT",
        "q_collision_physics": "ABSTAIN",
        "semantic_road": "ABSTAIN",
        "physical_dynamics": "ABSTAIN",
        "planning_safety": "ABSTAIN",
    }
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__three-actor-joint-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "JOINT_INTERACTION_DECISION.json", decision)
    wall = time.monotonic() - started
    checks = {
        "r69_r79_r80_and_sceneir_authorities_accepted": all(bool(gate["checks"]["passed"]) for gate in gates),
        "three_actor_proposal_bindings_exact": first_two_exact and third_exact,
        "actor_and_trajectory_denominators_exact": len(actor_ids) == int(config["composition"]["expected_actor_count"]) and all(len(rows) == expected_rows for rows in target_states.values()),
        "all_constant_translation_self_kinematics_accept": all(row["q_self_kinematics"] == "ACCEPT" for row in kinematic_rows),
        "all_individual_aabb_acceptances_recomputed": all(row["q_individual_aabb_interaction"] == "ACCEPT" for row in kinematic_rows),
        "joint_edit_creates_no_new_aabb_overlap": len(joint_new) <= maximum_new,
        "no_emergent_cross_edit_overlap": not emergent_new,
        "edited_actor_pairs_create_no_new_overlap": not edited_pair_new,
        "joint_compile_repeat_exact": repeat_exact,
        "collision_physics_semantic_dynamics_planning_safety_abstain": True,
        "packages_and_sceneir_sources_immutable": all(_sha256(path) == expected for path, expected in frozen.items()) and all(_sha256(path) == expected for path, expected in base_files.items()) and actor5_manifest == _verify_package(actor5_package),
        "wall_within_budget": wall <= float(config["resources"]["maximum_wall_seconds"]),
        "training_not_started": True,
        "confirmation_not_read": True,
    }
    checks["passed"] = all(checks.values())
    _write_json(run_dir / "R81_GATE.json", {"schema_version": "worldsim_v6.r81_gate.v1", "checks": checks, "decision": "accept_three_actor_joint_interaction_conformance" if checks["passed"] else "reject_and_search_three_actor_composition"})
    _write_json(run_dir / "RESOURCE_AUDIT.json", {"schema_version": "worldsim_v6.r81_resource_audit.v1", "gpu_used": False, "wall_seconds": wall, "disk_free_gib_at_start": shutil.disk_usage(run_root).free / (1024**3), "training_started": False, "confirmation_content_read": False})
    _write_json(run_dir / "SUMMARY.json", {
        "schema_version": "worldsim_v6.r81_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_development_three_actor_joint_interaction" if checks["passed"] else "rejected",
        "source_commit": source_commit,
        "actor_ids": list(edits),
        "translations_m": {key: value.tolist() for key, value in edits.items()},
        "individual_new_overlap_events": {key: len(value) for key, value in individual_new.items()},
        "joint_new_overlap_events": len(joint_new),
        "joint_removed_overlap_events": len(base_keys - joint_keys),
        "emergent_cross_edit_overlap_events": len(emergent_new),
        "edited_pair_new_overlap_events": len(edited_pair_new),
        "q_joint_aabb_interaction": decision["q_joint_aabb_interaction"],
        "claim_boundary": config["claim_boundary"],
    })
    tracked = ["R81_GATE.json", "SUMMARY.json", "JOINT_INTERACTION_DECISION.json", "RESOURCE_AUDIT.json"]
    _write_json(run_dir / "MANIFEST.json", {"schema_version": "worldsim_v6.r81_manifest.v1", "files": {name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)} for name in tracked}})
    _write_json(run_dir / "TERMINAL.json", {"schema_version": "worldsim_v6.terminal.v1", "status": "done" if checks["passed"] else "rejected", "manifest_sha256": _sha256(run_dir / "MANIFEST.json"), "summary_sha256": _sha256(run_dir / "SUMMARY.json")})
    print(run_dir, flush=True)
    return run_dir


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/r81_three_actor_joint_interaction_v1.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
