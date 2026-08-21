"""WorldSim V6 R59：验证 actor2 counterfactual 的 self-kinematics 与全 actor AABB interaction。"""

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


TASK_ID = "WS-V6-R59-SECOND-ACTOR-INTERACTION-FACTOR-01"


class R59ExperimentError(RuntimeError):
    """R59 正式实验合同失败。"""


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R59ExperimentError("正式 R59 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R59ExperimentError("R59 task_id 漂移")
    sources = config["sources"]
    r58_run = _resolve_runs_uri(sources["r58_run"])
    binding_run = _resolve_runs_uri(sources["sceneir_binding_run"])
    base_package = binding_run / sources["base_sceneir_package"]
    frozen_files = {
        r58_run / "MANIFEST.json": sources["r58_manifest_sha256"], r58_run / "R58_GATE.json": sources["r58_gate_sha256"],
        r58_run / "SUMMARY.json": sources["r58_summary_sha256"], r58_run / "INTERVENTION_COMPARISON.json": sources["r58_intervention_sha256"],
        binding_run / "MANIFEST.json": sources["sceneir_binding_manifest_sha256"], binding_run / "R13_SCENEIR_SENSOR_BINDING_GATE.json": sources["sceneir_binding_gate_sha256"],
        base_package / "MANIFEST.json": sources["base_sceneir_manifest_sha256"], base_package / "sceneir.json": sources["base_sceneir_document_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    r58_gate = json.loads((r58_run / "R58_GATE.json").read_text(encoding="utf-8"))
    binding_gate = json.loads((binding_run / "R13_SCENEIR_SENSOR_BINDING_GATE.json").read_text(encoding="utf-8"))
    r58_summary = json.loads((r58_run / "SUMMARY.json").read_text(encoding="utf-8"))
    base_manifest = json.loads((base_package / "MANIFEST.json").read_text(encoding="utf-8"))
    package_files = {base_package / relative: row["sha256"] for relative, row in base_manifest["files"].items()}
    for path, expected_sha in package_files.items():
        _verify(path, expected_sha)
    if shutil.disk_usage(run_root).free / (1024**3) < float(config["resources"]["minimum_disk_free_gib"]):
        raise R59ExperimentError("R59 磁盘资源不足")
    base = _replay_once(base_package, 1)
    cohort = config["cohort"]
    target_id = str(cohort["actor_id"])
    target_states = sorted((row for row in base["actor_states"] if row["actor_id"] == target_id), key=lambda row: int(row["timestamp_us"]))
    if len(target_states) != int(cohort["expected_trajectory_rows"]):
        raise R59ExperimentError("actor2 trajectory denominator 漂移")
    actor_ids = sorted({row["actor_id"] for row in base["actor_states"]})
    base_positions = np.asarray([row["centroid_world_m"] for row in target_states], dtype=np.float64)
    base_collision_keys = {(int(row["timestamp_us"]), tuple(row["actor_pair"])) for row in base["collision_labels"] if row["aabb_overlap"] and target_id in row["actor_pair"]}
    intervention = config["intervention"]
    delta = np.asarray(intervention["translation_delta_m"], dtype=np.float64)
    states1, collisions1 = _compile_intervention(base["actor_states"], target_id, delta)
    states2, collisions2 = _compile_intervention(base["actor_states"], target_id, delta)
    edited_target = sorted((row for row in states1 if row["actor_id"] == target_id), key=lambda row: int(row["timestamp_us"]))
    edited_positions = np.asarray([row["centroid_world_m"] for row in edited_target], dtype=np.float64)
    dt = float(cohort["timestamp_step_seconds"])
    velocity_error = float(np.max(np.abs(np.diff(base_positions, axis=0) / dt - np.diff(edited_positions, axis=0) / dt)))
    acceleration_error = float(np.max(np.abs(np.diff(np.diff(base_positions, axis=0) / dt, axis=0) / dt - np.diff(np.diff(edited_positions, axis=0) / dt, axis=0) / dt)))
    edited_collision_keys = {(int(row["timestamp_us"]), tuple(row["actor_pair"])) for row in collisions1 if row["aabb_overlap"] and target_id in row["actor_pair"]}
    new_collisions = sorted(edited_collision_keys - base_collision_keys)
    removed_collisions = sorted(base_collision_keys - edited_collision_keys)
    tolerance = float(config["thresholds"]["maximum_kinematic_invariance_error"])
    decision = {
        "schema_version": "worldsim_v6.r59_interaction_decision.v1", "intervention_id": intervention["id"], "actor_id": target_id,
        "translation_delta_m": delta.tolist(), "q_self_kinematics": "ACCEPT" if velocity_error <= tolerance and acceleration_error <= tolerance else "REJECT",
        "q_aabb_interaction": "ACCEPT" if not new_collisions else "REJECT", "q_contact": "ABSTAIN", "q_road_support": "ABSTAIN",
        "physical_trajectory_validity": "ABSTAIN", "baseline_target_overlap_events": len(base_collision_keys), "edited_target_overlap_events": len(edited_collision_keys),
        "new_overlap_events": len(new_collisions), "removed_overlap_events": len(removed_collisions),
        "new_overlap_examples": [{"timestamp_us": row[0], "actor_pair": list(row[1])} for row in new_collisions[:20]],
        "maximum_velocity_invariance_error": velocity_error, "maximum_acceleration_invariance_error": acceleration_error,
        "compile_repeat_exact": _content_sha256({"states": states1, "collisions": collisions1}) == _content_sha256({"states": states2, "collisions": collisions2}),
    }
    decision["joint_conformance_decision"] = "ACCEPT" if decision["q_self_kinematics"] == decision["q_aabb_interaction"] == "ACCEPT" else "REJECT"
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__actor2-interaction-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "INTERACTION_DECISION.json", decision)
    checks = {
        "r58_renderer_and_sceneir_binding_accepted": bool(r58_gate["checks"]["passed"] and binding_gate["checks"]["passed"]),
        "actor_and_intervention_bound_exact": int(r58_summary["actor_model_index"]) == int(cohort["actor_model_index"]) and r58_summary["translation_delta_m"] == intervention["translation_delta_m"],
        "complete_actor_and_trajectory_denominators": len(actor_ids) == int(cohort["expected_actor_count"]) and len(target_states) == int(cohort["expected_trajectory_rows"]),
        "constant_translation_self_kinematics_accept": decision["q_self_kinematics"] == "ACCEPT",
        "no_new_aabb_overlap_events": decision["q_aabb_interaction"] == "ACCEPT" and decision["new_overlap_events"] == 0,
        "compile_repeat_exact": bool(decision["compile_repeat_exact"]),
        "contact_road_physical_and_safety_abstain": True,
        "source_immutable": all(_sha256(path) == expected_sha for path, expected_sha in frozen_files.items()) and all(_sha256(path) == expected_sha for path, expected_sha in package_files.items()),
        "wall_within_budget": (time.monotonic() - started) <= float(config["resources"]["maximum_wall_seconds"]),
        "training_not_started": True, "confirmation_not_read": True,
    }
    checks["passed"] = all(checks.values())
    _write_json(run_dir / "R59_GATE.json", {"schema_version": "worldsim_v6.r59_gate.v1", "checks": checks, "decision": "accept_second_actor_interaction_factor" if checks["passed"] else "reject_second_actor_interaction_factor"})
    _write_json(run_dir / "SUMMARY.json", {
        "schema_version": "worldsim_v6.r59_summary.v1", "task_id": TASK_ID, "hypothesis_id": config["hypothesis_id"], "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_development_second_actor_interaction_factor" if checks["passed"] else "rejected", "source_commit": source_commit,
        "actor_count": len(actor_ids), "trajectory_rows": len(target_states), "new_overlap_events": len(new_collisions), "removed_overlap_events": len(removed_collisions),
        "q_self_kinematics": decision["q_self_kinematics"], "q_aabb_interaction": decision["q_aabb_interaction"], "joint_conformance_decision": decision["joint_conformance_decision"], "physical_trajectory_validity": "ABSTAIN", "claim_boundary": config["claim_boundary"],
    })
    tracked = ["R59_GATE.json", "SUMMARY.json", "INTERACTION_DECISION.json"]
    _write_json(run_dir / "MANIFEST.json", {"schema_version": "worldsim_v6.r59_manifest.v1", "files": {name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)} for name in tracked}})
    _write_json(run_dir / "TERMINAL.json", {"schema_version": "worldsim_v6.terminal.v1", "status": "done" if checks["passed"] else "rejected", "manifest_sha256": _sha256(run_dir / "MANIFEST.json"), "summary_sha256": _sha256(run_dir / "SUMMARY.json")})
    print(str(run_dir), flush=True)
    return run_dir


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/r59_second_actor_interaction_factor_v1.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
