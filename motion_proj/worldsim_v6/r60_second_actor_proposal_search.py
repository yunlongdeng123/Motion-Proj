"""WorldSim V6 R60：在冻结 x/z 网格搜索 actor2 interaction-safe translation。"""

from __future__ import annotations

import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from motion_proj.worldsim_v6.r12_dynamic_logsim import _replay_once
from motion_proj.worldsim_v6.r38_actor_interaction_factor import _compile_intervention, _content_sha256
from motion_proj.worldsim_v6.r44_verified_actor_proposal_bake import _git, _resolve_runs_uri, _sha256, _verify, _write_json


TASK_ID = "WS-V6-R60-SECOND-ACTOR-PROPOSAL-SEARCH-01"


class R60ExperimentError(RuntimeError):
    """R60 正式实验合同失败。"""


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R60ExperimentError("正式 R60 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R60ExperimentError("R60 task_id 漂移")
    sources = config["sources"]
    r59_run = _resolve_runs_uri(sources["r59_run"])
    r58_run = _resolve_runs_uri(sources["r58_run"])
    binding_run = _resolve_runs_uri(sources["sceneir_binding_run"])
    base_package = binding_run / sources["base_sceneir_package"]
    frozen_files = {
        r59_run / "MANIFEST.json": sources["r59_manifest_sha256"], r59_run / "R59_GATE.json": sources["r59_gate_sha256"],
        r59_run / "SUMMARY.json": sources["r59_summary_sha256"], r59_run / "INTERACTION_DECISION.json": sources["r59_decision_sha256"],
        r58_run / "R58_GATE.json": sources["r58_gate_sha256"], binding_run / "MANIFEST.json": sources["sceneir_binding_manifest_sha256"],
        binding_run / "R13_SCENEIR_SENSOR_BINDING_GATE.json": sources["sceneir_binding_gate_sha256"], base_package / "MANIFEST.json": sources["base_sceneir_manifest_sha256"],
        base_package / "sceneir.json": sources["base_sceneir_document_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    r59_gate = json.loads((r59_run / "R59_GATE.json").read_text(encoding="utf-8"))
    r59_decision = json.loads((r59_run / "INTERACTION_DECISION.json").read_text(encoding="utf-8"))
    r58_gate = json.loads((r58_run / "R58_GATE.json").read_text(encoding="utf-8"))
    binding_gate = json.loads((binding_run / "R13_SCENEIR_SENSOR_BINDING_GATE.json").read_text(encoding="utf-8"))
    base_manifest = json.loads((base_package / "MANIFEST.json").read_text(encoding="utf-8"))
    package_files = {base_package / relative: row["sha256"] for relative, row in base_manifest["files"].items()}
    for path, expected_sha in package_files.items():
        _verify(path, expected_sha)
    if shutil.disk_usage(run_root).free / (1024**3) < float(config["resources"]["minimum_disk_free_gib"]):
        raise R60ExperimentError("R60 磁盘资源不足")
    base = _replay_once(base_package, 1)
    cohort = config["cohort"]
    target_id = str(cohort["actor_id"])
    target_states = sorted((row for row in base["actor_states"] if row["actor_id"] == target_id), key=lambda row: int(row["timestamp_us"]))
    actor_ids = sorted({row["actor_id"] for row in base["actor_states"]})
    base_positions = np.asarray([row["centroid_world_m"] for row in target_states], dtype=np.float64)
    base_collision_keys = {(int(row["timestamp_us"]), tuple(row["actor_pair"])) for row in base["collision_labels"] if row["aabb_overlap"] and target_id in row["actor_pair"]}
    search = config["search"]
    candidates = [(float(x), float(z)) for x in search["x_values_m"] for z in search["z_values_m"] if not (bool(search["exclude_zero_translation"]) and float(x) == 0.0 and float(z) == 0.0)]
    dt = float(cohort["timestamp_step_seconds"])
    tolerance = float(config["thresholds"]["maximum_kinematic_invariance_error"])
    rows = []
    for x, z in candidates:
        delta = np.asarray([x, 0.0, z], dtype=np.float64)
        states, collisions = _compile_intervention(base["actor_states"], target_id, delta)
        edited_target = sorted((row for row in states if row["actor_id"] == target_id), key=lambda row: int(row["timestamp_us"]))
        edited_positions = np.asarray([row["centroid_world_m"] for row in edited_target], dtype=np.float64)
        velocity_error = float(np.max(np.abs(np.diff(base_positions, axis=0) / dt - np.diff(edited_positions, axis=0) / dt)))
        acceleration_error = float(np.max(np.abs(np.diff(np.diff(base_positions, axis=0) / dt, axis=0) / dt - np.diff(np.diff(edited_positions, axis=0) / dt, axis=0) / dt)))
        edited_keys = {(int(row["timestamp_us"]), tuple(row["actor_pair"])) for row in collisions if row["aabb_overlap"] and target_id in row["actor_pair"]}
        new_keys = edited_keys - base_collision_keys
        norm = float(np.linalg.norm(delta))
        accept = velocity_error <= tolerance and acceleration_error <= tolerance and not new_keys and norm >= float(search["minimum_translation_norm_m"])
        rows.append({
            "proposal_id": f"actor2_translate_x_{x:+.1f}_z_{z:+.1f}", "translation_delta_m": delta.tolist(), "translation_norm_m": norm,
            "q_self_kinematics": "ACCEPT" if velocity_error <= tolerance and acceleration_error <= tolerance else "REJECT",
            "q_aabb_interaction": "ACCEPT" if not new_keys else "REJECT", "new_overlap_events": len(new_keys), "removed_overlap_events": len(base_collision_keys - edited_keys),
            "maximum_velocity_invariance_error": velocity_error, "maximum_acceleration_invariance_error": acceleration_error,
            "joint_decision": "ACCEPT" if accept else "REJECT", "contact_road_physical_safety": "ABSTAIN",
        })
    accepted = [row for row in rows if row["joint_decision"] == "ACCEPT"]
    anchor = np.asarray(search["rejected_anchor_delta_m"], dtype=np.float64)
    for row in accepted:
        delta = np.asarray(row["translation_delta_m"], dtype=np.float64)
        row["selection_squared_distance_to_rejected_anchor"] = float(np.sum((delta - anchor) ** 2))
    selected = sorted(accepted, key=lambda row: (row["selection_squared_distance_to_rejected_anchor"], row["translation_delta_m"][0], row["translation_delta_m"][2]))[0] if accepted else None
    anchor_row = next(row for row in rows if row["translation_delta_m"] == search["rejected_anchor_delta_m"])
    selected_repeat_exact = False
    if selected is not None:
        delta = np.asarray(selected["translation_delta_m"], dtype=np.float64)
        states1, collisions1 = _compile_intervention(base["actor_states"], target_id, delta)
        states2, collisions2 = _compile_intervention(base["actor_states"], target_id, delta)
        selected_repeat_exact = _content_sha256({"states": states1, "collisions": collisions1}) == _content_sha256({"states": states2, "collisions": collisions2})
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__actor2-proposal-search-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_jsonl(run_dir / "PROPOSAL_CATALOG.jsonl", rows)
    _write_json(run_dir / "SELECTED_PROPOSAL.json", {"schema_version": "worldsim_v6.r60_selected_proposal.v1", "selection_rule": search["selection_rule"], "selected": selected})
    checks = {
        "r58_accepted_and_r59_rejected_preserved": bool(r58_gate["checks"]["passed"] and not r59_gate["checks"]["passed"] and r59_decision["new_overlap_events"] == 7),
        "sceneir_binding_accepted": bool(binding_gate["checks"]["passed"]),
        "actor_and_trajectory_denominators_exact": len(actor_ids) == int(cohort["expected_actor_count"]) and len(target_states) == int(cohort["expected_trajectory_rows"]),
        "candidate_denominator_exact": len(rows) == int(search["expected_candidate_count"]),
        "rejected_anchor_reproduced": anchor_row["joint_decision"] == "REJECT" and anchor_row["new_overlap_events"] == 7,
        "all_self_kinematics_accept": all(row["q_self_kinematics"] == "ACCEPT" for row in rows),
        "at_least_one_zero_new_overlap_candidate": bool(accepted),
        "selected_candidate_zero_new_overlap": selected is not None and selected["new_overlap_events"] == 0 and selected["joint_decision"] == "ACCEPT",
        "selected_differs_from_rejected_anchor": selected is not None and selected["translation_delta_m"] != search["rejected_anchor_delta_m"],
        "selected_compile_repeat_exact": selected_repeat_exact,
        "contact_road_physical_and_safety_abstain": True,
        "source_immutable": all(_sha256(path) == expected_sha for path, expected_sha in frozen_files.items()) and all(_sha256(path) == expected_sha for path, expected_sha in package_files.items()),
        "wall_within_budget": (time.monotonic() - started) <= float(config["resources"]["maximum_wall_seconds"]),
        "training_not_started": True, "confirmation_not_read": True,
    }
    checks["passed"] = all(checks.values())
    _write_json(run_dir / "R60_GATE.json", {"schema_version": "worldsim_v6.r60_gate.v1", "checks": checks, "decision": "accept_factor_aware_second_actor_proposal_search" if checks["passed"] else "reject_second_actor_proposal_search"})
    _write_json(run_dir / "SUMMARY.json", {
        "schema_version": "worldsim_v6.r60_summary.v1", "task_id": TASK_ID, "hypothesis_id": config["hypothesis_id"], "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_development_factor_aware_second_actor_proposal_search" if checks["passed"] else "rejected", "source_commit": source_commit,
        "candidate_count": len(rows), "accepted_candidate_count": len(accepted), "rejected_candidate_count": len(rows) - len(accepted),
        "selected_proposal_id": selected["proposal_id"] if selected else None, "selected_translation_delta_m": selected["translation_delta_m"] if selected else None,
        "selected_distance_to_rejected_anchor_m": math.sqrt(selected["selection_squared_distance_to_rejected_anchor"]) if selected else None,
        "physical_trajectory_validity": "ABSTAIN", "claim_boundary": config["claim_boundary"],
    })
    tracked = ["R60_GATE.json", "SUMMARY.json", "PROPOSAL_CATALOG.jsonl", "SELECTED_PROPOSAL.json"]
    _write_json(run_dir / "MANIFEST.json", {"schema_version": "worldsim_v6.r60_manifest.v1", "files": {name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)} for name in tracked}})
    _write_json(run_dir / "TERMINAL.json", {"schema_version": "worldsim_v6.terminal.v1", "status": "done" if checks["passed"] else "rejected", "manifest_sha256": _sha256(run_dir / "MANIFEST.json"), "summary_sha256": _sha256(run_dir / "SUMMARY.json")})
    print(str(run_dir), flush=True)
    return run_dir


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/r60_second_actor_proposal_search_v1.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
