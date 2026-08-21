"""WorldSim V6 R77: select and render an interaction-safe actor5 translation edit."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from motion_proj.worldsim_v6.r12_dynamic_logsim import _replay_once
from motion_proj.worldsim_v6.r38_actor_interaction_factor import _compile_intervention, _content_sha256
from motion_proj.worldsim_v6.r44_verified_actor_proposal_bake import _git, _resolve_runs_uri, _sha256, _verify, _write_json
from motion_proj.worldsim_v6.r46_detached_transform_package_logsim import _verify_package


TASK_ID = "WS-V6-R77-THIRD-ACTOR-EDIT-PROPOSAL-01"


class R77ExperimentError(RuntimeError):
    """The preregistered R77 experiment contract was violated."""


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _sensor_pass(row: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    errors = row["native_actor_field_max_error"]
    return bool(
        errors["means_m"] <= float(thresholds["maximum_means_error_m"])
        and errors["quaternions_wxyz"] <= float(thresholds["maximum_quaternion_error"])
        and all(errors[name] <= float(thresholds["maximum_static_field_error"]) for name in ("scales_m", "opacities", "view_dependent_rgb"))
        and row["full_sensor_rgb_mae"] <= float(thresholds["maximum_rgb_mae"])
        and row["full_sensor_rgb_p99_absolute_error"] <= float(thresholds["maximum_rgb_p99_absolute_error"])
        and row["full_sensor_depth_mae_m"] <= float(thresholds["maximum_depth_mae_m"])
        and row["full_sensor_opacity_mae"] <= float(thresholds["maximum_opacity_mae"])
    )


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R77ExperimentError("formal R77 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text())
    if config.get("task_id") != TASK_ID:
        raise R77ExperimentError("R77 task_id drift")
    sources = config["sources"]
    search = config["search"]
    resources = config["resources"]
    r76_run = _resolve_runs_uri(sources["r76_run"])
    r75_run = _resolve_runs_uri(sources["r75_run"])
    package = r75_run / "package"
    binding_run = _resolve_runs_uri(sources["sceneir_binding_run"])
    base_package = binding_run / sources["base_sceneir_package"]
    baseline_sensor = r76_run / sources["r76_frame57_sensor"]
    checkpoint = Path(sources["streetgs_checkpoint"])
    upstream = Path(sources["streetgs_upstream_root"])
    frozen_files = {
        r76_run / "MANIFEST.json": sources["r76_manifest_sha256"],
        r76_run / "R76_GATE.json": sources["r76_gate_sha256"],
        r76_run / "SUMMARY.json": sources["r76_summary_sha256"],
        baseline_sensor: sources["r76_frame57_sensor_sha256"],
        r75_run / "R75_GATE.json": sources["r75_gate_sha256"],
        package / "PACKAGE_MANIFEST.json": sources["r75_package_manifest_sha256"],
        binding_run / "MANIFEST.json": sources["sceneir_binding_manifest_sha256"],
        binding_run / "R13_SCENEIR_SENSOR_BINDING_GATE.json": sources["sceneir_binding_gate_sha256"],
        base_package / "MANIFEST.json": sources["base_sceneir_manifest_sha256"],
        base_package / "sceneir.json": sources["base_sceneir_document_sha256"],
        checkpoint: sources["streetgs_checkpoint_sha256"],
    }
    for path, expected in frozen_files.items():
        _verify(path, expected)
    if _git(upstream, "rev-parse", "HEAD") != sources["streetgs_upstream_commit"]:
        raise R77ExperimentError("StreetGS upstream commit drift")
    if shutil.disk_usage(run_root).free / (1024**3) < float(resources["minimum_disk_free_gib"]):
        raise R77ExperimentError("R77 disk resource insufficient")
    gates = [
        json.loads((r76_run / "R76_GATE.json").read_text()),
        json.loads((r75_run / "R75_GATE.json").read_text()),
        json.loads((binding_run / "R13_SCENEIR_SENSOR_BINDING_GATE.json").read_text()),
    ]
    package_manifest = _verify_package(package)
    base_manifest = json.loads((base_package / "MANIFEST.json").read_text())
    base_files = {base_package / relative: row["sha256"] for relative, row in base_manifest["files"].items()}
    for path, expected in base_files.items():
        _verify(path, expected)
    base = _replay_once(base_package, 1)
    target_id = str(config["cohort"]["actor_id"])
    target_states = sorted((row for row in base["actor_states"] if row["actor_id"] == target_id), key=lambda row: int(row["timestamp_us"]))
    actor_ids = sorted({row["actor_id"] for row in base["actor_states"]})
    base_positions = np.asarray([row["centroid_world_m"] for row in target_states], dtype=np.float64)
    base_collision_keys = {(int(row["timestamp_us"]), tuple(row["actor_pair"])) for row in base["collision_labels"] if row["aabb_overlap"] and target_id in row["actor_pair"]}
    candidates = [
        (float(x), float(z))
        for x in search["x_values_m"]
        for z in search["z_values_m"]
        if not (float(x) == 0.0 and float(z) == 0.0)
    ]
    dt = float(config["cohort"]["timestamp_step_seconds"])
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
        accepted = velocity_error <= tolerance and acceleration_error <= tolerance and not new_keys and norm >= float(search["minimum_translation_norm_m"])
        rows.append(
            {
                "proposal_id": f"actor5_translate_x_{x:+.1f}_z_{z:+.1f}",
                "translation_delta_m": delta.tolist(),
                "translation_norm_m": norm,
                "new_overlap_events": len(new_keys),
                "removed_overlap_events": len(base_collision_keys - edited_keys),
                "maximum_velocity_invariance_error": velocity_error,
                "maximum_acceleration_invariance_error": acceleration_error,
                "q_self_kinematics": "ACCEPT" if velocity_error <= tolerance and acceleration_error <= tolerance else "REJECT",
                "q_aabb_interaction": "ACCEPT" if not new_keys else "REJECT",
                "joint_decision": "ACCEPT" if accepted else "REJECT",
                "contact_road_physical_safety": "ABSTAIN",
            }
        )
    accepted_rows = [row for row in rows if row["joint_decision"] == "ACCEPT"]
    selected = sorted(accepted_rows, key=lambda row: (row["translation_norm_m"], row["translation_delta_m"][0], row["translation_delta_m"][2]))[0] if accepted_rows else None
    if selected is None:
        raise R77ExperimentError("no interaction-safe candidate in preregistered grid")
    selected_delta = np.asarray(selected["translation_delta_m"], dtype=np.float64)
    states1, collisions1 = _compile_intervention(base["actor_states"], target_id, selected_delta)
    states2, collisions2 = _compile_intervention(base["actor_states"], target_id, selected_delta)
    selected_repeat_exact = _content_sha256({"states": states1, "collisions": collisions1}) == _content_sha256({"states": states2, "collisions": collisions2})

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__actor5-edit-proposal-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_jsonl(run_dir / "PROPOSAL_CATALOG.jsonl", rows)
    _write_json(run_dir / "SELECTED_PROPOSAL.json", {"schema_version": "worldsim_v6.r77_selected_proposal.v1", "selection_rule": search["selection_rule"], "selected": selected})
    worker_dir = run_dir / "worker"
    delta_text = ",".join(str(float(value)) for value in selected["translation_delta_m"])
    command = [
        sources["drivestudio_python"], str(repo_root / "scripts/worldsim_v6/r36_actor_sensor_worker.py"),
        "--repo-root", str(repo_root), "--checkpoint", str(checkpoint), "--upstream-root", str(upstream),
        "--package", str(package), "--frames", str(search["sensor_frame_index"]), "--actor-model-index", str(config["cohort"]["actor_model_index"]),
        f"--translation-delta-m={delta_text}", "--output", str(worker_dir),
    ]
    completed = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, timeout=float(resources["maximum_worker_seconds"]))
    (run_dir / "worker.log").write_text(completed.stdout + "\n--- STDERR ---\n" + completed.stderr)
    if completed.returncode != 0:
        raise R77ExperimentError(f"sensor worker failed: rc={completed.returncode}")
    sensor_rows = _load_rows(worker_dir / "FRAME_METRICS.jsonl")
    if len(sensor_rows) != 1:
        raise R77ExperimentError("sensor denominator drift")
    sensor_row = sensor_rows[0]
    audit = json.loads((worker_dir / "WORKER_AUDIT.json").read_text())
    edited_sensor = worker_dir / sensor_row["sensor_path"]
    with np.load(baseline_sensor, allow_pickle=False) as baseline, np.load(edited_sensor, allow_pickle=False) as edited:
        absolute = np.abs(edited["compiled_rgb"].astype(np.float64) - baseline["compiled_rgb"].astype(np.float64))
    channel_axis = 0 if absolute.ndim == 3 and absolute.shape[0] == 3 else -1
    changed_pixels = int(np.sum(np.max(absolute, axis=channel_axis) > float(search["rgb_change_epsilon"])))
    comparison = {
        "schema_version": "worldsim_v6.r77_edit_sensor_comparison.v1",
        "frame_index": int(search["sensor_frame_index"]),
        "translation_delta_m": selected["translation_delta_m"],
        "changed_rgb_pixels_vs_logged": changed_pixels,
        "rgb_mae_vs_logged": float(absolute.mean()),
        "baseline_sensor_sha256": _sha256(baseline_sensor),
        "edited_sensor_sha256": _sha256(edited_sensor),
    }
    _write_json(run_dir / "EDIT_SENSOR_COMPARISON.json", comparison)
    wall_seconds = time.monotonic() - started
    checks = {
        "r76_r75_and_sceneir_authorities_accepted": all(bool(gate["checks"]["passed"]) for gate in gates),
        "actor_and_trajectory_denominators_exact": len(actor_ids) == int(config["cohort"]["expected_actor_count"]) and len(target_states) == int(config["cohort"]["expected_trajectory_rows"]),
        "candidate_denominator_exact": len(rows) == int(search["expected_candidate_count"]),
        "all_self_kinematics_accept": all(row["q_self_kinematics"] == "ACCEPT" for row in rows),
        "at_least_one_interaction_safe_candidate": bool(accepted_rows),
        "selected_by_preregistered_rule": selected == sorted(accepted_rows, key=lambda row: (row["translation_norm_m"], row["translation_delta_m"][0], row["translation_delta_m"][2]))[0],
        "selected_zero_new_overlap": selected["new_overlap_events"] == 0,
        "selected_compile_repeat_exact": selected_repeat_exact,
        "selected_sensor_nontrivial": changed_pixels >= int(search["minimum_changed_rgb_pixels_vs_logged"]),
        "selected_compiled_native_sensor_conformant": _sensor_pass(sensor_row, config["thresholds"]),
        "sensor_runtime_exact": sensor_row["translation_delta_m"] == selected["translation_delta_m"] and sensor_row["package_actor_frame_valid"] and sensor_row["compiled_repeat_exact"] and sensor_row["native_translation_state_restored_exact"],
        "package_checkpoint_and_sources_immutable": audit["package_manifest_sha256_before"] == audit["package_manifest_sha256_after"] == sources["r75_package_manifest_sha256"] and audit["checkpoint_sha256_before"] == audit["checkpoint_sha256_after"] == sources["streetgs_checkpoint_sha256"] and all(_sha256(path) == expected for path, expected in frozen_files.items()) and all(_sha256(path) == expected for path, expected in base_files.items()) and package_manifest == _verify_package(package),
        "contact_road_semantic_physical_planning_safety_abstain": True,
        "gpu_within_budget": audit["peak_torch_reserved_bytes"] / (1024**2) <= float(resources["maximum_peak_gpu_memory_mib"]),
        "wall_within_budget": wall_seconds <= float(resources["maximum_wall_seconds"]),
        "training_not_started": True,
        "confirmation_not_read": True,
    }
    checks["passed"] = all(checks.values())
    _write_json(run_dir / "R77_GATE.json", {"schema_version": "worldsim_v6.r77_gate.v1", "checks": checks, "decision": "accept_third_actor_edit_proposal" if checks["passed"] else "reject_or_expand_third_actor_edit_search"})
    _write_json(run_dir / "RESOURCE_AUDIT.json", {"schema_version": "worldsim_v6.r77_resource_audit.v1", "peak_torch_reserved_mib": audit["peak_torch_reserved_bytes"] / (1024**2), "worker_wall_seconds": audit["wall_seconds"], "wall_seconds": wall_seconds, "training_started": False, "confirmation_content_read": False})
    _write_json(
        run_dir / "SUMMARY.json",
        {
            "schema_version": "worldsim_v6.r77_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_third_actor_interaction_safe_visible_edit" if checks["passed"] else "rejected",
            "source_commit": source_commit,
            "candidate_count": len(rows),
            "accepted_candidate_count": len(accepted_rows),
            "selected_proposal_id": selected["proposal_id"],
            "selected_translation_delta_m": selected["translation_delta_m"],
            "selected_translation_norm_m": selected["translation_norm_m"],
            "changed_rgb_pixels_vs_logged": changed_pixels,
            "compiled_native_rgb_mae": sensor_row["full_sensor_rgb_mae"],
            "compiled_native_depth_mae_m": sensor_row["full_sensor_depth_mae_m"],
            "claim_boundary": config["claim_boundary"],
        },
    )
    tracked = ["R77_GATE.json", "SUMMARY.json", "RESOURCE_AUDIT.json", "PROPOSAL_CATALOG.jsonl", "SELECTED_PROPOSAL.json", "EDIT_SENSOR_COMPARISON.json", "worker.log", "worker/FRAME_METRICS.jsonl", "worker/WORKER_AUDIT.json", f"worker/{sensor_row['sensor_path']}"]
    _write_json(run_dir / "MANIFEST.json", {"schema_version": "worldsim_v6.r77_manifest.v1", "files": {name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)} for name in tracked}})
    _write_json(run_dir / "TERMINAL.json", {"schema_version": "worldsim_v6.terminal.v1", "status": "done" if checks["passed"] else "rejected", "manifest_sha256": _sha256(run_dir / "MANIFEST.json"), "summary_sha256": _sha256(run_dir / "SUMMARY.json")})
    print(run_dir, flush=True)
    return run_dir


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/r77_third_actor_edit_proposal_v1.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
