"""WorldSim V6 R48：让 StreetGS worker 直接消费 detached transform-owned actor package。"""

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

from motion_proj.worldsim_v6.r44_verified_actor_proposal_bake import _git, _resolve_runs_uri, _sha256, _verify, _write_json
from motion_proj.worldsim_v6.r46_detached_transform_package_logsim import _verify_package


TASK_ID = "WS-V6-R48-DETACHED-ACTOR-PACKAGE-SENSOR-RUNTIME-01"


class R48ExperimentError(RuntimeError):
    """R48 正式实验合同失败。"""


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R48ExperimentError("正式 R48 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R48ExperimentError("R48 task_id 漂移")
    sources = config["sources"]
    r47_run = _resolve_runs_uri(sources["r47_run"])
    detached_package = r47_run / sources["detached_package"]
    r43_run = _resolve_runs_uri(sources["r43_run"])
    r36_run = _resolve_runs_uri(sources["r36_run"])
    r43_sensor = r43_run / sources["r43_sensor"]
    baseline_sensor = r36_run / sources["r36_baseline_sensor"]
    checkpoint = Path(sources["streetgs_checkpoint"])
    upstream = Path(sources["streetgs_upstream_root"])
    frozen_files = {
        r47_run / "MANIFEST.json": sources["r47_manifest_sha256"],
        r47_run / "R47_GATE.json": sources["r47_gate_sha256"],
        r47_run / "SUMMARY.json": sources["r47_summary_sha256"],
        r47_run / "EVENT_TRAJECTORY.jsonl": sources["r47_event_trajectory_sha256"],
        r47_run / "IDENTITY_AUDIT.json": sources["r47_identity_audit_sha256"],
        detached_package / "PACKAGE_MANIFEST.json": sources["detached_package_manifest_sha256"],
        r43_run / "MANIFEST.json": sources["r43_manifest_sha256"],
        r43_run / "R43_GATE.json": sources["r43_gate_sha256"],
        r43_run / "VERIFIED_PROPOSAL.json": sources["r43_verified_proposal_sha256"],
        r43_run / "translate_world_x_m1p0_z_m0p5/FRAME_METRICS.jsonl": sources["r43_frame_metrics_sha256"],
        r43_sensor: sources["r43_sensor_sha256"],
        r36_run / "MANIFEST.json": sources["r36_manifest_sha256"],
        r36_run / "R36_GATE.json": sources["r36_gate_sha256"],
        baseline_sensor: sources["r36_baseline_sensor_sha256"],
        checkpoint: sources["streetgs_checkpoint_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    if _git(upstream, "rev-parse", "HEAD") != sources["streetgs_upstream_commit"]:
        raise R48ExperimentError("StreetGS upstream commit 漂移")
    r47_gate = json.loads((r47_run / "R47_GATE.json").read_text(encoding="utf-8"))
    r43_gate = json.loads((r43_run / "R43_GATE.json").read_text(encoding="utf-8"))
    r36_gate = json.loads((r36_run / "R36_GATE.json").read_text(encoding="utf-8"))
    verified = json.loads((r43_run / "VERIFIED_PROPOSAL.json").read_text(encoding="utf-8"))
    package_manifest = _verify_package(detached_package)
    runtime_contract = config["runtime_contract"]
    proposal_binding_exact = verified["proposal"]["proposal_id"] == runtime_contract["proposal_id"] and verified["proposal"]["translation_delta_m"] == runtime_contract["translation_delta_m"]
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R48ExperimentError("R48 磁盘资源不足")

    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / f"{now.strftime('%Y%m%dT%H%M%SZ')}__detached-sensor-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        worker_dir = run_dir / "worker"
        command = [
            sources["drivestudio_python"], str(repo_root / "scripts/worldsim_v6/r36_actor_sensor_worker.py"),
            "--repo-root", str(repo_root), "--checkpoint", str(checkpoint), "--upstream-root", str(upstream),
            "--package", str(detached_package), "--frames", str(runtime_contract["frame_index"]),
            "--actor-model-index", str(runtime_contract["actor_model_index"]), "--output", str(worker_dir),
        ]
        completed = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, timeout=float(config["resources"]["maximum_worker_seconds"]))
        (run_dir / "worker.log").write_text(completed.stdout + "\n--- STDERR ---\n" + completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise R48ExperimentError(f"detached package sensor worker 失败：rc={completed.returncode}")
        rows = [json.loads(line) for line in (worker_dir / "FRAME_METRICS.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(rows) != 1:
            raise R48ExperimentError("R48 必须恰有一帧")
        row = rows[0]
        audit = json.loads((worker_dir / "WORKER_AUDIT.json").read_text(encoding="utf-8"))
        sensor_path = worker_dir / row["sensor_path"]
        _verify(sensor_path, row["sensor_sha256"])
        actual_sensor = np.load(sensor_path, allow_pickle=False)
        expected_sensor = np.load(r43_sensor, allow_pickle=False)
        sensor_arrays_exact = set(actual_sensor.files) == set(expected_sensor.files) and all(np.array_equal(actual_sensor[name], expected_sensor[name]) for name in actual_sensor.files)
        baseline_rgb = np.load(baseline_sensor, allow_pickle=False)["native_rgb"].astype(np.float32)
        runtime_rgb = actual_sensor["native_rgb"].astype(np.float32)
        changed_pixels = int(np.count_nonzero(np.mean(np.abs(runtime_rgb - baseline_rgb), axis=-1) > float(config["thresholds"]["counterfactual_rgb_change_threshold"])))
        thresholds = config["thresholds"]
        actor_fields_pass = row["native_actor_field_max_error"]["means_m"] <= float(thresholds["maximum_means_error_m"]) and row["native_actor_field_max_error"]["quaternions_wxyz"] <= float(thresholds["maximum_quaternion_error"]) and all(row["native_actor_field_max_error"][key] <= float(thresholds["maximum_static_field_error"]) for key in ("scales_m", "opacities", "view_dependent_rgb"))
        sensors_pass = row["full_sensor_rgb_mae"] <= float(thresholds["maximum_rgb_mae"]) and row["full_sensor_rgb_p99_absolute_error"] <= float(thresholds["maximum_rgb_p99_absolute_error"]) and row["full_sensor_depth_mae_m"] <= float(thresholds["maximum_depth_mae_m"]) and row["full_sensor_opacity_mae"] <= float(thresholds["maximum_opacity_mae"])
        wall_seconds = time.monotonic() - started
        peak_mib = int(audit["peak_torch_reserved_bytes"]) / (1024**2)
        checks = {
            "r47_r43_r36_authorities_accepted": r47_gate["checks"]["passed"] and r43_gate["checks"]["passed"] and r36_gate["checks"]["passed"],
            "proposal_binding_exact": proposal_binding_exact and row["translation_delta_m"] == runtime_contract["translation_delta_m"],
            "direct_transform_owned_runtime_mode": audit["runtime_mode"] == runtime_contract["required_worker_runtime_mode"] and audit["translation_source"] == runtime_contract["required_translation_source"],
            "detached_package_manifest_immutable": audit["package_manifest_sha256_before"] == audit["package_manifest_sha256_after"] == sources["detached_package_manifest_sha256"],
            "counterfactual_effect_nontrivial": changed_pixels >= int(thresholds["minimum_counterfactual_changed_pixels"]),
            "actor_visible_support_nontrivial": row["actor_effect_pixels"] >= int(thresholds["minimum_actor_effect_pixels"]),
            "compiled_actor_fields_match_native": actor_fields_pass,
            "compiled_full_sensor_matches_native": sensors_pass,
            "detached_sensor_arrays_match_r43_exact": sensor_arrays_exact,
            "compiled_repeat_exact": row["compiled_repeat_exact"],
            "native_state_restored_exact": row["native_translation_state_restored_exact"],
            "checkpoint_immutable": audit["checkpoint_sha256_before"] == audit["checkpoint_sha256_after"] == sources["streetgs_checkpoint_sha256"],
            "physical_and_safety_validity_abstain": True,
            "source_immutable": all(_sha256(path) == expected_sha for path, expected_sha in frozen_files.items()) and package_manifest == _verify_package(detached_package),
            "gpu_within_budget": peak_mib <= float(config["resources"]["maximum_peak_gpu_memory_mib"]),
            "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True, "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        _write_json(run_dir / "R48_GATE.json", {
            "schema_version": "worldsim_v6.r48_gate.v1", "checks": checks,
            "decision": "accept_detached_actor_package_sensor_runtime" if checks["passed"] else "reject_or_repair_detached_actor_package_sensor_runtime",
        })
        _write_json(run_dir / "RESOURCE_AUDIT.json", {
            "schema_version": "worldsim_v6.r48_resource_audit.v1", "gpu_used": True,
            "peak_torch_reserved_mib": peak_mib, "worker_wall_seconds": float(audit["wall_seconds"]),
            "wall_seconds": wall_seconds, "disk_free_gib_at_start": free_gib,
            "training_started": False, "confirmation_content_read": False,
        })
        summary = {
            "schema_version": "worldsim_v6.r48_summary.v1", "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"], "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_detached_actor_package_sensor_runtime" if checks["passed"] else "rejected",
            "source_commit": source_commit, "proposal_id": runtime_contract["proposal_id"], "translation_delta_m": runtime_contract["translation_delta_m"],
            "counterfactual_changed_pixels": changed_pixels, "sensor_arrays_match_r43_exact": sensor_arrays_exact,
            "full_sensor_rgb_mae": row["full_sensor_rgb_mae"], "full_sensor_depth_mae_m": row["full_sensor_depth_mae_m"],
            "runtime_mode": audit["runtime_mode"], "translation_source": audit["translation_source"],
            "physical_trajectory_validity": "ABSTAIN", "safety_validity": "ABSTAIN", "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = ["R48_GATE.json", "SUMMARY.json", "RESOURCE_AUDIT.json", "worker.log", "worker/FRAME_METRICS.jsonl", "worker/WORKER_AUDIT.json", f"worker/{row['sensor_path']}"]
        _write_json(run_dir / "MANIFEST.json", {
            "schema_version": "worldsim_v6.r48_manifest.v1",
            "files": {name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)} for name in tracked},
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
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/r48_detached_actor_package_sensor_runtime_v1.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

