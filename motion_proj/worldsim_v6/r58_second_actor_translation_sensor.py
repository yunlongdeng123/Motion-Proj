"""WorldSim V6 R58：actor2 counterfactual translation 的 native sensor conformance。"""

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


TASK_ID = "WS-V6-R58-SECOND-ACTOR-TRANSLATION-SENSOR-01"


class R58ExperimentError(RuntimeError):
    """R58 正式实验合同失败。"""


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _row_pass(row: dict[str, Any], thresholds: dict[str, Any]) -> bool:
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
        raise R58ExperimentError("正式 R58 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R58ExperimentError("R58 task_id 漂移")
    sources = config["sources"]
    r57_run = _resolve_runs_uri(sources["r57_run"])
    r56_run = _resolve_runs_uri(sources["r56_run"])
    package = r56_run / "package"
    baseline_sensor = r57_run / sources["r57_frame98_sensor"]
    checkpoint = Path(sources["streetgs_checkpoint"])
    upstream = Path(sources["streetgs_upstream_root"])
    frozen_files = {
        r57_run / "MANIFEST.json": sources["r57_manifest_sha256"], r57_run / "R57_GATE.json": sources["r57_gate_sha256"],
        r57_run / "SUMMARY.json": sources["r57_summary_sha256"], baseline_sensor: sources["r57_frame98_sensor_sha256"],
        r56_run / "R56_GATE.json": sources["r56_gate_sha256"], package / "PACKAGE_MANIFEST.json": sources["r56_package_manifest_sha256"],
        checkpoint: sources["streetgs_checkpoint_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    if _git(upstream, "rev-parse", "HEAD") != sources["streetgs_upstream_commit"]:
        raise R58ExperimentError("StreetGS upstream commit 漂移")
    r57_gate = json.loads((r57_run / "R57_GATE.json").read_text(encoding="utf-8"))
    r56_gate = json.loads((r56_run / "R56_GATE.json").read_text(encoding="utf-8"))
    package_manifest = _verify_package(package)
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R58ExperimentError("R58 磁盘资源不足")
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__actor2-translation-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        worker_dir = run_dir / "worker"
        intervention = config["intervention"]
        delta_text = ",".join(str(float(value)) for value in intervention["translation_delta_m"])
        command = [
            sources["drivestudio_python"], str(repo_root / "scripts/worldsim_v6/r36_actor_sensor_worker.py"),
            "--repo-root", str(repo_root), "--checkpoint", str(checkpoint), "--upstream-root", str(upstream),
            "--package", str(package), "--frames", str(intervention["frame_index"]), "--actor-model-index", str(intervention["actor_model_index"]),
            f"--translation-delta-m={delta_text}", "--output", str(worker_dir),
        ]
        completed = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, timeout=float(config["resources"]["maximum_worker_seconds"]))
        (run_dir / "worker.log").write_text(completed.stdout + "\n--- STDERR ---\n" + completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise R58ExperimentError(f"actor2 translation worker 失败：rc={completed.returncode}")
        rows = _load_rows(worker_dir / "FRAME_METRICS.jsonl")
        if len(rows) != 1:
            raise R58ExperimentError("R58 worker 分母漂移")
        row = rows[0]
        audit = json.loads((worker_dir / "WORKER_AUDIT.json").read_text(encoding="utf-8"))
        edited_sensor = worker_dir / row["sensor_path"]
        with np.load(baseline_sensor, allow_pickle=False) as baseline, np.load(edited_sensor, allow_pickle=False) as edited:
            baseline_rgb = np.asarray(baseline["compiled_rgb"])
            edited_rgb = np.asarray(edited["compiled_rgb"])
        absolute = np.abs(edited_rgb.astype(np.float64) - baseline_rgb.astype(np.float64))
        channel_axis = 0 if absolute.ndim == 3 and absolute.shape[0] == 3 else -1
        changed_pixels = int(np.sum(np.max(absolute, axis=channel_axis) > float(intervention["rgb_change_epsilon"])))
        comparison = {"schema_version": "worldsim_v6.r58_intervention_comparison.v1", "frame_index": int(intervention["frame_index"]), "translation_delta_m": intervention["translation_delta_m"], "changed_rgb_pixels_vs_logged": changed_pixels, "rgb_mae_vs_logged": float(absolute.mean()), "baseline_sensor_sha256": _sha256(baseline_sensor), "edited_sensor_sha256": _sha256(edited_sensor)}
        _write_json(run_dir / "INTERVENTION_COMPARISON.json", comparison)
        peak_mib = audit["peak_torch_reserved_bytes"] / (1024**2)
        wall_seconds = time.monotonic() - started
        checks = {
            "r57_and_r56_authorities_accepted": bool(r57_gate["checks"]["passed"] and r56_gate["checks"]["passed"]),
            "frame_actor_and_translation_exact": row["frame_index"] == int(intervention["frame_index"]) and audit["actor_model_index"] == int(intervention["actor_model_index"]) and row["translation_delta_m"] == intervention["translation_delta_m"],
            "runtime_ownership_modes_exact": audit["runtime_mode"] == intervention["required_runtime_mode"] and audit["translation_source"] == intervention["required_translation_source"] and audit["lifecycle_source"] == intervention["required_lifecycle_source"],
            "package_lifecycle_exact": bool(row["package_actor_frame_valid"]) == bool(intervention["expected_frame_validity"]),
            "compiled_native_thresholds_pass": _row_pass(row, config["thresholds"]),
            "edited_sensor_nontrivial_vs_logged": changed_pixels >= int(intervention["minimum_changed_rgb_pixels_vs_logged"]),
            "compiled_repeat_and_native_restoration_exact": bool(row["compiled_repeat_exact"] and row["native_translation_state_restored_exact"]),
            "package_manifest_immutable": audit["package_manifest_sha256_before"] == audit["package_manifest_sha256_after"] == sources["r56_package_manifest_sha256"],
            "checkpoint_immutable": audit["checkpoint_sha256_before"] == audit["checkpoint_sha256_after"] == sources["streetgs_checkpoint_sha256"],
            "source_immutable": all(_sha256(path) == expected_sha for path, expected_sha in frozen_files.items()) and package_manifest == _verify_package(package),
            "edit_validity_and_unsupported_claims_abstain": True,
            "gpu_within_budget": peak_mib <= float(config["resources"]["maximum_peak_gpu_memory_mib"]),
            "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True, "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        _write_json(run_dir / "R58_GATE.json", {"schema_version": "worldsim_v6.r58_gate.v1", "checks": checks, "decision": "accept_second_actor_counterfactual_sensor_conformance" if checks["passed"] else "reject_or_repair_second_actor_counterfactual_sensor"})
        _write_json(run_dir / "RESOURCE_AUDIT.json", {"schema_version": "worldsim_v6.r58_resource_audit.v1", "peak_torch_reserved_mib": peak_mib, "worker_wall_seconds": audit["wall_seconds"], "wall_seconds": wall_seconds, "disk_free_gib_at_start": free_gib, "training_started": False, "confirmation_content_read": False})
        _write_json(run_dir / "SUMMARY.json", {
            "schema_version": "worldsim_v6.r58_summary.v1", "task_id": TASK_ID, "hypothesis_id": config["hypothesis_id"], "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_second_actor_counterfactual_sensor_conformance" if checks["passed"] else "rejected", "source_commit": source_commit,
            "actor_model_index": int(intervention["actor_model_index"]), "frame_index": int(intervention["frame_index"]), "translation_delta_m": intervention["translation_delta_m"],
            "changed_rgb_pixels_vs_logged": changed_pixels, "rgb_mae_vs_logged": comparison["rgb_mae_vs_logged"], "compiled_native_rgb_mae": row["full_sensor_rgb_mae"], "compiled_native_depth_mae_m": row["full_sensor_depth_mae_m"], "edit_validity": "ABSTAIN", "claim_boundary": config["claim_boundary"],
        })
        tracked = ["R58_GATE.json", "SUMMARY.json", "RESOURCE_AUDIT.json", "INTERVENTION_COMPARISON.json", "worker.log", "worker/FRAME_METRICS.jsonl", "worker/WORKER_AUDIT.json", f"worker/{row['sensor_path']}"]
        _write_json(run_dir / "MANIFEST.json", {"schema_version": "worldsim_v6.r58_manifest.v1", "files": {name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)} for name in tracked}})
        _write_json(run_dir / "TERMINAL.json", {"schema_version": "worldsim_v6.terminal.v1", "status": "done" if checks["passed"] else "rejected", "manifest_sha256": _sha256(run_dir / "MANIFEST.json"), "summary_sha256": _sha256(run_dir / "SUMMARY.json")})
        print(str(run_dir), flush=True)
        return run_dir
    except Exception as error:
        _write_json(run_dir / "TERMINAL.json", {"schema_version": "worldsim_v6.terminal.v1", "status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/r58_second_actor_translation_sensor_v1.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
