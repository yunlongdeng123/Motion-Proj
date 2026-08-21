"""WorldSim V6 R51：验证 geometry/transform/lifecycle 三 owner 的五帧 native sensor runtime。"""

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


TASK_ID = "WS-V6-R51-LIFECYCLE-AWARE-SENSOR-RUNTIME-01"


class R51ExperimentError(RuntimeError):
    """R51 正式实验合同失败。"""


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _rows_pass(rows: list[dict[str, Any]], thresholds: dict[str, Any]) -> bool:
    for row in rows:
        errors = row["native_actor_field_max_error"]
        if errors["means_m"] > float(thresholds["maximum_means_error_m"]) or errors["quaternions_wxyz"] > float(thresholds["maximum_quaternion_error"]):
            return False
        if any(errors[key] > float(thresholds["maximum_static_field_error"]) for key in ("scales_m", "opacities", "view_dependent_rgb")):
            return False
        if row["full_sensor_rgb_mae"] > float(thresholds["maximum_rgb_mae"]) or row["full_sensor_rgb_p99_absolute_error"] > float(thresholds["maximum_rgb_p99_absolute_error"]):
            return False
        if row["full_sensor_depth_mae_m"] > float(thresholds["maximum_depth_mae_m"]) or row["full_sensor_opacity_mae"] > float(thresholds["maximum_opacity_mae"]):
            return False
    return True


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R51ExperimentError("正式 R51 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R51ExperimentError("R51 task_id 漂移")
    sources = config["sources"]
    r50_run = _resolve_runs_uri(sources["r50_run"])
    package = r50_run / "package"
    r49_run = _resolve_runs_uri(sources["r49_run"])
    r48_run = _resolve_runs_uri(sources["r48_run"])
    r48_sensor = r48_run / sources["r48_frame57_sensor"]
    checkpoint = Path(sources["streetgs_checkpoint"])
    upstream = Path(sources["streetgs_upstream_root"])
    frozen_files = {
        r50_run / "MANIFEST.json": sources["r50_manifest_sha256"],
        r50_run / "R50_GATE.json": sources["r50_gate_sha256"],
        r50_run / "SUMMARY.json": sources["r50_summary_sha256"],
        package / "PACKAGE_MANIFEST.json": sources["r50_package_manifest_sha256"],
        package / "TRAJECTORY_GEOMETRY.json": sources["r50_trajectory_geometry_sha256"],
        package / "RUNTIME_CONTRACT.json": sources["r50_runtime_contract_sha256"],
        r49_run / "R49_GATE.json": sources["r49_gate_sha256"],
        r49_run / "SUMMARY.json": sources["r49_summary_sha256"],
        r48_run / "R48_GATE.json": sources["r48_gate_sha256"],
        r48_sensor: sources["r48_frame57_sensor_sha256"],
        checkpoint: sources["streetgs_checkpoint_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    if _git(upstream, "rev-parse", "HEAD") != sources["streetgs_upstream_commit"]:
        raise R51ExperimentError("StreetGS upstream commit 漂移")
    r50_gate = json.loads((r50_run / "R50_GATE.json").read_text(encoding="utf-8"))
    r49_gate = json.loads((r49_run / "R49_GATE.json").read_text(encoding="utf-8"))
    r48_gate = json.loads((r48_run / "R48_GATE.json").read_text(encoding="utf-8"))
    package_manifest = _verify_package(package)
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R51ExperimentError("R51 磁盘资源不足")

    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / f"{now.strftime('%Y%m%dT%H%M%SZ')}__lifecycle-sensor-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        worker_dir = run_dir / "worker"
        contract = config["runtime_contract"]
        command = [
            sources["drivestudio_python"], str(repo_root / "scripts/worldsim_v6/r36_actor_sensor_worker.py"),
            "--repo-root", str(repo_root), "--checkpoint", str(checkpoint), "--upstream-root", str(upstream),
            "--package", str(package), "--frames", ",".join(str(value) for value in contract["frame_indices"]),
            "--actor-model-index", str(contract["actor_model_index"]), "--output", str(worker_dir),
        ]
        completed = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, timeout=float(config["resources"]["maximum_worker_seconds"]))
        (run_dir / "worker.log").write_text(completed.stdout + "\n--- STDERR ---\n" + completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise R51ExperimentError(f"lifecycle-aware worker 失败：rc={completed.returncode}")
        rows = _load_rows(worker_dir / "FRAME_METRICS.jsonl")
        audit = json.loads((worker_dir / "WORKER_AUDIT.json").read_text(encoding="utf-8"))
        by_frame = {int(row["frame_index"]): row for row in rows}
        frames = [int(value) for value in contract["frame_indices"]]
        validity = [bool(by_frame[frame]["package_actor_frame_valid"]) for frame in frames]
        frame57_sensor = worker_dir / by_frame[57]["sensor_path"]
        frame57_current = np.load(frame57_sensor, allow_pickle=False)
        frame57_reference = np.load(r48_sensor, allow_pickle=False)
        frame57_arrays_exact = set(frame57_current.files) == set(frame57_reference.files) and all(np.array_equal(frame57_current[name], frame57_reference[name]) for name in frame57_current.files)
        inactive_rows = [by_frame[frame] for frame, valid in zip(frames, validity) if not valid]
        peak_mib = int(audit["peak_torch_reserved_bytes"]) / (1024**2)
        wall_seconds = time.monotonic() - started
        checks = {
            "r50_and_r48_authorities_accepted_r49_rejected": r50_gate["checks"]["passed"] and r48_gate["checks"]["passed"] and not r49_gate["checks"]["passed"],
            "five_frame_denominator_exact": sorted(by_frame) == sorted(frames),
            "runtime_ownership_modes_exact": audit["runtime_mode"] == contract["required_runtime_mode"] and audit["translation_source"] == contract["required_translation_source"] and audit["lifecycle_source"] == contract["required_lifecycle_source"],
            "translation_and_lifecycle_binding_exact": all(row["translation_delta_m"] == contract["translation_delta_m"] for row in rows) and validity == contract["expected_frame_validity"],
            "all_compiled_native_thresholds_pass": _rows_pass(rows, config["thresholds"]),
            "inactive_frames_zero_opacity_and_sensor_error": all(row["native_actor_field_max_error"]["opacities"] == 0.0 and row["full_sensor_rgb_mae"] == 0.0 and row["full_sensor_depth_mae_m"] == 0.0 and row["full_sensor_opacity_mae"] == 0.0 for row in inactive_rows),
            "active_frame57_sensor_preserved_exact": frame57_arrays_exact and _sha256(frame57_sensor) == sources["r48_frame57_sensor_sha256"],
            "known_visible_frame_effect_nontrivial": by_frame[57]["actor_effect_pixels"] >= int(contract["minimum_frame57_actor_effect_pixels"]),
            "compiled_repeats_and_native_restoration_exact": all(row["compiled_repeat_exact"] and row["native_translation_state_restored_exact"] for row in rows),
            "package_manifest_immutable": audit["package_manifest_sha256_before"] == audit["package_manifest_sha256_after"] == sources["r50_package_manifest_sha256"],
            "checkpoint_immutable": audit["checkpoint_sha256_before"] == audit["checkpoint_sha256_after"] == sources["streetgs_checkpoint_sha256"],
            "physical_and_safety_validity_abstain": True,
            "source_immutable": all(_sha256(path) == expected_sha for path, expected_sha in frozen_files.items()) and package_manifest == _verify_package(package),
            "gpu_within_budget": peak_mib <= float(config["resources"]["maximum_peak_gpu_memory_mib"]),
            "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True, "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        _write_json(run_dir / "R51_GATE.json", {
            "schema_version": "worldsim_v6.r51_gate.v1", "checks": checks,
            "decision": "accept_lifecycle_aware_actor_sensor_runtime" if checks["passed"] else "reject_or_repair_lifecycle_aware_sensor_runtime",
        })
        _write_json(run_dir / "RESOURCE_AUDIT.json", {
            "schema_version": "worldsim_v6.r51_resource_audit.v1", "gpu_used": True,
            "peak_torch_reserved_mib": peak_mib, "worker_wall_seconds": float(audit["wall_seconds"]),
            "wall_seconds": wall_seconds, "disk_free_gib_at_start": free_gib,
            "training_started": False, "confirmation_content_read": False,
        })
        summary = {
            "schema_version": "worldsim_v6.r51_summary.v1", "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"], "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_lifecycle_aware_actor_sensor_runtime" if checks["passed"] else "rejected",
            "source_commit": source_commit, "proposal_id": contract["proposal_id"], "frame_indices": frames,
            "frame_validity": validity, "compiled_native_pass_frame_count": sum(_rows_pass([row], config["thresholds"]) for row in rows),
            "inactive_zero_error_frame_count": sum(row["full_sensor_rgb_mae"] == row["full_sensor_depth_mae_m"] == row["full_sensor_opacity_mae"] == 0.0 for row in inactive_rows),
            "frame57_sensor_preserved_exact": frame57_arrays_exact,
            "maximum_rgb_mae": max(row["full_sensor_rgb_mae"] for row in rows),
            "maximum_depth_mae_m": max(row["full_sensor_depth_mae_m"] for row in rows),
            "physical_trajectory_validity": "ABSTAIN", "safety_validity": "ABSTAIN", "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = ["R51_GATE.json", "SUMMARY.json", "RESOURCE_AUDIT.json", "worker.log", "worker/FRAME_METRICS.jsonl", "worker/WORKER_AUDIT.json"]
        tracked.extend(f"worker/{row['sensor_path']}" for row in rows)
        _write_json(run_dir / "MANIFEST.json", {
            "schema_version": "worldsim_v6.r51_manifest.v1",
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
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/r51_lifecycle_aware_sensor_runtime_v1.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

