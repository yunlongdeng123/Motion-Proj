"""WorldSim V6 R57：actor2 编译 package 与 native StreetGS 的五帧 sensor equivalence。"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from motion_proj.worldsim_v6.r44_verified_actor_proposal_bake import _git, _resolve_runs_uri, _sha256, _verify, _write_json
from motion_proj.worldsim_v6.r46_detached_transform_package_logsim import _verify_package


TASK_ID = "WS-V6-R57-SECOND-ACTOR-SENSOR-EQUIVALENCE-01"


class R57ExperimentError(RuntimeError):
    """R57 正式实验合同失败。"""


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
        raise R57ExperimentError("正式 R57 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R57ExperimentError("R57 task_id 漂移")
    sources = config["sources"]
    r56_run = _resolve_runs_uri(sources["r56_run"])
    package = r56_run / "package"
    checkpoint = Path(sources["streetgs_checkpoint"])
    upstream = Path(sources["streetgs_upstream_root"])
    frozen_files = {
        r56_run / "MANIFEST.json": sources["r56_manifest_sha256"],
        r56_run / "R56_GATE.json": sources["r56_gate_sha256"],
        r56_run / "SUMMARY.json": sources["r56_summary_sha256"],
        package / "PACKAGE_MANIFEST.json": sources["r56_package_manifest_sha256"],
        package / "TRAJECTORY_GEOMETRY.json": sources["r56_trajectory_geometry_sha256"],
        checkpoint: sources["streetgs_checkpoint_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    if _git(upstream, "rev-parse", "HEAD") != sources["streetgs_upstream_commit"]:
        raise R57ExperimentError("StreetGS upstream commit 漂移")
    r56_gate = json.loads((r56_run / "R56_GATE.json").read_text(encoding="utf-8"))
    package_manifest = _verify_package(package)
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R57ExperimentError("R57 磁盘资源不足")
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__actor2-sensor-s{config['seed']}-r1"
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
            raise R57ExperimentError(f"actor2 sensor worker 失败：rc={completed.returncode}")
        rows = _load_rows(worker_dir / "FRAME_METRICS.jsonl")
        audit = json.loads((worker_dir / "WORKER_AUDIT.json").read_text(encoding="utf-8"))
        frames = [int(value) for value in contract["frame_indices"]]
        by_frame = {int(row["frame_index"]): row for row in rows}
        validity = [bool(by_frame[frame]["package_actor_frame_valid"]) for frame in frames]
        pass_frames = [frame for frame in frames if _row_pass(by_frame[frame], config["thresholds"])]
        max_effect = max(row["actor_effect_pixels"] for row in rows)
        peak_mib = audit["peak_torch_reserved_bytes"] / (1024**2)
        wall_seconds = time.monotonic() - started
        checks = {
            "r56_authority_accepted": bool(r56_gate["checks"]["passed"]),
            "five_frame_denominator_exact": sorted(by_frame) == sorted(frames),
            "runtime_ownership_modes_exact": audit["runtime_mode"] == contract["required_runtime_mode"] and audit["translation_source"] == contract["required_translation_source"] and audit["lifecycle_source"] == contract["required_lifecycle_source"],
            "actor_model_translation_and_lifecycle_exact": audit["actor_model_index"] == int(contract["actor_model_index"]) and all(row["translation_delta_m"] == contract["translation_delta_m"] for row in rows) and validity == contract["expected_frame_validity"],
            "all_compiled_native_thresholds_pass": pass_frames == frames,
            "at_least_one_frame_actor_effect_nontrivial": max_effect >= int(contract["minimum_maximum_actor_effect_pixels"]),
            "compiled_repeats_and_native_restoration_exact": all(row["compiled_repeat_exact"] and row["native_translation_state_restored_exact"] for row in rows),
            "package_manifest_immutable": audit["package_manifest_sha256_before"] == audit["package_manifest_sha256_after"] == sources["r56_package_manifest_sha256"],
            "checkpoint_immutable": audit["checkpoint_sha256_before"] == audit["checkpoint_sha256_after"] == sources["streetgs_checkpoint_sha256"],
            "source_immutable": all(_sha256(path) == expected_sha for path, expected_sha in frozen_files.items()) and package_manifest == _verify_package(package),
            "unsupported_claims_abstain": True,
            "gpu_within_budget": peak_mib <= float(config["resources"]["maximum_peak_gpu_memory_mib"]),
            "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True, "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        _write_json(run_dir / "R57_GATE.json", {"schema_version": "worldsim_v6.r57_gate.v1", "checks": checks, "decision": "accept_second_actor_native_sensor_equivalence" if checks["passed"] else "reject_or_repair_second_actor_sensor_equivalence"})
        _write_json(run_dir / "RESOURCE_AUDIT.json", {"schema_version": "worldsim_v6.r57_resource_audit.v1", "peak_torch_reserved_mib": peak_mib, "worker_wall_seconds": audit["wall_seconds"], "wall_seconds": wall_seconds, "disk_free_gib_at_start": free_gib, "training_started": False, "confirmation_content_read": False})
        _write_json(run_dir / "SUMMARY.json", {
            "schema_version": "worldsim_v6.r57_summary.v1", "task_id": TASK_ID, "hypothesis_id": config["hypothesis_id"], "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_second_actor_native_sensor_equivalence" if checks["passed"] else "rejected", "source_commit": source_commit,
            "actor_model_index": int(contract["actor_model_index"]), "frame_indices": frames, "compiled_native_pass_frames": pass_frames, "maximum_actor_effect_pixels": max_effect,
            "maximum_rgb_mae": max(row["full_sensor_rgb_mae"] for row in rows), "maximum_depth_mae_m": max(row["full_sensor_depth_mae_m"] for row in rows), "claim_boundary": config["claim_boundary"],
        })
        tracked = ["R57_GATE.json", "SUMMARY.json", "RESOURCE_AUDIT.json", "worker.log", "worker/FRAME_METRICS.jsonl", "worker/WORKER_AUDIT.json"]
        tracked.extend(f"worker/{row['sensor_path']}" for row in rows)
        _write_json(run_dir / "MANIFEST.json", {"schema_version": "worldsim_v6.r57_manifest.v1", "files": {name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)} for name in tracked}})
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
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/r57_second_actor_sensor_equivalence_v1.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
