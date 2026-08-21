"""WorldSim V6 R49：跨五个时间点对照 detached package 与 legacy actor sensor runtime。"""

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


TASK_ID = "WS-V6-R49-MULTIFRAME-DETACHED-SENSOR-RUNTIME-01"


class R49ExperimentError(RuntimeError):
    """R49 正式实验合同失败。"""


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _run_worker(
    name: str,
    package: Path,
    delta: list[float] | None,
    repo_root: Path,
    run_dir: Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources = config["sources"]
    contract = config["runtime_contract"]
    output = run_dir / name
    command = [
        sources["drivestudio_python"], str(repo_root / "scripts/worldsim_v6/r36_actor_sensor_worker.py"),
        "--repo-root", str(repo_root), "--checkpoint", sources["streetgs_checkpoint"],
        "--upstream-root", sources["streetgs_upstream_root"], "--package", str(package),
        "--frames", ",".join(str(value) for value in contract["frame_indices"]),
        "--actor-model-index", str(contract["actor_model_index"]), "--output", str(output),
    ]
    if delta is not None:
        command.append(f"--translation-delta-m={','.join(str(value) for value in delta)}")
    completed = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, timeout=float(config["resources"]["maximum_worker_seconds"]))
    (run_dir / f"{name}.log").write_text(completed.stdout + "\n--- STDERR ---\n" + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise R49ExperimentError(f"{name} worker 失败：rc={completed.returncode}")
    return _load_rows(output / "FRAME_METRICS.jsonl"), json.loads((output / "WORKER_AUDIT.json").read_text(encoding="utf-8"))


def _sensor_arrays_exact(left: Path, right: Path) -> bool:
    left_data = np.load(left, allow_pickle=False)
    right_data = np.load(right, allow_pickle=False)
    return set(left_data.files) == set(right_data.files) and all(np.array_equal(left_data[name], right_data[name]) for name in left_data.files)


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
        raise R49ExperimentError("正式 R49 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R49ExperimentError("R49 task_id 漂移")
    sources = config["sources"]
    r48_run = _resolve_runs_uri(sources["r48_run"])
    r47_run = _resolve_runs_uri(sources["r47_run"])
    detached_package = r47_run / sources["detached_package"]
    r35_run = _resolve_runs_uri(sources["r35_run"])
    legacy_package = r35_run / "package"
    checkpoint = Path(sources["streetgs_checkpoint"])
    upstream = Path(sources["streetgs_upstream_root"])
    frozen_files = {
        r48_run / "MANIFEST.json": sources["r48_manifest_sha256"],
        r48_run / "R48_GATE.json": sources["r48_gate_sha256"],
        r48_run / "SUMMARY.json": sources["r48_summary_sha256"],
        r47_run / "R47_GATE.json": sources["r47_gate_sha256"],
        r47_run / "EVENT_TRAJECTORY.jsonl": sources["r47_event_trajectory_sha256"],
        detached_package / "PACKAGE_MANIFEST.json": sources["detached_package_manifest_sha256"],
        r35_run / "MANIFEST.json": sources["r35_manifest_sha256"],
        legacy_package / "PACKAGE_MANIFEST.json": sources["r35_package_manifest_sha256"],
        checkpoint: sources["streetgs_checkpoint_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    if _git(upstream, "rev-parse", "HEAD") != sources["streetgs_upstream_commit"]:
        raise R49ExperimentError("StreetGS upstream commit 漂移")
    r48_gate = json.loads((r48_run / "R48_GATE.json").read_text(encoding="utf-8"))
    r47_gate = json.loads((r47_run / "R47_GATE.json").read_text(encoding="utf-8"))
    detached_manifest = _verify_package(detached_package)
    legacy_manifest = _verify_package(legacy_package)
    events = {int(row["sequence_index"]): row for row in _load_rows(r47_run / "EVENT_TRAJECTORY.jsonl")}
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R49ExperimentError("R49 磁盘资源不足")

    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / f"{now.strftime('%Y%m%dT%H%M%SZ')}__multiframe-sensor-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        direct_rows, direct_audit = _run_worker("direct_detached", detached_package, None, repo_root, run_dir, config)
        legacy_rows, legacy_audit = _run_worker("legacy_r35_delta", legacy_package, config["runtime_contract"]["translation_delta_m"], repo_root, run_dir, config)
        direct_by_frame = {int(row["frame_index"]): row for row in direct_rows}
        legacy_by_frame = {int(row["frame_index"]): row for row in legacy_rows}
        frames = [int(value) for value in config["runtime_contract"]["frame_indices"]]
        comparisons = []
        for frame in frames:
            direct_sensor = run_dir / "direct_detached" / direct_by_frame[frame]["sensor_path"]
            legacy_sensor = run_dir / "legacy_r35_delta" / legacy_by_frame[frame]["sensor_path"]
            comparisons.append({
                "frame_index": frame, "timestamp_us": frame * 100000,
                "sensor_arrays_exact": _sensor_arrays_exact(direct_sensor, legacy_sensor),
                "direct_sensor_sha256": _sha256(direct_sensor), "legacy_sensor_sha256": _sha256(legacy_sensor),
                "direct_event_state_sha256": events[frame]["materialized_state_sha256"],
                "direct_event_sha256": events[frame]["trajectory_event_sha256"],
                "actor_effect_pixels": int(direct_by_frame[frame]["actor_effect_pixels"]),
            })
        (run_dir / "FRAME_COMPARISONS.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in comparisons), encoding="utf-8"
        )
        contract = config["runtime_contract"]
        pre = events[int(contract["pre_stationary_frame"])]
        boundary = events[int(contract["stationary_boundary_frame"])]
        tail = events[int(contract["stationary_tail_frame"])]
        peak_mib = max(int(direct_audit["peak_torch_reserved_bytes"]), int(legacy_audit["peak_torch_reserved_bytes"])) / (1024**2)
        wall_seconds = time.monotonic() - started
        checks = {
            "r48_and_r47_authorities_accepted": r48_gate["checks"]["passed"] and r47_gate["checks"]["passed"],
            "five_frame_denominator_exact": sorted(direct_by_frame) == sorted(legacy_by_frame) == sorted(frames),
            "runtime_modes_exact": direct_audit["runtime_mode"] == "transform_owned_package_direct" and direct_audit["translation_source"] == "package_transform_trajectory" and legacy_audit["runtime_mode"] == "legacy_materialized_geometry_with_cli_translation" and legacy_audit["translation_source"] == "cli_argument",
            "translation_binding_exact": all(row["translation_delta_m"] == contract["translation_delta_m"] for row in direct_rows + legacy_rows),
            "all_sensor_arrays_cross_path_exact": all(row["sensor_arrays_exact"] and row["direct_sensor_sha256"] == row["legacy_sensor_sha256"] for row in comparisons),
            "all_compiled_native_thresholds_pass": _rows_pass(direct_rows, config["thresholds"]) and _rows_pass(legacy_rows, config["thresholds"]),
            "all_compiled_repeats_and_state_restorations_exact": all(row["compiled_repeat_exact"] and row["native_translation_state_restored_exact"] for row in direct_rows + legacy_rows),
            "known_visible_frame_effect_nontrivial": direct_by_frame[57]["actor_effect_pixels"] >= int(contract["minimum_frame57_actor_effect_pixels"]),
            "stationary_state_and_event_identity_relation_preserved": pre["materialized_state_sha256"] != boundary["materialized_state_sha256"] and boundary["materialized_state_sha256"] == tail["materialized_state_sha256"] and boundary["trajectory_event_sha256"] != tail["trajectory_event_sha256"],
            "both_packages_immutable": direct_audit["package_manifest_sha256_before"] == direct_audit["package_manifest_sha256_after"] == sources["detached_package_manifest_sha256"] and legacy_audit["package_manifest_sha256_before"] == legacy_audit["package_manifest_sha256_after"] == sources["r35_package_manifest_sha256"],
            "checkpoint_immutable_both_workers": all(audit["checkpoint_sha256_before"] == audit["checkpoint_sha256_after"] == sources["streetgs_checkpoint_sha256"] for audit in [direct_audit, legacy_audit]),
            "physical_and_safety_validity_abstain": True,
            "source_immutable": all(_sha256(path) == expected_sha for path, expected_sha in frozen_files.items()) and detached_manifest == _verify_package(detached_package) and legacy_manifest == _verify_package(legacy_package),
            "gpu_within_budget": peak_mib <= float(config["resources"]["maximum_peak_gpu_memory_mib"]),
            "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True, "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        _write_json(run_dir / "R49_GATE.json", {
            "schema_version": "worldsim_v6.r49_gate.v1", "checks": checks,
            "decision": "accept_multiframe_detached_actor_sensor_runtime" if checks["passed"] else "reject_or_repair_multiframe_detached_sensor_runtime",
        })
        _write_json(run_dir / "RESOURCE_AUDIT.json", {
            "schema_version": "worldsim_v6.r49_resource_audit.v1", "gpu_used": True,
            "peak_torch_reserved_mib": peak_mib, "worker_wall_seconds_sum": float(direct_audit["wall_seconds"]) + float(legacy_audit["wall_seconds"]),
            "wall_seconds": wall_seconds, "disk_free_gib_at_start": free_gib,
            "training_started": False, "confirmation_content_read": False,
        })
        summary = {
            "schema_version": "worldsim_v6.r49_summary.v1", "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"], "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_multiframe_detached_actor_sensor_runtime" if checks["passed"] else "rejected",
            "source_commit": source_commit, "proposal_id": contract["proposal_id"], "frame_indices": frames,
            "cross_path_exact_sensor_frame_count": sum(row["sensor_arrays_exact"] for row in comparisons),
            "stationary_state_shared_across_distinct_events": boundary["materialized_state_sha256"] == tail["materialized_state_sha256"] and boundary["trajectory_event_sha256"] != tail["trajectory_event_sha256"],
            "maximum_rgb_mae": max(row["full_sensor_rgb_mae"] for row in direct_rows),
            "maximum_depth_mae_m": max(row["full_sensor_depth_mae_m"] for row in direct_rows),
            "physical_trajectory_validity": "ABSTAIN", "safety_validity": "ABSTAIN", "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = ["R49_GATE.json", "SUMMARY.json", "FRAME_COMPARISONS.jsonl", "RESOURCE_AUDIT.json", "direct_detached.log", "legacy_r35_delta.log", "direct_detached/FRAME_METRICS.jsonl", "direct_detached/WORKER_AUDIT.json", "legacy_r35_delta/FRAME_METRICS.jsonl", "legacy_r35_delta/WORKER_AUDIT.json"]
        for name, rows in [("direct_detached", direct_rows), ("legacy_r35_delta", legacy_rows)]:
            tracked.extend(f"{name}/{row['sensor_path']}" for row in rows)
        _write_json(run_dir / "MANIFEST.json", {
            "schema_version": "worldsim_v6.r49_manifest.v1",
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
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/r49_multiframe_detached_sensor_runtime_v1.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

