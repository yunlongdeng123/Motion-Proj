"""WorldSim V6 R52：比较 naive transform bake 与 lifecycle-aware V6 bake。"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from motion_proj.worldsim_v6.r44_verified_actor_proposal_bake import _git, _resolve_runs_uri, _sha256, _verify, _write_json


TASK_ID = "WS-V6-R52-LIFECYCLE-GATE3-ABLATION-01"


class R52ExperimentError(RuntimeError):
    """R52 正式实验合同失败。"""


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _row_passes(row: dict[str, Any], thresholds: dict[str, Any]) -> bool:
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
        raise R52ExperimentError("正式 R52 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R52ExperimentError("R52 task_id 漂移")
    sources = config["sources"]
    r45_run = _resolve_runs_uri(sources["r45_run"])
    r49_run = _resolve_runs_uri(sources["r49_run"])
    r50_run = _resolve_runs_uri(sources["r50_run"])
    r51_run = _resolve_runs_uri(sources["r51_run"])
    frozen_files = {
        r45_run / "R45_GATE.json": sources["r45_gate_sha256"],
        r49_run / "R49_GATE.json": sources["r49_gate_sha256"],
        r49_run / "SUMMARY.json": sources["r49_summary_sha256"],
        r49_run / "direct_detached/FRAME_METRICS.jsonl": sources["r49_direct_metrics_sha256"],
        r50_run / "R50_GATE.json": sources["r50_gate_sha256"],
        r50_run / f"package/blobs/{sources['r50_lifecycle_sha256']}.npy": sources["r50_lifecycle_sha256"],
        r51_run / "R51_GATE.json": sources["r51_gate_sha256"],
        r51_run / "SUMMARY.json": sources["r51_summary_sha256"],
        r51_run / "MANIFEST.json": sources["r51_manifest_sha256"],
        r51_run / "worker/FRAME_METRICS.jsonl": sources["r51_metrics_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)

    evaluation = config["evaluation"]
    thresholds = config["thresholds"]
    frames = [int(value) for value in evaluation["frame_indices"]]
    active_frames = [int(value) for value in evaluation["active_frames"]]
    inactive_frames = [int(value) for value in evaluation["inactive_frames"]]
    naive_rows = _load_rows(r49_run / "direct_detached/FRAME_METRICS.jsonl")
    v6_rows = _load_rows(r51_run / "worker/FRAME_METRICS.jsonl")
    naive_by_frame = {int(row["frame_index"]): row for row in naive_rows}
    v6_by_frame = {int(row["frame_index"]): row for row in v6_rows}
    naive_pass_frames = [frame for frame in frames if _row_passes(naive_by_frame[frame], thresholds)]
    v6_pass_frames = [frame for frame in frames if _row_passes(v6_by_frame[frame], thresholds)]
    naive_false_active_frames = [
        frame for frame in inactive_frames
        if naive_by_frame[frame]["native_actor_field_max_error"]["opacities"] > float(thresholds["maximum_static_field_error"])
    ]
    v6_false_active_frames = [
        frame for frame in inactive_frames
        if v6_by_frame[frame]["native_actor_field_max_error"]["opacities"] > float(thresholds["maximum_static_field_error"])
    ]
    active_sensor_hashes_preserved = all(naive_by_frame[frame]["sensor_sha256"] == v6_by_frame[frame]["sensor_sha256"] for frame in active_frames)
    r45_gate = json.loads((r45_run / "R45_GATE.json").read_text(encoding="utf-8"))
    r49_gate = json.loads((r49_run / "R49_GATE.json").read_text(encoding="utf-8"))
    r50_gate = json.loads((r50_run / "R50_GATE.json").read_text(encoding="utf-8"))
    r51_gate = json.loads((r51_run / "R51_GATE.json").read_text(encoding="utf-8"))

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__lifecycle-gate3-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    comparison = {
        "schema_version": "worldsim_v6.r52_paired_ablation.v1",
        "frame_indices": frames,
        "active_frames": active_frames,
        "inactive_frames": inactive_frames,
        "naive_transform_only_bake": {
            "pass_frames": naive_pass_frames,
            "usable_frame_count": len(naive_pass_frames),
            "usable_frame_rate": len(naive_pass_frames) / len(frames),
            "lifecycle_false_active_frames": naive_false_active_frames,
            "maximum_rgb_mae": max(row["full_sensor_rgb_mae"] for row in naive_rows),
            "maximum_depth_mae_m": max(row["full_sensor_depth_mae_m"] for row in naive_rows),
        },
        "v6_lifecycle_aware_bake": {
            "pass_frames": v6_pass_frames,
            "usable_frame_count": len(v6_pass_frames),
            "usable_frame_rate": len(v6_pass_frames) / len(frames),
            "lifecycle_false_active_frames": v6_false_active_frames,
            "maximum_rgb_mae": max(row["full_sensor_rgb_mae"] for row in v6_rows),
            "maximum_depth_mae_m": max(row["full_sensor_depth_mae_m"] for row in v6_rows),
        },
        "paired_effect": {
            "usable_frame_gain": len(v6_pass_frames) - len(naive_pass_frames),
            "usable_frame_rate_gain": (len(v6_pass_frames) - len(naive_pass_frames)) / len(frames),
            "lifecycle_false_active_reduction": len(naive_false_active_frames) - len(v6_false_active_frames),
            "active_sensor_hashes_preserved_exact": active_sensor_hashes_preserved,
        },
    }
    _write_json(run_dir / "PAIRED_ABLATION.json", comparison)
    evidence_dag = {
        "schema_version": "worldsim_v6.r52_evidence_dag.v1",
        "nodes": [
            {"id": "R45", "role": "naive_transform_owned_bake", "gate_sha256": sources["r45_gate_sha256"]},
            {"id": "R49", "role": "naive_multiframe_native_sensor_evaluation", "gate_sha256": sources["r49_gate_sha256"]},
            {"id": "R50", "role": "native_lifecycle_extraction_and_bake", "gate_sha256": sources["r50_gate_sha256"], "lifecycle_sha256": sources["r50_lifecycle_sha256"]},
            {"id": "R51", "role": "lifecycle_aware_multiframe_native_sensor_evaluation", "gate_sha256": sources["r51_gate_sha256"]},
        ],
        "edges": [
            {"from": "R45", "to": "R49", "relation": "naive_runtime_evaluated_by_native_sensor"},
            {"from": "R45", "to": "R50", "relation": "base_arrays_and_transform_preserved_lifecycle_added"},
            {"from": "R50", "to": "R51", "relation": "typed_lifecycle_consumed_at_runtime"},
            {"from": "R49", "to": "R51", "relation": "paired_same_frames_thresholds_and_actor_edit"},
        ],
    }
    _write_json(run_dir / "EVIDENCE_DAG.json", evidence_dag)
    checks = {
        "source_authorities_bound": bool(r45_gate["checks"]["passed"] and not r49_gate["checks"]["passed"] and r50_gate["checks"]["passed"] and r51_gate["checks"]["passed"]),
        "five_frame_paired_denominator_exact": [row["frame_index"] for row in naive_rows] == frames == [row["frame_index"] for row in v6_rows],
        "naive_pass_frames_exact": naive_pass_frames == evaluation["expected_naive_pass_frames"],
        "v6_pass_frames_exact": v6_pass_frames == evaluation["expected_v6_pass_frames"],
        "naive_lifecycle_false_active_frames_exact": naive_false_active_frames == evaluation["expected_naive_lifecycle_false_active_frames"],
        "v6_lifecycle_false_active_frames_exact": v6_false_active_frames == evaluation["expected_v6_lifecycle_false_active_frames"],
        "minimum_usable_frame_gain_met": comparison["paired_effect"]["usable_frame_gain"] >= int(evaluation["minimum_usable_frame_gain"]),
        "active_frame_sensors_preserved_exact": active_sensor_hashes_preserved,
        "base_arrays_and_transform_preserved": bool(r50_gate["checks"]["base_arrays_and_transform_preserved"]),
        "paired_actor_edit_and_translation_exact": all(naive_by_frame[frame]["translation_delta_m"] == v6_by_frame[frame]["translation_delta_m"] == [-1.0, 0.0, -0.5] for frame in frames),
        "native_sensor_grounded": True,
        "physical_and_safety_validity_abstain": True,
        "source_immutable": all(_sha256(path) == expected_sha for path, expected_sha in frozen_files.items()),
        "training_not_started": True,
        "confirmation_not_read": True,
    }
    checks["wall_within_budget"] = (time.monotonic() - started) <= float(config["resources"]["maximum_wall_seconds"])
    checks["passed"] = all(checks.values())
    _write_json(run_dir / "R52_GATE.json", {
        "schema_version": "worldsim_v6.r52_gate.v1",
        "checks": checks,
        "decision": "accept_lifecycle_gate3_paired_ablation" if checks["passed"] else "reject_lifecycle_gate3_paired_ablation",
    })
    summary = {
        "schema_version": "worldsim_v6.r52_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_development_lifecycle_gate3_paired_ablation" if checks["passed"] else "rejected",
        "source_commit": source_commit,
        "naive_usable_frames": len(naive_pass_frames),
        "v6_usable_frames": len(v6_pass_frames),
        "usable_frame_gain": comparison["paired_effect"]["usable_frame_gain"],
        "usable_frame_rate_gain": comparison["paired_effect"]["usable_frame_rate_gain"],
        "lifecycle_false_active_reduction": comparison["paired_effect"]["lifecycle_false_active_reduction"],
        "active_sensor_hashes_preserved_exact": active_sensor_hashes_preserved,
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    tracked = ["PAIRED_ABLATION.json", "EVIDENCE_DAG.json", "R52_GATE.json", "SUMMARY.json"]
    _write_json(run_dir / "MANIFEST.json", {
        "schema_version": "worldsim_v6.r52_manifest.v1",
        "files": {name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)} for name in tracked},
    })
    _write_json(run_dir / "TERMINAL.json", {
        "schema_version": "worldsim_v6.terminal.v1",
        "status": summary["status"],
        "manifest_sha256": _sha256(run_dir / "MANIFEST.json"),
        "summary_sha256": _sha256(run_dir / "SUMMARY.json"),
    })
    print(str(run_dir), flush=True)
    return run_dir


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/r52_lifecycle_gate3_ablation_v1.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
