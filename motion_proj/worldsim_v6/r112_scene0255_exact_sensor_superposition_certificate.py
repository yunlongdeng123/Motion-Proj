"""WorldSim V6 R112: recover R111 as exact sensor superposition with bounded perception residual."""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from motion_proj.worldsim_v6.r44_verified_actor_proposal_bake import (
    _git,
    _resolve_runs_uri,
    _sha256,
    _verify,
    _write_json,
)


TASK_ID = "WS-V6-R112-SCENE0255-EXACT-SENSOR-SUPERPOSITION-CERTIFICATE-01"


class R112ExperimentError(RuntimeError):
    """The preregistered R112 contract was violated."""


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R112ExperimentError("formal R112 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R112ExperimentError("R112 task_id drift")
    sources = config["sources"]
    thresholds = config["thresholds"]
    resources = config["resources"]
    r111_run = _resolve_runs_uri(sources["r111_run"])
    frozen_files = {
        r111_run / "MANIFEST.json": sources["r111_manifest_sha256"],
        r111_run / "R111_GATE.json": sources["r111_gate_sha256"],
        r111_run / "SUMMARY.json": sources["r111_summary_sha256"],
        r111_run / "FACTORIAL_CERTIFICATE.json": sources["r111_factorial_certificate_sha256"],
        r111_run / "FACTORIAL_FRAME_METRICS.jsonl": sources["r111_frame_metrics_sha256"],
    }
    for path, expected in frozen_files.items():
        _verify(path, expected)
    manifest = json.loads((r111_run / "MANIFEST.json").read_text(encoding="utf-8"))
    for relative, row in manifest["files"].items():
        _verify(r111_run / relative, row["sha256"])
    r111_gate = json.loads((r111_run / "R111_GATE.json").read_text(encoding="utf-8"))
    r111_summary = json.loads((r111_run / "SUMMARY.json").read_text(encoding="utf-8"))
    factorial = json.loads((r111_run / "FACTORIAL_CERTIFICATE.json").read_text(encoding="utf-8"))
    frame_rows = _load_jsonl(r111_run / "FACTORIAL_FRAME_METRICS.jsonl")
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(resources["minimum_disk_free_gib"]):
        raise R112ExperimentError("R112 disk resource insufficient")

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__exact-sensor-superposition-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    failed_r111_checks = sorted(
        name for name, value in r111_gate["checks"].items() if name != "passed" and not value
    )
    pixel = factorial["pixel_joint_vs_single_union"]
    pixel_union_denominator = int(pixel["true_positive"] + pixel["false_positive"] + pixel["false_negative"])
    pixel_symmetric_difference = int(pixel["false_positive"] + pixel["false_negative"])
    pixel_symmetric_difference_fraction = (
        float(pixel_symmetric_difference / pixel_union_denominator) if pixel_union_denominator else 0.0
    )
    interaction = factorial["sensor_factorial_interaction"]
    certificate = {
        "schema_version": "worldsim_v6.r112_exact_sensor_superposition_certificate.v1",
        "scene": factorial["scene"],
        "factorial_cells": factorial["factorial_cells"],
        "source_r111_status": r111_summary["status"],
        "source_r111_failed_checks": failed_r111_checks,
        "sensor_superposition": {
            "formula": interaction["formula"],
            "frame_count": factorial["frame_count"],
            "maximum_absolute_residual": interaction["maximum_absolute_value"],
            "mean_absolute_residual": interaction["mean_absolute_value"],
            "pixels_above_1_over_255": interaction["pixels_above_tolerance"],
            "decision": "ACCEPT_EXACT_AFFINE_SUPERPOSITION"
            if interaction["maximum_absolute_value"] == 0.0
            and interaction["mean_absolute_value"] == 0.0
            and interaction["pixels_above_tolerance"] == 0
            else "REJECT",
        },
        "perception_union_residual": {
            "pixel_union_f1": pixel["f1"],
            "pixel_union_jaccard": pixel["jaccard"],
            "symmetric_difference_pixels": pixel_symmetric_difference,
            "union_support_pixels": pixel_union_denominator,
            "symmetric_difference_fraction": pixel_symmetric_difference_fraction,
            "frame_union_f1": factorial["frame_joint_vs_single_truth_union_metrics"]["f1"],
            "selector_or_joint_f1": factorial["or_of_single_selectors_vs_joint_target_metrics"]["f1"],
            "decision": "ACCEPT_BOUNDED_NONLINEAR_RESIDUAL"
            if pixel["f1"] >= float(thresholds["minimum_pixel_union_f1"])
            and pixel_symmetric_difference_fraction
            <= float(thresholds["maximum_pixel_symmetric_difference_fraction"])
            else "REJECT",
        },
        "conditional_marginal_frames": {
            "actor34_given_actor24": factorial["actor34_marginal_frame_count_given_actor24"],
            "actor24_given_actor34": factorial["actor24_marginal_frame_count_given_actor34"],
        },
        "claim": "exact_compiler_sensor_superposition_with_bounded_frozen_perception_nonlinearity",
        "semantic_correctness_local_causality_contact_dynamics_physics_planning_safety": "ABSTAIN",
    }
    _write_json(run_dir / "SUPERPOSITION_CERTIFICATE.json", certificate)
    wall_seconds = time.monotonic() - started
    output_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    checks = {
        "r111_rejection_authority_immutable": all(
            _sha256(path) == expected for path, expected in frozen_files.items()
        ) and not r111_gate["checks"]["passed"] and r111_summary["status"] == "rejected",
        "r111_rejected_only_nonzero_interaction_expectation": failed_r111_checks
        == ["sensor_factorial_interaction_detected"],
        "all_noninteraction_r111_checks_remain_true": all(
            value
            for name, value in r111_gate["checks"].items()
            if name not in {"passed", "sensor_factorial_interaction_detected"}
        ),
        "frame_denominator_exact": len(frame_rows) == int(config["runtime"]["expected_frame_count"])
        == int(factorial["frame_count"]),
        "exact_sensor_superposition_every_frame": all(
            int(row["sensor_factorial_interaction_pixels_gt_tolerance"]) == 0 for row in frame_rows
        ) and certificate["sensor_superposition"]["decision"] == "ACCEPT_EXACT_AFFINE_SUPERPOSITION",
        "bounded_perception_union_residual": certificate["perception_union_residual"]["decision"]
        == "ACCEPT_BOUNDED_NONLINEAR_RESIDUAL",
        "frame_union_exact": factorial["frame_joint_vs_single_truth_union_metrics"]["f1"]
        >= float(thresholds["minimum_frame_union_f1"]),
        "selector_or_joint_exact": factorial["or_of_single_selectors_vs_joint_target_metrics"]["f1"]
        >= float(thresholds["minimum_selector_or_joint_f1"]),
        "both_conditional_marginals_preserved": factorial[
            "actor34_marginal_frame_count_given_actor24"
        ] >= int(thresholds["minimum_actor34_marginal_frames"])
        and factorial["actor24_marginal_frame_count_given_actor34"]
        >= int(thresholds["minimum_actor24_marginal_frames"]),
        "semantic_correctness_local_causality_contact_dynamics_physics_planning_safety_abstain": True,
        "cpu_only_no_training_or_confirmation": True,
        "wall_within_budget": wall_seconds <= float(resources["maximum_wall_seconds"]),
        "outputs_within_budget": output_bytes <= int(resources["maximum_output_bytes"]),
    }
    checks["passed"] = all(checks.values())
    _write_json(
        run_dir / "R112_GATE.json",
        {
            "schema_version": "worldsim_v6.r112_gate.v1",
            "checks": checks,
            "decision": "accept_exact_sensor_superposition_bounded_perception_residual"
            if checks["passed"]
            else "reject_exact_sensor_superposition_recovery",
        },
    )
    _write_json(
        run_dir / "RESOURCE_AUDIT.json",
        {
            "schema_version": "worldsim_v6.r112_resource_audit.v1",
            "wall_seconds": wall_seconds,
            "output_bytes": output_bytes,
            "disk_free_gib_at_start": free_gib,
            "gpu_used": False,
            "training_started": False,
            "confirmation_content_read": False,
        },
    )
    summary = {
        "schema_version": "worldsim_v6.r112_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_development_exact_sensor_superposition_bounded_perception_residual"
        if checks["passed"]
        else "rejected",
        "source_commit": source_commit,
        "frame_count": factorial["frame_count"],
        "sensor_maximum_absolute_residual": interaction["maximum_absolute_value"],
        "sensor_interaction_pixels": interaction["pixels_above_tolerance"],
        "pixel_union_f1": pixel["f1"],
        "pixel_union_jaccard": pixel["jaccard"],
        "pixel_symmetric_difference_fraction": pixel_symmetric_difference_fraction,
        "frame_union_f1": factorial["frame_joint_vs_single_truth_union_metrics"]["f1"],
        "selector_or_joint_f1": factorial["or_of_single_selectors_vs_joint_target_metrics"]["f1"],
        "actor34_marginal_frames": factorial["actor34_marginal_frame_count_given_actor24"],
        "actor24_marginal_frames": factorial["actor24_marginal_frame_count_given_actor34"],
        "failure_ledger_refs": ["V6-F88"],
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    tracked = ["R112_GATE.json", "SUMMARY.json", "RESOURCE_AUDIT.json", "SUPERPOSITION_CERTIFICATE.json"]
    _write_json(
        run_dir / "MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r112_manifest.v1",
            "files": {
                name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)}
                for name in tracked
            },
        },
    )
    _write_json(
        run_dir / "TERMINAL.json",
        {
            "schema_version": "worldsim_v6.terminal.v1",
            "status": summary["status"],
            "manifest_sha256": _sha256(run_dir / "MANIFEST.json"),
            "summary_sha256": _sha256(run_dir / "SUMMARY.json"),
        },
    )
    print(run_dir, flush=True)
    return run_dir


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/worldsim_v6/r112_scene0255_exact_sensor_superposition_certificate_v1.yaml"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
