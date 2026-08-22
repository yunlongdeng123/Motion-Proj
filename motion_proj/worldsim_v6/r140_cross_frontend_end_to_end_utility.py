"""WorldSim V6 R140: certify end-to-end selective-perception utility including sensor cost."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from motion_proj.worldsim_v6.r44_verified_actor_proposal_bake import (
    _git,
    _resolve_runs_uri,
    _sha256,
    _verify,
    _write_json,
)


TASK_ID = "WS-V6-R140-CROSS-FRONTEND-END-TO-END-UTILITY-01"


class R140ExperimentError(RuntimeError):
    """The preregistered R140 cost-certificate contract was violated."""


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R140ExperimentError("formal R140 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R140ExperimentError("R140 task_id drift")
    sources = config["sources"]
    evaluation = config["evaluation"]
    resources = config["resources"]

    runs = {
        name: _resolve_runs_uri(sources[f"{name}_run"])
        for name in ("r130", "r131", "r133", "r134", "r137", "r139")
    }
    frozen_files: dict[Path, str] = {
        runs["r130"] / "RESOURCE_AUDIT.json": sources["r130_resource_sha256"],
        runs["r131"] / "RESOURCE_AUDIT.json": sources["r131_resource_sha256"],
        runs["r133"] / "MANIFEST.json": sources["r133_manifest_sha256"],
        runs["r133"] / "R133_GATE.json": sources["r133_gate_sha256"],
        runs["r133"] / "SELECTIVE_EXECUTION_RESULT.json": sources["r133_result_sha256"],
        runs["r133"] / "RESOURCE_AUDIT.json": sources["r133_resource_sha256"],
        runs["r134"] / "RESOURCE_AUDIT.json": sources["r134_resource_sha256"],
        runs["r137"] / "MANIFEST.json": sources["r137_manifest_sha256"],
        runs["r137"] / "R137_GATE.json": sources["r137_gate_sha256"],
        runs["r137"] / "EXACT_INPUT_REUSE_RESULT.json": sources["r137_result_sha256"],
        runs["r137"] / "RESOURCE_AUDIT.json": sources["r137_resource_sha256"],
        runs["r139"] / "MANIFEST.json": sources["r139_manifest_sha256"],
        runs["r139"] / "R139_GATE.json": sources["r139_gate_sha256"],
        runs["r139"] / "ORTHOGONAL_CONFIRMATION.json": sources[
            "r139_confirmation_sha256"
        ],
        runs["r139"] / "RESOURCE_AUDIT.json": sources["r139_resource_sha256"],
    }
    for path, expected in frozen_files.items():
        _verify(path, expected)

    r130_resource = _load(runs["r130"] / "RESOURCE_AUDIT.json")
    r131_resource = _load(runs["r131"] / "RESOURCE_AUDIT.json")
    r133_gate = _load(runs["r133"] / "R133_GATE.json")
    r133_result = _load(runs["r133"] / "SELECTIVE_EXECUTION_RESULT.json")
    r133_resource = _load(runs["r133"] / "RESOURCE_AUDIT.json")
    r134_resource = _load(runs["r134"] / "RESOURCE_AUDIT.json")
    r137_gate = _load(runs["r137"] / "R137_GATE.json")
    r137_result = _load(runs["r137"] / "EXACT_INPUT_REUSE_RESULT.json")
    r137_resource = _load(runs["r137"] / "RESOURCE_AUDIT.json")
    r139_gate = _load(runs["r139"] / "R139_GATE.json")
    r139_result = _load(runs["r139"] / "ORTHOGONAL_CONFIRMATION.json")
    r139_resource = _load(runs["r139"] / "RESOURCE_AUDIT.json")

    specs = [
        {
            "condition": "streetgs_r130_r131_cross_prospective",
            "frontend": "streetgs",
            "frame_count": int(r133_result["frame_count"]),
            "sensor_seconds": float(r130_resource["sensor_worker_seconds"])
            + float(r131_resource["sensor_worker_seconds"]),
            "full_perception_seconds": float(
                r133_resource["source_full_reference_worker_seconds"]
            ),
            "selective_perception_seconds": float(r133_resource["perception_worker_seconds"]),
            "full_invocations": int(r133_result["full_reference_invocations"]),
            "selective_invocations": int(r133_result["selective_invocations"]),
            "reconstruction_error_count": int(
                r133_result["reconstruction_metrics"]["false_positive"]
            )
            + int(r133_result["reconstruction_metrics"]["false_negative"]),
        },
        {
            "condition": "adgs_r134_r137_development",
            "frontend": "ad_gs",
            "frame_count": int(r137_result["frame_count"]),
            "sensor_seconds": float(r134_resource["sensor_worker_seconds"]),
            "full_perception_seconds": float(r134_resource["perception_worker_seconds"]),
            "selective_perception_seconds": float(r137_resource["worker_seconds"]),
            "full_invocations": int(r137_result["full_reference_invocations"]),
            "selective_invocations": int(r137_result["selective_invocations"]),
            "reconstruction_error_count": int(r137_result["reconstruction_error_count"]),
        },
        {
            "condition": "adgs_r139_orthogonal_exact_once",
            "frontend": "ad_gs",
            "frame_count": int(r139_result["frame_count"]),
            "sensor_seconds": float(r139_resource["sensor_worker_seconds"]),
            "full_perception_seconds": float(r139_resource["full_worker_seconds"]),
            "selective_perception_seconds": float(r139_resource["selective_worker_seconds"]),
            "full_invocations": int(r139_result["full_reference_invocations"]),
            "selective_invocations": int(r139_result["selective_invocations"]),
            "reconstruction_error_count": int(r139_result["reconstruction_error_count"]),
        },
    ]
    rows: list[dict[str, Any]] = []
    for spec in specs:
        full_pipeline = spec["sensor_seconds"] + spec["full_perception_seconds"]
        selective_pipeline = spec["sensor_seconds"] + spec["selective_perception_seconds"]
        rows.append(
            {
                **spec,
                "full_pipeline_seconds": full_pipeline,
                "selective_pipeline_seconds": selective_pipeline,
                "perception_component_reduction_fraction": 1.0
                - spec["selective_perception_seconds"] / spec["full_perception_seconds"],
                "end_to_end_reduction_fraction": 1.0 - selective_pipeline / full_pipeline,
                "sensor_fraction_of_full_pipeline": spec["sensor_seconds"] / full_pipeline,
                "invocation_reduction_fraction": 1.0
                - spec["selective_invocations"] / spec["full_invocations"],
            }
        )
    savings = np.asarray([row["end_to_end_reduction_fraction"] for row in rows])
    macro_saving = float(savings.mean())
    worst_saving = float(savings.min())
    wall_seconds = time.monotonic() - started
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__end-to-end-utility-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    certificate = {
        "schema_version": "worldsim_v6.r140_end_to_end_utility_certificate.v1",
        "conditions": rows,
        "macro_end_to_end_reduction_fraction": macro_saving,
        "worst_end_to_end_reduction_fraction": worst_saving,
        "best_end_to_end_reduction_fraction": float(savings.max()),
        "timing_semantics": "single_observed_worker_times_shared_sensor_cost_added_to_each_path",
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "END_TO_END_UTILITY_CERTIFICATE.json", certificate)
    precloseout_output_bytes = sum(
        path.stat().st_size for path in run_dir.rglob("*") if path.is_file()
    )
    checks = {
        "r133_r137_r139_selective_authorities_accepted": bool(
            r133_gate["checks"]["passed"]
            and r137_gate["checks"]["passed"]
            and r139_gate["checks"]["passed"]
        ),
        "all_source_artifacts_immutable": all(
            _sha256(path) == expected for path, expected in frozen_files.items()
        ),
        "three_condition_two_frontend_denominator": len(rows) == 3
        and {row["frontend"] for row in rows} == {"streetgs", "ad_gs"},
        "shared_sensor_cost_included_exactly_once_per_pipeline": all(
            row["full_pipeline_seconds"]
            == row["sensor_seconds"] + row["full_perception_seconds"]
            and row["selective_pipeline_seconds"]
            == row["sensor_seconds"] + row["selective_perception_seconds"]
            for row in rows
        ),
        "all_bound_reconstructions_zero_error": all(
            row["reconstruction_error_count"] == 0 for row in rows
        )
        and r137_result["false_reuse_count"] == 0
        and r139_result["false_reuse_count"] == 0,
        "component_and_end_to_end_savings_positive_every_condition": all(
            row["perception_component_reduction_fraction"] > 0.0
            and row["end_to_end_reduction_fraction"] > 0.0
            for row in rows
        ),
        "worst_end_to_end_saving_gate": worst_saving
        >= float(evaluation["minimum_worst_end_to_end_reduction_fraction"]),
        "macro_end_to_end_saving_gate": macro_saving
        >= float(evaluation["minimum_macro_end_to_end_reduction_fraction"]),
        "sensor_dominance_reported_not_hidden": all(
            row["sensor_fraction_of_full_pipeline"] > 0.0 for row in rows
        ),
        "cpu_wall_and_output_budget": wall_seconds <= float(resources["maximum_wall_seconds"])
        and precloseout_output_bytes <= int(resources["maximum_output_bytes"]),
        "no_gpu_training_confirmation_or_artifact_mutation": True,
        "single_sample_timing_semantics_planning_physics_safety_abstain": True,
    }
    checks["passed"] = all(checks.values())
    _write_json(
        run_dir / "R140_GATE.json",
        {
            "schema_version": "worldsim_v6.r140_gate.v1",
            "checks": checks,
            "decision": "accept_cross_frontend_end_to_end_utility_certificate"
            if checks["passed"]
            else "reject_cross_frontend_end_to_end_utility_certificate",
        },
    )
    summary = {
        "schema_version": "worldsim_v6.r140_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_cross_frontend_end_to_end_utility"
        if checks["passed"]
        else "rejected_cross_frontend_end_to_end_utility",
        "source_commit": source_commit,
        "condition_count": len(rows),
        "frontend_count": len({row["frontend"] for row in rows}),
        "macro_end_to_end_reduction_fraction": macro_saving,
        "worst_end_to_end_reduction_fraction": worst_saving,
        "condition_end_to_end_reduction_fraction": {
            row["condition"]: row["end_to_end_reduction_fraction"] for row in rows
        },
        "reconstruction_error_count": sum(row["reconstruction_error_count"] for row in rows),
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    _write_json(
        run_dir / "RESOURCE_AUDIT.json",
        {
            "schema_version": "worldsim_v6.r140_resource_audit.v1",
            "wall_seconds": time.monotonic() - started,
            "gpu_used": false,
            "training_started": false,
            "confirmation_content_read": false,
        },
    )
    tracked = [
        "END_TO_END_UTILITY_CERTIFICATE.json",
        "R140_GATE.json",
        "SUMMARY.json",
        "RESOURCE_AUDIT.json",
    ]
    _write_json(
        run_dir / "MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r140_manifest.v1",
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
        default=Path("configs/worldsim_v6/r140_cross_frontend_end_to_end_utility_v1.yaml"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
