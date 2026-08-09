#!/usr/bin/env python
"""聚合 A4-P1 候选资格、资源快照与冻结选择裁决。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

from omegaconf import OmegaConf


PROJECT = Path("/root/autodl-tmp/motion_proj")
PROTOCOL = PROJECT / "configs/worldsim_v3/a4_p1_contribution_prune_protocol_v1.yaml"


from motion_proj.worldsim_v3.contribution_prune import select_largest_eligible_arm
from scripts.run_worldsim_v3_a4_p1_prune import (
    cgroup_memory_events,
    directory_bytes,
    load_stage,
    sha256_file,
    write_stage,
)
from scripts.validate_worldsim_v3_a4_p1_contribution_prune_protocol import (
    validate_inputs,
    validate_schema,
)


PRIOR_STAGES = (
    "input_audit",
    "contribution_scan",
    "materialize_p1_b05",
    "evaluate_p1_source_and_b05",
    "materialize_p1_b10",
    "evaluate_p1_b10",
    "materialize_p1_b20",
    "evaluate_p1_b20",
    "runtime_profile_all_arms",
)


def load_json_artifact(run_dir: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    path = run_dir / record["path"]
    if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"A4-P1 artifact drift: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resource_snapshot(
    run_dir: Path,
    manifest: Mapping[str, Any],
    stages: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """在选择前对可观测的全局资源上界做一次冻结快照。"""
    resource_stages = [
        stages[name]
        for name in (
            "contribution_scan",
            "evaluate_p1_source_and_b05",
            "evaluate_p1_b10",
            "evaluate_p1_b20",
            "runtime_profile_all_arms",
        )
    ]
    before = manifest["preflight"]["cgroup_memory_events"]
    after = cgroup_memory_events()
    measured = {
        "wall_time_seconds": (
            datetime.now(timezone.utc) - datetime.fromisoformat(manifest["started_at"])
        ).total_seconds(),
        "peak_torch_allocated_mib": max(float(row["peak_torch_allocated_mib"]) for row in resource_stages),
        "peak_torch_reserved_mib": max(float(row["peak_torch_reserved_mib"]) for row in resource_stages),
        "peak_nvidia_process_memory_mib_sampled": max(
            int(row["peak_nvidia_process_memory_mib_sampled"]) for row in resource_stages
        ),
        "peak_cgroup_memory_bytes_sampled": max(
            int(row["peak_cgroup_memory_bytes_sampled"]) for row in resource_stages
        ),
        "run_bytes": directory_bytes(run_dir),
        "disk_free_bytes": shutil.disk_usage(run_dir).free,
        "oom_events_delta": after.get("oom", 0) - before.get("oom", 0),
        "oom_kill_events_delta": after.get("oom_kill", 0) - before.get("oom_kill", 0),
        "sampling_errors": [
            error
            for row in resource_stages
            for error in row.get("sampling_errors", [])
        ],
    }
    ceilings = protocol["resource_ceilings"]
    violations = {
        "wall_time_seconds": measured["wall_time_seconds"] > float(ceilings["wall_time_seconds"]),
        "peak_torch_allocated_mib": measured["peak_torch_allocated_mib"] > float(ceilings["peak_torch_allocated_mib"]),
        "peak_torch_reserved_mib": measured["peak_torch_reserved_mib"] > float(ceilings["peak_torch_reserved_mib"]),
        "peak_nvidia_process_memory_mib_sampled": measured["peak_nvidia_process_memory_mib_sampled"] > int(ceilings["peak_nvidia_process_memory_mib_sampled"]),
        "peak_cgroup_memory_bytes": measured["peak_cgroup_memory_bytes_sampled"] > int(ceilings["peak_cgroup_memory_bytes"]),
        "run_bytes": measured["run_bytes"] > int(ceilings["run_bytes"]),
        "disk_free_floor_bytes": measured["disk_free_bytes"] < int(ceilings["disk_free_floor_bytes"]),
        "oom_events_delta": measured["oom_events_delta"] != int(ceilings["oom_events_delta"]),
        "oom_kill_events_delta": measured["oom_kill_events_delta"] != int(ceilings["oom_kill_events_delta"]),
        "sampling_errors": bool(measured["sampling_errors"]),
    }
    return {
        "status": "failed" if any(violations.values()) else "passed",
        "measured": measured,
        "ceilings": ceilings,
        "violations": violations,
    }


def build_arm_rows(
    run_dir: Path,
    protocol: Mapping[str, Any],
    stages: Mapping[str, Mapping[str, Any]],
    resources_pass: bool,
) -> list[dict[str, Any]]:
    quality_records: dict[str, Mapping[str, Any]] = {}
    for stage_name in (
        "evaluate_p1_source_and_b05",
        "evaluate_p1_b10",
        "evaluate_p1_b20",
    ):
        quality_records.update(stages[stage_name]["quality_artifacts"])
    runtime = stages["runtime_profile_all_arms"]["arm_rows"]
    source_bytes = int(protocol["selected_asset"]["checkpoint"]["bytes"])
    rows = []
    for arm, fraction in (("p1-source", 0.0), ("p1-b05", 0.05), ("p1-b10", 0.10), ("p1-b20", 0.20)):
        quality = load_json_artifact(run_dir, quality_records[arm])
        if arm == "p1-source":
            checkpoint_bytes = source_bytes
            materialized = None
            expected_counts_exact = quality["expected_model_counts_exact"]
            checkpoint_reload_exact = quality["checkpoint_reload_exact"] and runtime[arm]["checkpoint_reload_exact"]
        else:
            materialized = stages[f"materialize_{arm.replace('-', '_')}"]
            checkpoint_bytes = int(materialized["candidate_checkpoint"]["bytes"])
            expected_counts_exact = quality["expected_model_counts_exact"] and materialized["candidate_grid_and_removal_counts_exact"]
            checkpoint_reload_exact = quality["checkpoint_reload_exact"] and runtime[arm]["checkpoint_reload_exact"]
        rows.append(
            {
                "id": arm,
                "prune_fraction": fraction,
                "checkpoint_bytes": checkpoint_bytes,
                "checkpoint_bytes_strictly_less_than_source": arm != "p1-source" and checkpoint_bytes < source_bytes,
                "checkpoint_byte_reduction": source_bytes - checkpoint_bytes,
                "candidate_checkpoint_reload_exact": checkpoint_reload_exact,
                "expected_counts_exact": expected_counts_exact,
                "all_quality_safeguards_pass": bool(quality["all_quality_safeguards_pass"]),
                "all_quality_endpoints_complete_and_finite": bool(quality["all_endpoints_complete_and_finite"]),
                "source_inputs_unchanged": all(
                    stages[name].get("source_inputs_unchanged", True)
                    for name in PRIOR_STAGES
                ),
                "resources_within_frozen_ceilings": resources_pass,
                "quality_artifact": quality_records[arm],
                "runtime": runtime[arm],
                "materialization_stage": materialized,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    protocol = OmegaConf.to_container(OmegaConf.load(args.protocol), resolve=True)
    validate_schema(protocol)
    validate_inputs(protocol)
    manifest_path = args.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stages = {
        name: load_stage(args.run_dir, manifest, name)
        for name in PRIOR_STAGES
    }
    resources = resource_snapshot(args.run_dir, manifest, stages, protocol)
    arms = build_arm_rows(args.run_dir, protocol, stages, resources["status"] == "passed")
    selection = select_largest_eligible_arm(arms)
    selected = next(row for row in arms if row["id"] == selection["selected_arm"])
    ledger = {
        name: {
            "status": "done",
            "duration_seconds": float(stages[name].get("duration_seconds", 0.0)),
            "stage_sha256": manifest["stage_hashes"][name],
            "minimum_rerun_unit": stages[name]["minimum_rerun_unit"],
        }
        for name in PRIOR_STAGES
    }
    stage = {
        "status": "done",
        "stage": "aggregate",
        "duration_seconds": 0.0,
        "arms": arms,
        "selection": selection,
        "selected_checkpoint_bytes": selected["checkpoint_bytes"],
        "resource_snapshot": resources,
        "stage_ledger": ledger,
        "selection_order": "largest_prune_fraction_then_arm_id",
        "runtime_values_used_for_quality_selection": False,
        "post_hoc_arm_or_threshold_added": False,
        "minimum_rerun_unit": "aggregate_and_downstream",
    }
    write_stage(args.run_dir, manifest, "aggregate", stage)
    print(json.dumps({"status": "aggregate_complete", "selection": selection}, sort_keys=True))


if __name__ == "__main__":
    main()
