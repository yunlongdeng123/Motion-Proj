#!/usr/bin/env python
"""聚合 A4-P2 完整性、质量、资源与冻结选择裁决。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

from omegaconf import OmegaConf


PROJECT = Path("/root/autodl-tmp/motion_proj")
PROTOCOL = PROJECT / "configs/worldsim_v3/a4_p2_mixed_precision_protocol_v1.yaml"
PRIOR_STAGES = (
    "input_audit",
    "source_dtype_audit",
    "materialize_p2_gs_param_fp16",
    "evaluate_p2_source_and_candidate",
    "runtime_profile_both_arms",
)


from motion_proj.worldsim_v3.mixed_precision import select_precision_arm
from scripts.run_worldsim_v3_a4_p2_precision import (
    cgroup_memory_events,
    directory_bytes,
    load_stage,
    sha256_file,
    write_stage,
)
from scripts.validate_worldsim_v3_a4_p2_mixed_precision_protocol import (
    validate_inputs,
    validate_schema,
)


def load_json_artifact(run_dir: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    path = run_dir / record["path"]
    if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"A4-P2 artifact drift: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resource_snapshot(
    run_dir: Path,
    manifest: Mapping[str, Any],
    stages: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """在选择前冻结全局资源上界快照。"""
    resource_stages = [
        stages["evaluate_p2_source_and_candidate"],
        stages["runtime_profile_both_arms"],
    ]
    before = manifest["preflight"]["cgroup_memory_events"]
    after = cgroup_memory_events()
    measured = {
        "wall_time_seconds": (
            datetime.now(timezone.utc) - datetime.fromisoformat(manifest["started_at"])
        ).total_seconds(),
        "peak_torch_allocated_mib": max(
            float(row["peak_torch_allocated_mib"]) for row in resource_stages
        ),
        "peak_torch_reserved_mib": max(
            float(row["peak_torch_reserved_mib"]) for row in resource_stages
        ),
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
            error for row in resource_stages for error in row.get("sampling_errors", [])
        ],
    }
    ceilings = protocol["resource_ceilings"]
    violations = {
        "wall_time_seconds": measured["wall_time_seconds"]
        > float(ceilings["wall_time_seconds"]),
        "peak_torch_allocated_mib": measured["peak_torch_allocated_mib"]
        > float(ceilings["peak_torch_allocated_mib"]),
        "peak_torch_reserved_mib": measured["peak_torch_reserved_mib"]
        > float(ceilings["peak_torch_reserved_mib"]),
        "peak_nvidia_process_memory_mib_sampled": measured[
            "peak_nvidia_process_memory_mib_sampled"
        ]
        > int(ceilings["peak_nvidia_process_memory_mib_sampled"]),
        "peak_cgroup_memory_bytes": measured["peak_cgroup_memory_bytes_sampled"]
        > int(ceilings["peak_cgroup_memory_bytes"]),
        "run_bytes": measured["run_bytes"] > int(ceilings["run_bytes"]),
        "disk_free_floor_bytes": measured["disk_free_bytes"]
        < int(ceilings["disk_free_floor_bytes"]),
        "oom_events_delta": measured["oom_events_delta"]
        != int(ceilings["oom_events_delta"]),
        "oom_kill_events_delta": measured["oom_kill_events_delta"]
        != int(ceilings["oom_kill_events_delta"]),
        "sampling_errors": bool(measured["sampling_errors"]),
    }
    return {
        "status": "failed" if any(violations.values()) else "passed",
        "measured": measured,
        "ceilings": ceilings,
        "violations": violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    protocol = OmegaConf.to_container(OmegaConf.load(args.protocol), resolve=True)
    validate_schema(protocol)
    validate_inputs(protocol)
    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    stages = {name: load_stage(args.run_dir, manifest, name) for name in PRIOR_STAGES}
    resources = resource_snapshot(args.run_dir, manifest, stages, protocol)
    evaluation = stages["evaluate_p2_source_and_candidate"]
    quality = {
        arm: load_json_artifact(args.run_dir, record)
        for arm, record in evaluation["quality_artifacts"].items()
    }
    runtime = stages["runtime_profile_both_arms"]["arm_rows"]
    materialized = stages["materialize_p2_gs_param_fp16"]
    source_bytes = int(protocol["selected_asset"]["checkpoint"]["bytes"])
    source_inputs_exact = all(
        stages[name].get("source_inputs_unchanged", True) for name in PRIOR_STAGES
    )
    candidate = {
        "id": "p2-gs-param-fp16",
        "checkpoint_conversion_and_preservation_exact": bool(
            materialized["converted_field_set_and_float16_bytes_exact"]
            and materialized["converted_field_roundtrip_error_complete_and_finite"]
            and materialized["preserved_fields_and_checkpoint_schema_exact"]
            and materialized["candidate_registry_counts_and_indices_exact"]
            and materialized["candidate_checkpoint_write_count"] == 1
            and not materialized["source_checkpoint_copied"]
        ),
        "candidate_checkpoint_reload_and_runtime_dtype_exact": bool(
            quality["p2-gs-param-fp16"]["checkpoint_reload_exact"]
            and quality["p2-gs-param-fp16"]["runtime_converted_field_audit"]["exact"]
            and runtime["p2-gs-param-fp16"]["checkpoint_reload_exact"]
            and runtime["p2-gs-param-fp16"]["runtime_converted_field_audit"]["exact"]
        ),
        "renderer_input_float32_exact": bool(
            quality["p2-gs-param-fp16"]["renderer_input_dtypes"][
                "all_renderer_inputs_float32"
            ]
            and quality["p2-gs-param-fp16"]["renderer_input_dtypes"][
                "spherical_harmonics_inputs_float32"
            ]
            and runtime["p2-gs-param-fp16"]["render_input_dtypes"][
                "all_renderer_inputs_float32"
            ]
            and runtime["p2-gs-param-fp16"]["render_input_dtypes"][
                "spherical_harmonics_inputs_float32"
            ]
        ),
        "source_baseline_replay_exact": bool(
            quality["p2-source"]["baseline_p1_replay_pass"]
        ),
        "all_quality_safeguards_pass": bool(
            quality["p2-gs-param-fp16"]["all_quality_safeguards_pass"]
        ),
        "all_quality_endpoints_complete_and_finite": bool(
            quality["p2-source"]["all_endpoints_complete_and_finite"]
            and quality["p2-gs-param-fp16"]["all_endpoints_complete_and_finite"]
        ),
        "checkpoint_bytes": int(materialized["candidate_checkpoint"]["bytes"]),
        "checkpoint_bytes_strictly_less_than_source": int(
            materialized["candidate_checkpoint"]["bytes"]
        )
        < source_bytes,
        "checkpoint_byte_reduction": source_bytes
        - int(materialized["candidate_checkpoint"]["bytes"]),
        "source_inputs_unchanged": source_inputs_exact,
        "resources_within_frozen_ceilings": resources["status"] == "passed",
        "quality_artifact": evaluation["quality_artifacts"]["p2-gs-param-fp16"],
        "runtime": runtime["p2-gs-param-fp16"],
        "materialization_stage": materialized,
    }
    source = {
        "id": "p2-source",
        "checkpoint_bytes": source_bytes,
        "checkpoint_byte_reduction": 0,
        "source_baseline_replay_exact": bool(quality["p2-source"]["baseline_p1_replay_pass"]),
        "all_quality_endpoints_complete_and_finite": bool(
            quality["p2-source"]["all_endpoints_complete_and_finite"]
        ),
        "source_inputs_unchanged": source_inputs_exact,
        "quality_artifact": evaluation["quality_artifacts"]["p2-source"],
        "runtime": runtime["p2-source"],
    }
    selection = select_precision_arm(candidate)
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
        "arms": [source, candidate],
        "selection": selection,
        "selected_checkpoint_bytes": (
            candidate["checkpoint_bytes"]
            if selection["selected_arm"] == "p2-gs-param-fp16"
            else source_bytes
        ),
        "resource_snapshot": resources,
        "stage_ledger": ledger,
        "runtime_values_used_for_quality_selection": False,
        "post_hoc_field_policy_or_threshold_added": False,
        "minimum_rerun_unit": "aggregate_and_downstream",
    }
    write_stage(args.run_dir, manifest, "aggregate", stage)
    print(json.dumps({"status": "aggregate_complete", "selection": selection}, sort_keys=True))


if __name__ == "__main__":
    main()
