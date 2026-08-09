#!/usr/bin/env python
"""聚合 A4-P3 package 完整性、质量、资源与冻结选择裁决。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping

from omegaconf import OmegaConf


PROJECT = Path("/root/autodl-tmp/motion_proj")
PROTOCOL = PROJECT / "configs/worldsim_v3/a4_p3_chunk_protocol_v1.yaml"
PRIOR_STAGES = (
    "input_audit",
    "source_layout_audit",
    "materialize_chunk_package",
    "reassemble_and_hash_audit",
    "evaluate_source_and_chunk",
    "runtime_profile_both_arms",
)


from motion_proj.worldsim_v3.chunk_package import select_chunk_arm
from scripts.run_worldsim_v3_a4_p3_chunk import (
    cgroup_memory_events,
    directory_bytes,
    load_stage,
    sha256_file,
    write_stage,
)
from scripts.validate_worldsim_v3_a4_p3_chunk_protocol import (
    validate_inputs,
    validate_schema,
)


def load_json_artifact(run_dir: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    path = run_dir / str(record["path"])
    if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"A4-P3 artifact drift: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resource_snapshot(
    run_dir: Path,
    manifest: Mapping[str, Any],
    stages: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """在选择前冻结全局资源上界快照。"""
    resource_stages = [
        stages["source_layout_audit"],
        stages["materialize_chunk_package"],
        stages["reassemble_and_hash_audit"],
        stages["evaluate_source_and_chunk"],
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


def adapter_exact(result: Mapping[str, Any]) -> bool:
    """核对 P2 mixed persistent 与 FP32 renderer/SH 输入。"""
    adapter = result["renderer_input_dtypes"] if "renderer_input_dtypes" in result else result["render_input_dtypes"]
    return bool(
        result["runtime_converted_field_audit"]["exact"]
        and adapter["all_renderer_inputs_float32"]
        and adapter["spherical_harmonics_inputs_float32"]
        and not adapter["autocast_enabled"]
        and int(result["persistent_parameter_bytes_by_dtype"]["bytes_by_dtype"].get("float16", 0))
        > 0
    )


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
    evaluation = stages["evaluate_source_and_chunk"]
    quality = {
        arm: load_json_artifact(args.run_dir, record)
        for arm, record in evaluation["quality_artifacts"].items()
    }
    runtime_stage = stages["runtime_profile_both_arms"]
    runtime = runtime_stage["arm_rows"]
    materialized = stages["materialize_chunk_package"]
    reassembled = stages["reassemble_and_hash_audit"]
    package_audit = reassembled["package_audit"]
    comparison = reassembled["checkpoint_comparison"]
    source_inputs_exact = all(
        stages[name].get("source_inputs_unchanged", True) for name in PRIOR_STAGES
    )
    adapter_contract_exact = all(
        adapter_exact(row)
        for row in (
            quality["p3-source"],
            quality["p3-chunk-package"],
            runtime["p3-source"],
            runtime["p3-chunk-package"],
        )
    )
    runtime_matrix_exact = runtime_stage["matrix"] == {
        "frames": [10, 100, 190],
        "cameras": [0, 1, 2],
        "sample_count_per_arm": 9,
    } and all(
        row["matrix_complete_and_unique"]
        and row["resolution_exact"]
        and row["synchronized_timing_complete"]
        and math.isfinite(float(row["render_p50_seconds"]))
        and math.isfinite(float(row["render_p95_seconds"]))
        and math.isfinite(float(row["aggregate_fps"]))
        for row in runtime.values()
    )
    candidate = {
        "id": "p3-chunk-package",
        "exact_static_and_actor_asset_inventory": bool(
            materialized["package_counts_exact"]
            and package_audit["static_cell_membership_exact"]
            and package_audit["actor_membership_exact"]
            and package_audit["indices_unique_disjoint_exhaustive"]
        ),
        "exact_row_fields_and_shared_skeleton_without_duplication": bool(
            package_audit["row_fields_exact"]
            and reassembled["shared_state_preserved_and_source_rows_not_duplicated"]
            and comparison["non_tensor_values_exact"]
        ),
        "exact_package_manifest_hashes_bytes_counts_bounds_and_indices": bool(
            package_audit["manifest_records_exact"]
            and package_audit["payload_files_verified"] == 158
            and materialized["package_file_count_including_manifest"] == 159
        ),
        "bitwise_exact_full_checkpoint_reassembly_and_reload": bool(
            comparison["all_exact"]
            and quality["p3-chunk-package"]["checkpoint_reload_exact"]
            and runtime["p3-chunk-package"]["checkpoint_reload_exact"]
        ),
        "p2_mixed_precision_runtime_adapter_exact": adapter_contract_exact,
        "source_baseline_replay_matches_p2_exact": bool(
            quality["p3-source"]["baseline_p2_replay_pass"]
        ),
        "all_57_rgb_hashes_and_all_31_quality_endpoints_exact": bool(
            quality["p3-chunk-package"]["candidate_per_view_rgb_sha256_exact"]
            and quality["p3-chunk-package"]["candidate_source_endpoint_replay_pass"]
            and quality["p3-chunk-package"]["all_exact_quality_gates_pass"]
        ),
        "all_quality_endpoints_complete_and_finite": bool(
            quality["p3-source"]["all_endpoints_complete_and_finite"]
            and quality["p3-chunk-package"]["all_endpoints_complete_and_finite"]
        ),
        "source_inputs_unchanged": source_inputs_exact,
        "resources_within_frozen_ceilings": resources["status"] == "passed",
        "package_bytes": int(materialized["package_bytes_including_manifest"]),
        "source_checkpoint_bytes": int(protocol["selected_asset"]["checkpoint"]["bytes"]),
        "package_minus_source_bytes": int(materialized["package_bytes_including_manifest"])
        - int(protocol["selected_asset"]["checkpoint"]["bytes"]),
        "quality_artifact": evaluation["quality_artifacts"]["p3-chunk-package"],
        "runtime": runtime["p3-chunk-package"],
        "materialization_stage": materialized,
        "reassembly_stage": reassembled,
    }
    source = {
        "id": "p3-source",
        "checkpoint_bytes": int(protocol["selected_asset"]["checkpoint"]["bytes"]),
        "source_baseline_replay_matches_p2_exact": bool(
            quality["p3-source"]["baseline_p2_replay_pass"]
        ),
        "all_quality_endpoints_complete_and_finite": quality["p3-source"][
            "all_endpoints_complete_and_finite"
        ],
        "source_inputs_unchanged": source_inputs_exact,
        "quality_artifact": evaluation["quality_artifacts"]["p3-source"],
        "runtime": runtime["p3-source"],
    }
    selection = select_chunk_arm(candidate)
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
        "selected_storage_bytes": (
            candidate["package_bytes"]
            if selection["selected_arm"] == "p3-chunk-package"
            else source["checkpoint_bytes"]
        ),
        "resource_snapshot": resources,
        "stage_ledger": ledger,
        "runtime_matrix_exact": runtime_matrix_exact,
        "runtime_values_used_for_selection": False,
        "post_hoc_chunk_size_merge_field_policy_or_threshold_added": False,
        "minimum_rerun_unit": "aggregate_and_downstream",
    }
    write_stage(args.run_dir, manifest, "aggregate", stage)
    print(json.dumps({"status": "aggregate_complete", "selection": selection}, sort_keys=True))


if __name__ == "__main__":
    main()
