#!/usr/bin/env python
"""收口 A4-P3 的 21 项审计、资源证据、选择裁决与终态。"""

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
_ACTIVE_RUN_DIR: Path | None = None
ALL_STAGES = (
    "input_audit",
    "source_layout_audit",
    "materialize_chunk_package",
    "reassemble_and_hash_audit",
    "evaluate_source_and_chunk",
    "runtime_profile_both_arms",
    "aggregate",
    "resume_audit",
)
RESOURCE_STAGES = (
    "source_layout_audit",
    "materialize_chunk_package",
    "reassemble_and_hash_audit",
    "evaluate_source_and_chunk",
    "runtime_profile_both_arms",
)


from motion_proj.worldsim_v3.chunk_package import select_chunk_arm
from scripts.aggregate_worldsim_v3_a4_p3 import adapter_exact
from scripts.run_worldsim_v3_a4_p3_chunk import (
    atomic_json,
    cgroup_memory_events,
    directory_bytes,
    directory_digest,
    load_stage,
    sha256_file,
)
from scripts.validate_worldsim_v3_a4_p3_chunk_protocol import (
    validate_inputs,
    validate_schema,
)


def load_json_artifact(run_dir: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    path = run_dir / str(record["path"])
    if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"A4-P3 finalization artifact drift: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_resource_audit(
    run_dir: Path,
    manifest: Mapping[str, Any],
    stages: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """按全运行 wall/disk/OOM 与五个资源阶段的峰值收口。"""
    resources = [stages[name] for name in RESOURCE_STAGES]
    before = manifest["preflight"]["cgroup_memory_events"]
    after = cgroup_memory_events()
    measured = {
        "wall_time_seconds": (
            datetime.now(timezone.utc) - datetime.fromisoformat(manifest["started_at"])
        ).total_seconds(),
        "peak_torch_allocated_mib": max(
            float(row["peak_torch_allocated_mib"]) for row in resources
        ),
        "peak_torch_reserved_mib": max(
            float(row["peak_torch_reserved_mib"]) for row in resources
        ),
        "peak_nvidia_process_memory_mib_sampled": max(
            int(row["peak_nvidia_process_memory_mib_sampled"]) for row in resources
        ),
        "peak_cgroup_memory_bytes_sampled": max(
            int(row["peak_cgroup_memory_bytes_sampled"]) for row in resources
        ),
        "run_bytes_before_finalization": directory_bytes(run_dir),
        "disk_free_bytes_after": shutil.disk_usage(run_dir).free,
        "oom_events_delta": after.get("oom", 0) - before.get("oom", 0),
        "oom_kill_events_delta": after.get("oom_kill", 0) - before.get("oom_kill", 0),
        "sampling_errors": [
            error for row in resources for error in row.get("sampling_errors", [])
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
        "run_bytes": measured["run_bytes_before_finalization"]
        > int(ceilings["run_bytes"]),
        "disk_free_floor_bytes": measured["disk_free_bytes_after"]
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
    global _ACTIVE_RUN_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    _ACTIVE_RUN_DIR = args.run_dir
    protocol = OmegaConf.to_container(OmegaConf.load(args.protocol), resolve=True)
    validate_schema(protocol)
    current_inputs = validate_inputs(protocol)
    manifest_path = args.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stages = {name: load_stage(args.run_dir, manifest, name) for name in ALL_STAGES}
    resource_audit = build_resource_audit(args.run_dir, manifest, stages, protocol)
    resource_path = args.run_dir / "artifacts" / "resource_audit.json"
    atomic_json(resource_path, resource_audit)

    selected = protocol["selected_asset"]
    source_unchanged = all(
        sha256_file(Path(selected[name]["path"])) == selected[name]["sha256"]
        for name in ("checkpoint", "source_config", "actor_registry")
    )
    masks_contract = protocol["baseline_quality"]["actor_masks"]
    current_masks = directory_digest(Path(masks_contract["path"]), masks_contract["file_glob"])
    expected_masks = {
        "sha256": masks_contract["sha256"],
        "file_count": int(masks_contract["file_count"]),
        "total_bytes": int(masks_contract["total_bytes"]),
    }
    layout = stages["source_layout_audit"]
    materialized = stages["materialize_chunk_package"]
    reassembled = stages["reassemble_and_hash_audit"]
    evaluation = stages["evaluate_source_and_chunk"]
    quality = {
        arm: load_json_artifact(args.run_dir, record)
        for arm, record in evaluation["quality_artifacts"].items()
    }
    runtime_stage = stages["runtime_profile_both_arms"]
    runtime = runtime_stage["arm_rows"]
    aggregate = stages["aggregate"]
    resume = stages["resume_audit"]
    candidate_row = next(row for row in aggregate["arms"] if row["id"] == "p3-chunk-package")
    selection_replay = select_chunk_arm(candidate_row)
    package_manifest = load_json_artifact(args.run_dir, materialized["package_manifest"])
    package_audit = reassembled["package_audit"]
    comparison = reassembled["checkpoint_comparison"]
    forbidden_media = [
        str(path.relative_to(args.run_dir))
        for path in args.run_dir.rglob("*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".mp4", ".webm"}
    ]
    source_rows = quality["p3-source"]["baseline_p2_replay_rows"]
    candidate_rows = quality["p3-chunk-package"]["candidate_source_replay_rows"]
    rgb_rows = quality["p3-chunk-package"]["candidate_source_rgb_rows"]
    adapters_exact = all(
        adapter_exact(row)
        for row in (
            quality["p3-source"],
            quality["p3-chunk-package"],
            runtime["p3-source"],
            runtime["p3-chunk-package"],
        )
    )
    actor14 = package_manifest["actor_assets"][14]
    package_resume_artifacts = resume["actions"]["materialize_chunk_package"][
        "validated_artifacts"
    ]
    runtime_matrix_exact = aggregate["runtime_matrix_exact"] and all(
        row["matrix_complete_and_unique"]
        and row["resolution_exact"]
        and row["synchronized_timing_complete"]
        and math.isfinite(float(row["render_p50_seconds"]))
        and math.isfinite(float(row["render_p95_seconds"]))
        and math.isfinite(float(row["aggregate_fps"]))
        for row in runtime.values()
    )
    audits = {
        "all_fingerprinted_inputs_exact": current_inputs == manifest["input_audits"],
        "p2_canonical_evidence_done_and_mixed_selected": stages["input_audit"]["p2_status"]
        == "done"
        and stages["input_audit"]["p2_selected_arm"] == "p2-gs-param-fp16"
        and stages["input_audit"]["p2_method_state"]
        == "selected_mixed_precision_parameter_storage_fp32_render"
        and stages["input_audit"]["p2_audit_count"] == 19
        and stages["input_audit"]["p2_all_audits_pass"]
        and stages["input_audit"]["p2_terminal"] == {"failure": None, "status": "done"},
        "source_checkpoint_registry_config_and_masks_unchanged": source_unchanged
        and current_masks == expected_masks,
        "source_precision_and_row_tensor_schema_exact": layout[
            "source_precision_and_row_tensor_schema_exact"
        ]
        and layout["source_layout"]["row_tensor_schema"]["Background"][
            "row_tensor_count"
        ]
        == 25
        and layout["source_layout"]["row_tensor_schema"]["RigidNodes"][
            "row_tensor_count"
        ]
        == 26,
        "static_grid_contract_and_source_inventory_exact": layout[
            "static_grid_contract_and_source_inventory_exact"
        ]
        and layout["source_layout"]["static_inventory"]["occupied_chunk_count"] == 133
        and layout["source_layout"]["static_inventory"]["inventory_sha256"]
        == protocol["static_chunk_contract"]["expected_source_inventory"]["inventory_sha256"],
        "static_chunk_indices_sorted_disjoint_exhaustive_and_cell_membership_exact": package_audit[
            "indices_unique_disjoint_exhaustive"
        ]
        and package_audit["static_cell_membership_exact"]
        and package_audit["model_coverage"]["Background"]["missing"] == 0
        and package_audit["model_coverage"]["Background"]["duplicated"] == 0,
        "actor_assets_24_counts_indices_and_empty_actor_exact": package_audit[
            "indices_unique_disjoint_exhaustive"
        ]
        and package_audit["actor_membership_exact"]
        and len(package_manifest["actor_assets"]) == 24
        and actor14["actor_index"] == 14
        and actor14["availability"] == "unavailable_empty"
        and actor14["row_count"] == 0
        and package_audit["model_coverage"]["RigidNodes"]["missing"] == 0
        and package_audit["model_coverage"]["RigidNodes"]["duplicated"] == 0,
        "chunk_row_field_dtypes_shapes_and_values_exact": package_audit[
            "row_fields_exact"
        ]
        and comparison["tensor_shape_dtype_value_sha256_exact"],
        "shared_state_preserved_and_source_rows_not_duplicated": reassembled[
            "shared_state_preserved_and_source_rows_not_duplicated"
        ]
        and comparison["non_tensor_values_exact"],
        "package_manifest_file_hashes_bytes_counts_and_bounds_exact": materialized[
            "package_counts_exact"
        ]
        and materialized["package_file_count_including_manifest"] == 159
        and package_audit["manifest_records_exact"]
        and package_audit["payload_files_verified"] == 158,
        "reassembled_checkpoint_schema_dtype_and_tensor_hashes_exact": comparison[
            "recursive_container_schema_exact"
        ]
        and comparison["tensor_paths_exact"]
        and comparison["tensor_shape_dtype_value_sha256_exact"]
        and comparison["all_exact"],
        "reassembled_checkpoint_reload_counts_registry_and_adapter_exact": quality[
            "p3-chunk-package"
        ]["checkpoint_reload_exact"]
        and runtime["p3-chunk-package"]["checkpoint_reload_exact"]
        and quality["p3-chunk-package"]["runtime_converted_field_audit"]["exact"]
        and runtime["p3-chunk-package"]["runtime_converted_field_audit"]["exact"],
        "p2_runtime_precision_adapter_contract_reused_exactly": adapters_exact,
        "source_baseline_replay_matches_p2_exact": len(source_rows) == 31
        and all(row["passed"] for row in source_rows)
        and quality["p3-source"]["baseline_p2_replay_pass"],
        "frozen_actor_masks_reused_without_regeneration": current_masks == expected_masks
        and evaluation["frozen_masks_reused_exactly"]
        and not evaluation["candidate_mask_regeneration_performed"],
        "all_57_rgb_hashes_and_31_quality_endpoints_exact": len(rgb_rows) == 57
        and all(row["passed"] for row in rgb_rows)
        and len(candidate_rows) == 31
        and all(row["passed"] for row in candidate_rows)
        and quality["p3-chunk-package"]["all_exact_quality_gates_pass"],
        "runtime_matrix_and_statistics_complete": runtime_matrix_exact
        and runtime_stage["runtime_rgb_sha256_exact_between_arms"]
        and runtime_stage["performance_values_report_only"]
        and not runtime_stage["selective_loading_or_view_culling_performed"],
        "no_training_optimizer_source_mutation_or_raw_media": not forbidden_media
        and not stages["input_audit"]["training_optimizer_or_source_mutation_performed"]
        and not layout["training_optimizer_or_source_mutation_performed"]
        and not materialized["training_optimizer_or_source_mutation_performed"]
        and not reassembled["training_optimizer_or_source_mutation_performed"]
        and evaluation["no_optimizer_constructed_or_step_executed"]
        and runtime_stage["no_optimizer_constructed_or_step_executed"]
        and not evaluation["raw_render_media_written"]
        and not runtime_stage["raw_render_media_written"]
        and not materialized["source_checkpoint_copied"]
        and not materialized["persistent_reassembled_checkpoint_written"]
        and not reassembled["persistent_reassembled_checkpoint_written"],
        "resources_within_frozen_ceilings": resource_audit["status"] == "passed",
        "selection_and_exact_fallback_rule_exact": aggregate["selection"]
        == selection_replay
        and (
            (
                selection_replay["selected_arm"] == "p3-source"
                and selection_replay["fallback_exact_alias"]
            )
            or (
                selection_replay["selected_arm"] == "p3-chunk-package"
                and not selection_replay["fallback_exact_alias"]
                and candidate_row[
                    "all_57_rgb_hashes_and_all_31_quality_endpoints_exact"
                ]
            )
        )
        and not aggregate["runtime_values_used_for_selection"]
        and not aggregate["post_hoc_chunk_size_merge_field_policy_or_threshold_added"],
        "dry_run_resume_reuses_completed_stages_without_torch_or_gpu": resume[
            "all_completed_stages_reused"
        ]
        and not resume["torch_imported"]
        and not resume["gpu_launch_observed"]
        and len(resume["actions"]) == 7
        and len(package_resume_artifacts) == 159,
    }
    if set(audits) != set(protocol["required_audits"]):
        raise RuntimeError("A4-P3 required audit schema drift")
    if not all(audits.values()):
        failed = sorted(name for name, passed in audits.items() if not passed)
        raise RuntimeError("A4-P3 required audits failed: " + ", ".join(failed))

    selection = aggregate["selection"]
    if selection["selected_arm"] == "p3-source":
        selected_asset = {
            "arm": "p3-source",
            "storage": "immutable_p2_selected_checkpoint_exact_alias",
            "checkpoint": selected["checkpoint"],
            "source_config": selected["source_config"],
            "actor_registry": selected["actor_registry"],
        }
    else:
        selected_asset = {
            "arm": "p3-chunk-package",
            "storage": "run_local_exact_chunk_package",
            "package_root": materialized["package_root"],
            "package_manifest": materialized["package_manifest"],
            "package_bytes": materialized["package_bytes_including_manifest"],
            "package_counts": materialized["package_counts"],
            "source_config": selected["source_config"],
            "actor_registry": selected["actor_registry"],
            "source_checkpoint": selected["checkpoint"],
        }
    stage_ledger = dict(aggregate["stage_ledger"])
    for name, stage in (("aggregate", aggregate), ("resume_audit", resume)):
        stage_ledger[name] = {
            "status": "done",
            "duration_seconds": float(stage["duration_seconds"]),
            "stage_sha256": manifest["stage_hashes"][name],
            "minimum_rerun_unit": stage["minimum_rerun_unit"],
        }
    summary = {
        "status": "done",
        "task_id": protocol["task_id"],
        "profile_id": protocol["profile_id"],
        "scene": protocol["scene"],
        "seed": int(protocol["seed"]),
        "protocol_sha256": sha256_file(args.protocol),
        "project_commit": manifest["project_commit"],
        "p3_experiment_terminal": "done",
        "method_state": selection["method_state"],
        "selection": selection,
        "selected_asset": selected_asset,
        "arms": aggregate["arms"],
        "package_manifest": materialized["package_manifest"],
        "quality_artifacts": evaluation["quality_artifacts"],
        "runtime": runtime,
        "resources": resource_audit,
        "resources_sha256": sha256_file(resource_path),
        "stage_ledger": stage_ledger,
        "audits": audits,
        "claim_boundary": protocol["claim_boundary"],
        "next_action": "close_p3_and_a4_then_prepare_ws_v3_r0_integration",
    }
    summary_path = args.run_dir / "summary.json"
    atomic_json(summary_path, summary)
    manifest["status"] = "done"
    manifest["summary_sha256"] = sha256_file(summary_path)
    manifest["resource_audit_sha256"] = sha256_file(resource_path)
    manifest["package_manifest_sha256"] = materialized["package_manifest"]["sha256"]
    manifest["source_inputs_after"] = {
        name: sha256_file(Path(selected[name]["path"]))
        for name in ("checkpoint", "source_config", "actor_registry")
    }
    atomic_json(manifest_path, manifest, replace=True)
    atomic_json(
        args.run_dir / "terminal.json", {"status": "done", "failure": None}, replace=True
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if _ACTIVE_RUN_DIR is not None and _ACTIVE_RUN_DIR.exists():
            atomic_json(
                _ACTIVE_RUN_DIR / "terminal.json",
                {
                    "status": "blocked",
                    "failure": {
                        "code": "A4_P3_FINALIZATION_FAILED",
                        "detail": f"{type(error).__name__}: {error}",
                    },
                },
                replace=True,
            )
        raise
