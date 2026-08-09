#!/usr/bin/env python
"""收口 A4-P2 的 19 项审计、资源证据、选择裁决与终态。"""

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
PROTOCOL = PROJECT / "configs/worldsim_v3/a4_p2_mixed_precision_protocol_v1.yaml"
_ACTIVE_RUN_DIR: Path | None = None
ALL_STAGES = (
    "input_audit",
    "source_dtype_audit",
    "materialize_p2_gs_param_fp16",
    "evaluate_p2_source_and_candidate",
    "runtime_profile_both_arms",
    "aggregate",
    "resume_audit",
)
RESOURCE_STAGES = (
    "evaluate_p2_source_and_candidate",
    "runtime_profile_both_arms",
)


from motion_proj.worldsim_v3.mixed_precision import select_precision_arm
from scripts.run_worldsim_v3_a4_p2_precision import (
    atomic_json,
    cgroup_memory_events,
    directory_bytes,
    directory_digest,
    load_stage,
    sha256_file,
)
from scripts.validate_worldsim_v3_a4_p2_mixed_precision_protocol import (
    validate_inputs,
    validate_schema,
)


def load_json_artifact(run_dir: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    path = run_dir / record["path"]
    if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"A4-P2 finalization artifact drift: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_resource_audit(
    run_dir: Path,
    manifest: Mapping[str, Any],
    stages: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
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
    materialized = stages["materialize_p2_gs_param_fp16"]
    evaluation = stages["evaluate_p2_source_and_candidate"]
    quality = {
        arm: load_json_artifact(args.run_dir, record)
        for arm, record in evaluation["quality_artifacts"].items()
    }
    runtime_stage = stages["runtime_profile_both_arms"]
    runtime = runtime_stage["arm_rows"]
    aggregate = stages["aggregate"]
    resume = stages["resume_audit"]
    candidate_row = next(row for row in aggregate["arms"] if row["id"] == "p2-gs-param-fp16")
    selection_replay = select_precision_arm(candidate_row)
    forbidden_media = [
        str(path.relative_to(args.run_dir))
        for path in args.run_dir.rglob("*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".mp4", ".webm"}
    ]
    quality_rows = quality["p2-gs-param-fp16"]["quality_safeguard_rows"]
    candidate_quality_adapter = quality["p2-gs-param-fp16"]["renderer_input_dtypes"]
    candidate_runtime_adapter = runtime["p2-gs-param-fp16"]["render_input_dtypes"]
    source_quality_adapter = quality["p2-source"]["renderer_input_dtypes"]
    source_runtime_adapter = runtime["p2-source"]["render_input_dtypes"]
    source_inventory = runtime["p2-source"]["persistent_parameter_bytes_by_dtype"]
    candidate_inventory = runtime["p2-gs-param-fp16"]["persistent_parameter_bytes_by_dtype"]
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
    audits = {
        "all_fingerprinted_inputs_exact": current_inputs == manifest["input_audits"],
        "p1_canonical_evidence_done_and_source_selected": stages["input_audit"]["p1_status"]
        == "done"
        and stages["input_audit"]["p1_selected_arm"] == "p1-source"
        and stages["input_audit"]["p1_fallback_exact_alias"]
        and stages["input_audit"]["p1_terminal"] == {"failure": None, "status": "done"},
        "source_checkpoint_registry_config_and_masks_unchanged": source_unchanged
        and current_masks == expected_masks,
        "source_checkpoint_dtype_schema_exact": stages["source_dtype_audit"][
            "source_checkpoint_dtype_schema_exact"
        ],
        "converted_field_set_and_float16_bytes_exact": materialized[
            "converted_field_set_and_float16_bytes_exact"
        ]
        and materialized["candidate_checkpoint_write_count"] == 1
        and not materialized["source_checkpoint_copied"],
        "converted_field_roundtrip_error_complete_and_finite": materialized[
            "converted_field_roundtrip_error_complete_and_finite"
        ],
        "preserved_fields_and_checkpoint_schema_exact": materialized[
            "preserved_fields_and_checkpoint_schema_exact"
        ],
        "candidate_registry_counts_and_indices_exact": materialized[
            "candidate_registry_counts_and_indices_exact"
        ],
        "runtime_persistent_parameter_dtypes_exact": quality["p2-gs-param-fp16"][
            "runtime_converted_field_audit"
        ]["exact"]
        and runtime["p2-gs-param-fp16"]["runtime_converted_field_audit"]["exact"]
        and quality["p2-source"]["runtime_converted_field_audit"]["exact"]
        and runtime["p2-source"]["runtime_converted_field_audit"]["exact"],
        "renderer_inputs_float32_and_autocast_disabled": candidate_quality_adapter[
            "all_renderer_inputs_float32"
        ]
        and candidate_runtime_adapter["all_renderer_inputs_float32"]
        and candidate_quality_adapter["spherical_harmonics_inputs_float32"]
        and candidate_runtime_adapter["spherical_harmonics_inputs_float32"]
        and source_quality_adapter["all_renderer_inputs_float32"]
        and source_runtime_adapter["all_renderer_inputs_float32"]
        and source_quality_adapter["spherical_harmonics_inputs_float32"]
        and source_runtime_adapter["spherical_harmonics_inputs_float32"]
        and not candidate_quality_adapter["autocast_enabled"]
        and not candidate_runtime_adapter["autocast_enabled"],
        "source_baseline_replay_matches_p1_exact": quality["p2-source"][
            "baseline_p1_replay_pass"
        ]
        and len(quality["p2-source"]["baseline_replay_rows"]) == 31
        and all(row["passed"] for row in quality["p2-source"]["baseline_replay_rows"]),
        "frozen_actor_masks_reused_without_regeneration": current_masks == expected_masks
        and evaluation["frozen_masks_reused_exactly"]
        and not evaluation["candidate_mask_regeneration_performed"],
        "all_quality_endpoints_complete_and_finite": all(
            result["all_endpoints_complete_and_finite"] for result in quality.values()
        ),
        "candidate_quality_safeguards_applied_exactly": len(quality_rows) == 31
        and all(isinstance(row["passed"], bool) for row in quality_rows),
        "checkpoint_reduction_and_runtime_matrix_exact": materialized[
            "checkpoint_bytes_strictly_less_than_source"
        ]
        and int(materialized["candidate_checkpoint"]["bytes"])
        < int(selected["checkpoint"]["bytes"])
        and runtime_matrix_exact
        and int(candidate_inventory["bytes_by_dtype"].get("float16", 0)) > 0
        and int(candidate_inventory["total_bytes"]) < int(source_inventory["total_bytes"]),
        "no_training_optimizer_source_mutation_or_raw_media": not forbidden_media
        and not stages["input_audit"]["training_optimizer_or_source_mutation_performed"]
        and not stages["source_dtype_audit"]["training_optimizer_or_source_mutation_performed"]
        and not materialized["training_optimizer_or_source_mutation_performed"]
        and evaluation["no_optimizer_constructed_or_step_executed"]
        and runtime_stage["no_optimizer_constructed_or_step_executed"]
        and not evaluation["raw_render_media_written"]
        and not runtime_stage["raw_render_media_written"],
        "resources_within_frozen_ceilings": resource_audit["status"] == "passed",
        "selection_and_exact_fallback_rule_exact": aggregate["selection"] == selection_replay
        and (
            (
                selection_replay["selected_arm"] == "p2-source"
                and selection_replay["fallback_exact_alias"]
            )
            or (
                selection_replay["selected_arm"] == "p2-gs-param-fp16"
                and not selection_replay["fallback_exact_alias"]
                and candidate_row["all_quality_safeguards_pass"]
            )
        )
        and not aggregate["post_hoc_field_policy_or_threshold_added"],
        "dry_run_resume_reuses_completed_stages_without_gpu_launch": resume[
            "all_completed_stages_reused"
        ]
        and not resume["torch_imported"]
        and not resume["gpu_launch_observed"]
        and len(resume["actions"]) == 6,
    }
    if set(audits) != set(protocol["required_audits"]):
        raise RuntimeError("A4-P2 required audit schema drift")
    if not all(audits.values()):
        failed = sorted(name for name, passed in audits.items() if not passed)
        raise RuntimeError("A4-P2 required audits failed: " + ", ".join(failed))

    selection = aggregate["selection"]
    if selection["selected_arm"] == "p2-source":
        selected_asset = {
            "arm": "p2-source",
            "storage": "immutable_source_reference_exact_alias",
            "checkpoint": selected["checkpoint"],
            "actor_registry": selected["actor_registry"],
        }
    else:
        selected_asset = {
            "arm": "p2-gs-param-fp16",
            "storage": "run_local_atomic_mixed_precision_candidate",
            "checkpoint": materialized["candidate_checkpoint"],
            "actor_registry": materialized["candidate_registry"],
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
        "p2_experiment_terminal": "done",
        "method_state": selection["method_state"],
        "selection": selection,
        "selected_asset": selected_asset,
        "arms": aggregate["arms"],
        "quality_artifacts": evaluation["quality_artifacts"],
        "runtime": runtime,
        "resources": resource_audit,
        "resources_sha256": sha256_file(resource_path),
        "stage_ledger": stage_ledger,
        "audits": audits,
        "claim_boundary": protocol["claim_boundary"],
        "next_action": "close_p2_then_freeze_minimum_p3_chunk_protocol",
    }
    summary_path = args.run_dir / "summary.json"
    atomic_json(summary_path, summary)
    manifest["status"] = "done"
    manifest["summary_sha256"] = sha256_file(summary_path)
    manifest["resource_audit_sha256"] = sha256_file(resource_path)
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
                        "code": "A4_P2_FINALIZATION_FAILED",
                        "detail": f"{type(error).__name__}: {error}",
                    },
                },
                replace=True,
            )
        raise
