#!/usr/bin/env python
"""收口 A4-P1 的 21 项审计、资源证据、选择裁决与终态。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np
from omegaconf import OmegaConf


PROJECT = Path("/root/autodl-tmp/motion_proj")
PROTOCOL = PROJECT / "configs/worldsim_v3/a4_p1_contribution_prune_protocol_v1.yaml"
_ACTIVE_RUN_DIR: Path | None = None


from motion_proj.worldsim_v3.contribution_prune import array_sha256, select_largest_eligible_arm
from scripts.run_worldsim_v3_a4_p1_prune import (
    atomic_json,
    cgroup_memory_events,
    directory_bytes,
    directory_digest,
    load_stage,
    sha256_file,
)
from scripts.validate_worldsim_v3_a4_p1_contribution_prune_protocol import (
    validate_inputs,
    validate_schema,
)


ALL_STAGES = (
    "input_audit",
    "contribution_scan",
    "materialize_p1_b05",
    "evaluate_p1_source_and_b05",
    "materialize_p1_b10",
    "evaluate_p1_b10",
    "materialize_p1_b20",
    "evaluate_p1_b20",
    "runtime_profile_all_arms",
    "aggregate",
    "resume_audit",
)
RESOURCE_STAGES = (
    "contribution_scan",
    "evaluate_p1_source_and_b05",
    "evaluate_p1_b10",
    "evaluate_p1_b20",
    "runtime_profile_all_arms",
)


def load_json_artifact(run_dir: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    path = run_dir / record["path"]
    if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"A4-P1 finalization artifact drift: {path}")
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
        "peak_torch_allocated_mib": max(float(row["peak_torch_allocated_mib"]) for row in resources),
        "peak_torch_reserved_mib": max(float(row["peak_torch_reserved_mib"]) for row in resources),
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
        "wall_time_seconds": measured["wall_time_seconds"] > float(ceilings["wall_time_seconds"]),
        "peak_torch_allocated_mib": measured["peak_torch_allocated_mib"] > float(ceilings["peak_torch_allocated_mib"]),
        "peak_torch_reserved_mib": measured["peak_torch_reserved_mib"] > float(ceilings["peak_torch_reserved_mib"]),
        "peak_nvidia_process_memory_mib_sampled": measured["peak_nvidia_process_memory_mib_sampled"] > int(ceilings["peak_nvidia_process_memory_mib_sampled"]),
        "peak_cgroup_memory_bytes": measured["peak_cgroup_memory_bytes_sampled"] > int(ceilings["peak_cgroup_memory_bytes"]),
        "run_bytes": measured["run_bytes_before_finalization"] > int(ceilings["run_bytes"]),
        "disk_free_floor_bytes": measured["disk_free_bytes_after"] < int(ceilings["disk_free_floor_bytes"]),
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


def contribution_artifact_exact(
    run_dir: Path, stage: Mapping[str, Any], protocol: Mapping[str, Any]
) -> bool:
    record = stage["score_artifact"]
    path = run_dir / record["path"]
    if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
        return False
    with np.load(path, allow_pickle=False) as loaded:
        names = set(loaded.files)
        required = set(protocol["contribution_contract"]["score_artifact_required_arrays"])
        if not required.issubset(names) or names != set(record["arrays"]):
            return False
        return all(
            str(loaded[name].dtype) == meta["dtype"]
            and list(loaded[name].shape) == meta["shape"]
            and array_sha256(loaded[name]) == meta["sha256"]
            for name, meta in record["arrays"].items()
        )


def quality_artifacts(
    run_dir: Path, stages: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    records = {}
    for stage_name in (
        "evaluate_p1_source_and_b05",
        "evaluate_p1_b10",
        "evaluate_p1_b20",
    ):
        records.update(stages[stage_name]["quality_artifacts"])
    return {arm: load_json_artifact(run_dir, record) for arm, record in records.items()}


def expected_removal_exact(stage: Mapping[str, Any]) -> bool:
    fraction = float(stage["prune_fraction"])
    removed = sum(int(row["removed_count"]) for row in stage["per_asset_removed_count"])
    counts = stage["source_and_candidate_model_counts"]
    source_total = sum(int(value) for value in counts["source"].values())
    candidate_total = sum(int(value) for value in counts["candidate"].values())
    return (
        stage["candidate_grid_and_removal_counts_exact"]
        and removed == source_total - candidate_total
        and all(
            int(row["removed_count"]) == math.floor(int(row["source_count"]) * fraction)
            for row in stage["per_asset_removed_count"]
        )
    )


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
    materializations = [
        stages[name]
        for name in ("materialize_p1_b05", "materialize_p1_b10", "materialize_p1_b20")
    ]
    evaluations = [
        stages[name]
        for name in ("evaluate_p1_source_and_b05", "evaluate_p1_b10", "evaluate_p1_b20")
    ]
    quality = quality_artifacts(args.run_dir, stages)
    runtime = stages["runtime_profile_all_arms"]
    aggregate = stages["aggregate"]
    resume = stages["resume_audit"]
    mask_contract = protocol["baseline_quality"]["actor_masks"]
    current_masks = directory_digest(Path(mask_contract["path"]), mask_contract["file_glob"])
    forbidden_media = [
        str(path.relative_to(args.run_dir))
        for path in args.run_dir.rglob("*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".mp4", ".webm"}
    ]
    selection_replay = select_largest_eligible_arm(aggregate["arms"])
    selected_row = next(
        row for row in aggregate["arms"] if row["id"] == aggregate["selection"]["selected_arm"]
    )
    candidate_quality_rows = [
        row
        for arm, result in quality.items()
        if arm != "p1-source"
        for row in result["quality_safeguard_rows"]
    ]
    runtime_rows = runtime["arm_rows"]
    audits = {
        "all_fingerprinted_inputs_exact": current_inputs == manifest["input_audits"],
        "p0_and_p5_canonical_evidence_done": stages["input_audit"]["p0_status"] == "done"
        and stages["input_audit"]["p5_status"] == "done"
        and stages["input_audit"]["p5_terminal"] == {"failure": None, "status": "done"},
        "source_checkpoint_registry_and_config_unchanged": source_unchanged,
        "contribution_ranking_uses_train_only": stages["contribution_scan"]["ranking_partition"] == "train_only"
        and not stages["contribution_scan"]["heldout_influenced_ranking"],
        "contribution_score_schema_and_hashes_exact": contribution_artifact_exact(
            args.run_dir, stages["contribution_scan"], protocol
        ),
        "heldout_contribution_is_audit_only": not stages["contribution_scan"]["heldout_influenced_ranking"]
        and set(stages["contribution_scan"]["train_frames"]).isdisjoint(
            stages["contribution_scan"]["heldout_audit_frames"]
        ),
        "candidate_grid_and_removal_counts_exact": all(expected_removal_exact(stage) for stage in materializations),
        "every_row_aligned_tensor_and_ancestry_field_pruned_by_same_mask": all(
            stage["row_alignment_audit"]["exact"] for stage in materializations
        ),
        "invariant_checkpoint_fields_exact": all(
            stage["invariant_field_hashes_before_after"]["exact"]
            and stage["checkpoint_schema_exact"]
            for stage in materializations
        ),
        "candidate_checkpoints_reload_and_counts_exact": all(
            quality[arm]["checkpoint_reload_exact"]
            and quality[arm]["expected_model_counts_exact"]
            and runtime_rows[arm]["checkpoint_reload_exact"]
            for arm in ("p1-b05", "p1-b10", "p1-b20")
        ),
        "unavailable_actor_remains_explicitly_empty": all(
            stage["unavailable_actor_remains_explicitly_empty"] for stage in materializations
        )
        and all(result["unavailable_actor_remains_explicitly_empty"] for result in quality.values()),
        "frozen_baseline_masks_reused_exactly": current_masks
        == {
            "sha256": mask_contract["sha256"],
            "file_count": int(mask_contract["file_count"]),
            "total_bytes": int(mask_contract["total_bytes"]),
        }
        and all(stage["frozen_masks_reused_exactly"] for stage in evaluations),
        "baseline_replay_matches_frozen_metrics": quality["p1-source"]["baseline_historical_replay_pass"]
        and all(row["passed"] for row in quality["p1-source"]["historical_replay_rows"]),
        "all_quality_endpoints_complete_and_finite": all(
            result["all_endpoints_complete_and_finite"] for result in quality.values()
        ),
        "candidate_quality_safeguards_applied_exactly": len(candidate_quality_rows) == 3 * 31
        and all(isinstance(row["passed"], bool) for row in candidate_quality_rows),
        "checkpoint_and_gaussian_reduction_exact": all(
            expected_removal_exact(stage)
            and int(stage["candidate_checkpoint"]["bytes"]) < int(selected["checkpoint"]["bytes"])
            for stage in materializations
        ),
        "runtime_matrix_and_statistics_exact": runtime["matrix"]
        == {
            "frames": [10, 100, 190],
            "cameras": [0, 1, 2],
            "sample_count_per_arm": 9,
        }
        and all(
            row["matrix_complete_and_unique"]
            and row["resolution_exact"]
            and row["synchronized_timing_complete"]
            and math.isfinite(float(row["render_p50_seconds"]))
            and math.isfinite(float(row["render_p95_seconds"]))
            and math.isfinite(float(row["aggregate_fps"]))
            for row in runtime_rows.values()
        ),
        "no_training_optimizer_or_raw_render_media": not forbidden_media
        and all(stage["no_optimizer_constructed_or_step_executed"] for stage in [stages["contribution_scan"], *evaluations, runtime])
        and all(not stage["raw_render_media_written"] for stage in [stages["contribution_scan"], *evaluations, runtime]),
        "resources_within_frozen_ceilings": resource_audit["status"] == "passed",
        "selection_and_fallback_rule_exact": aggregate["selection"] == selection_replay
        and (
            (selection_replay["selected_arm"] == "p1-source" and selection_replay["fallback_exact_alias"])
            or (
                selection_replay["selected_arm"] != "p1-source"
                and not selection_replay["fallback_exact_alias"]
                and selected_row["all_quality_safeguards_pass"]
            )
        )
        and not aggregate["post_hoc_arm_or_threshold_added"],
        "dry_run_resume_reuses_completed_stages_without_gpu_launch": resume["all_completed_stages_reused"]
        and not resume["torch_imported"]
        and not resume["gpu_launch_observed"]
        and len(resume["actions"]) == 10,
    }
    if set(audits) != set(protocol["required_audits"]):
        raise RuntimeError("A4-P1 required audit schema drift")
    if not all(audits.values()):
        failed = sorted(name for name, passed in audits.items() if not passed)
        raise RuntimeError("A4-P1 required audits failed: " + ", ".join(failed))

    selection = aggregate["selection"]
    if selection["selected_arm"] == "p1-source":
        selected_asset = {
            "arm": "p1-source",
            "storage": "immutable_source_reference_exact_alias",
            "checkpoint": selected["checkpoint"],
            "actor_registry": selected["actor_registry"],
        }
    else:
        materialized = stages[f"materialize_{selection['selected_arm'].replace('-', '_')}"]
        selected_asset = {
            "arm": selection["selected_arm"],
            "storage": "run_local_atomic_candidate_checkpoint",
            "checkpoint": materialized["candidate_checkpoint"],
            "actor_registry": materialized["candidate_registry"],
        }
    stage_ledger = dict(aggregate["stage_ledger"])
    stage_ledger["aggregate"] = {
        "status": "done",
        "duration_seconds": float(aggregate["duration_seconds"]),
        "stage_sha256": manifest["stage_hashes"]["aggregate"],
        "minimum_rerun_unit": aggregate["minimum_rerun_unit"],
    }
    stage_ledger["resume_audit"] = {
        "status": "done",
        "duration_seconds": float(resume["duration_seconds"]),
        "stage_sha256": manifest["stage_hashes"]["resume_audit"],
        "minimum_rerun_unit": resume["minimum_rerun_unit"],
    }
    summary = {
        "status": "done",
        "task_id": protocol["task_id"],
        "profile_id": protocol["profile_id"],
        "scene": protocol["scene"],
        "seed": int(protocol["seed"]),
        "protocol_sha256": sha256_file(args.protocol),
        "project_commit": manifest["project_commit"],
        "p1_experiment_terminal": "done",
        "method_state": selection["method_state"],
        "selection": selection,
        "selected_asset": selected_asset,
        "arms": aggregate["arms"],
        "quality_artifacts": {
            arm: next(
                record
                for stage in evaluations
                for name, record in stage["quality_artifacts"].items()
                if name == arm
            )
            for arm in quality
        },
        "runtime": runtime["arm_rows"],
        "resources": resource_audit,
        "resources_sha256": sha256_file(resource_path),
        "stage_ledger": stage_ledger,
        "audits": audits,
        "claim_boundary": protocol["claim_boundary"],
        "next_action": "close_p1_then_freeze_and_execute_minimum_p2_or_p3",
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
    atomic_json(args.run_dir / "terminal.json", {"status": "done", "failure": None}, replace=True)
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
                        "code": "A4_P1_FINALIZATION_FAILED",
                        "detail": f"{type(error).__name__}: {error}",
                    },
                },
                replace=True,
            )
        raise
