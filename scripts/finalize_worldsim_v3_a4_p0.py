#!/usr/bin/env python
"""收口 A4-P0 profile 的 stage、资源与恢复审计。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from omegaconf import OmegaConf


PROJECT = Path("/root/autodl-tmp/motion_proj")
PROTOCOL = PROJECT / "configs/worldsim_v3/a4_p0_profile_protocol_v2.yaml"
_ACTIVE_RUN_DIR: Path | None = None


from scripts.run_worldsim_v3_a3_s_b_paired_smoke import (
    atomic_json,
    cgroup_memory_events,
    directory_bytes,
    sha256_file,
)
from scripts.validate_worldsim_v3_a4_p0_profile_protocol import (
    validate_inputs,
    validate_schema,
)


def load_stage(run_dir: Path, manifest: dict, name: str) -> dict:
    path = run_dir / "stages" / f"{name}.json"
    if sha256_file(path) != manifest["stage_hashes"][name]:
        raise RuntimeError(f"A4-P0 stage hash drift: {name}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_resource_audit(
    *, run_dir: Path, manifest: dict, runtime: dict, protocol: dict
) -> dict:
    ceilings = protocol["resource_ceilings"]
    started = datetime.fromisoformat(manifest["started_at"])
    wall = (datetime.now(timezone.utc) - started).total_seconds()
    events_before = manifest["preflight"]["cgroup_memory_events"]
    events_after = cgroup_memory_events()
    measured = {
        "wall_time_seconds": wall,
        "peak_torch_allocated_mib": runtime["peak_torch_allocated_mib"],
        "peak_torch_reserved_mib": runtime["peak_torch_reserved_mib"],
        "peak_nvidia_process_memory_mib_sampled": runtime[
            "peak_nvidia_process_memory_mib_sampled"
        ],
        "peak_cgroup_memory_bytes_sampled": runtime["peak_cgroup_memory_bytes_sampled"],
        "run_bytes_before_finalization": directory_bytes(run_dir),
        "disk_free_bytes_after": shutil.disk_usage(run_dir).free,
        "oom_events_delta": events_after.get("oom", 0) - events_before.get("oom", 0),
        "oom_kill_events_delta": events_after.get("oom_kill", 0)
        - events_before.get("oom_kill", 0),
    }
    violations = {
        "wall_time_seconds": measured["wall_time_seconds"] > ceilings["wall_time_seconds"],
        "peak_torch_allocated_mib": measured["peak_torch_allocated_mib"]
        > ceilings["peak_torch_allocated_mib"],
        "peak_torch_reserved_mib": measured["peak_torch_reserved_mib"]
        > ceilings["peak_torch_reserved_mib"],
        "peak_nvidia_process_memory_mib_sampled": measured[
            "peak_nvidia_process_memory_mib_sampled"
        ]
        is None
        or measured["peak_nvidia_process_memory_mib_sampled"]
        > ceilings["peak_nvidia_process_memory_mib_sampled"],
        "peak_cgroup_memory_bytes": measured["peak_cgroup_memory_bytes_sampled"]
        is None
        or measured["peak_cgroup_memory_bytes_sampled"]
        > ceilings["peak_cgroup_memory_bytes"],
        "resource_sampling_errors": bool(runtime["sampling_errors"]),
        "run_bytes": measured["run_bytes_before_finalization"] > ceilings["run_bytes"],
        "disk_free_floor_bytes": measured["disk_free_bytes_after"]
        < ceilings["disk_free_floor_bytes"],
        "oom_events_delta": measured["oom_events_delta"] != ceilings["oom_events_delta"],
        "oom_kill_events_delta": measured["oom_kill_events_delta"]
        != ceilings["oom_kill_events_delta"],
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
    validate_inputs(protocol)
    manifest_path = args.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stages = {
        name: load_stage(args.run_dir, manifest, name)
        for name in ("inventory", "runtime_probe", "aggregate", "resume_audit")
    }
    runtime = stages["runtime_probe"]
    resume = stages["resume_audit"]
    checkpoint = Path(protocol["selected_asset"]["checkpoint"]["path"])
    registry = Path(protocol["selected_asset"]["actor_registry"]["path"])
    forbidden = [
        str(path.relative_to(args.run_dir))
        for path in args.run_dir.rglob("*")
        if path.suffix.lower() in {".pth", ".png", ".jpg", ".jpeg", ".mp4"}
    ]
    resource_audit = build_resource_audit(
        run_dir=args.run_dir,
        manifest=manifest,
        runtime=runtime,
        protocol=protocol,
    )
    resource_path = args.run_dir / "artifacts" / "resource_audit.json"
    atomic_json(resource_path, resource_audit)
    audits = {
        "selected_input_hashes_and_bytes_exact": True,
        "historical_evidence_hashes_exact": True,
        "no_rejected_r1_input": manifest["input_audits"][
            "selected_asset.checkpoint"
        ]["sha256"]
        == protocol["selected_asset"]["checkpoint"]["sha256"],
        "no_training_or_optimizer": runtime[
            "no_optimizer_constructed_or_step_executed"
        ],
        "no_checkpoint_written": not forbidden,
        "source_checkpoint_sha_before_after_exact": sha256_file(checkpoint)
        == protocol["selected_asset"]["checkpoint"]["sha256"],
        "actor_registry_sha_before_after_exact": sha256_file(registry)
        == protocol["selected_asset"]["actor_registry"]["sha256"],
        "warmup_rgb_hash_repeat_exact": runtime["warmup_rgb_hash_repeat_exact"],
        "measured_matrix_complete_and_unique": runtime[
            "measured_matrix_complete_and_unique"
        ],
        "native_resolution_exact": runtime["native_resolution_exact"],
        "synchronized_timing_complete": runtime["synchronized_timing_complete"],
        "resources_within_frozen_ceilings": resource_audit["status"] == "passed",
        "dry_run_resume_reuses_completed_stages_without_gpu_launch": not resume[
            "gpu_launch_observed"
        ]
        and not resume["torch_imported"]
        and all(
            row["action"] == "reuse_completed_stage"
            for row in resume["actions"].values()
        ),
    }
    expected_audits = protocol["required_audits"]
    if set(audits) != set(expected_audits):
        raise RuntimeError("A4-P0 required audit schema drift")
    if not all(audits.values()):
        failed = sorted(name for name, value in audits.items() if not value)
        raise RuntimeError("A4-P0 required audits failed: " + ", ".join(failed))
    aggregate = stages["aggregate"]
    aggregate["stage_ledger"]["failure_recovery"] = {
        "source": "read_only_no_torch_resume_audit",
        "status": "done",
        "wall_time_seconds": resume["dry_run_seconds"],
        "input_bytes": sum(
            (args.run_dir / row["path"]).stat().st_size
            for row in resume["actions"].values()
        ),
        "output_bytes": (
            args.run_dir / "stages" / "resume_audit.json"
        ).stat().st_size,
        "filesystem_cache": "not_applicable_read_only_hash_audit",
        "minimum_rerun_unit": resume["minimum_rerun_unit"],
    }
    summary = {
        "status": "done",
        "task_id": protocol["task_id"],
        "profile_id": protocol["profile_id"],
        "seed": protocol["seed"],
        "scene": protocol["scene"],
        "protocol_sha256": sha256_file(args.protocol),
        "selected_asset": {
            "role": protocol["selected_asset"]["role"],
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "actor_registry_sha256": sha256_file(registry),
        },
        "asset_inventory": stages["inventory"],
        "performance": {
            key: runtime[key]
            for key in (
                "prepare_dataset_seconds",
                "trainer_construction_seconds",
                "process_cold_checkpoint_load_seconds",
                "process_warm_checkpoint_reload_seconds",
                "filesystem_cache",
                "runtime_render_warmup_seconds",
                "runtime_render_sample_seconds",
                "runtime_render_p50_seconds",
                "runtime_render_p95_seconds",
                "runtime_render_fps",
            )
        },
        "stage_ledger": aggregate["stage_ledger"],
        "runtime_rows": runtime["runtime_rows"],
        "resources": resource_audit,
        "resources_sha256": sha256_file(resource_path),
        "audits": audits,
        "claim_boundary": protocol["claim_boundary"],
        "next_action": "close_p0_then_freeze_p1_p2_p3_p5_decision",
        "project_commit": manifest["project_commit"],
    }
    summary_path = args.run_dir / "summary.json"
    atomic_json(summary_path, summary)
    manifest["status"] = "done"
    manifest["summary_sha256"] = sha256_file(summary_path)
    manifest["resource_audit_sha256"] = sha256_file(resource_path)
    atomic_json(manifest_path, manifest, replace=True)
    atomic_json(
        args.run_dir / "terminal.json",
        {"status": "done", "failure": None},
        replace=True,
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
                        "code": "A4_P0_FINALIZATION_FAILED",
                        "detail": f"{type(error).__name__}: {error}",
                    },
                },
                replace=True,
            )
        raise
