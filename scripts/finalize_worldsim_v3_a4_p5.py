#!/usr/bin/env python
"""收口 A4-P5 registry、reload、资源与恢复审计。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from omegaconf import OmegaConf


PROJECT = Path("/root/autodl-tmp/motion_proj")
PROTOCOL = PROJECT / "configs/worldsim_v3/a4_p5_registry_resume_protocol_v1.yaml"
_ACTIVE_RUN_DIR: Path | None = None


from scripts.run_worldsim_v3_a4_p5_registry import (
    atomic_json,
    canonical_sha256,
    cgroup_memory_events,
    directory_bytes,
    sha256_file,
)
from scripts.validate_worldsim_v3_a4_p5_registry_resume_protocol import (
    validate_inputs,
    validate_schema,
)


def load_stage(run_dir: Path, manifest: dict, name: str) -> dict:
    path = run_dir / "stages" / f"{name}.json"
    if sha256_file(path) != manifest["stage_hashes"][name]:
        raise RuntimeError(f"A4-P5 stage hash drift: {name}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_resource_audit(
    *, run_dir: Path, manifest: dict, reload_stage: dict, protocol: dict
) -> dict:
    ceilings = protocol["resource_ceilings"]
    started = datetime.fromisoformat(manifest["started_at"])
    wall = (datetime.now(timezone.utc) - started).total_seconds()
    before = manifest["preflight"]["cgroup_memory_events"]
    after = cgroup_memory_events()
    measured = {
        "wall_time_seconds": wall,
        "peak_torch_allocated_mib": reload_stage["peak_torch_allocated_mib"],
        "peak_torch_reserved_mib": reload_stage["peak_torch_reserved_mib"],
        "peak_nvidia_process_memory_mib_sampled": reload_stage[
            "peak_nvidia_process_memory_mib_sampled"
        ],
        "peak_cgroup_memory_bytes_sampled": reload_stage[
            "peak_cgroup_memory_bytes_sampled"
        ],
        "run_bytes_before_finalization": directory_bytes(run_dir),
        "disk_free_bytes_after": shutil.disk_usage(run_dir).free,
        "oom_events_delta": after.get("oom", 0) - before.get("oom", 0),
        "oom_kill_events_delta": after.get("oom_kill", 0)
        - before.get("oom_kill", 0),
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
        "resource_sampling_errors": bool(reload_stage["sampling_errors"]),
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
        for name in (
            "input_audit",
            "registry_materialize",
            "reload_smoke",
            "aggregate",
            "resume_audit",
        )
    }
    registry_path = args.run_dir / stages["registry_materialize"]["registry"]["path"]
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    canonical_expected = registry["registry_sha256"]
    canonical_payload = dict(registry)
    canonical_payload.pop("registry_sha256")
    selected = protocol["selected_asset"]
    checkpoint = Path(selected["checkpoint"]["path"])
    source_registry = Path(selected["actor_registry"]["path"])
    reload_stage = stages["reload_smoke"]
    resume = stages["resume_audit"]
    forbidden = [
        str(path.relative_to(args.run_dir))
        for path in args.run_dir.rglob("*")
        if path.suffix.lower() in {".pth", ".png", ".jpg", ".jpeg", ".mp4"}
    ]
    resource_audit = build_resource_audit(
        run_dir=args.run_dir,
        manifest=manifest,
        reload_stage=reload_stage,
        protocol=protocol,
    )
    resource_path = args.run_dir / "artifacts" / "resource_audit.json"
    atomic_json(resource_path, resource_audit)
    contract = protocol["registry_contract"]
    expected_actor = contract["actor_assets"]
    totals = registry["totals"]
    audits = {
        "selected_input_hashes_and_bytes_exact": len(manifest["input_audits"]) == 9,
        "p0_canonical_evidence_exact_and_done": stages["input_audit"]["p0_terminal"]
        == {"failure": None, "status": "done"},
        "reference_only_no_checkpoint_copy_or_rewrite": not forbidden
        and not stages["registry_materialize"]["checkpoint_copy_or_rewrite_performed"]
        and all(
            row["storage"] == "external_immutable_reference"
            for row in registry["source_assets"].values()
        ),
        "compact_registry_schema_exact": registry["schema_version"]
        == contract["schema_version"]
        and set(registry) == set(contract["required_top_level_fields"])
        and canonical_sha256(canonical_payload) == canonical_expected
        and registry_path.stat().st_size <= int(contract["output_bytes_ceiling"]),
        "static_asset_inventory_exact": totals["static_asset_count"] == 1
        and totals["static_gaussian_count"]
        == int(contract["static_asset"]["gaussian_count"])
        and registry["static_assets"][0]["independently_extractable"] is False,
        "actor_asset_counts_and_availability_exact": totals["actor_asset_count"]
        == int(expected_actor["actor_count"])
        and totals["actor_gaussian_count"] == int(expected_actor["gaussian_count"])
        and totals["available_actor_count"]
        == int(expected_actor["available_actor_count"])
        and totals["unavailable_actor_count"]
        == int(expected_actor["unavailable_actor_count"]),
        "actor_index_counts_and_hashes_exact_after_reload": reload_stage[
            "all_actor_indices_exact"
        ]
        and len(reload_stage["actor_rows"]) == int(expected_actor["actor_count"]),
        "model_gaussian_counts_exact_after_reload": reload_stage[
            "model_gaussian_counts"
        ]
        == {
            "Background": int(contract["static_asset"]["gaussian_count"]),
            "RigidNodes": int(expected_actor["gaussian_count"]),
        },
        "unavailable_actor_remains_explicitly_empty": reload_stage[
            "unavailable_actor_remains_explicitly_empty"
        ]
        and sum(
            row["actual_gaussian_count"] == 0 for row in reload_stage["actor_rows"]
        )
        == int(expected_actor["unavailable_actor_count"]),
        "source_checkpoint_and_registry_sha_before_after_exact": sha256_file(checkpoint)
        == selected["checkpoint"]["sha256"]
        and sha256_file(source_registry) == selected["actor_registry"]["sha256"],
        "no_training_optimizer_render_or_media": reload_stage[
            "no_optimizer_constructed_or_step_executed"
        ]
        and reload_stage["render_count"] == 0
        and not forbidden,
        "exactly_one_checkpoint_load": reload_stage["checkpoint_load_count"] == 1,
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
    if set(audits) != set(protocol["required_audits"]):
        raise RuntimeError("A4-P5 required audit schema drift")
    if not all(audits.values()):
        failed = sorted(name for name, value in audits.items() if not value)
        raise RuntimeError("A4-P5 required audits failed: " + ", ".join(failed))
    stage_ledger = stages["aggregate"]["stage_ledger"]
    stage_ledger["resume_audit"] = {
        "status": "done",
        "wall_time_seconds": resume["dry_run_seconds"],
        "input_bytes": sum(
            (args.run_dir / row["path"]).stat().st_size
            for row in resume["actions"].values()
        ),
        "output_bytes": (args.run_dir / "stages" / "resume_audit.json").stat().st_size,
        "filesystem_cache": "not_applicable_read_only_hash_audit",
        "minimum_rerun_unit": resume["minimum_rerun_unit"],
    }
    summary = {
        "status": "done",
        "task_id": protocol["task_id"],
        "profile_id": protocol["profile_id"],
        "scene": protocol["scene"],
        "seed": protocol["seed"],
        "protocol_sha256": sha256_file(args.protocol),
        "project_commit": manifest["project_commit"],
        "deployment_registry": {
            **stages["registry_materialize"]["registry"],
            "schema_version": registry["schema_version"],
            "totals": totals,
        },
        "reload": {
            key: reload_stage[key]
            for key in (
                "dataset_prepare_seconds",
                "trainer_construction_seconds",
                "checkpoint_load_seconds",
                "reload_total_seconds",
                "checkpoint_load_count",
                "checkpoint_loader",
                "filesystem_cache",
                "render_count",
                "model_gaussian_counts",
            )
        },
        "resources": resource_audit,
        "resources_sha256": sha256_file(resource_path),
        "recovery": {
            "dry_run_seconds": resume["dry_run_seconds"],
            "minimum_rerun_unit": resume["minimum_rerun_unit"],
            "gpu_launch_observed": resume["gpu_launch_observed"],
        },
        "stage_ledger": stage_ledger,
        "audits": audits,
        "claim_boundary": protocol["claim_boundary"],
        "next_action": "close_p5_then_freeze_p1_p2_p3_decision",
    }
    summary_path = args.run_dir / "summary.json"
    atomic_json(summary_path, summary)
    manifest["status"] = "done"
    manifest["summary_sha256"] = sha256_file(summary_path)
    manifest["resource_audit_sha256"] = sha256_file(resource_path)
    manifest["checkpoint_sha256_after_finalization"] = sha256_file(checkpoint)
    manifest["actor_registry_sha256_after_finalization"] = sha256_file(source_registry)
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
                        "code": "A4_P5_FINALIZATION_FAILED",
                        "detail": f"{type(error).__name__}: {error}",
                    },
                },
                replace=True,
            )
        raise
