#!/usr/bin/env python
"""无 torch、无 GPU 地审计 A4-P1 已完成阶段可被恢复复用。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

from omegaconf import OmegaConf


PROJECT = Path("/root/autodl-tmp/motion_proj")
PROTOCOL = PROJECT / "configs/worldsim_v3/a4_p1_contribution_prune_protocol_v1.yaml"


from scripts.run_worldsim_v3_a4_p1_prune import (
    load_stage,
    nvidia_compute_rows,
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
    "aggregate",
)


def validate_artifact(run_dir: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    path = run_dir / str(record["path"])
    actual_bytes = path.stat().st_size
    actual_sha = sha256_file(path)
    if actual_bytes != int(record["bytes"]) or actual_sha != record["sha256"]:
        raise RuntimeError(f"A4-P1 resume artifact drift: {path}")
    return {
        "path": str(path.relative_to(run_dir)),
        "bytes": actual_bytes,
        "sha256": actual_sha,
    }


def stage_artifacts(
    run_dir: Path, name: str, stage: Mapping[str, Any]
) -> list[dict[str, Any]]:
    records = []
    if name == "contribution_scan":
        records.append(validate_artifact(run_dir, stage["score_artifact"]))
    elif name.startswith("materialize_"):
        records.append(validate_artifact(run_dir, stage["candidate_checkpoint"]))
        records.append(validate_artifact(run_dir, stage["candidate_registry"]))
    elif name.startswith("evaluate_"):
        records.extend(
            validate_artifact(run_dir, record)
            for record in stage["quality_artifacts"].values()
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    torch_before = "torch" in sys.modules
    gpu_before = nvidia_compute_rows()
    started = time.perf_counter()
    protocol = OmegaConf.to_container(OmegaConf.load(args.protocol), resolve=True)
    validate_schema(protocol)
    validate_inputs(protocol)
    manifest_path = args.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actions = {}
    for name in PRIOR_STAGES:
        stage = load_stage(args.run_dir, manifest, name)
        artifacts = stage_artifacts(args.run_dir, name, stage)
        stage_path = args.run_dir / "stages" / f"{name}.json"
        actions[name] = {
            "action": "reuse_completed_stage",
            "path": str(stage_path.relative_to(args.run_dir)),
            "sha256": manifest["stage_hashes"][name],
            "bytes": stage_path.stat().st_size,
            "validated_artifacts": artifacts,
        }
    gpu_after = nvidia_compute_rows()
    torch_after = "torch" in sys.modules
    stage = {
        "status": "done",
        "stage": "resume_audit",
        "duration_seconds": time.perf_counter() - started,
        "dry_run_seconds": time.perf_counter() - started,
        "actions": actions,
        "first_invalid_stage": None,
        "torch_imported_before": torch_before,
        "torch_imported_after": torch_after,
        "torch_imported": torch_before or torch_after,
        "gpu_compute_rows_before": gpu_before,
        "gpu_compute_rows_after": gpu_after,
        "gpu_launch_observed": gpu_before != gpu_after or bool(gpu_after),
        "all_completed_stages_reused": all(
            row["action"] == "reuse_completed_stage" for row in actions.values()
        ),
        "minimum_rerun_unit": "none_all_completed_stages_reused",
    }
    write_stage(args.run_dir, manifest, "resume_audit", stage)
    print(json.dumps({"status": "resume_audit_complete", "seconds": stage["dry_run_seconds"]}))


if __name__ == "__main__":
    main()
