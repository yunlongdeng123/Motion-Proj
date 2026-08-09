#!/usr/bin/env python
"""验证 A4-P3 全阶段恢复语义，且进程不得导入 torch 或启动 GPU。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
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
    "aggregate",
)


from scripts.run_worldsim_v3_a4_p3_chunk import (
    load_stage,
    nvidia_compute_rows,
    sha256_file,
    write_stage,
)
from scripts.validate_worldsim_v3_a4_p3_chunk_protocol import (
    validate_inputs,
    validate_schema,
)


def validate_artifact(run_dir: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    """不导入 torch 地校验一个 artifact。"""
    path = run_dir / str(record["path"])
    actual_bytes = path.stat().st_size
    actual_sha = sha256_file(path)
    if actual_bytes != int(record["bytes"]) or actual_sha != record["sha256"]:
        raise RuntimeError(f"A4-P3 resume artifact drift: {path}")
    return {
        "path": str(path.relative_to(run_dir)),
        "bytes": actual_bytes,
        "sha256": actual_sha,
    }


def validate_package_payloads(
    run_dir: Path, materialized: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """按 package manifest 校验 skeleton 与全部 157 个 data assets。"""
    manifest_record = validate_artifact(run_dir, materialized["package_manifest"])
    manifest_path = run_dir / str(materialized["package_manifest"]["path"])
    package = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    records = [package["skeleton"], *package["static_assets"], *package["actor_assets"]]
    validated = [manifest_record]
    for record in records:
        path = root / str(record["path"])
        actual_bytes = path.stat().st_size
        actual_sha = sha256_file(path)
        if actual_bytes != int(record["bytes"]) or actual_sha != record["sha256"]:
            raise RuntimeError(f"A4-P3 resume package payload drift: {path}")
        validated.append(
            {
                "path": str(path.relative_to(run_dir)),
                "bytes": actual_bytes,
                "sha256": actual_sha,
            }
        )
    return validated


def stage_artifacts(
    run_dir: Path,
    name: str,
    stage: Mapping[str, Any],
    materialized: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """返回一个阶段所有需复用的持久 artifact。"""
    if name == "materialize_chunk_package":
        return validate_package_payloads(run_dir, stage)
    if name == "reassemble_and_hash_audit":
        return [validate_artifact(run_dir, stage["package_manifest"])]
    if name == "evaluate_source_and_chunk":
        return [
            validate_artifact(run_dir, record)
            for record in stage["quality_artifacts"].values()
        ]
    return []


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
    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    actions = {}
    materialized = None
    for name in PRIOR_STAGES:
        stage = load_stage(args.run_dir, manifest, name)
        if name == "materialize_chunk_package":
            materialized = stage
        artifacts = stage_artifacts(args.run_dir, name, stage, materialized)
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
