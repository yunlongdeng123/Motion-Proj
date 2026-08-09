#!/usr/bin/env python
"""聚合 A4-P5 已完成 materialize/reload stage，等待独立 resume audit。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


from scripts.run_worldsim_v3_a4_p5_registry import atomic_json, sha256_file, write_stage


_ACTIVE_RUN_DIR: Path | None = None


def load_stage(run_dir: Path, manifest: dict, name: str) -> dict:
    path = run_dir / "stages" / f"{name}.json"
    if sha256_file(path) != manifest["stage_hashes"][name]:
        raise RuntimeError(f"A4-P5 stage hash drift: {name}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_aggregate(run_dir: Path, manifest: dict) -> dict:
    input_stage = load_stage(run_dir, manifest, "input_audit")
    materialize = load_stage(run_dir, manifest, "registry_materialize")
    reload_stage = load_stage(run_dir, manifest, "reload_smoke")
    stage_paths = {
        name: run_dir / "stages" / f"{name}.json"
        for name in ("input_audit", "registry_materialize", "reload_smoke")
    }
    return {
        "status": "done",
        "stage": "aggregate",
        "stage_ledger": {
            "input_audit": {
                "status": "done",
                "input_bytes": input_stage["input_bytes"],
                "output_bytes": stage_paths["input_audit"].stat().st_size,
                "filesystem_cache": "not_applicable_hash_and_json_audit",
                "minimum_rerun_unit": input_stage["minimum_rerun_unit"],
            },
            "registry_materialize": {
                "status": "done",
                "input_bytes": manifest["input_audits"][
                    "selected_asset.actor_registry"
                ]["bytes"],
                "output_bytes": materialize["registry"]["bytes"],
                "filesystem_cache": "not_applicable_reference_only_json",
                "minimum_rerun_unit": materialize["minimum_rerun_unit"],
            },
            "reload_smoke": {
                "status": "done",
                "wall_time_seconds": reload_stage["reload_total_seconds"],
                "input_bytes": sum(
                    manifest["input_audits"][name]["bytes"]
                    for name in (
                        "selected_asset.checkpoint",
                        "selected_asset.source_config",
                        "selected_asset.actor_registry",
                    )
                ),
                "output_bytes": stage_paths["reload_smoke"].stat().st_size,
                "filesystem_cache": reload_stage["filesystem_cache"],
                "minimum_rerun_unit": reload_stage["minimum_rerun_unit"],
            },
            "aggregate": {
                "status": "done",
                "input_bytes": sum(path.stat().st_size for path in stage_paths.values()),
                "output_bytes": None,
                "output_bytes_missing_reason": "aggregate_stage_size_unknown_before_atomic_write",
                "filesystem_cache": "not_applicable_json_aggregation",
                "minimum_rerun_unit": "aggregate_only",
            },
            "resume_audit": {
                "status": None,
                "missing_reason": "resume_audit_runs_in_separate_no_torch_process",
                "minimum_rerun_unit": "resume_audit_only",
            },
        },
        "registry": materialize["registry"],
        "reload": {
            "model_gaussian_counts": reload_stage["model_gaussian_counts"],
            "actor_count": len(reload_stage["actor_rows"]),
            "all_actor_indices_exact": reload_stage["all_actor_indices_exact"],
        },
    }


def main() -> None:
    global _ACTIVE_RUN_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    _ACTIVE_RUN_DIR = args.run_dir
    manifest_path = args.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    aggregate = build_aggregate(args.run_dir, manifest)
    write_stage(args.run_dir, manifest, "aggregate", aggregate)
    manifest["aggregate_complete"] = True
    atomic_json(manifest_path, manifest, replace=True)
    print(json.dumps({"status": "aggregate_complete", "run_dir": str(args.run_dir)}))


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
                        "code": "A4_P5_AGGREGATE_FAILED",
                        "detail": f"{type(error).__name__}: {error}",
                    },
                },
                replace=True,
            )
        raise
