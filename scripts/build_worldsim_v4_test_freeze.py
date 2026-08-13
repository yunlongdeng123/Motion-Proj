#!/usr/bin/env python3
"""Build the sole committed V4 test freeze before any test-quality read."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from typing import Any

import yaml

from motion_proj.worldsim_v4.test_freeze import (
    FREEZE_SCHEMA,
    TASK_ID,
    canonical_json_bytes,
    exclusive_json,
    sha256_file,
    validate_execution_plan,
)


class TestFreezeBuildError(RuntimeError):
    pass


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TestFreezeBuildError(f"YAML root must be mapping: {path}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TestFreezeBuildError(f"JSON root must be mapping: {path}")
    return value


def git(project_root: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(project_root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode:
        raise TestFreezeBuildError(process.stderr.strip())
    return process.stdout.strip()


def build_freeze(
    *,
    project_root: Path,
    config_path: Path,
    inventory_path: Path,
    cohort_path: Path,
    validation_run: Path,
    p0_scope_path: Path,
    metrics_path: Path,
    run_root: Path,
    stamp: str,
    first_run_id: int,
) -> dict[str, Any]:
    if git(project_root, "status", "--porcelain"):
        raise TestFreezeBuildError("freeze build requires clean source worktree")
    source_commit = git(project_root, "rev-parse", "HEAD")
    source_tree = git(project_root, "rev-parse", "HEAD^{tree}")
    config = load_yaml(config_path)
    inventory = load_yaml(inventory_path)
    cohort = load_yaml(cohort_path)
    p0 = load_yaml(p0_scope_path)
    metrics = load_yaml(metrics_path)
    summary_path = validation_run / "summary.json"
    manifest_path = validation_run / "manifest.json"
    status_path = validation_run / "status.json"
    validation = load_json(summary_path)
    status = load_json(status_path)
    if (
        validation.get("validation_gate_passed") is not True
        or validation.get("test_freeze_authorized") is not True
        or validation.get("test_quality_read") is not False
        or status.get("summary_sha256") != sha256_file(summary_path)
        or status.get("manifest_sha256") != sha256_file(manifest_path)
    ):
        raise TestFreezeBuildError("M3 validation does not authorize test freeze")
    scene_order = cohort.get("freeze", {}).get("scene_roles", {}).get("test")
    if (
        not isinstance(scene_order, list)
        or len(scene_order) != 18
        or inventory.get("scene_order") != scene_order
        or list(inventory.get("scenes", {})) != scene_order
        or any(row.get("partition") != "test" for row in inventory["scenes"].values())
        or inventory.get("test_quality_read") is not False
        or config.get("test_protocol", {}).get("scene_order") != scene_order
        or config.get("trajectory", {}).get("selected_parameters")
        != validation.get("selected_parameters")
    ):
        raise TestFreezeBuildError("test split/assets/config order drift")
    execution_plan = []
    for index, scene in enumerate(scene_order):
        run_id = first_run_id + index
        attempt_id = f"m3-test-{index + 1:02d}-{scene.replace('-', '')}-s0-r{run_id}"
        execution_plan.append(
            {
                "ordinal": index + 1,
                "scene": scene,
                "attempt_id": attempt_id,
                "run_dir": str(
                    run_root
                    / f"{stamp}__m3-test-exact-once-{scene.replace('-', '')}-s0-r{run_id}"
                ),
            }
        )
    ledger_dir = run_root / f"{stamp}__m3-test-exact-once-ledger-s0"
    payload = {
        "schema_version": FREEZE_SCHEMA,
        "task_id": TASK_ID,
        "status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "source_tree": source_tree,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "config_sha256": sha256_file(config_path),
        "split": {"path": str(cohort_path), "sha256": sha256_file(cohort_path)},
        "split_sha256": sha256_file(cohort_path),
        "test_asset_inventory": {
            "path": str(inventory_path),
            "sha256": sha256_file(inventory_path),
        },
        "test_asset_inventory_sha256": sha256_file(inventory_path),
        "validation_freeze": {
            "path": str(validation_run),
            "summary_sha256": sha256_file(summary_path),
            "manifest_sha256": sha256_file(manifest_path),
            "status_sha256": sha256_file(status_path),
        },
        "method_selection": {
            "m2_router": config["execution_gate"]["m2_frozen_router"],
            "m2_matched_non_router": config["execution_gate"]["m2_frozen_matched_non_router"],
            "m3_parameters": config["trajectory"]["selected_parameters"],
            "operations": config["clip"]["operations"],
            "camera_ids": config["clip"]["camera_ids"],
            "clip_protocol": config["clip"]["sample_protocol"],
        },
        "thresholds": {
            **config["gates"],
            "minimum_rendered_effect_pixels": config["operations"]["minimum_rendered_effect_pixels"],
        },
        "baseline_list": {
            "matched_v4": list(p0["baselines"]),
            "temporal_arms": list(config["ablations"]["arms"]),
            "test_comparison": ["FRAME_INDEPENDENT", "FULL_WARP_REGULARIZED"],
        },
        "metrics_list": list(config["metrics"]),
        "metrics_contract": {"path": str(metrics_path), "sha256": sha256_file(metrics_path)},
        "statistics": metrics["statistics"],
        "scene_order": scene_order,
        "execution_plan": execution_plan,
        "ledger_dir": str(ledger_dir),
        "runner_python": inventory["drivestudio_python"],
        "resources": {
            "required_gpu": "NVIDIA GeForce RTX 3090",
            "minimum_disk_free_gib": 20,
            "maximum_gpu_used_at_attempt_start_mib": 2048,
        },
        "test_read_count": 1,
        "parameter_search": False,
        "threshold_search": False,
        "test_authorized": True,
    }
    validate_execution_plan(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--validation-run", type=Path, required=True)
    parser.add_argument("--p0-scope", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--first-run-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = build_freeze(
        project_root=args.project_root.resolve(),
        config_path=args.config.resolve(),
        inventory_path=args.inventory.resolve(),
        cohort_path=args.cohort.resolve(),
        validation_run=args.validation_run.resolve(),
        p0_scope_path=args.p0_scope.resolve(),
        metrics_path=args.metrics.resolve(),
        run_root=args.run_root.resolve(),
        stamp=args.stamp,
        first_run_id=args.first_run_id,
    )
    exclusive_json(args.output.resolve(), payload)
    print(json.dumps({"path": str(args.output), "sha256": sha256_file(args.output)}))


if __name__ == "__main__":
    main()
