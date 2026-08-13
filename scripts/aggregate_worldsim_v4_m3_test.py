#!/usr/bin/env python3
"""Aggregate the committed exact-once 18-scene M3 test without retuning."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from motion_proj.worldsim_v4.statistics import paired_scene_test, summarize_scenes
from motion_proj.worldsim_v4.test_freeze import (
    TASK_ID,
    committed_freeze,
    load_mapping,
    sha256_file,
    validate_execution_plan,
)
from scripts.run_worldsim_v4_m3_scene import atomic_json, write_jsonl
from scripts.run_worldsim_v4_m3_test_exact_once import completion_payload


class M3TestAggregateError(RuntimeError):
    pass


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise M3TestAggregateError(f"YAML root must be mapping: {path}")
    return value


def finite_mean(values: Iterable[float]) -> float | None:
    data = [float(value) for value in values]
    if not data:
        return None
    if not all(math.isfinite(value) for value in data):
        raise M3TestAggregateError("test aggregate contains non-finite metric")
    return float(np.mean(data))


def relative_improvement(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None or baseline <= 0.0:
        return None
    return (baseline - candidate) / baseline


def unique_sequence(summary: Mapping[str, Any], operation: str, arm: str) -> Mapping[str, Any]:
    rows = [
        row
        for row in summary["sequences"]
        if row["operation"] == operation and row["arm"] == arm
    ]
    if len(rows) != 1:
        raise M3TestAggregateError(
            f"{summary['scene']} {operation}/{arm} must be unique"
        )
    return rows[0]


def manifest(run_dir: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name not in {"manifest.json", "status.json"}:
            files[path.relative_to(run_dir).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return {
        "schema_version": "worldsim_v4_m3_test_aggregate_manifest_v1",
        "task_id": TASK_ID,
        "status": "done",
        "files": files,
        "test_quality_read": True,
        "test_source_content_reread": False,
    }


def aggregate(freeze_path: Path, project_root: Path, run_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    if run_dir.exists():
        raise FileExistsError(run_dir)
    freeze, provenance = committed_freeze(freeze_path, project_root)
    plan = validate_execution_plan(freeze)
    inventory_path = Path(freeze["test_asset_inventory"]["path"])
    inventory = load_yaml(inventory_path)
    if (
        sha256_file(inventory_path) != freeze["test_asset_inventory_sha256"]
        or inventory.get("scene_order") != freeze["scene_order"]
    ):
        raise M3TestAggregateError("test inventory differs from freeze")
    ledger = Path(freeze["ledger_dir"])
    terminal = load_mapping(ledger / "terminal.json")
    if (
        terminal.get("state") != "done"
        or terminal.get("attempt_count") != 18
        or terminal.get("completion_count") != 18
        or terminal.get("freeze_sha256") != provenance["freeze_sha256"]
        or terminal.get("scene_order") != freeze["scene_order"]
    ):
        raise M3TestAggregateError("test ledger terminal contract drift")
    attempts = sorted((ledger / "attempts").glob("*.json"))
    completions = sorted((ledger / "completions").glob("*.json"))
    if len(attempts) != 18 or len(completions) != 18:
        raise M3TestAggregateError("exact-once attempt/completion count drift")

    operations = list(freeze["method_selection"]["operations"])
    baseline_arm, candidate_arm = freeze["baseline_list"]["test_comparison"]
    selected = freeze["method_selection"]["m3_parameters"]
    paired_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    all_sequences: list[dict[str, Any]] = []
    for planned in plan:
        attempt_id = planned["attempt_id"]
        completion_path = ledger / "completions" / f"{attempt_id}.json"
        expected_completion = completion_payload(
            freeze=freeze,
            provenance=provenance,
            planned=planned,
            inventory=inventory,
        )
        if load_mapping(completion_path) != expected_completion:
            raise M3TestAggregateError("test completion differs from run evidence")
        summary = load_mapping(Path(planned["run_dir"]) / "summary.json")
        scene = planned["scene"]
        if summary["status"] == "abstain":
            scene_rows.append(
                {
                    "scene": scene,
                    "status": "abstain",
                    "reason": summary["reason"],
                    "retained_in_denominator": True,
                    "source_run": planned["run_dir"],
                }
            )
            continue
        if (
            summary["status"] != "done"
            or summary["parameters"] != selected
            or set(summary["operations"]) != set(operations)
            or not summary["checkpoint_immutable"]
            or not summary["rollback_exact"]
            or summary.get("development_optimization_read") is not False
            or summary.get("validation_optimization_read") is not False
        ):
            raise M3TestAggregateError(f"test scene output drift: {scene}")
        all_sequences.extend(summary["sequences"])
        scene_pairs = []
        for operation in operations:
            baseline = unique_sequence(summary, operation, baseline_arm)
            candidate = unique_sequence(summary, operation, candidate_arm)
            row = {
                "scene": scene,
                "operation": operation,
                "baseline_arm": baseline_arm,
                "candidate_arm": candidate_arm,
                "baseline_operation_success": baseline["operation_success"],
                "candidate_operation_success": candidate["operation_success"],
                "baseline_warp_l1_delta": baseline["warp_l1_delta"],
                "candidate_warp_l1_delta": candidate["warp_l1_delta"],
                "baseline_temporal_lpips": baseline["temporal_lpips"],
                "candidate_temporal_lpips": candidate["temporal_lpips"],
                "baseline_identity_switch": baseline["identity_switch"],
                "candidate_identity_switch": candidate["identity_switch"],
                "candidate_semantic_reintroduction_pixels": candidate[
                    "semantic_reintroduction_pixels"
                ],
                "candidate_rollback_exact": candidate["rollback_exact"],
                "non_target_psnr_delta": candidate["non_target_psnr"]
                - baseline["non_target_psnr"],
                "non_target_ssim_delta": candidate["non_target_ssim"]
                - baseline["non_target_ssim"],
                "non_target_lpips_alex_delta": candidate["non_target_lpips_alex"]
                - baseline["non_target_lpips_alex"],
            }
            paired_rows.append(row)
            scene_pairs.append(row)
        scene_rows.append(
            {
                "scene": scene,
                "status": "done",
                "reason": None,
                "retained_in_denominator": True,
                "source_run": planned["run_dir"],
                "baseline_warp_l1_delta": finite_mean(
                    row["baseline_warp_l1_delta"] for row in scene_pairs
                ),
                "candidate_warp_l1_delta": finite_mean(
                    row["candidate_warp_l1_delta"] for row in scene_pairs
                ),
                "baseline_temporal_lpips": finite_mean(
                    row["baseline_temporal_lpips"] for row in scene_pairs
                ),
                "candidate_temporal_lpips": finite_mean(
                    row["candidate_temporal_lpips"] for row in scene_pairs
                ),
                "non_target_psnr_delta": finite_mean(
                    row["non_target_psnr_delta"] for row in scene_pairs
                ),
                "non_target_ssim_delta": finite_mean(
                    row["non_target_ssim_delta"] for row in scene_pairs
                ),
                "non_target_lpips_alex_delta": finite_mean(
                    row["non_target_lpips_alex_delta"] for row in scene_pairs
                ),
            }
        )

    baseline_warp = finite_mean(row["baseline_warp_l1_delta"] for row in paired_rows)
    candidate_warp = finite_mean(row["candidate_warp_l1_delta"] for row in paired_rows)
    baseline_tlpips = finite_mean(row["baseline_temporal_lpips"] for row in paired_rows)
    candidate_tlpips = finite_mean(row["candidate_temporal_lpips"] for row in paired_rows)
    warp_improvement = relative_improvement(baseline_warp, candidate_warp)
    tlpips_improvement = relative_improvement(baseline_tlpips, candidate_tlpips)
    baseline_success = sum(bool(row["baseline_operation_success"]) for row in paired_rows)
    candidate_success = sum(bool(row["candidate_operation_success"]) for row in paired_rows)
    baseline_identity = sum(int(row["baseline_identity_switch"]) for row in paired_rows)
    candidate_identity = sum(int(row["candidate_identity_switch"]) for row in paired_rows)
    thresholds = freeze["thresholds"]
    identity_passed = (
        bool(paired_rows)
        and bool(thresholds["allow_identity_zero_remains_zero"])
        and baseline_identity == 0
        and candidate_identity == 0
    )
    if baseline_identity > 0:
        identity_passed = (
            (baseline_identity - candidate_identity) / baseline_identity
            >= float(thresholds["identity_switch_relative_reduction_min"])
        )
    temporal_passed = (
        warp_improvement is not None
        and tlpips_improvement is not None
        and max(warp_improvement, tlpips_improvement)
        >= float(thresholds["temporal_error_relative_improvement_min"])
    )
    rollback_fraction = finite_mean(
        float(bool(row["rollback_exact"])) for row in all_sequences
    )
    gate_rows = {
        "operation_success_non_degradation": {
            "baseline": baseline_success,
            "candidate": candidate_success,
            "passed": bool(paired_rows) and candidate_success >= baseline_success,
        },
        "temporal_error_relative_improvement": {
            "warp_l1": warp_improvement,
            "temporal_lpips": tlpips_improvement,
            "threshold": thresholds["temporal_error_relative_improvement_min"],
            "rule": "either",
            "passed": temporal_passed,
        },
        "identity_switch": {
            "baseline": baseline_identity,
            "candidate": candidate_identity,
            "passed": identity_passed,
        },
        "deleted_semantic_reintroduction": {
            "value": sum(
                int(row["candidate_semantic_reintroduction_pixels"])
                for row in paired_rows
                if row["operation"] == "REMOVE"
            ),
            "threshold": thresholds["deleted_semantic_reintroduction"],
            "passed": bool(paired_rows)
            and sum(
                int(row["candidate_semantic_reintroduction_pixels"])
                for row in paired_rows
                if row["operation"] == "REMOVE"
            )
            == int(thresholds["deleted_semantic_reintroduction"]),
        },
        "rollback_exact_fraction": {
            "value": rollback_fraction,
            "threshold": thresholds["rollback_exact_fraction"],
            "passed": rollback_fraction == float(thresholds["rollback_exact_fraction"]),
        },
    }
    test_gate_passed = all(row["passed"] for row in gate_rows.values())
    scene_statistics = {
        metric: summarize_scenes(scene_rows, metric)
        for metric in (
            "baseline_warp_l1_delta",
            "candidate_warp_l1_delta",
            "baseline_temporal_lpips",
            "candidate_temporal_lpips",
            "non_target_psnr_delta",
            "non_target_ssim_delta",
            "non_target_lpips_alex_delta",
        )
    }
    paired_statistics = {
        "warp_l1": paired_scene_test(
            {row["scene"]: row.get("candidate_warp_l1_delta") for row in scene_rows},
            {row["scene"]: row.get("baseline_warp_l1_delta") for row in scene_rows},
            higher_is_better=False,
        ),
        "temporal_lpips": paired_scene_test(
            {row["scene"]: row.get("candidate_temporal_lpips") for row in scene_rows},
            {row["scene"]: row.get("baseline_temporal_lpips") for row in scene_rows},
            higher_is_better=False,
        ),
    }
    run_dir.mkdir(parents=True)
    write_jsonl(run_dir / "paired_metrics.jsonl", paired_rows)
    write_jsonl(run_dir / "scene_metrics.jsonl", scene_rows)
    snapshots = run_dir / "source_snapshot"
    snapshots.mkdir()
    for source in (
        freeze_path,
        inventory_path,
        Path(freeze["config"]["path"]),
        ledger / "terminal.json",
        Path(__file__),
    ):
        shutil.copy2(source, snapshots / source.name)
    summary = {
        "schema_version": "worldsim_v4_m3_test_aggregate_summary_v1",
        "task_id": TASK_ID,
        "status": "done",
        "partition": "test",
        "scene_denominator": 18,
        "evaluable_scene_count": sum(row["status"] == "done" for row in scene_rows),
        "abstain_scene_count": sum(row["status"] == "abstain" for row in scene_rows),
        "scene_operation_denominator": len(paired_rows),
        "scenes": scene_rows,
        "selected_parameters": selected,
        "baseline_arm": baseline_arm,
        "candidate_arm": candidate_arm,
        "aggregate": {
            "baseline_warp_l1_delta": baseline_warp,
            "candidate_warp_l1_delta": candidate_warp,
            "warp_l1_relative_improvement": warp_improvement,
            "baseline_temporal_lpips": baseline_tlpips,
            "candidate_temporal_lpips": candidate_tlpips,
            "temporal_lpips_relative_improvement": tlpips_improvement,
            "non_target_psnr_delta": finite_mean(
                row["non_target_psnr_delta"] for row in paired_rows
            ),
            "non_target_ssim_delta": finite_mean(
                row["non_target_ssim_delta"] for row in paired_rows
            ),
            "non_target_lpips_alex_delta": finite_mean(
                row["non_target_lpips_alex_delta"] for row in paired_rows
            ),
        },
        "scene_statistics": scene_statistics,
        "paired_statistics": paired_statistics,
        "gates": gate_rows,
        "test_gate_passed": test_gate_passed,
        "conclusion": "confirmed" if test_gate_passed else "not_confirmed",
        "freeze_commit": provenance["freeze_commit"],
        "freeze_sha256": provenance["freeze_sha256"],
        "exact_once_attempt_count": 18,
        "exact_once_completion_count": 18,
        "test_quality_read": True,
        "test_source_content_reread_by_aggregator": False,
        "parameter_search": False,
        "threshold_search": False,
        "duration_seconds": time.monotonic() - started,
    }
    atomic_json(run_dir / "summary.json", summary)
    atomic_json(
        run_dir / "fingerprint.json",
        {
            "task_id": TASK_ID,
            "freeze_commit": provenance["freeze_commit"],
            "freeze_sha256": provenance["freeze_sha256"],
            "ledger_terminal_sha256": sha256_file(ledger / "terminal.json"),
            "inventory_sha256": sha256_file(inventory_path),
            "test_quality_read": True,
            "test_source_content_reread": False,
        },
    )
    atomic_json(run_dir / "manifest.json", manifest(run_dir))
    atomic_json(
        run_dir / "status.json",
        {
            "task_id": TASK_ID,
            "status": "done",
            "summary_sha256": sha256_file(run_dir / "summary.json"),
            "manifest_sha256": sha256_file(run_dir / "manifest.json"),
            "fingerprint_sha256": sha256_file(run_dir / "fingerprint.json"),
            "test_quality_read": True,
            "test_source_content_reread": False,
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = aggregate(
        args.freeze.resolve(), args.project_root.resolve(), args.run_dir.resolve()
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "test_gate_passed": summary["test_gate_passed"],
                "conclusion": summary["conclusion"],
                "aggregate": summary["aggregate"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
