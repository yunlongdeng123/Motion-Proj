#!/usr/bin/env python3
"""对冻结参数的六场 M3 validation 做 scene×operation balanced 聚合。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_worldsim_v4_m3_scene import (  # noqa: E402
    TASK_ID,
    atomic_json,
    git_dirty,
    git_head,
    output_manifest,
    sha256_file,
    verify_binding,
    write_jsonl,
)


class M3ValidationError(RuntimeError):
    pass


def verify_run(binding: Mapping[str, Any], label: str) -> tuple[Path, dict[str, Any]]:
    run = Path(str(binding["path"])).resolve()
    summary_path = run / "summary.json"
    manifest_path = run / "manifest.json"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise M3ValidationError(f"{label} 缺 summary/manifest")
    if sha256_file(summary_path) != binding["summary_sha256"]:
        raise M3ValidationError(f"{label} summary SHA 不匹配")
    if sha256_file(manifest_path) != binding["manifest_sha256"]:
        raise M3ValidationError(f"{label} manifest SHA 不匹配")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("test_quality_read") is not False:
        raise M3ValidationError(f"{label} 触碰了 test quality")
    return run, summary


def mean(values: Iterable[float]) -> float:
    result = [float(value) for value in values]
    if not result or not all(math.isfinite(value) for value in result):
        raise M3ValidationError("聚合指标缺失或非有限")
    return float(np.mean(result))


def relative_improvement(baseline: float, candidate: float) -> float:
    if not math.isfinite(baseline) or baseline <= 0.0 or not math.isfinite(candidate):
        raise M3ValidationError("relative improvement 需要 baseline > 0")
    return (baseline - candidate) / baseline


def unique_sequence(
    summary: Mapping[str, Any], operation: str, arm: str
) -> Mapping[str, Any]:
    matches = [
        row
        for row in summary["sequences"]
        if row["operation"] == operation and row["arm"] == arm
    ]
    if len(matches) != 1:
        raise M3ValidationError(
            f"{summary['scene']} {operation}/{arm} 必须唯一"
        )
    return matches[0]


def run(*, config_path: Path, run_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    if run_dir.exists():
        raise FileExistsError(run_dir)
    if git_dirty():
        raise M3ValidationError("formal validation aggregator 要求 clean worktree")
    run_dir.mkdir(parents=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    m3_config_path = verify_binding(config["m3_config"], "M3 config")
    inventory_path = verify_binding(config["scene_inventory"], "scene inventory")
    m3_config = yaml.safe_load(m3_config_path.read_text(encoding="utf-8"))
    inventory = yaml.safe_load(inventory_path.read_text(encoding="utf-8"))
    _, development = verify_run(config["development_freeze"], "development freeze")
    selected = dict(development["selected_parameters"])
    if selected != m3_config["trajectory"]["selected_parameters"]:
        raise M3ValidationError("M3 config 参数与 development freeze 不一致")
    validation_inventory = {
        scene: row
        for scene, row in inventory["scenes"].items()
        if row["partition"] == "validation"
    }
    if len(validation_inventory) != int(config["aggregation"]["scene_denominator"]):
        raise M3ValidationError("validation scene denominator 不一致")
    if set(validation_inventory) != set(config["validation_scene_runs"]):
        raise M3ValidationError("validation scene run inventory 不完整")
    operations = list(config["aggregation"]["operations"])
    baseline_arm = config["aggregation"]["baseline_arm"]
    candidate_arm = config["aggregation"]["candidate_arm"]
    scene_rows = []
    paired_rows = []
    all_sequences = []
    evaluable_count = 0
    for scene in validation_inventory:
        _, summary = verify_run(config["validation_scene_runs"][scene], scene)
        if summary["scene"] != scene or summary["partition"] != "validation":
            raise M3ValidationError(f"validation identity 不一致: {scene}")
        if summary["project_git_dirty"]:
            raise M3ValidationError(f"validation source 非 clean: {scene}")
        if summary.get("development_content_read") is not False:
            raise M3ValidationError(f"validation run 反读 development content: {scene}")
        if summary.get("development_optimization_read") is not False:
            raise M3ValidationError(f"validation run 执行 optimization: {scene}")
        if summary.get("validation_optimization_read") is not False:
            raise M3ValidationError(f"validation run 执行 validation optimization: {scene}")
        if summary["status"] == "abstain":
            scene_rows.append(
                {
                    "scene": scene,
                    "status": "abstain",
                    "reason": summary["reason"],
                    "retained_in_denominator": True,
                    "source_run": config["validation_scene_runs"][scene]["path"],
                }
            )
            continue
        if summary["status"] != "done":
            raise M3ValidationError(f"validation scene 非 done/abstain: {scene}")
        if summary["parameters"] != selected:
            raise M3ValidationError(f"validation 参数漂移: {scene}")
        if set(summary["operations"]) != set(operations):
            raise M3ValidationError(f"validation operation 不完整: {scene}")
        if set(summary["arms"]) != {
            "FRAME_INDEPENDENT",
            "LINEAR",
            "CUBIC_BSPLINE",
            "CUBIC_BSPLINE_TEMPORAL_EVIDENCE",
            "FULL_WARP_REGULARIZED",
        }:
            raise M3ValidationError(f"validation ablation arms 不完整: {scene}")
        if not summary["checkpoint_immutable"] or not summary["rollback_exact"]:
            raise M3ValidationError(f"validation rollback/checkpoint 非 exact: {scene}")
        evaluable_count += 1
        all_sequences.extend(summary["sequences"])
        scene_rows.append(
            {
                "scene": scene,
                "status": "evaluable",
                "reason": None,
                "retained_in_denominator": True,
                "source_run": config["validation_scene_runs"][scene]["path"],
                "sequence_count": len(summary["sequences"]),
            }
        )
        for operation in operations:
            baseline = unique_sequence(summary, operation, baseline_arm)
            candidate = unique_sequence(summary, operation, candidate_arm)
            paired_rows.append(
                {
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
                    "non_target_lpips_alex_delta": candidate[
                        "non_target_lpips_alex"
                    ]
                    - baseline["non_target_lpips_alex"],
                }
            )
    if evaluable_count == 0:
        raise M3ValidationError("validation 无可评场景")
    expected_pairs = evaluable_count * len(operations)
    if len(paired_rows) != expected_pairs:
        raise M3ValidationError("scene×operation paired denominator 不完整")
    baseline_warp = mean(row["baseline_warp_l1_delta"] for row in paired_rows)
    candidate_warp = mean(row["candidate_warp_l1_delta"] for row in paired_rows)
    baseline_tlpips = mean(row["baseline_temporal_lpips"] for row in paired_rows)
    candidate_tlpips = mean(row["candidate_temporal_lpips"] for row in paired_rows)
    warp_improvement = relative_improvement(baseline_warp, candidate_warp)
    tlpips_improvement = relative_improvement(baseline_tlpips, candidate_tlpips)
    baseline_success = sum(
        bool(row["baseline_operation_success"]) for row in paired_rows
    )
    candidate_success = sum(
        bool(row["candidate_operation_success"]) for row in paired_rows
    )
    baseline_identity = sum(int(row["baseline_identity_switch"]) for row in paired_rows)
    candidate_identity = sum(
        int(row["candidate_identity_switch"]) for row in paired_rows
    )
    gates = config["gates"]
    identity_passed = (
        baseline_identity == 0 and candidate_identity == 0
        if gates["allow_identity_zero_remains_zero"]
        else False
    )
    if baseline_identity > 0:
        identity_passed = (
            (baseline_identity - candidate_identity) / baseline_identity
            >= float(gates["identity_switch_relative_reduction_min"])
        )
    gate_rows = {
        "operation_success_non_degradation": {
            "baseline": baseline_success,
            "candidate": candidate_success,
            "passed": candidate_success >= baseline_success,
        },
        "temporal_error_relative_improvement": {
            "warp_l1": warp_improvement,
            "temporal_lpips": tlpips_improvement,
            "threshold": float(gates["temporal_error_relative_improvement_min"]),
            "rule": "either",
            "passed": max(warp_improvement, tlpips_improvement)
            >= float(gates["temporal_error_relative_improvement_min"]),
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
            "threshold": int(gates["deleted_semantic_reintroduction"]),
        },
        "rollback_exact_fraction": {
            "value": mean(
                float(bool(row["rollback_exact"])) for row in all_sequences
            ),
            "threshold": float(gates["rollback_exact_fraction"]),
        },
    }
    gate_rows["deleted_semantic_reintroduction"]["passed"] = (
        gate_rows["deleted_semantic_reintroduction"]["value"]
        == gate_rows["deleted_semantic_reintroduction"]["threshold"]
    )
    gate_rows["rollback_exact_fraction"]["passed"] = (
        gate_rows["rollback_exact_fraction"]["value"]
        == gate_rows["rollback_exact_fraction"]["threshold"]
    )
    validation_gate_passed = all(row["passed"] for row in gate_rows.values())
    write_jsonl(run_dir / "paired_metrics.jsonl", paired_rows)
    source_snapshot = run_dir / "source_snapshot"
    source_snapshot.mkdir()
    for path in (config_path, m3_config_path, inventory_path, Path(__file__)):
        shutil.copy2(path, source_snapshot / path.name)
    summary = {
        "schema_version": "worldsim_v4_m3_validation_summary_v1",
        "task_id": TASK_ID,
        "status": "done",
        "partition": "validation",
        "scene_denominator": len(validation_inventory),
        "evaluable_scene_count": evaluable_count,
        "abstain_scene_count": len(validation_inventory) - evaluable_count,
        "scene_operation_denominator": len(paired_rows),
        "scenes": scene_rows,
        "selected_parameters": selected,
        "reduction": config["aggregation"]["reduction"],
        "baseline_arm": baseline_arm,
        "candidate_arm": candidate_arm,
        "aggregate": {
            "baseline_warp_l1_delta": baseline_warp,
            "candidate_warp_l1_delta": candidate_warp,
            "warp_l1_relative_improvement": warp_improvement,
            "baseline_temporal_lpips": baseline_tlpips,
            "candidate_temporal_lpips": candidate_tlpips,
            "temporal_lpips_relative_improvement": tlpips_improvement,
            "non_target_psnr_delta": mean(
                row["non_target_psnr_delta"] for row in paired_rows
            ),
            "non_target_ssim_delta": mean(
                row["non_target_ssim_delta"] for row in paired_rows
            ),
            "non_target_lpips_alex_delta": mean(
                row["non_target_lpips_alex_delta"] for row in paired_rows
            ),
        },
        "gates": gate_rows,
        "validation_gate_passed": validation_gate_passed,
        "test_freeze_authorized": validation_gate_passed,
        "limitations": [
            "仅 3/6 validation scenes 可评，其余三场按冻结 actor registry 保留 abstain",
            "REMOVE 因 remove_bypass_exact 在五臂间完全一致，零 temporal improvement 仍保留在九序列聚合",
            "证据只支持冻结三操作/三前向相机/2–4 秒 clip，不外推到任意角色或视角",
        ],
        "development_content_read": False,
        "development_optimization_read": False,
        "validation_content_read": True,
        "validation_optimization_read": False,
        "test_quality_read": False,
        "project_git_head": git_head(),
        "project_git_dirty": git_dirty(),
        "duration_seconds": time.monotonic() - started,
    }
    atomic_json(run_dir / "summary.json", summary)
    atomic_json(
        run_dir / "fingerprint.json",
        {
            "task_id": TASK_ID,
            "project_git_head": git_head(),
            "project_git_dirty": git_dirty(),
            "validation_config_sha256": sha256_file(config_path),
            "m3_config_sha256": sha256_file(m3_config_path),
            "scene_inventory_sha256": sha256_file(inventory_path),
            "development_freeze_summary_sha256": config["development_freeze"][
                "summary_sha256"
            ],
            "test_quality_read": False,
        },
    )
    manifest = output_manifest(run_dir)
    atomic_json(run_dir / "manifest.json", manifest)
    atomic_json(
        run_dir / "status.json",
        {
            "task_id": TASK_ID,
            "status": "done",
            "summary_sha256": sha256_file(run_dir / "summary.json"),
            "manifest_sha256": sha256_file(run_dir / "manifest.json"),
            "fingerprint_sha256": sha256_file(run_dir / "fingerprint.json"),
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs/worldsim_v4/m3_validation_selection_v1.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run(config_path=args.config.resolve(), run_dir=args.run_dir.resolve())
    print(
        json.dumps(
            {
                "status": summary["status"],
                "validation_gate_passed": summary["validation_gate_passed"],
                "test_freeze_authorized": summary["test_freeze_authorized"],
                "aggregate": summary["aggregate"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
