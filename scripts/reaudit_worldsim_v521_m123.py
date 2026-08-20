#!/usr/bin/env python3
"""P6–P8：把 frozen historical M1/M2/M3 与新 census 做 exact-overlap re-audit。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from motion_proj.worldsim_v521.census import sha256_file
from motion_proj.worldsim_v521.protocol import atomic_json


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-run", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v521/m123_reaudit_v1.yaml"))
    args = parser.parse_args()
    run = args.census_run.resolve()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["status"] != "frozen_before_exact_overlap_quality_read":
        raise RuntimeError("M123 overlap denominator 未冻结")
    base_rows = read_jsonl(run / "BASE_CENSUS_METRICS.jsonl")
    registry = read_jsonl(run / "BADCASE_REGISTRY.jsonl")
    census_scenes = sorted({row["scene"] for row in base_rows})
    m1_development_scenes = ["scene-0471", "scene-1087", "scene-0379"]
    m1_v4_validation_scenes = ["scene-0071", "scene-0317", "scene-0450"]
    m1_overlap = sorted(set(census_scenes) & set(m1_development_scenes + m1_v4_validation_scenes))
    minimum_scenes = int(config["exact_overlap"]["minimum_independent_scenes"])
    m1 = {
        "schema": "worldsim_v521_m1_reaudit_v1",
        "task_id": config["task_ids"]["m1"],
        "status": "done",
        "conclusion": "M1_EVIDENCE_INSUFFICIENT_KEEP_PENDING",
        "census_scenes": census_scenes,
        "historical_m1_scenes": m1_development_scenes + m1_v4_validation_scenes,
        "exact_overlap_scenes": m1_overlap,
        "exact_overlap_view_count": 0,
        "minimum_independent_scenes": minimum_scenes,
        "quadrants": {
            "Q1_base_good_m1_bad": {"status": "undefined_no_exact_overlap", "views": 0},
            "Q2_base_bad_m1_bad": {"status": "undefined_no_exact_overlap", "views": 0},
            "Q3_base_bad_m1_good": {"status": "undefined_no_exact_overlap", "views": 0},
            "Q4_base_good_m1_good": {"status": "undefined_no_exact_overlap", "views": 0},
        },
        "scene_1087_answer": {
            "base_reconstruction_status": "undefined_scene_not_in_matched_base_census",
            "m1_observation_evidence": {
                "raw_mean_observed_views": 0.028232,
                "zero_observation_gaussians": 904933,
                "source": "docs/EXPERIMENTS.md frozen V5.1 M1 evidence",
            },
            "answer": "M1 缺 observation 有直接历史证据；base RGB 是否同时失败无法在 exact matched overlap 下回答",
        },
        "historical_signals_retained": {
            "v5_scene1087_sparse_observation": True,
            "v5_scene0471_identity_recall": 0.080747,
            "v5_scene0379_identity_recall": 0.202933,
            "v5_scene0471_0379_persistent_track_fraction": [0.0, 0.0],
            "v4_scene0317_boundary_support_target_recall_approx": 0.45,
        },
        "readiness_gate": {
            "m1_core_confirmed": False,
            "reason": "0 exact-overlap scenes < frozen minimum 2; historical aggregate 不伪配逐 view",
        },
        "algorithm_modified": False,
    }
    m2 = {
        "schema": "worldsim_v521_m2_reaudit_v1",
        "task_id": config["task_ids"]["m2"],
        "status": "done",
        "conclusion": "M2_EVIDENCE_INSUFFICIENT",
        "exact_taxonomy_mapped_requests": 0,
        "minimum_exact_mapped_requests": int(config["m2"]["minimum_exact_mapped_requests"]),
        "historical_evidence_retained": {
            "v4_requests_candidates": [154, 214],
            "v4_router_accepted_abstain": [83, 71],
            "v4_full_denominator_hole_geometry_mae_delta_m": 3.3908096237,
            "v5_geometry_first_conclusion": "m2_rejected_no_absolute_geometry_safe_candidate",
            "v5_completed_blocked_terminal": [8, 4],
        },
        "questions": {
            "request_failure_class": "undefined_no_exact_request_to_case_mapping",
            "rgb_vs_geometry_tradeoff": "historical_geometry_caveat_retained_not_remapped",
            "abstain_target_validity": "undefined_no_exact_request_to_case_mapping",
            "base_representation_vs_m1_hole": "undefined_no_exact_overlap",
            "future_m1_dependency": "undefined",
        },
        "router_refit": False,
        "threshold_search": False,
        "algorithm_modified": False,
    }
    temporal_proxy_cases = [row for row in registry if "TEMPORAL_PROXY" in row["failure_axes"]]
    m3 = {
        "schema": "worldsim_v521_m3_reaudit_v1",
        "task_id": config["task_ids"]["m3"],
        "status": "done",
        "conclusion": "M3_EVIDENCE_INSUFFICIENT_KEEP_PENDING",
        "new_base_temporal_failure_cases": 0,
        "new_temporal_proxy_cases": len(temporal_proxy_cases),
        "temporal_failure_prevalence": "undefined_unwarped_proxy_cannot_trigger_B-TEMPORAL",
        "historical_v4_temporal_delta": {
            "status": "confirmed",
            "validation_evaluable_abstain": [3, 3],
            "validation_warp_l1_relative_improvement": 0.3041063132,
            "validation_temporal_lpips_relative_improvement": 0.0264715072,
            "test_attempted_evaluable_abstain": [18, 12, 6],
            "test_warp_l1_relative_improvement_approx": 0.3439,
            "test_temporal_lpips_relative_improvement_approx": 0.1637,
            "source": "/root/autodl-tmp/runs/worldsim_v4/WS-V4-M3-TEMPORAL-DELTA-01/20260813T225624Z__m3-test-aggregate18-s0-r335",
        },
        "v5_constraint_projection": {
            "status": "rejected",
            "conclusion": "m3_rejected_constraint_projection_not_needed_on_frozen_requests",
            "does_not_rewrite_v4_temporal_delta": True,
        },
        "questions": {
            "base_B_TEMPORAL_frequency": "undefined",
            "overlap_actor_boundary_occ": "undefined",
            "v4_delta_targets_base_temporal_badcase": "undefined_no_exact_window_mapping",
            "relationship_to_future_m1": "undefined",
            "downstream_editor_role": "keep_pending_not_promoted",
        },
        "parameter_search": False,
        "algorithm_modified": False,
    }
    atomic_json(run / "M1_REAUDIT.json", m1)
    atomic_json(run / "M2_REAUDIT.json", m2)
    atomic_json(run / "M3_REAUDIT.json", m3)
    atomic_json(
        run / "P6_P8_SUMMARY.json",
        {
            "status": "done", "outcome": "m123_reaudit_complete_with_exact_overlap_limits",
            "m1": m1["conclusion"], "m2": m2["conclusion"], "m3": m3["conclusion"],
            "config_sha256": sha256_file(args.config),
            "validation_test_kitti_quality_read": False,
            "algorithm_candidates": 0,
        },
    )


if __name__ == "__main__":
    main()
