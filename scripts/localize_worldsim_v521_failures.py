#!/usr/bin/env python3
"""P5：用完整 Discovery census denominator 做 driving-condition localization。"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from motion_proj.worldsim_v521.protocol import atomic_json


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mse_from_psnr(value: float | None) -> float | None:
    return None if value is None else float(10.0 ** (-float(value) / 10.0))


def bootstrap_scene_rate(rates: list[float], seed: int = 52105, samples: int = 10000) -> dict[str, Any]:
    if len(rates) < 2:
        return {"status": "undefined_insufficient_denominator", "ci95": None}
    rng = np.random.default_rng(seed)
    values = np.asarray(rates, dtype=np.float64)
    estimates = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return {
        "status": "done", "samples": samples, "seed": seed,
        "ci95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))],
    }


def correlation(rows: list[dict[str, Any]], x_name: str, y_name: str, x: list[float], y: list[float]) -> dict[str, Any]:
    scenes = {row["scene"] for row in rows}
    if len(x) < 10 or len(scenes) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return {
            "status": "undefined_insufficient_denominator", "x": x_name, "y": y_name,
            "valid_rows": len(x), "valid_scenes": len(scenes),
        }
    result = spearmanr(np.asarray(x), np.asarray(y))
    return {
        "status": "done", "x": x_name, "y": y_name, "valid_rows": len(x),
        "valid_scenes": len(scenes), "rho": float(result.statistic), "pvalue_descriptive": float(result.pvalue),
        "causal_interpretation": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-run", required=True, type=Path)
    args = parser.parse_args()
    run = args.census_run.resolve()
    if json.loads((run / "P3_SUMMARY.json").read_text(encoding="utf-8")).get("outcome") != "p3_gate_pass":
        raise RuntimeError("P5 只接受 frozen P3")
    base_rows = read_jsonl(run / "BASE_CENSUS_METRICS.jsonl")
    temporal_rows = read_jsonl(run / "TEMPORAL_CENSUS_METRICS.jsonl")
    registry = read_jsonl(run / "BADCASE_REGISTRY.jsonl")
    analysis: dict[str, Any] = {}
    for base in sorted({row["base"] for row in base_rows}):
        rows = [row for row in base_rows if row["base"] == base]
        cases = [row for row in registry if row["base"] == base and row["entity_kind"] == "view"]
        case_keys = {(row["scene"], row["canonical_sample_index"], row["camera"]) for row in cases}
        per_scene = {}
        scene_rates = []
        for scene in sorted({row["scene"] for row in rows}):
            selected = [row for row in rows if row["scene"] == scene]
            count = sum((row["scene"], row["canonical_sample_index"], row["camera"]) in case_keys for row in selected)
            rate = count / len(selected)
            scene_rates.append(rate)
            per_scene[scene] = {"denominator": len(selected), "badcase_count": count, "badcase_rate": rate}
        actor_rows = [row for row in rows if row["metrics"]["actor"]["status"] == "done"]
        areas = [float(row["actor_context"]["image_area_ratio"]) for row in actor_rows]
        actor_psnr = [float(row["metrics"]["actor"]["psnr"]) for row in actor_rows]
        static_actor_ratios, boundary_actor_ratios = [], []
        for row in actor_rows:
            actor_mse = mse_from_psnr(row["metrics"]["actor"]["psnr"])
            static_mse = mse_from_psnr(row["metrics"]["static"]["psnr"])
            boundary_mse = mse_from_psnr(row["metrics"]["boundary"]["psnr"])
            if actor_mse and static_mse is not None:
                static_actor_ratios.append(actor_mse / max(static_mse, 1e-15))
            if actor_mse and boundary_mse is not None:
                boundary_actor_ratios.append(boundary_mse / max(actor_mse, 1e-15))
        temporal = [row for row in temporal_rows if row["base"] == base]
        temporal_values = [float(row["metrics"]["global_residual_change_l1"]) for row in temporal]
        analysis[base] = {
            "full_view_denominator": len(rows),
            "labeled_view_cases": len(cases),
            "per_scene": per_scene,
            "scene_balanced_badcase_rate_mean": float(np.mean(scene_rates)),
            "scene_balanced_badcase_rate_median": float(np.median(scene_rates)),
            "scene_bootstrap_badcase_rate": bootstrap_scene_rate(scene_rates),
            "failure_class_counts": dict(sorted(Counter(label for row in cases for label in row["failure_class"]).items())),
            "actor_area_ratio": {
                "valid_rows": len(areas),
                "median": None if not areas else float(np.median(areas)),
                "q10_q90": None if not areas else [float(np.quantile(areas, 0.1)), float(np.quantile(areas, 0.9))],
            },
            "actor_area_vs_actor_psnr": correlation(actor_rows, "actor_image_area_ratio", "actor_psnr", areas, actor_psnr),
            "actor_to_static_mse_ratio": {
                "valid_rows": len(static_actor_ratios),
                "median": None if not static_actor_ratios else float(np.median(static_actor_ratios)),
            },
            "boundary_to_actor_mse_ratio": {
                "valid_rows": len(boundary_actor_ratios),
                "median": None if not boundary_actor_ratios else float(np.median(boundary_actor_ratios)),
            },
            "temporal_proxy": {
                "valid_windows": len(temporal_values),
                "median": None if not temporal_values else float(np.median(temporal_values)),
                "q90": None if not temporal_values else float(np.quantile(temporal_values, 0.9)),
                "failure_prevalence": "undefined_unwarped_proxy_cannot_trigger_B-TEMPORAL",
            },
            "condition_coverage": {
                "actor_area": {"defined": len(actor_rows), "undefined": len(rows) - len(actor_rows)},
                "distance": {"defined": 0, "undefined": len(rows)},
                "visibility": {"defined": 0, "undefined": len(rows)},
                "speed": {"defined": 0, "undefined": len(rows)},
                "lidar_support": {"defined": 0, "undefined": len(rows)},
                "occlusion_transition": {"defined": 0, "undefined": len(rows)},
            },
        }
    questions = {
        "dynamic_actor_vs_static_background": "partially_answered_by_actor_to_static_mse_ratio",
        "boundary_vs_actor_interior": "answered_by_boundary_to_actor_mse_ratio_and_B-BOUNDARY_prevalence",
        "distance_small_area_lidar_relation": "small_area_only; distance_and_lidar_undefined",
        "occlusion_independent_frequency": "undefined_no_audited_visibility_transition",
        "good_single_frame_temporal_inconsistency": "undefined_unwarped_proxy_only",
        "sparse_observation_identity_independent_of_rgb": "undefined_no_exact_M1_overlap_or_identity_denominator",
    }
    atomic_json(
        run / "P5_LOCALIZATION.json",
        {
            "schema": "worldsim_v521_failure_localization_v1",
            "task_id": "WS-V521-P5-FAILURE-LOCALIZATION-01",
            "status": "done", "outcome": "p5_gate_pass_with_explicit_undefined_axes",
            "analysis": analysis, "core_question_status": questions,
            "denominator_policy": "full_discovery_census_including_undefined_coverage",
            "panel_only_statistics": False, "causal_claims": False,
        },
    )


if __name__ == "__main__":
    main()
