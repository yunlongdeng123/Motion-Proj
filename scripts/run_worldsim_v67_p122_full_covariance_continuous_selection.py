"""Select a full-covariance score for prospective continuous-cost evaluation."""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from motion_proj.worldsim_v67.actor_state_reliability import spearman_correlation
from scripts.run_worldsim_v67_p86_direct_trajectory_reliability import _select_by_scene
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import (
    DirectionalActorGaussian, _actor_entries,
)
from scripts.run_worldsim_v67_p114_monotone_tail_risk import (
    _crossing_probability, _trajectory_tail_features,
)
from scripts.run_worldsim_v67_p117_full_covariance_actor_uncertainty import (
    CorrelatedActorGaussian, _predict as _predict_full_covariance,
)
from scripts.run_worldsim_v67_p119_ranked_range_tail import _head_features
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost


@torch.no_grad()
def _full_covariance_crossing_probability(
    arrays: dict[str, np.ndarray], model: CorrelatedActorGaussian,
    feature_mean: np.ndarray, feature_scale: np.ndarray,
    target_mean: np.ndarray, target_scale: np.ndarray,
) -> np.ndarray:
    actor_features, _, inverse = _actor_entries(arrays)
    point_count = int(arrays["actor_position_error_vector_ego_profile_m"].shape[1])
    mean, scale, correlation = _predict_full_covariance(
        model.eval(), actor_features, feature_mean, feature_scale, target_mean, target_scale,
    )
    row_mean = mean.reshape(-1, point_count, 2)[inverse]
    row_scale = scale.reshape(-1, point_count, 2)[inverse]
    row_correlation = correlation.reshape(-1, point_count)[inverse]
    normal = np.asarray(arrays["query_boundary_normal_ego_profile"], dtype=np.float32)
    projected_mean = np.sum(normal * row_mean, axis=2)
    nx_sx = normal[:, :, 0] * row_scale[:, :, 0]
    ny_sy = normal[:, :, 1] * row_scale[:, :, 1]
    projected_scale = np.sqrt(np.maximum(
        np.square(nx_sx) + np.square(ny_sy) + 2.0 * row_correlation * nx_sx * ny_sy,
        1e-8,
    ))
    predicted = np.asarray(arrays["predicted_separation_profile_m"], dtype=np.float32)
    radius = np.asarray(arrays["occupancy_interaction_radius_m"], dtype=np.float32)[:, None]
    signed = predicted - radius
    margin = (np.abs(signed) + np.sign(signed) * projected_mean) / projected_scale
    return torch.special.ndtr(torch.from_numpy(-margin).cuda()).cpu().numpy()


def _evaluate(
    arrays: dict[str, np.ndarray], full_probability: np.ndarray,
    p109_probability: np.ndarray, top_k: int, coverage: float, floor: float,
) -> dict[str, float | int]:
    full_grouped = _trajectory_tail_features(arrays, full_probability, top_k)
    p109_grouped = _trajectory_tail_features(arrays, p109_probability, top_k)
    _, full_score = _head_features(full_grouped)
    _, p109_score = _head_features(p109_grouped)
    actual_cost, scenes = _continuous_cost(arrays, floor)
    if not np.array_equal(scenes, full_grouped["scene_index"]):
        raise RuntimeError("P122 continuous cost groups are not aligned")
    clearance = full_grouped["clearance_score"]
    full_selected = _select_by_scene(full_score, scenes, coverage)
    p109_selected = _select_by_scene(p109_score, scenes, coverage)
    clearance_selected = _select_by_scene(clearance, scenes, coverage)
    full_spearman = spearman_correlation(actual_cost, full_score)
    p109_spearman = spearman_correlation(actual_cost, p109_score)
    return {
        "trajectory_count": int(len(actual_cost)),
        "selected_trajectory_count": int(len(full_selected)),
        "all_mean_boundary_state_cost": float(actual_cost.mean()),
        "full_covariance_selected_mean_cost": float(actual_cost[full_selected].mean()),
        "p109_selected_mean_cost": float(actual_cost[p109_selected].mean()),
        "clearance_selected_mean_cost": float(actual_cost[clearance_selected].mean()),
        "full_covariance_cost_spearman": full_spearman,
        "p109_cost_spearman": p109_spearman,
        "clearance_cost_spearman": spearman_correlation(actual_cost, clearance),
        "full_covariance_spearman_gain_over_p109": full_spearman - p109_spearman,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_dir = args.runs_root / "worldsim_v67" / config["task_id"] / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    started = time.monotonic()
    full_checkpoint = torch.load(
        args.runs_root / config["frozen_p117"]["run"] / config["frozen_p117"]["artifact"],
        map_location="cuda",
    )
    full_model = CorrelatedActorGaussian(20, full_checkpoint["hidden_dimensions"]).cuda()
    full_model.load_state_dict(full_checkpoint["model_state_dict"])
    p109_checkpoint = torch.load(
        args.runs_root / config["frozen_p109"]["run"] / config["frozen_p109"]["artifact"],
        map_location="cuda",
    )
    p109_model = DirectionalActorGaussian(20, p109_checkpoint["hidden_dimensions"]).cuda()
    p109_model.load_state_dict(p109_checkpoint["model_state_dict"])
    rows_root = args.runs_root / config["rows"]["run"]
    results = {}
    torch.cuda.reset_peak_memory_stats()
    for cohort in config["development_cohorts"]:
        cohort_root = args.runs_root / cohort.get("run", config["rows"]["run"])
        arrays = dict(np.load(cohort_root / cohort["artifact"], allow_pickle=False))
        full_probability = _full_covariance_crossing_probability(
            arrays, full_model,
            np.asarray(full_checkpoint["feature_mean"], dtype=np.float32),
            np.asarray(full_checkpoint["feature_scale"], dtype=np.float32),
            np.asarray(full_checkpoint["target_mean"], dtype=np.float32),
            np.asarray(full_checkpoint["target_scale"], dtype=np.float32),
        )
        p109_probability, _ = _crossing_probability(
            arrays, p109_model,
            np.asarray(p109_checkpoint["feature_mean"], dtype=np.float32),
            np.asarray(p109_checkpoint["feature_scale"], dtype=np.float32),
            np.asarray(p109_checkpoint["target_mean"], dtype=np.float32),
            np.asarray(p109_checkpoint["target_scale"], dtype=np.float32),
        )
        results[cohort["name"]] = _evaluate(
            arrays, full_probability, p109_probability,
            int(config["evaluation"]["top_k_crossing_probabilities"]),
            float(config["selection"]["coverage_fraction"]),
            float(config["boundary_state_cost"]["clearance_floor_m"]),
        )
        print(json.dumps({cohort["name"]: results[cohort["name"]]}, indent=2), flush=True)
    gains = [float(value["full_covariance_spearman_gain_over_p109"]) for value in results.values()]
    decisions = {
        "no_selected_cost_regression": all(
            value["full_covariance_selected_mean_cost"] <= value["p109_selected_mean_cost"]
            for value in results.values()
        ),
        "minimum_mean_spearman_gain": float(np.mean(gains))
        >= float(config["decision"]["minimum_mean_spearman_gain_over_p109"]),
    }
    verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"], "development_evaluations": results,
        "decision_checks": decisions,
        "resources": {"gpu": torch.cuda.get_device_name(0),
                      "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                      "wall_seconds": time.monotonic() - started},
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({
        "status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict,
                      "decision_checks": decisions}, indent=2), flush=True)


if __name__ == "__main__":
    main()
