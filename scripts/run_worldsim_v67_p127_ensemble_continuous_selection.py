"""Evaluate the frozen P126 ensemble on continuous boundary-state cost."""

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
from scripts.run_worldsim_v67_p104_temporal_flip_supervision import _aligned_group_max
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import (
    DirectionalActorGaussian, _actor_entries, _predict,
)
from scripts.run_worldsim_v67_p114_monotone_tail_risk import (
    _crossing_probability, _trajectory_tail_features,
)
from scripts.run_worldsim_v67_p119_ranked_range_tail import _head_features
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost


def _ensemble_trajectory_score(
    arrays: dict[str, np.ndarray], models: list[DirectionalActorGaussian],
    feature_mean: np.ndarray, feature_scale: np.ndarray,
    target_mean: np.ndarray, target_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    actor_features, _, inverse = _actor_entries(arrays)
    point_count = int(arrays["actor_position_error_vector_ego_profile_m"].shape[1])
    member_means, member_scales = [], []
    for model in models:
        mean, scale = _predict(
            model.eval(), actor_features, feature_mean, feature_scale, target_mean, target_scale,
        )
        member_means.append(mean.reshape(-1, point_count, 2)[inverse])
        member_scales.append(scale.reshape(-1, point_count, 2)[inverse])
    means = np.stack(member_means, axis=0)
    scales = np.stack(member_scales, axis=0)
    normal = np.asarray(arrays["query_boundary_normal_ego_profile"], dtype=np.float32)
    projected_member_mean = np.sum(normal[None] * means, axis=3)
    projected_mean = projected_member_mean.mean(axis=0)
    aleatoric_variance = np.mean(np.sum(np.square(normal[None] * scales), axis=3), axis=0)
    total_variance = np.maximum(aleatoric_variance + projected_member_mean.var(axis=0), 1e-8)
    predicted = np.asarray(arrays["predicted_separation_profile_m"], dtype=np.float32)
    radius = np.asarray(arrays["occupancy_interaction_radius_m"], dtype=np.float32)[:, None]
    signed = predicted - radius
    row_score = np.max(-(
        np.abs(signed) + np.sign(signed) * projected_mean
    ) / np.sqrt(total_variance), axis=1)
    keys = np.stack((
        arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["query_id"],
    ), axis=1)
    identities = np.unique(keys, axis=0)
    return _aligned_group_max(keys, row_score, identities), identities[:, 0].astype(np.int32)


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
    ensemble = torch.load(
        args.runs_root / config["frozen_p126"]["run"] / config["frozen_p126"]["artifact"],
        map_location="cuda",
    )
    feature_mean = np.asarray(ensemble["feature_mean"], dtype=np.float32)
    feature_scale = np.asarray(ensemble["feature_scale"], dtype=np.float32)
    target_mean = np.asarray(ensemble["target_mean"], dtype=np.float32)
    target_scale = np.asarray(ensemble["target_scale"], dtype=np.float32)
    models = []
    for state in ensemble["member_state_dicts"]:
        model = DirectionalActorGaussian(20, ensemble["hidden_dimensions"]).cuda()
        model.load_state_dict(state)
        models.append(model.eval())
    p109 = torch.load(
        args.runs_root / config["frozen_p109"]["run"] / config["frozen_p109"]["artifact"],
        map_location="cuda",
    )
    p109_model = DirectionalActorGaussian(20, p109["hidden_dimensions"]).cuda()
    p109_model.load_state_dict(p109["model_state_dict"])
    p109_model.eval()
    p109_feature_mean = np.asarray(p109["feature_mean"], dtype=np.float32)
    p109_feature_scale = np.asarray(p109["feature_scale"], dtype=np.float32)
    p109_target_mean = np.asarray(p109["target_mean"], dtype=np.float32)
    p109_target_scale = np.asarray(p109["target_scale"], dtype=np.float32)
    floor = float(config["boundary_state_cost"]["clearance_floor_m"])
    coverage = float(config["selection"]["coverage_fraction"])
    top_k = int(config["score"]["top_k_crossing_probabilities"])
    results = {}
    torch.cuda.reset_peak_memory_stats()
    for cohort in config["development_cohorts"]:
        cohort_root = args.runs_root / cohort["run"]
        arrays = dict(np.load(cohort_root / cohort["artifact"], allow_pickle=False))
        ensemble_score, scenes = _ensemble_trajectory_score(
            arrays, models, feature_mean, feature_scale, target_mean, target_scale,
        )
        p109_probability, _ = _crossing_probability(
            arrays, p109_model, p109_feature_mean, p109_feature_scale,
            p109_target_mean, p109_target_scale,
        )
        grouped = _trajectory_tail_features(arrays, p109_probability, top_k)
        _, p109_score = _head_features(grouped)
        actual_cost, cost_scenes = _continuous_cost(arrays, floor)
        if not (np.array_equal(scenes, cost_scenes) and np.array_equal(scenes, grouped["scene_index"])):
            raise RuntimeError("P127 trajectory grouping is not aligned")
        clearance = grouped["clearance_score"]
        ensemble_selected = _select_by_scene(ensemble_score, scenes, coverage)
        p109_selected = _select_by_scene(p109_score, scenes, coverage)
        clearance_selected = _select_by_scene(clearance, scenes, coverage)
        mean_cost = float(actual_cost.mean())
        results[cohort["name"]] = {
            "trajectory_count": int(len(actual_cost)),
            "selected_trajectory_count": int(len(ensemble_selected)),
            "all_mean_boundary_state_cost": mean_cost,
            "ensemble_selected_mean_cost": float(actual_cost[ensemble_selected].mean()),
            "p109_selected_mean_cost": float(actual_cost[p109_selected].mean()),
            "clearance_selected_mean_cost": float(actual_cost[clearance_selected].mean()),
            "ensemble_selected_cost_reduction": float(
                (mean_cost - actual_cost[ensemble_selected].mean()) / max(mean_cost, 1e-12)
            ),
            "ensemble_cost_spearman": spearman_correlation(actual_cost, ensemble_score),
            "p109_cost_spearman": spearman_correlation(actual_cost, p109_score),
            "clearance_cost_spearman": spearman_correlation(actual_cost, clearance),
        }
        results[cohort["name"]]["ensemble_spearman_gain_over_p109"] = float(
            results[cohort["name"]]["ensemble_cost_spearman"]
            - results[cohort["name"]]["p109_cost_spearman"]
        )
        print(json.dumps({cohort["name"]: results[cohort["name"]]}, indent=2), flush=True)
    gains = [value["ensemble_spearman_gain_over_p109"] for value in results.values()]
    decisions = {
        "no_selected_cost_regression": all(
            value["ensemble_selected_mean_cost"] <= value["p109_selected_mean_cost"]
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
