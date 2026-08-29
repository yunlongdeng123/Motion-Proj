"""Ablate only P126 between-member projected-mean variance."""

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
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian, _actor_entries, _predict
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost
from scripts.run_worldsim_v67_p127_ensemble_continuous_selection import _ensemble_trajectory_score
from scripts.run_worldsim_v67_p158_crps_actor_ensemble import _subset


def _aleatoric_only_score(
    arrays: dict[str, np.ndarray], models: list[DirectionalActorGaussian],
    feature_mean: np.ndarray, feature_scale: np.ndarray, target_mean: np.ndarray, target_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    actor_features, _, inverse = _actor_entries(arrays)
    point_count = int(arrays["actor_position_error_vector_ego_profile_m"].shape[1])
    means, scales = [], []
    for model in models:
        mean, scale = _predict(model, actor_features, feature_mean, feature_scale, target_mean, target_scale)
        means.append(mean.reshape(-1, point_count, 2)[inverse])
        scales.append(scale.reshape(-1, point_count, 2)[inverse])
    means = np.stack(means)
    scales = np.stack(scales)
    normal = np.asarray(arrays["query_boundary_normal_ego_profile"], dtype=np.float32)
    projected_member_mean = np.sum(normal[None] * means, axis=3)
    projected_mean = projected_member_mean.mean(axis=0)
    aleatoric_variance = np.mean(np.sum(np.square(normal[None] * scales), axis=3), axis=0)
    epistemic_variance = projected_member_mean.var(axis=0)
    predicted = np.asarray(arrays["predicted_separation_profile_m"], dtype=np.float32)
    radius = np.asarray(arrays["occupancy_interaction_radius_m"], dtype=np.float32)[:, None]
    signed = predicted - radius
    row_score = np.max(-(
        np.abs(signed) + np.sign(signed) * projected_mean
    ) / np.sqrt(np.maximum(aleatoric_variance, 1e-8)), axis=1)
    keys = np.stack((
        arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["query_id"],
    ), axis=1)
    identities = np.unique(keys, axis=0)
    fraction = float(np.mean(epistemic_variance / np.maximum(aleatoric_variance + epistemic_variance, 1e-8)))
    return _aligned_group_max(keys, row_score, identities), identities[:, 0].astype(np.int32), fraction


def _evaluate(
    arrays: dict[str, np.ndarray], models: list[DirectionalActorGaussian],
    feature_mean: np.ndarray, feature_scale: np.ndarray, target_mean: np.ndarray, target_scale: np.ndarray,
    coverage: float, floor: float,
) -> dict[str, float | int]:
    full_score, scenes = _ensemble_trajectory_score(
        arrays, models, feature_mean, feature_scale, target_mean, target_scale,
    )
    control_score, control_scenes, fraction = _aleatoric_only_score(
        arrays, models, feature_mean, feature_scale, target_mean, target_scale,
    )
    actual_cost, cost_scenes = _continuous_cost(arrays, floor)
    if not (np.array_equal(scenes, control_scenes) and np.array_equal(scenes, cost_scenes)):
        raise RuntimeError("P161 trajectory grouping is not aligned")
    full_selected = _select_by_scene(full_score, scenes, coverage)
    control_selected = _select_by_scene(control_score, scenes, coverage)
    full_rank = spearman_correlation(actual_cost, full_score)
    control_rank = spearman_correlation(actual_cost, control_score)
    return {
        "row_count": int(len(arrays["features"])), "trajectory_count": int(len(actual_cost)),
        "selected_trajectory_count": int(len(full_selected)),
        "full_p126_selected_mean_cost": float(actual_cost[full_selected].mean()),
        "aleatoric_only_selected_mean_cost": float(actual_cost[control_selected].mean()),
        "full_minus_aleatoric_selected_cost": float(actual_cost[full_selected].mean() - actual_cost[control_selected].mean()),
        "full_p126_cost_spearman": full_rank, "aleatoric_only_cost_spearman": control_rank,
        "full_minus_aleatoric_spearman_gain": float(full_rank - control_rank),
        "mean_projected_epistemic_fraction": fraction,
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
    frozen = torch.load(
        args.runs_root / config["frozen_p126"]["run"] / config["frozen_p126"]["artifact"], map_location="cuda",
    )
    feature_mean = np.asarray(frozen["feature_mean"], dtype=np.float32)
    feature_scale = np.asarray(frozen["feature_scale"], dtype=np.float32)
    target_mean = np.asarray(frozen["target_mean"], dtype=np.float32)
    target_scale = np.asarray(frozen["target_scale"], dtype=np.float32)
    models = []
    for state in frozen["member_state_dicts"]:
        model = DirectionalActorGaussian(20, frozen["hidden_dimensions"]).cuda()
        model.load_state_dict(state)
        models.append(model.eval())
    coverage = float(config["selection"]["coverage_fraction"])
    floor = float(config["boundary_state_cost"]["clearance_floor_m"])
    torch.cuda.reset_peak_memory_stats()
    decision_results = {}
    for cohort in config["decision_cohorts"]:
        arrays = dict(np.load(args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False))
        decision_results[cohort["name"]] = _evaluate(
            arrays, models, feature_mean, feature_scale, target_mean, target_scale, coverage, floor,
        )
        print(json.dumps({cohort["name"]: decision_results[cohort["name"]]}, indent=2), flush=True)
    diagnostic_spec = config["post_confirmation_diagnostic"]
    diagnostic_arrays = dict(np.load(
        args.runs_root / diagnostic_spec["run"] / diagnostic_spec["artifact"], allow_pickle=False,
    ))
    diagnostic_results = {}
    for horizon in diagnostic_spec["horizons_seconds"]:
        key = str(float(horizon))
        diagnostic_results[key] = _evaluate(
            _subset(diagnostic_arrays, float(horizon)), models,
            feature_mean, feature_scale, target_mean, target_scale, coverage, floor,
        )
        print(json.dumps({f"P147_H{key}": diagnostic_results[key]}, indent=2), flush=True)
    gains = [row["full_minus_aleatoric_spearman_gain"] for row in decision_results.values()]
    decisions = {
        "full_p126_cost_noninferior": all(row["full_p126_selected_mean_cost"] <= row["aleatoric_only_selected_mean_cost"] for row in decision_results.values()),
        "minimum_mean_full_p126_spearman_gain": float(np.mean(gains)) >= float(config["decision"]["minimum_mean_full_minus_aleatoric_spearman_gain"]),
    }
    verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict, "role": config["role"],
        "consumed_development_evaluations": decision_results,
        "post_confirmation_consumed_p147_diagnostic": diagnostic_results,
        "decision_checks": decisions, "mean_full_minus_aleatoric_spearman_gain": float(np.mean(gains)),
        "resources": {"gpu": torch.cuda.get_device_name(0), "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                      "wall_seconds": time.monotonic() - started},
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({
        "status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict, "decision_checks": decisions}, indent=2), flush=True)


if __name__ == "__main__":
    main()
