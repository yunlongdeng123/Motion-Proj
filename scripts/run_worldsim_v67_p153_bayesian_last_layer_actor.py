"""Fit a full Bayesian last-layer covariance on the frozen P109 Actor model."""

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
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian, _actor_entries
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost


@torch.no_grad()
def _hidden_and_distribution(
    model: DirectionalActorGaussian, features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = model.network[:-1](features)
    mean, scale = model(features)
    augmented = torch.cat((hidden, torch.ones((len(hidden), 1), device=hidden.device)), dim=1)
    return augmented, mean, scale


@torch.no_grad()
def _fit_precision(
    model: DirectionalActorGaussian, features: torch.Tensor,
    batch_size: int, prior_precision: float,
) -> torch.Tensor:
    dimension = int(model.network[-1].in_features) + 1
    precision = torch.eye(dimension, dtype=torch.float64, device="cuda")[None].repeat(2, 1, 1)
    precision *= prior_precision
    for start in range(0, len(features), batch_size):
        hidden, _, scale = _hidden_and_distribution(model, features[start:start + batch_size])
        hidden = hidden.double()
        weights = scale.double().square().reciprocal()
        for axis in range(2):
            weighted = hidden * torch.sqrt(weights[:, axis:axis + 1])
            precision[axis].add_(weighted.T @ weighted)
    return torch.linalg.inv(precision).float()


@torch.no_grad()
def _predict_with_leverage(
    model: DirectionalActorGaussian, raw_features: np.ndarray,
    feature_mean: np.ndarray, feature_scale: np.ndarray,
    target_mean: np.ndarray, target_scale: np.ndarray,
    covariance: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means, scales, epistemic = [], [], []
    for start in range(0, len(raw_features), 65536):
        batch = torch.from_numpy((raw_features[start:start + 65536] - feature_mean) / feature_scale).cuda()
        hidden, mean, scale = _hidden_and_distribution(model, batch)
        axis_variance = torch.stack([
            torch.sum((hidden @ covariance[axis]) * hidden, dim=1) for axis in range(2)
        ], dim=1)
        means.append((mean * torch.from_numpy(target_scale).cuda() + torch.from_numpy(target_mean).cuda()).cpu().numpy())
        scales.append((scale * torch.from_numpy(target_scale).cuda()).cpu().numpy())
        epistemic.append((axis_variance * torch.from_numpy(np.square(target_scale)).cuda()).cpu().numpy())
    return np.concatenate(means), np.concatenate(scales), np.concatenate(epistemic)


def _trajectory_score(
    arrays: dict[str, np.ndarray], model: DirectionalActorGaussian,
    feature_mean: np.ndarray, feature_scale: np.ndarray,
    target_mean: np.ndarray, target_scale: np.ndarray, covariance: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, float]:
    actor_features, _, inverse = _actor_entries(arrays)
    point_count = int(arrays["actor_position_error_vector_ego_profile_m"].shape[1])
    mean, scale, epistemic = _predict_with_leverage(
        model, actor_features, feature_mean, feature_scale, target_mean, target_scale, covariance,
    )
    row_mean = mean.reshape(-1, point_count, 2)[inverse]
    row_scale = scale.reshape(-1, point_count, 2)[inverse]
    row_epistemic = epistemic.reshape(-1, point_count, 2)[inverse]
    normal = np.asarray(arrays["query_boundary_normal_ego_profile"], dtype=np.float32)
    projected_mean = np.sum(normal * row_mean, axis=2)
    aleatoric_variance = np.sum(np.square(normal * row_scale), axis=2)
    epistemic_variance = np.sum(np.square(normal) * row_epistemic, axis=2)
    total_variance = np.maximum(aleatoric_variance + epistemic_variance, 1e-8)
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
    return (
        _aligned_group_max(keys, row_score, identities), identities[:, 0].astype(np.int32),
        float(np.mean(epistemic_variance / total_variance)),
    )


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
    source = dict(np.load(args.runs_root / config["source_rows"]["run"] / config["source_rows"]["artifact"], allow_pickle=False))
    raw_features, _, _ = _actor_entries(source)
    checkpoint = torch.load(args.runs_root / config["frozen_p109"]["run"] / config["frozen_p109"]["artifact"], map_location="cuda")
    model = DirectionalActorGaussian(20, checkpoint["hidden_dimensions"]).cuda()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_scale = np.asarray(checkpoint["feature_scale"], dtype=np.float32)
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
    target_scale = np.asarray(checkpoint["target_scale"], dtype=np.float32)
    features = torch.from_numpy((raw_features - feature_mean) / feature_scale).cuda()
    posterior = config["posterior"]
    torch.cuda.reset_peak_memory_stats()
    covariance = _fit_precision(model, features, int(posterior["batch_size"]), float(posterior["prior_precision"]))
    torch.save({"last_layer_covariance": covariance.cpu(), "prior_precision": float(posterior["prior_precision"])}, run_dir / config["model_artifact"])
    floor = float(config["boundary_state_cost"]["clearance_floor_m"])
    coverage = float(config["selection"]["coverage_fraction"])
    results = {}
    for cohort in config["development_cohorts"]:
        arrays = dict(np.load(args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False))
        score, scenes, epistemic_fraction = _trajectory_score(
            arrays, model, feature_mean, feature_scale, target_mean, target_scale, covariance,
        )
        actual_cost, cost_scenes = _continuous_cost(arrays, floor)
        if not np.array_equal(scenes, cost_scenes):
            raise RuntimeError("P153 trajectory grouping is not aligned")
        selected = _select_by_scene(score, scenes, coverage)
        reference = config["frozen_p126_comparisons"][cohort["name"]]
        model_spearman = spearman_correlation(actual_cost, score)
        results[cohort["name"]] = {
            "trajectory_count": int(len(actual_cost)), "selected_trajectory_count": int(len(selected)),
            "last_layer_selected_mean_cost": float(actual_cost[selected].mean()),
            "p126_selected_mean_cost": float(reference["selected_cost"]),
            "last_layer_cost_spearman": model_spearman,
            "p126_cost_spearman": float(reference["spearman"]),
            "spearman_gain_over_p126": float(model_spearman - float(reference["spearman"])),
            "mean_projected_epistemic_fraction": epistemic_fraction,
        }
        print(json.dumps({cohort["name"]: results[cohort["name"]]}, indent=2), flush=True)
    gains = [row["spearman_gain_over_p126"] for row in results.values()]
    decisions = {
        "no_selected_cost_regression": all(row["last_layer_selected_mean_cost"] <= row["p126_selected_mean_cost"] for row in results.values()),
        "minimum_mean_spearman_gain": float(np.mean(gains)) >= float(config["decision"]["minimum_mean_spearman_gain"]),
    }
    verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "posterior_fit": {"actor_time_tokens": int(len(features)), "last_layer_dimension": int(covariance.shape[-1]),
                          "prior_precision": float(posterior["prior_precision"])},
        "development_evaluations": results, "decision_checks": decisions,
        "mean_spearman_gain": float(np.mean(gains)),
        "resources": {"gpu": torch.cuda.get_device_name(0), "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                      "wall_seconds": time.monotonic() - started},
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict, "decision_checks": decisions}, indent=2), flush=True)


if __name__ == "__main__":
    main()
