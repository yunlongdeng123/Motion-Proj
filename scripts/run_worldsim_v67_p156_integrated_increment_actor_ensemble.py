"""Train Actor residual-velocity ensembles and integrate their distributions over time."""

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

from motion_proj.worldsim_v67.actor_state_reliability import ACTOR_FEATURE_NAMES, spearman_correlation
from scripts.run_worldsim_v67_p86_direct_trajectory_reliability import _select_by_scene
from scripts.run_worldsim_v67_p104_temporal_flip_supervision import _aligned_group_max
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian, _predict
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost


def _increment_tokens(
    arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keys = np.stack((
        arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["actor_id"],
    ), axis=1)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    actor = np.asarray(arrays["features"], dtype=np.float32)[first, :len(ACTOR_FEATURE_NAMES)]
    horizon = np.asarray(arrays["horizon_seconds"], dtype=np.float32)[first]
    residual = np.asarray(arrays["actor_position_error_vector_ego_profile_m"], dtype=np.float32)[first]
    interval_count = residual.shape[1] - 1
    delta_t = horizon / interval_count
    midpoint = delta_t[:, None] * (np.arange(interval_count, dtype=np.float32)[None] + 0.5)
    features = np.concatenate((
        np.broadcast_to(actor[:, None, :], (len(actor), interval_count, actor.shape[1])),
        midpoint[:, :, None], np.broadcast_to(horizon[:, None, None], (len(actor), interval_count, 1)),
    ), axis=2)
    velocity = np.diff(residual, axis=1) / delta_t[:, None, None]
    return features.reshape(-1, features.shape[-1]), velocity.reshape(-1, 2), inverse, delta_t


def _trajectory_score(
    arrays: dict[str, np.ndarray], models: list[DirectionalActorGaussian],
    feature_mean: np.ndarray, feature_scale: np.ndarray,
    target_mean: np.ndarray, target_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    actor_features, _, inverse, delta_t = _increment_tokens(arrays)
    point_count = int(arrays["actor_position_error_vector_ego_profile_m"].shape[1])
    interval_count = point_count - 1
    member_means, member_variances = [], []
    for model in models:
        velocity_mean, velocity_scale = _predict(
            model.eval(), actor_features, feature_mean, feature_scale, target_mean, target_scale,
        )
        velocity_mean = velocity_mean.reshape(-1, interval_count, 2)
        velocity_scale = velocity_scale.reshape(-1, interval_count, 2)
        increment_mean = velocity_mean * delta_t[:, None, None]
        increment_variance = np.square(velocity_scale * delta_t[:, None, None])
        position_mean = np.concatenate((
            np.zeros((len(delta_t), 1, 2), dtype=np.float32), np.cumsum(increment_mean, axis=1),
        ), axis=1)
        position_variance = np.concatenate((
            np.zeros((len(delta_t), 1, 2), dtype=np.float32), np.cumsum(increment_variance, axis=1),
        ), axis=1)
        member_means.append(position_mean[inverse])
        member_variances.append(position_variance[inverse])
    means = np.stack(member_means, axis=0)
    variances = np.stack(member_variances, axis=0)
    normal = np.asarray(arrays["query_boundary_normal_ego_profile"], dtype=np.float32)
    projected_member_mean = np.sum(normal[None] * means, axis=3)
    projected_mean = projected_member_mean.mean(axis=0)
    epistemic_variance = projected_member_mean.var(axis=0)
    aleatoric_variance = np.mean(np.sum(np.square(normal[None]) * variances, axis=3), axis=0)
    total_variance = np.maximum(epistemic_variance + aleatoric_variance, 1e-8)
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
    raw_features, raw_target, _, _ = _increment_tokens(source)
    feature_mean = raw_features.mean(0)
    feature_scale = raw_features.std(0).clip(min=1e-4)
    target_mean = raw_target.mean(0)
    target_scale = raw_target.std(0).clip(min=0.05)
    features = torch.from_numpy((raw_features - feature_mean) / feature_scale).cuda()
    targets = torch.from_numpy((raw_target - target_mean) / target_scale).cuda()
    model_config = config["model"]
    models = []
    final_losses = {}
    torch.cuda.reset_peak_memory_stats()
    for seed_value in config["member_seeds"]:
        seed_value = int(seed_value)
        torch.manual_seed(seed_value)
        model = DirectionalActorGaussian(features.shape[1], model_config["hidden_dimensions"]).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"]))
        final_loss = 0.0
        for step in range(int(model_config["steps"])):
            index = torch.randint(len(features), (int(model_config["batch_size"]),), device="cuda")
            mean, scale = model(features[index])
            normalized = (targets[index] - mean) / scale
            loss = (0.5 * normalized.square() + torch.log(scale)).sum(1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
            if step % 500 == 0 or step + 1 == int(model_config["steps"]):
                print(f"P156 increment seed={seed_value} step={step + 1} nll={final_loss:.6f}", flush=True)
        final_losses[str(seed_value)] = final_loss
        models.append(model.eval())
    torch.save({
        "feature_mean": feature_mean, "feature_scale": feature_scale, "target_mean": target_mean,
        "target_scale": target_scale, "hidden_dimensions": model_config["hidden_dimensions"],
        "member_state_dicts": [model.state_dict() for model in models],
    }, run_dir / config["model_artifact"])
    floor = float(config["boundary_state_cost"]["clearance_floor_m"])
    coverage = float(config["selection"]["coverage_fraction"])
    results = {}
    for cohort in config["development_cohorts"]:
        arrays = dict(np.load(args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False))
        score, scenes, epistemic_fraction = _trajectory_score(
            arrays, models, feature_mean, feature_scale, target_mean, target_scale,
        )
        actual_cost, cost_scenes = _continuous_cost(arrays, floor)
        if not np.array_equal(scenes, cost_scenes):
            raise RuntimeError("P156 trajectory grouping is not aligned")
        selected = _select_by_scene(score, scenes, coverage)
        reference = config["frozen_p126_comparisons"][cohort["name"]]
        model_spearman = spearman_correlation(actual_cost, score)
        results[cohort["name"]] = {
            "trajectory_count": int(len(actual_cost)), "selected_trajectory_count": int(len(selected)),
            "integrated_increment_selected_mean_cost": float(actual_cost[selected].mean()),
            "p126_selected_mean_cost": float(reference["selected_cost"]),
            "integrated_increment_cost_spearman": model_spearman,
            "p126_cost_spearman": float(reference["spearman"]),
            "spearman_gain_over_p126": float(model_spearman - float(reference["spearman"])),
            "mean_projected_epistemic_fraction": epistemic_fraction,
        }
        print(json.dumps({cohort["name"]: results[cohort["name"]]}, indent=2), flush=True)
    gains = [row["spearman_gain_over_p126"] for row in results.values()]
    decisions = {
        "no_selected_cost_regression": all(row["integrated_increment_selected_mean_cost"] <= row["p126_selected_mean_cost"] for row in results.values()),
        "minimum_mean_spearman_gain": float(np.mean(gains)) >= float(config["decision"]["minimum_mean_spearman_gain"]),
    }
    verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "training": {"increment_time_tokens": int(len(features)), "interval_count": 8,
                     "source_horizons_seconds": sorted(np.unique(source["horizon_seconds"]).tolist()),
                     "member_final_velocity_nll": final_losses},
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
