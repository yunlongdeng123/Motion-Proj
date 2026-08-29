"""Train a three-member full-covariance Actor residual deep ensemble."""

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
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import _actor_entries
from scripts.run_worldsim_v67_p117_full_covariance_actor_uncertainty import (
    CorrelatedActorGaussian, _predict,
)
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost


def _trajectory_score(
    arrays: dict[str, np.ndarray], models: list[CorrelatedActorGaussian],
    feature_mean: np.ndarray, feature_scale: np.ndarray,
    target_mean: np.ndarray, target_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    actor_features, _, inverse = _actor_entries(arrays)
    point_count = int(arrays["actor_position_error_vector_ego_profile_m"].shape[1])
    member_means, member_scales, member_correlations = [], [], []
    for model in models:
        mean, scale, correlation = _predict(
            model.eval(), actor_features, feature_mean, feature_scale, target_mean, target_scale,
        )
        member_means.append(mean.reshape(-1, point_count, 2)[inverse])
        member_scales.append(scale.reshape(-1, point_count, 2)[inverse])
        member_correlations.append(correlation.reshape(-1, point_count)[inverse])
    means = np.stack(member_means)
    scales = np.stack(member_scales)
    correlations = np.stack(member_correlations)
    normal = np.asarray(arrays["query_boundary_normal_ego_profile"], dtype=np.float32)
    projected_member_mean = np.sum(normal[None] * means, axis=3)
    nx_sx = normal[None, :, :, 0] * scales[:, :, :, 0]
    ny_sy = normal[None, :, :, 1] * scales[:, :, :, 1]
    projected_member_variance = (
        np.square(nx_sx) + np.square(ny_sy) + 2.0 * correlations * nx_sx * ny_sy
    )
    projected_mean = projected_member_mean.mean(axis=0)
    epistemic = projected_member_mean.var(axis=0)
    total_variance = np.maximum(projected_member_variance.mean(axis=0) + epistemic, 1e-8)
    signed = (
        np.asarray(arrays["predicted_separation_profile_m"], dtype=np.float32)
        - np.asarray(arrays["occupancy_interaction_radius_m"], dtype=np.float32)[:, None]
    )
    row_score = np.max(-(
        np.abs(signed) + np.sign(signed) * projected_mean
    ) / np.sqrt(total_variance), axis=1)
    keys = np.stack((
        arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["query_id"],
    ), axis=1)
    identities = np.unique(keys, axis=0)
    return (
        _aligned_group_max(keys, row_score, identities),
        identities[:, 0].astype(np.int32),
        float(np.mean(epistemic / total_variance)),
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
    source = dict(np.load(
        args.runs_root / config["source_rows"]["run"] / config["source_rows"]["artifact"],
        allow_pickle=False,
    ))
    raw_features, raw_target, _ = _actor_entries(source)
    frozen = torch.load(
        args.runs_root / config["frozen_p117"]["run"] / config["frozen_p117"]["artifact"],
        map_location="cuda",
    )
    feature_mean = np.asarray(frozen["feature_mean"], dtype=np.float32)
    feature_scale = np.asarray(frozen["feature_scale"], dtype=np.float32)
    target_mean = np.asarray(frozen["target_mean"], dtype=np.float32)
    target_scale = np.asarray(frozen["target_scale"], dtype=np.float32)
    features = torch.from_numpy((raw_features - feature_mean) / feature_scale).cuda()
    targets = torch.from_numpy((raw_target - target_mean) / target_scale).cuda()
    model_config = config["model"]
    seed0 = CorrelatedActorGaussian(features.shape[1], model_config["hidden_dimensions"]).cuda()
    seed0.load_state_dict(frozen["model_state_dict"])
    models = [seed0.eval()]
    final_losses = {}
    torch.cuda.reset_peak_memory_stats()
    for seed_value in config["new_member_seeds"]:
        seed_value = int(seed_value)
        torch.manual_seed(seed_value)
        model = CorrelatedActorGaussian(features.shape[1], model_config["hidden_dimensions"]).cuda()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(model_config["learning_rate"]),
            weight_decay=float(model_config["weight_decay"]),
        )
        final_loss = 0.0
        for step in range(int(model_config["steps"])):
            index = torch.randint(
                len(features), (int(model_config["batch_size"]),), device="cuda",
            )
            mean, scale, correlation = model(features[index])
            residual = (targets[index] - mean) / scale
            one_minus_rho2 = 1.0 - correlation.square()
            mahalanobis = (
                residual[:, 0].square() + residual[:, 1].square()
                - 2.0 * correlation * residual[:, 0] * residual[:, 1]
            ) / one_minus_rho2
            loss = (
                torch.log(scale).sum(1) + 0.5 * torch.log(one_minus_rho2)
                + 0.5 * mahalanobis
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
            if step % 500 == 0 or step + 1 == int(model_config["steps"]):
                print(
                    f"P138 full-cov ensemble seed={seed_value} step={step + 1} nll={final_loss:.6f}",
                    flush=True,
                )
        final_losses[str(seed_value)] = final_loss
        models.append(model.eval())
    torch.save({
        "feature_mean": feature_mean, "feature_scale": feature_scale,
        "target_mean": target_mean, "target_scale": target_scale,
        "hidden_dimensions": model_config["hidden_dimensions"],
        "member_seeds": [0] + [int(x) for x in config["new_member_seeds"]],
        "member_state_dicts": [model.state_dict() for model in models],
    }, run_dir / config["model_artifact"])
    floor = float(config["boundary_state_cost"]["clearance_floor_m"])
    coverage = float(config["selection"]["coverage_fraction"])
    results = {}
    for cohort in config["development_cohorts"]:
        arrays = dict(np.load(
            args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False,
        ))
        score, scenes, epistemic_fraction = _trajectory_score(
            arrays, models, feature_mean, feature_scale, target_mean, target_scale,
        )
        actual_cost, cost_scenes = _continuous_cost(arrays, floor)
        if not np.array_equal(scenes, cost_scenes):
            raise RuntimeError("P138 trajectory grouping is not aligned")
        selected = _select_by_scene(score, scenes, coverage)
        reference = config["frozen_p126_comparisons"][cohort["name"]]
        model_spearman = spearman_correlation(actual_cost, score)
        results[cohort["name"]] = {
            "trajectory_count": int(len(actual_cost)),
            "selected_trajectory_count": int(len(selected)),
            "full_cov_ensemble_selected_mean_cost": float(actual_cost[selected].mean()),
            "diagonal_ensemble_selected_mean_cost": float(reference["selected_cost"]),
            "full_cov_ensemble_cost_spearman": model_spearman,
            "diagonal_ensemble_cost_spearman": float(reference["spearman"]),
            "spearman_gain_over_diagonal_ensemble": float(
                model_spearman - float(reference["spearman"])
            ),
            "mean_projected_epistemic_fraction": epistemic_fraction,
        }
        print(json.dumps({cohort["name"]: results[cohort["name"]]}, indent=2), flush=True)
    gains = [x["spearman_gain_over_diagonal_ensemble"] for x in results.values()]
    decisions = {
        "no_selected_cost_regression_from_diagonal_ensemble": all(
            x["full_cov_ensemble_selected_mean_cost"] <= x["diagonal_ensemble_selected_mean_cost"]
            for x in results.values()
        ),
        "minimum_mean_spearman_gain": float(np.mean(gains))
        >= float(config["decision"]["minimum_mean_spearman_gain_over_diagonal_ensemble"]),
    }
    verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "training": {"actor_time_tokens": int(len(features)), "members": 3,
                     "new_member_final_nll": final_losses},
        "development_evaluations": results, "decision_checks": decisions,
        "mean_spearman_gain_over_diagonal_ensemble": float(np.mean(gains)),
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
