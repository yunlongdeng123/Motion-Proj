"""Learn a task-conditioned standardized-residual correction on frozen P126 uncertainty."""

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
    DirectionalActorGaussian, _actor_entries, _predict as _predict_actor,
)
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost
from scripts.run_worldsim_v67_p142_task_conditioned_projected_ensemble import (
    ConditionalProjectedGaussian, _conditional_tokens, _predict as _predict_correction,
)


def _base_distribution(
    arrays: dict[str, np.ndarray], models: list[DirectionalActorGaussian],
    feature_mean: np.ndarray, feature_scale: np.ndarray,
    target_mean: np.ndarray, target_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    actor_features, _, inverse = _actor_entries(arrays)
    point_count = int(arrays["actor_position_error_vector_ego_profile_m"].shape[1])
    member_means, member_scales = [], []
    for model in models:
        mean, scale = _predict_actor(
            model, actor_features, feature_mean, feature_scale, target_mean, target_scale,
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
    return projected_mean.astype(np.float32), np.sqrt(total_variance).astype(np.float32)


def _correction_data(
    arrays: dict[str, np.ndarray], base_mean: np.ndarray, base_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    conditional_features, projected_target = _conditional_tokens(arrays)
    augmented = np.concatenate((
        conditional_features,
        base_mean.reshape(-1, 1),
        np.log(base_scale).reshape(-1, 1),
    ), axis=1)
    standardized_residual = (
        projected_target - base_mean.reshape(-1)
    ) / base_scale.reshape(-1)
    return augmented.astype(np.float32), standardized_residual.astype(np.float32)


def _trajectory_score(
    arrays: dict[str, np.ndarray], correction_models: list[ConditionalProjectedGaussian],
    correction_feature_mean: np.ndarray, correction_feature_scale: np.ndarray,
    correction_target_mean: float, correction_target_scale: float,
    base_models: list[DirectionalActorGaussian], base_feature_mean: np.ndarray,
    base_feature_scale: np.ndarray, base_target_mean: np.ndarray,
    base_target_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    base_mean, base_scale = _base_distribution(
        arrays, base_models, base_feature_mean, base_feature_scale,
        base_target_mean, base_target_scale,
    )
    raw_features, _ = _correction_data(arrays, base_mean, base_scale)
    point_count = base_mean.shape[1]
    member_means, member_scales = [], []
    for model in correction_models:
        mean, scale = _predict_correction(
            model, raw_features, correction_feature_mean, correction_feature_scale,
            correction_target_mean, correction_target_scale,
        )
        member_means.append(mean.reshape(-1, point_count))
        member_scales.append(scale.reshape(-1, point_count))
    correction_means = np.stack(member_means, axis=0)
    correction_scales = np.stack(member_scales, axis=0)
    correction_mean = correction_means.mean(axis=0)
    projected_mean = base_mean + base_scale * correction_mean
    correction_variance = (
        np.square(correction_scales).mean(axis=0) + correction_means.var(axis=0)
    )
    total_variance = np.maximum(np.square(base_scale) * correction_variance, 1e-8)
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
        _aligned_group_max(keys, row_score, identities),
        identities[:, 0].astype(np.int32),
        float(np.mean(correction_means.var(axis=0) / np.maximum(correction_variance, 1e-8))),
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
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8",
    )
    started = time.monotonic()
    frozen = torch.load(
        args.runs_root / config["frozen_p126"]["run"] / config["frozen_p126"]["artifact"],
        map_location="cuda",
    )
    base_feature_mean = np.asarray(frozen["feature_mean"], dtype=np.float32)
    base_feature_scale = np.asarray(frozen["feature_scale"], dtype=np.float32)
    base_target_mean = np.asarray(frozen["target_mean"], dtype=np.float32)
    base_target_scale = np.asarray(frozen["target_scale"], dtype=np.float32)
    base_models = []
    for state_dict in frozen["member_state_dicts"]:
        model = DirectionalActorGaussian(20, frozen["hidden_dimensions"]).cuda()
        model.load_state_dict(state_dict)
        base_models.append(model.eval())
    source = dict(np.load(
        args.runs_root / config["source_rows"]["run"] / config["source_rows"]["artifact"],
        allow_pickle=False,
    ))
    source_base_mean, source_base_scale = _base_distribution(
        source, base_models, base_feature_mean, base_feature_scale,
        base_target_mean, base_target_scale,
    )
    raw_features, raw_target = _correction_data(source, source_base_mean, source_base_scale)
    feature_mean = raw_features.mean(0)
    feature_scale = raw_features.std(0).clip(min=1e-4)
    target_mean = float(raw_target.mean())
    target_scale = float(max(raw_target.std(), 0.05))
    features = torch.from_numpy((raw_features - feature_mean) / feature_scale).cuda()
    targets = torch.from_numpy((raw_target - target_mean) / target_scale).cuda()
    model_config = config["model"]
    correction_models = []
    final_losses = {}
    torch.cuda.reset_peak_memory_stats()
    for seed_value in config["member_seeds"]:
        seed_value = int(seed_value)
        torch.manual_seed(seed_value)
        model = ConditionalProjectedGaussian(
            features.shape[1], model_config["hidden_dimensions"],
        ).cuda()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(model_config["learning_rate"]),
            weight_decay=float(model_config["weight_decay"]),
        )
        final_loss = 0.0
        for step in range(int(model_config["steps"])):
            index = torch.randint(
                len(features), (int(model_config["batch_size"]),), device="cuda",
            )
            mean, scale = model(features[index])
            residual = (targets[index] - mean) / scale
            loss = (0.5 * residual.square() + torch.log(scale)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
            if step % 500 == 0 or step + 1 == int(model_config["steps"]):
                print(
                    f"P143 conditional-residual seed={seed_value} step={step + 1} "
                    f"nll={final_loss:.6f}", flush=True,
                )
        final_losses[str(seed_value)] = final_loss
        correction_models.append(model.eval())
    torch.save({
        "feature_mean": feature_mean, "feature_scale": feature_scale,
        "target_mean": target_mean, "target_scale": target_scale,
        "hidden_dimensions": model_config["hidden_dimensions"],
        "member_seeds": [int(x) for x in config["member_seeds"]],
        "member_state_dicts": [model.state_dict() for model in correction_models],
        "frozen_base": config["frozen_p126"],
    }, run_dir / config["model_artifact"])
    floor = float(config["boundary_state_cost"]["clearance_floor_m"])
    coverage = float(config["selection"]["coverage_fraction"])
    results = {}
    for cohort in config["development_cohorts"]:
        arrays = dict(np.load(
            args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False,
        ))
        score, scenes, epistemic_fraction = _trajectory_score(
            arrays, correction_models, feature_mean, feature_scale, target_mean, target_scale,
            base_models, base_feature_mean, base_feature_scale, base_target_mean, base_target_scale,
        )
        actual_cost, cost_scenes = _continuous_cost(arrays, floor)
        if not np.array_equal(scenes, cost_scenes):
            raise RuntimeError("P143 trajectory grouping is not aligned")
        selected = _select_by_scene(score, scenes, coverage)
        reference = config["frozen_p126_comparisons"][cohort["name"]]
        model_spearman = spearman_correlation(actual_cost, score)
        results[cohort["name"]] = {
            "trajectory_count": int(len(actual_cost)),
            "selected_trajectory_count": int(len(selected)),
            "corrected_selected_mean_cost": float(actual_cost[selected].mean()),
            "p126_selected_mean_cost": float(reference["selected_cost"]),
            "corrected_cost_spearman": model_spearman,
            "p126_cost_spearman": float(reference["spearman"]),
            "spearman_gain_over_p126": float(model_spearman - float(reference["spearman"])),
            "mean_correction_epistemic_fraction": epistemic_fraction,
        }
        print(json.dumps({cohort["name"]: results[cohort["name"]]}, indent=2), flush=True)
    gains = [x["spearman_gain_over_p126"] for x in results.values()]
    decisions = {
        "no_selected_cost_regression": all(
            x["corrected_selected_mean_cost"] <= x["p126_selected_mean_cost"]
            for x in results.values()
        ),
        "minimum_mean_spearman_gain": float(np.mean(gains))
        >= float(config["decision"]["minimum_mean_spearman_gain"]),
    }
    verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "training": {"conditional_time_tokens": int(len(features)),
                     "standardized_target_mean": target_mean,
                     "standardized_target_scale": target_scale,
                     "member_final_nll": final_losses},
        "development_evaluations": results, "decision_checks": decisions,
        "mean_spearman_gain": float(np.mean(gains)),
        "resources": {"gpu": torch.cuda.get_device_name(0),
                      "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                      "wall_seconds": time.monotonic() - started},
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    (run_dir / "status.json").write_text(json.dumps({
        "status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict,
                      "decision_checks": decisions}, indent=2), flush=True)


if __name__ == "__main__":
    main()
