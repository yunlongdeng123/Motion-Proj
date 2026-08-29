"""Directly model oriented rectangle support residual along each query normal."""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional
import yaml

from motion_proj.worldsim_v67.actor_state_reliability import ACTOR_FEATURE_NAMES, spearman_correlation
from scripts.run_worldsim_v67_p86_direct_trajectory_reliability import _select_by_scene
from scripts.run_worldsim_v67_p104_temporal_flip_supervision import _aligned_group_max
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian, _actor_entries, _predict
from scripts.run_worldsim_v67_p158_crps_actor_ensemble import _subset
from scripts.run_worldsim_v67_p162_oriented_footprint_actor_ensemble import YawStore, _support, _yaw_entries


class SupportResidualGaussian(nn.Module):
    def __init__(self, input_dimension: int, hidden_dimensions: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = input_dimension
        for hidden in hidden_dimensions:
            layers.extend((nn.Linear(width, int(hidden)), nn.SiLU()))
            width = int(hidden)
        layers.append(nn.Linear(width, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.network(features)
        return output[:, 0], functional.softplus(output[:, 1]) + 0.02


def _support_entries(
    arrays: dict[str, np.ndarray], store: YawStore,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, _, _, predicted_heading, actual_heading = _yaw_entries(arrays, store)
    point_count = predicted_heading.shape[1]
    actor = np.asarray(arrays["features"], dtype=np.float32)[:, :len(ACTOR_FEATURE_NAMES)]
    normal = np.asarray(arrays["query_boundary_normal_ego_profile"], dtype=np.float32)
    length = actor[:, 11, None]
    width = actor[:, 12, None]
    predicted_support, _ = _support(predicted_heading, normal, length, width)
    actual_support, _ = _support(actual_heading, normal, length, width)
    fractions = np.linspace(0.0, 1.0, point_count, dtype=np.float32)
    features = np.concatenate((
        np.broadcast_to(actor[:, None, :], (len(actor), point_count, actor.shape[1])),
        np.broadcast_to(fractions[None, :, None], (len(actor), point_count, 1)),
        normal,
        np.sin(predicted_heading)[..., None], np.cos(predicted_heading)[..., None],
    ), axis=2)
    return features.reshape(-1, features.shape[-1]), (actual_support - predicted_support).reshape(-1), predicted_support, actual_support


@torch.no_grad()
def _predict_support(
    model: SupportResidualGaussian, features: np.ndarray,
    feature_mean: np.ndarray, feature_scale: np.ndarray, target_mean: float, target_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    means, scales = [], []
    for start in range(0, len(features), 131072):
        batch = torch.from_numpy((features[start:start + 131072] - feature_mean) / feature_scale).cuda()
        mean, scale = model(batch)
        means.append(mean.cpu().numpy() * target_scale + target_mean)
        scales.append(scale.cpu().numpy() * target_scale)
    return np.concatenate(means), np.concatenate(scales)


def _evaluate(
    arrays: dict[str, np.ndarray], position_models: list[DirectionalActorGaussian], support_models: list[SupportResidualGaussian],
    position_norm: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    support_norm: tuple[np.ndarray, np.ndarray, float, float], store: YawStore,
    coverage: float, floor: float, ego_half_width: float,
) -> dict[str, float | int]:
    point_count = int(arrays["actor_position_error_vector_ego_profile_m"].shape[1])
    pos_features, _, pos_inverse = _actor_entries(arrays)
    pos_member_mean, pos_member_scale = [], []
    for model in position_models:
        mean, scale = _predict(model, pos_features, *position_norm)
        pos_member_mean.append(mean.reshape(-1, point_count, 2)[pos_inverse])
        pos_member_scale.append(scale.reshape(-1, point_count, 2)[pos_inverse])
    pos_means = np.stack(pos_member_mean)
    pos_scales = np.stack(pos_member_scale)
    normal = np.asarray(arrays["query_boundary_normal_ego_profile"], dtype=np.float32)
    projected_member_mean = np.sum(normal[None] * pos_means, axis=3)
    position_mean = projected_member_mean.mean(axis=0)
    position_variance = np.mean(np.sum(np.square(normal[None] * pos_scales), axis=3), axis=0) + projected_member_mean.var(axis=0)

    support_features, support_target, predicted_support, actual_support = _support_entries(arrays, store)
    member_mean, member_scale = [], []
    for model in support_models:
        mean, scale = _predict_support(model, support_features, *support_norm)
        member_mean.append(mean.reshape(-1, point_count))
        member_scale.append(scale.reshape(-1, point_count))
    means = np.stack(member_mean)
    scales = np.stack(member_scale)
    support_mean = means.mean(axis=0)
    support_variance = np.square(scales).mean(axis=0) + means.var(axis=0)
    predicted_distance = np.asarray(arrays["predicted_separation_profile_m"], dtype=np.float32)
    predicted_clearance = predicted_distance - (ego_half_width + predicted_support)
    candidate_mean = position_mean - support_mean
    candidate_variance = position_variance + support_variance
    candidate_row_score = np.max(-(
        np.abs(predicted_clearance) + np.sign(predicted_clearance) * candidate_mean
    ) / np.sqrt(np.maximum(candidate_variance, 1e-8)), axis=1)
    baseline_row_score = np.max(-(
        np.abs(predicted_clearance) + np.sign(predicted_clearance) * position_mean
    ) / np.sqrt(np.maximum(position_variance, 1e-8)), axis=1)
    position_residual = np.sum(
        normal * np.asarray(arrays["actor_position_error_vector_ego_profile_m"], dtype=np.float32), axis=2,
    )
    boundary_residual = position_residual - support_target.reshape(-1, point_count)
    row_cost = np.max(np.abs(boundary_residual) / np.maximum(np.abs(predicted_clearance), floor), axis=1)
    keys = np.stack((
        arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["query_id"],
    ), axis=1)
    identities, inverse = np.unique(keys, axis=0, return_inverse=True)
    actual_cost = np.zeros(len(identities), dtype=np.float32)
    np.maximum.at(actual_cost, inverse, row_cost)
    candidate_score = _aligned_group_max(keys, candidate_row_score, identities)
    baseline_score = _aligned_group_max(keys, baseline_row_score, identities)
    scenes = identities[:, 0].astype(np.int32)
    candidate_selected = _select_by_scene(candidate_score, scenes, coverage)
    baseline_selected = _select_by_scene(baseline_score, scenes, coverage)
    candidate_rank = spearman_correlation(actual_cost, candidate_score)
    baseline_rank = spearman_correlation(actual_cost, baseline_score)
    return {
        "row_count": int(len(arrays["features"])), "trajectory_count": int(len(actual_cost)),
        "selected_trajectory_count": int(len(candidate_selected)),
        "direct_support_selected_mean_cost": float(actual_cost[candidate_selected].mean()),
        "position_only_selected_mean_cost": float(actual_cost[baseline_selected].mean()),
        "support_minus_position_selected_cost": float(actual_cost[candidate_selected].mean() - actual_cost[baseline_selected].mean()),
        "direct_support_cost_spearman": candidate_rank, "position_only_cost_spearman": baseline_rank,
        "spearman_gain_over_position_only": float(candidate_rank - baseline_rank),
        "mean_absolute_support_residual_m": float(np.mean(np.abs(actual_support - predicted_support))),
        "mean_support_standard_deviation_m": float(np.mean(np.sqrt(np.maximum(support_variance, 1e-8)))),
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
    store = YawStore([Path(value) for value in config["processed_roots"]])
    source = dict(np.load(args.runs_root / config["source_rows"]["run"] / config["source_rows"]["artifact"], allow_pickle=False))
    raw_features, raw_target, _, _ = _support_entries(source, store)
    feature_mean = raw_features.mean(0)
    feature_scale = raw_features.std(0).clip(min=1e-4)
    target_mean = float(raw_target.mean())
    target_scale = float(max(raw_target.std(), 0.02))
    features = torch.from_numpy((raw_features - feature_mean) / feature_scale).cuda()
    targets = torch.from_numpy((raw_target - target_mean) / target_scale).cuda()
    model_config = config["model"]
    support_models = []
    final_losses = {}
    torch.cuda.reset_peak_memory_stats()
    for seed_value in config["member_seeds"]:
        seed_value = int(seed_value)
        torch.manual_seed(seed_value)
        model = SupportResidualGaussian(features.shape[1], model_config["hidden_dimensions"]).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"]))
        final_loss = 0.0
        for step in range(int(model_config["steps_per_member"])):
            index = torch.randint(len(features), (int(model_config["batch_size"]),), device="cuda")
            mean, scale = model(features[index])
            residual = (targets[index] - mean) / scale
            loss = (0.5 * residual.square() + torch.log(scale)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
            if step % 1000 == 0 or step + 1 == int(model_config["steps_per_member"]):
                print(f"P163 support seed={seed_value} step={step + 1} nll={final_loss:.6f}", flush=True)
        final_losses[str(seed_value)] = final_loss
        support_models.append(model.eval())
    torch.save({
        "feature_mean": feature_mean, "feature_scale": feature_scale, "target_mean": target_mean, "target_scale": target_scale,
        "hidden_dimensions": model_config["hidden_dimensions"], "member_seeds": config["member_seeds"],
        "member_state_dicts": [model.state_dict() for model in support_models],
    }, run_dir / config["model_artifact"])
    frozen = torch.load(args.runs_root / config["frozen_p126"]["run"] / config["frozen_p126"]["artifact"], map_location="cuda")
    position_models = []
    for state in frozen["member_state_dicts"]:
        model = DirectionalActorGaussian(20, frozen["hidden_dimensions"]).cuda()
        model.load_state_dict(state)
        position_models.append(model.eval())
    position_norm = tuple(np.asarray(frozen[name], dtype=np.float32) for name in ("feature_mean", "feature_scale", "target_mean", "target_scale"))
    support_norm = (feature_mean, feature_scale, target_mean, target_scale)
    coverage = float(config["selection"]["coverage_fraction"])
    floor = float(config["boundary_state_cost"]["clearance_floor_m"])
    ego_half_width = float(config["ego_half_width_m"])
    decision_results = {}
    for cohort in config["decision_cohorts"]:
        arrays = dict(np.load(args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False))
        decision_results[cohort["name"]] = _evaluate(
            arrays, position_models, support_models, position_norm, support_norm, store, coverage, floor, ego_half_width,
        )
        print(json.dumps({cohort["name"]: decision_results[cohort["name"]]}, indent=2), flush=True)
    diagnostic_spec = config["post_confirmation_diagnostic"]
    diagnostic_arrays = dict(np.load(args.runs_root / diagnostic_spec["run"] / diagnostic_spec["artifact"], allow_pickle=False))
    diagnostic_results = {}
    for horizon in diagnostic_spec["horizons_seconds"]:
        key = str(float(horizon))
        diagnostic_results[key] = _evaluate(
            _subset(diagnostic_arrays, float(horizon)), position_models, support_models,
            position_norm, support_norm, store, coverage, floor, ego_half_width,
        )
        print(json.dumps({f"P147_H{key}": diagnostic_results[key]}, indent=2), flush=True)
    gains = [row["spearman_gain_over_position_only"] for row in decision_results.values()]
    decisions = {
        "no_selected_cost_regression": all(row["direct_support_selected_mean_cost"] <= row["position_only_selected_mean_cost"] for row in decision_results.values()),
        "minimum_mean_spearman_gain": float(np.mean(gains)) >= float(config["decision"]["minimum_mean_spearman_gain_over_position_only"]),
    }
    verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict, "role": config["role"],
        "training": {"query_time_tokens": int(len(features)), "member_final_nll": final_losses},
        "consumed_development_evaluations": decision_results,
        "post_confirmation_consumed_p147_diagnostic": diagnostic_results,
        "decision_checks": decisions, "mean_spearman_gain": float(np.mean(gains)),
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
