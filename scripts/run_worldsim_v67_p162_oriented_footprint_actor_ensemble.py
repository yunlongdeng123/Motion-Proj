"""Extend Actor reliability from position to linearized oriented-footprint boundary state."""

from __future__ import annotations

import argparse
import json
import math
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


def _wrap(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


class YawResidualGaussian(nn.Module):
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


class YawStore:
    def __init__(self, roots: list[Path]) -> None:
        self.roots = roots
        self.cache: dict[int, dict[int, dict[int, float]]] = {}

    def scene(self, scene: int) -> dict[int, dict[int, float]]:
        if scene not in self.cache:
            candidates = [root / f"{scene:03d}" / "instances" / "instances_info.json" for root in self.roots]
            path = next((candidate for candidate in candidates if candidate.is_file()), candidates[-1])
            raw = json.loads(path.read_text(encoding="utf-8"))
            tables = {}
            for actor_id, actor in raw.items():
                annotation = actor["frame_annotations"]
                tables[int(actor_id)] = {
                    int(frame): float(math.atan2(matrix[1][0], matrix[0][0]))
                    for frame, matrix in zip(annotation["frame_idx"], annotation["obj_to_world"])
                }
            self.cache[scene] = tables
        return self.cache[scene]


def _yaw_entries(
    arrays: dict[str, np.ndarray], store: YawStore,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keys = np.stack((
        arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["actor_id"],
    ), axis=1)
    unique, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    actor = np.asarray(arrays["features"], dtype=np.float32)[first, :len(ACTOR_FEATURE_NAMES)]
    point_count = int(arrays["actor_position_error_vector_ego_profile_m"].shape[1])
    fractions = np.linspace(0.0, 1.0, point_count, dtype=np.float32)
    target = np.zeros((len(unique), point_count), dtype=np.float32)
    predicted_relative = np.zeros_like(target)
    actual_relative = np.zeros_like(target)
    for index, (scene, horizon_tenths, anchor, actor_id) in enumerate(unique):
        horizon = float(horizon_tenths) / 10.0
        offsets = np.rint(np.linspace(0, int(horizon_tenths), point_count)).astype(np.int32)
        table = store.scene(int(scene))[int(actor_id)]
        current_yaw = table[int(anchor)]
        actual_yaw = np.asarray([table[int(anchor + offset)] for offset in offsets], dtype=np.float32)
        heading_delta = math.atan2(float(actor[index, 5]), float(actor[index, 4]))
        times = fractions * horizon
        predicted_relative[index] = heading_delta + float(actor[index, 3]) * times
        actual_relative[index] = heading_delta + _wrap(actual_yaw - current_yaw)
        target[index] = _wrap(actual_relative[index] - predicted_relative[index])
    features = np.concatenate((
        np.broadcast_to(actor[:, None, :], (len(actor), point_count, actor.shape[1])),
        np.broadcast_to(fractions[None, :, None], (len(actor), point_count, 1)),
    ), axis=2)
    return (
        features.reshape(-1, features.shape[-1]), target.reshape(-1), inverse,
        predicted_relative[inverse], actual_relative[inverse],
    )


@torch.no_grad()
def _predict_yaw(
    model: YawResidualGaussian, features: np.ndarray,
    feature_mean: np.ndarray, feature_scale: np.ndarray, target_mean: float, target_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    means, scales = [], []
    for start in range(0, len(features), 65536):
        batch = torch.from_numpy((features[start:start + 65536] - feature_mean) / feature_scale).cuda()
        mean, scale = model(batch)
        means.append(mean.cpu().numpy() * target_scale + target_mean)
        scales.append(scale.cpu().numpy() * target_scale)
    return np.concatenate(means), np.concatenate(scales)


def _support(
    heading: np.ndarray, normal: np.ndarray, length: np.ndarray, width: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cosine = normal[..., 0] * np.cos(heading) + normal[..., 1] * np.sin(heading)
    sine = -normal[..., 0] * np.sin(heading) + normal[..., 1] * np.cos(heading)
    value = 0.5 * (length * np.abs(cosine) + width * np.abs(sine))
    derivative = 0.5 * (length * np.sign(cosine) * sine - width * np.sign(sine) * cosine)
    return value, derivative


def _evaluate(
    arrays: dict[str, np.ndarray], position_models: list[DirectionalActorGaussian], yaw_models: list[YawResidualGaussian],
    position_norm: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    yaw_norm: tuple[np.ndarray, np.ndarray, float, float], store: YawStore,
    coverage: float, floor: float, ego_half_width: float,
) -> dict[str, float | int]:
    pos_features, _, pos_inverse = _actor_entries(arrays)
    point_count = int(arrays["actor_position_error_vector_ego_profile_m"].shape[1])
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

    yaw_features, _, yaw_inverse, predicted_heading, actual_heading = _yaw_entries(arrays, store)
    yaw_member_mean, yaw_member_scale = [], []
    for model in yaw_models:
        mean, scale = _predict_yaw(model, yaw_features, *yaw_norm)
        yaw_member_mean.append(mean.reshape(-1, point_count)[yaw_inverse])
        yaw_member_scale.append(scale.reshape(-1, point_count)[yaw_inverse])
    yaw_means = np.stack(yaw_member_mean)
    yaw_scales = np.stack(yaw_member_scale)
    yaw_mean = yaw_means.mean(axis=0)
    yaw_variance = np.square(yaw_scales).mean(axis=0) + yaw_means.var(axis=0)

    length = np.asarray(arrays["features"][:, 11], dtype=np.float32)[:, None]
    width = np.asarray(arrays["features"][:, 12], dtype=np.float32)[:, None]
    predicted_support, support_derivative = _support(predicted_heading, normal, length, width)
    actual_support, _ = _support(actual_heading, normal, length, width)
    predicted_distance = np.asarray(arrays["predicted_separation_profile_m"], dtype=np.float32)
    predicted_clearance = predicted_distance - (ego_half_width + predicted_support)
    candidate_mean = position_mean - support_derivative * yaw_mean
    candidate_variance = position_variance + np.square(support_derivative) * yaw_variance
    candidate_row_score = np.max(-(
        np.abs(predicted_clearance) + np.sign(predicted_clearance) * candidate_mean
    ) / np.sqrt(np.maximum(candidate_variance, 1e-8)), axis=1)
    baseline_row_score = np.max(-(
        np.abs(predicted_clearance) + np.sign(predicted_clearance) * position_mean
    ) / np.sqrt(np.maximum(position_variance, 1e-8)), axis=1)

    position_residual = np.sum(
        normal * np.asarray(arrays["actor_position_error_vector_ego_profile_m"], dtype=np.float32), axis=2,
    )
    boundary_residual = position_residual - (actual_support - predicted_support)
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
        "oriented_yaw_selected_mean_cost": float(actual_cost[candidate_selected].mean()),
        "position_only_selected_mean_cost": float(actual_cost[baseline_selected].mean()),
        "yaw_minus_position_selected_cost": float(actual_cost[candidate_selected].mean() - actual_cost[baseline_selected].mean()),
        "oriented_yaw_cost_spearman": candidate_rank, "position_only_cost_spearman": baseline_rank,
        "spearman_gain_over_position_only": float(candidate_rank - baseline_rank),
        "mean_absolute_yaw_residual_rad": float(np.mean(np.abs(_wrap(actual_heading - predicted_heading)))),
        "mean_yaw_standard_deviation_rad": float(np.mean(np.sqrt(np.maximum(yaw_variance, 1e-8)))),
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
    raw_features, raw_target, _, _, _ = _yaw_entries(source, store)
    feature_mean = raw_features.mean(0)
    feature_scale = raw_features.std(0).clip(min=1e-4)
    target_mean = float(raw_target.mean())
    target_scale = float(max(raw_target.std(), 0.02))
    features = torch.from_numpy((raw_features - feature_mean) / feature_scale).cuda()
    targets = torch.from_numpy((raw_target - target_mean) / target_scale).cuda()
    model_config = config["model"]
    yaw_models = []
    final_losses = {}
    torch.cuda.reset_peak_memory_stats()
    for seed_value in config["member_seeds"]:
        seed_value = int(seed_value)
        torch.manual_seed(seed_value)
        model = YawResidualGaussian(features.shape[1], model_config["hidden_dimensions"]).cuda()
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
                print(f"P162 yaw seed={seed_value} step={step + 1} nll={final_loss:.6f}", flush=True)
        final_losses[str(seed_value)] = final_loss
        yaw_models.append(model.eval())
    torch.save({
        "feature_mean": feature_mean, "feature_scale": feature_scale, "target_mean": target_mean, "target_scale": target_scale,
        "hidden_dimensions": model_config["hidden_dimensions"], "member_seeds": config["member_seeds"],
        "member_state_dicts": [model.state_dict() for model in yaw_models],
    }, run_dir / config["model_artifact"])
    frozen = torch.load(args.runs_root / config["frozen_p126"]["run"] / config["frozen_p126"]["artifact"], map_location="cuda")
    position_models = []
    for state in frozen["member_state_dicts"]:
        model = DirectionalActorGaussian(20, frozen["hidden_dimensions"]).cuda()
        model.load_state_dict(state)
        position_models.append(model.eval())
    position_norm = tuple(np.asarray(frozen[name], dtype=np.float32) for name in ("feature_mean", "feature_scale", "target_mean", "target_scale"))
    yaw_norm = (feature_mean, feature_scale, target_mean, target_scale)
    coverage = float(config["selection"]["coverage_fraction"])
    floor = float(config["boundary_state_cost"]["clearance_floor_m"])
    ego_half_width = float(config["ego_half_width_m"])
    decision_results = {}
    for cohort in config["decision_cohorts"]:
        arrays = dict(np.load(args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False))
        decision_results[cohort["name"]] = _evaluate(
            arrays, position_models, yaw_models, position_norm, yaw_norm, store, coverage, floor, ego_half_width,
        )
        print(json.dumps({cohort["name"]: decision_results[cohort["name"]]}, indent=2), flush=True)
    diagnostic_spec = config["post_confirmation_diagnostic"]
    diagnostic_arrays = dict(np.load(args.runs_root / diagnostic_spec["run"] / diagnostic_spec["artifact"], allow_pickle=False))
    diagnostic_results = {}
    for horizon in diagnostic_spec["horizons_seconds"]:
        key = str(float(horizon))
        diagnostic_results[key] = _evaluate(
            _subset(diagnostic_arrays, float(horizon)), position_models, yaw_models,
            position_norm, yaw_norm, store, coverage, floor, ego_half_width,
        )
        print(json.dumps({f"P147_H{key}": diagnostic_results[key]}, indent=2), flush=True)
    gains = [row["spearman_gain_over_position_only"] for row in decision_results.values()]
    decisions = {
        "no_selected_cost_regression": all(row["oriented_yaw_selected_mean_cost"] <= row["position_only_selected_mean_cost"] for row in decision_results.values()),
        "minimum_mean_spearman_gain": float(np.mean(gains)) >= float(config["decision"]["minimum_mean_spearman_gain_over_position_only"]),
    }
    verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict, "role": config["role"],
        "training": {"actor_time_tokens": int(len(features)), "member_final_nll": final_losses},
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
