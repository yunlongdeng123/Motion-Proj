"""Learn joint multi-Actor residual dependence around frozen P126 marginals."""

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
import yaml

from motion_proj.worldsim_v67.actor_state_reliability import ACTOR_FEATURE_NAMES, spearman_correlation
from scripts.run_worldsim_v67_p86_direct_trajectory_reliability import _select_by_scene
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import (
    DirectionalActorGaussian, _actor_entries, _predict,
)
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost
from scripts.run_worldsim_v67_p127_ensemble_continuous_selection import _ensemble_trajectory_score
from scripts.run_worldsim_v67_p158_crps_actor_ensemble import _subset


class JointActorDiffusion(nn.Module):
    def __init__(
        self, state_dimension: int, condition_dimension: int, hidden_dimension: int,
        layer_count: int, head_count: int, diffusion_steps: int,
    ) -> None:
        super().__init__()
        self.state_encoder = nn.Linear(state_dimension, hidden_dimension)
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dimension, hidden_dimension), nn.SiLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
        )
        self.time_embedding = nn.Embedding(diffusion_steps, hidden_dimension)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dimension, nhead=head_count,
            dim_feedforward=hidden_dimension * 2, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layer_count)
        self.output = nn.Sequential(nn.LayerNorm(hidden_dimension), nn.Linear(hidden_dimension, state_dimension))

    def forward(
        self, noisy_state: torch.Tensor, condition: torch.Tensor,
        mask: torch.Tensor, timestep: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.state_encoder(noisy_state) + self.condition_encoder(condition)
        hidden = hidden + self.time_embedding(timestep)[:, None, :]
        hidden = self.transformer(hidden, src_key_padding_mask=~mask)
        return self.output(hidden) * mask[..., None].to(hidden.dtype)


def _base_actor_distribution(
    arrays: dict[str, np.ndarray], models: list[DirectionalActorGaussian],
    feature_mean: np.ndarray, feature_scale: np.ndarray,
    target_mean: np.ndarray, target_scale: np.ndarray,
) -> dict[str, np.ndarray]:
    actor_features, target, row_to_actor = _actor_entries(arrays)
    point_count = int(arrays["actor_position_error_vector_ego_profile_m"].shape[1])
    actor_count = len(actor_features) // point_count
    member_means, member_scales = [], []
    for model in models:
        mean, scale = _predict(
            model.eval(), actor_features, feature_mean, feature_scale, target_mean, target_scale,
        )
        member_means.append(mean.reshape(actor_count, point_count, 2))
        member_scales.append(scale.reshape(actor_count, point_count, 2))
    means = np.stack(member_means)
    scales = np.stack(member_scales)
    base_mean = means.mean(axis=0)
    base_variance = np.mean(np.square(scales) + np.square(means), axis=0) - np.square(base_mean)
    base_scale = np.sqrt(np.maximum(base_variance, 1e-6))
    keys = np.stack((
        arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["actor_id"],
    ), axis=1)
    _, first, _ = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    actor_condition = np.asarray(arrays["features"], dtype=np.float32)[first, :len(ACTOR_FEATURE_NAMES)]
    actor_keys = keys[first]
    return {
        "condition": actor_condition,
        "target": target.reshape(actor_count, point_count, 2),
        "base_mean": base_mean, "base_scale": base_scale,
        "actor_keys": actor_keys, "row_to_actor": row_to_actor,
    }


def _pack_groups(actor: dict[str, np.ndarray], maximum_actors: int) -> dict[str, np.ndarray]:
    group_keys, actor_to_group = np.unique(actor["actor_keys"][:, :3], axis=0, return_inverse=True)
    counts = np.bincount(actor_to_group)
    if int(counts.max()) > maximum_actors:
        raise RuntimeError(f"P165 maximum group size {int(counts.max())} exceeds {maximum_actors}")
    group_count = len(group_keys)
    point_count = actor["target"].shape[1]
    condition = np.zeros((group_count, maximum_actors, actor["condition"].shape[1]), dtype=np.float32)
    target = np.zeros((group_count, maximum_actors, point_count * 2), dtype=np.float32)
    base_mean = np.zeros_like(target)
    base_scale = np.ones_like(target)
    mask = np.zeros((group_count, maximum_actors), dtype=bool)
    actor_slot = np.empty(len(actor_to_group), dtype=np.int32)
    for group_index in range(group_count):
        indices = np.flatnonzero(actor_to_group == group_index)
        slots = np.arange(len(indices), dtype=np.int32)
        actor_slot[indices] = slots
        condition[group_index, slots] = actor["condition"][indices]
        target[group_index, slots] = actor["target"][indices].reshape(len(indices), -1)
        base_mean[group_index, slots] = actor["base_mean"][indices].reshape(len(indices), -1)
        base_scale[group_index, slots] = actor["base_scale"][indices].reshape(len(indices), -1)
        mask[group_index, slots] = True
    innovation = (target - base_mean) / np.maximum(base_scale, 1e-4)
    innovation *= mask[..., None]
    return {
        "group_keys": group_keys, "condition": condition, "innovation": innovation,
        "base_mean": base_mean, "base_scale": base_scale, "mask": mask,
        "actor_to_group": actor_to_group.astype(np.int32), "actor_slot": actor_slot,
    }


@torch.no_grad()
def _sample_joint_innovation(
    model: JointActorDiffusion, condition: np.ndarray, mask: np.ndarray,
    condition_mean: np.ndarray, condition_scale: np.ndarray,
    alpha_bar: torch.Tensor, sample_count: int, inference_steps: int,
    group_batch_size: int, seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    diffusion_steps = len(alpha_bar)
    timesteps = np.unique(np.rint(np.linspace(diffusion_steps - 1, 0, inference_steps)).astype(np.int64))[::-1]
    outputs = []
    state_dimension = model.state_encoder.in_features
    for start in range(0, len(condition), group_batch_size):
        end = min(start + group_batch_size, len(condition))
        batch_condition = torch.from_numpy((condition[start:end] - condition_mean) / condition_scale).cuda()
        batch_mask = torch.from_numpy(mask[start:end]).cuda()
        batch_condition = batch_condition.repeat_interleave(sample_count, dim=0)
        batch_mask = batch_mask.repeat_interleave(sample_count, dim=0)
        state = torch.randn(
            (len(batch_condition), condition.shape[1], state_dimension), device="cuda",
        ) * batch_mask[..., None]
        for index, timestep_value in enumerate(timesteps):
            timestep = torch.full((len(state),), int(timestep_value), dtype=torch.long, device="cuda")
            predicted_noise = model(state, batch_condition, batch_mask, timestep)
            current_alpha = alpha_bar[int(timestep_value)]
            predicted_clean = (
                state - torch.sqrt(1.0 - current_alpha) * predicted_noise
            ) / torch.sqrt(current_alpha)
            if index + 1 == len(timesteps):
                state = predicted_clean
            else:
                next_alpha = alpha_bar[int(timesteps[index + 1])]
                state = torch.sqrt(next_alpha) * predicted_clean + torch.sqrt(1.0 - next_alpha) * predicted_noise
            state *= batch_mask[..., None]
        state = state.reshape(end - start, sample_count, condition.shape[1], state_dimension)
        outputs.append(state.permute(1, 0, 2, 3).cpu().numpy())
    return np.concatenate(outputs, axis=1)


def _joint_trajectory_score(
    arrays: dict[str, np.ndarray], actor: dict[str, np.ndarray], packed: dict[str, np.ndarray],
    model: JointActorDiffusion, condition_norm: tuple[np.ndarray, np.ndarray],
    alpha_bar: torch.Tensor, config: dict,
) -> tuple[np.ndarray, np.ndarray, float]:
    sampling = config["sampling"]
    innovation = _sample_joint_innovation(
        model, packed["condition"], packed["mask"], *condition_norm, alpha_bar,
        int(sampling["sample_count"]), int(sampling["inference_steps"]),
        int(sampling["group_batch_size"]), int(sampling["seed"]),
    )
    residual = packed["base_mean"][None] + packed["base_scale"][None] * innovation
    residual = residual.reshape(*residual.shape[:3], -1, 2)
    row_actor = actor["row_to_actor"]
    row_residual = residual[
        :, packed["actor_to_group"][row_actor], packed["actor_slot"][row_actor], :, :,
    ]
    normal = np.asarray(arrays["query_boundary_normal_ego_profile"], dtype=np.float32)
    projected_error = np.abs(np.sum(normal[None] * row_residual, axis=3))
    signed = np.asarray(arrays["predicted_separation_profile_m"], dtype=np.float32) - np.asarray(
        arrays["occupancy_interaction_radius_m"], dtype=np.float32,
    )[:, None]
    floor = float(config["boundary_state_cost"]["clearance_floor_m"])
    row_cost = np.max(projected_error / np.maximum(np.abs(signed), floor)[None], axis=2)
    keys = np.stack((
        arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["query_id"],
    ), axis=1)
    identities, inverse = np.unique(keys, axis=0, return_inverse=True)
    trajectory_samples = np.zeros((len(row_cost), len(identities)), dtype=np.float32)
    for sample_index in range(len(row_cost)):
        np.maximum.at(trajectory_samples[sample_index], inverse, row_cost[sample_index])
    quantile = float(sampling["trajectory_cost_quantile"])
    score = np.quantile(trajectory_samples, quantile, axis=0).astype(np.float32)
    return score, identities[:, 0].astype(np.int32), float(np.mean(trajectory_samples))


def _evaluate(
    arrays: dict[str, np.ndarray], base_models: list[DirectionalActorGaussian],
    feature_mean: np.ndarray, feature_scale: np.ndarray,
    target_mean: np.ndarray, target_scale: np.ndarray,
    model: JointActorDiffusion, condition_norm: tuple[np.ndarray, np.ndarray],
    alpha_bar: torch.Tensor, config: dict,
) -> dict[str, float | int]:
    actor = _base_actor_distribution(
        arrays, base_models, feature_mean, feature_scale, target_mean, target_scale,
    )
    packed = _pack_groups(actor, int(config["model"]["maximum_actors_per_group"]))
    joint_score, scenes, mean_sample_cost = _joint_trajectory_score(
        arrays, actor, packed, model, condition_norm, alpha_bar, config,
    )
    p126_score, p126_scenes = _ensemble_trajectory_score(
        arrays, base_models, feature_mean, feature_scale, target_mean, target_scale,
    )
    actual_cost, cost_scenes = _continuous_cost(
        arrays, float(config["boundary_state_cost"]["clearance_floor_m"]),
    )
    if not (np.array_equal(scenes, p126_scenes) and np.array_equal(scenes, cost_scenes)):
        raise RuntimeError("P165 trajectory grouping is not aligned")
    coverage = float(config["selection"]["coverage_fraction"])
    joint_selected = _select_by_scene(joint_score, scenes, coverage)
    p126_selected = _select_by_scene(p126_score, scenes, coverage)
    joint_rank = spearman_correlation(actual_cost, joint_score)
    p126_rank = spearman_correlation(actual_cost, p126_score)
    return {
        "row_count": int(len(arrays["features"])), "actor_group_count": int(len(packed["group_keys"])),
        "trajectory_count": int(len(actual_cost)), "selected_trajectory_count": int(len(joint_selected)),
        "joint_diffusion_selected_mean_cost": float(actual_cost[joint_selected].mean()),
        "p126_selected_mean_cost": float(actual_cost[p126_selected].mean()),
        "joint_minus_p126_selected_cost": float(actual_cost[joint_selected].mean() - actual_cost[p126_selected].mean()),
        "joint_diffusion_cost_spearman": joint_rank, "p126_cost_spearman": p126_rank,
        "spearman_gain_over_p126": float(joint_rank - p126_rank),
        "mean_sampled_trajectory_cost": mean_sample_cost,
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
    base_models = []
    for state in frozen["member_state_dicts"]:
        base = DirectionalActorGaussian(20, frozen["hidden_dimensions"]).cuda()
        base.load_state_dict(state)
        base_models.append(base.eval())

    source = dict(np.load(
        args.runs_root / config["source_rows"]["run"] / config["source_rows"]["artifact"], allow_pickle=False,
    ))
    source_actor = _base_actor_distribution(
        source, base_models, feature_mean, feature_scale, target_mean, target_scale,
    )
    source_packed = _pack_groups(source_actor, int(config["model"]["maximum_actors_per_group"]))
    valid_condition = source_packed["condition"][source_packed["mask"]]
    condition_mean = valid_condition.mean(axis=0)
    condition_scale = valid_condition.std(axis=0).clip(min=1e-4)
    condition_norm = (condition_mean, condition_scale)

    model_config = config["model"]
    diffusion_config = config["diffusion"]
    torch.manual_seed(int(config["seed"]))
    model = JointActorDiffusion(
        state_dimension=source_packed["innovation"].shape[2],
        condition_dimension=source_packed["condition"].shape[2],
        hidden_dimension=int(model_config["hidden_dimension"]),
        layer_count=int(model_config["layer_count"]), head_count=int(model_config["head_count"]),
        diffusion_steps=int(diffusion_config["steps"]),
    ).cuda()
    beta = torch.linspace(
        float(diffusion_config["beta_start"]), float(diffusion_config["beta_end"]),
        int(diffusion_config["steps"]), device="cuda",
    )
    alpha_bar = torch.cumprod(1.0 - beta, dim=0)
    condition = torch.from_numpy((source_packed["condition"] - condition_mean) / condition_scale).cuda()
    innovation = torch.from_numpy(source_packed["innovation"]).cuda()
    mask = torch.from_numpy(source_packed["mask"]).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    final_loss = 0.0
    torch.cuda.reset_peak_memory_stats()
    for step in range(int(model_config["training_steps"])):
        index = torch.randint(len(condition), (int(model_config["batch_size"]),), device="cuda")
        batch_mask = mask[index]
        timestep = torch.randint(len(alpha_bar), (len(index),), device="cuda")
        noise = torch.randn_like(innovation[index]) * batch_mask[..., None]
        current_alpha = alpha_bar[timestep, None, None]
        noisy = torch.sqrt(current_alpha) * innovation[index] + torch.sqrt(1.0 - current_alpha) * noise
        predicted = model(noisy, condition[index], batch_mask, timestep)
        squared = (predicted - noise).square() * batch_mask[..., None]
        loss = squared.sum() / (batch_mask.sum() * innovation.shape[2]).clamp(min=1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step % 1000 == 0 or step + 1 == int(model_config["training_steps"]):
            print(f"P165 joint diffusion step={step + 1} noise_mse={final_loss:.6f}", flush=True)

    torch.save({
        "model_state_dict": model.state_dict(), "condition_mean": condition_mean,
        "condition_scale": condition_scale, "beta": beta.cpu().numpy(),
    }, run_dir / config["model_artifact"])

    results = {}
    for cohort in config["decision_cohorts"]:
        arrays = dict(np.load(args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False))
        results[cohort["name"]] = _evaluate(
            arrays, base_models, feature_mean, feature_scale, target_mean, target_scale,
            model.eval(), condition_norm, alpha_bar, config,
        )
        print(json.dumps({cohort["name"]: results[cohort["name"]]}, indent=2), flush=True)

    diagnostic_spec = config["post_confirmation_diagnostic"]
    diagnostic_arrays = dict(np.load(
        args.runs_root / diagnostic_spec["run"] / diagnostic_spec["artifact"], allow_pickle=False,
    ))
    diagnostic_results = {}
    for horizon in diagnostic_spec["horizons_seconds"]:
        key = str(float(horizon))
        diagnostic_results[key] = _evaluate(
            _subset(diagnostic_arrays, float(horizon)), base_models, feature_mean, feature_scale,
            target_mean, target_scale, model.eval(), condition_norm, alpha_bar, config,
        )
        print(json.dumps({f"P147_H{key}": diagnostic_results[key]}, indent=2), flush=True)

    gains = [row["spearman_gain_over_p126"] for row in results.values()]
    decisions = {
        "no_selected_cost_regression": all(
            row["joint_diffusion_selected_mean_cost"] <= row["p126_selected_mean_cost"] for row in results.values()
        ),
        "minimum_mean_spearman_gain": float(np.mean(gains))
        >= float(config["decision"]["minimum_mean_spearman_gain_over_p126"]),
    }
    verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "training": {
            "actor_group_count": int(len(source_packed["group_keys"])),
            "unique_actor_states": int(len(source_actor["actor_keys"])),
            "final_noise_mse": final_loss,
        },
        "consumed_development_evaluations": results,
        "post_confirmation_consumed_p147_diagnostic": diagnostic_results,
        "decision_checks": decisions, "mean_spearman_gain": float(np.mean(gains)),
        "resources": {
            "gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "wall_seconds": time.monotonic() - started,
        },
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({
        "status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict, "decision_checks": decisions}, indent=2), flush=True)


if __name__ == "__main__":
    main()
