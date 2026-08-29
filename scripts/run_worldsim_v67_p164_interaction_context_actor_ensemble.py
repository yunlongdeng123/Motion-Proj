"""Condition frozen P126 members on nearby Actor interactions through residual adapters."""

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
import yaml

from motion_proj.worldsim_v67.actor_state_reliability import ACTOR_FEATURE_NAMES, spearman_correlation
from scripts.run_worldsim_v67_p86_direct_trajectory_reliability import _select_by_scene
from scripts.run_worldsim_v67_p104_temporal_flip_supervision import _aligned_group_max
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost
from scripts.run_worldsim_v67_p158_crps_actor_ensemble import _subset


def _neighbor_features(actor: np.ndarray, group_keys: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    feature_count = 20
    neighbors = np.zeros((len(actor), count, feature_count), dtype=np.float32)
    mask = np.zeros((len(actor), count), dtype=bool)
    _, group_inverse = np.unique(group_keys, axis=0, return_inverse=True)
    for group_index in range(int(group_inverse.max()) + 1):
        indices = np.flatnonzero(group_inverse == group_index)
        if len(indices) < 2:
            continue
        position = actor[indices, 6:8]
        delta_position = position[None, :, :] - position[:, None, :]
        distance = np.linalg.norm(delta_position, axis=2)
        np.fill_diagonal(distance, np.inf)
        slot_count = min(count, len(indices) - 1)
        local_neighbor = np.argsort(distance, axis=1)[:, :slot_count]
        neighbor_index = indices[local_neighbor]
        target_index = indices[:, None]
        relative_position = actor[neighbor_index, 6:8] - actor[target_index, 6:8]
        relative_velocity = actor[neighbor_index, 8:10] - actor[target_index, 8:10]
        neighbor_distance = np.linalg.norm(relative_position, axis=2, keepdims=True).clip(min=1e-4)
        radial_closing = -np.sum(relative_position * relative_velocity, axis=2, keepdims=True) / neighbor_distance
        encoded = np.concatenate((
            relative_position, relative_velocity, neighbor_distance, radial_closing,
            actor[neighbor_index, 0:4], actor[neighbor_index, 4:6],
            actor[neighbor_index, 11:15], actor[neighbor_index, 15:19],
        ), axis=2)
        neighbors[indices, :slot_count] = encoded
        mask[indices, :slot_count] = True
    return neighbors, mask


def _context_entries(
    arrays: dict[str, np.ndarray], neighbor_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keys = np.stack((
        arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["actor_id"],
    ), axis=1)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    actor = np.asarray(arrays["features"], dtype=np.float32)[first, :len(ACTOR_FEATURE_NAMES)]
    residual = np.asarray(arrays["actor_position_error_vector_ego_profile_m"], dtype=np.float32)[first]
    point_count = residual.shape[1]
    fractions = np.linspace(0.0, 1.0, point_count, dtype=np.float32)
    target_features = np.concatenate((
        np.broadcast_to(actor[:, None, :], (len(actor), point_count, actor.shape[1])),
        np.broadcast_to(fractions[None, :, None], (len(actor), point_count, 1)),
    ), axis=2)
    neighbors, mask = _neighbor_features(actor, keys[first, :3], neighbor_count)
    return target_features.reshape(-1, target_features.shape[-1]), residual.reshape(-1, 2), inverse, neighbors, mask


class InteractionResidualAdapter(nn.Module):
    def __init__(self, target_dimension: int, neighbor_dimension: int, embedding_dimension: int) -> None:
        super().__init__()
        self.target_encoder = nn.Sequential(
            nn.Linear(target_dimension, embedding_dimension), nn.SiLU(),
            nn.Linear(embedding_dimension, embedding_dimension), nn.SiLU(),
        )
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(neighbor_dimension, embedding_dimension), nn.SiLU(),
            nn.Linear(embedding_dimension, embedding_dimension), nn.SiLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(embedding_dimension * 3, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 4),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self, target: torch.Tensor, neighbors: torch.Tensor, mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_embedding = self.target_encoder(target)
        neighbor_embedding = self.neighbor_encoder(neighbors)
        valid = mask[..., None].to(neighbor_embedding.dtype)
        neighbor_embedding = neighbor_embedding * valid
        logits = torch.sum(neighbor_embedding * target_embedding[:, None, :], dim=2) / math.sqrt(target_embedding.shape[1])
        logits = logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=1) * mask.to(logits.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        context = torch.sum(weights[..., None] * neighbor_embedding, dim=1)
        output = self.output(torch.cat((target_embedding, context, target_embedding * context), dim=1))
        return output[:, :2], output[:, 2:].clamp(min=-2.0, max=2.0)


@torch.no_grad()
def _predict_member(
    base: DirectionalActorGaussian, adapter: InteractionResidualAdapter,
    target_features: np.ndarray, neighbors: np.ndarray, mask: np.ndarray,
    target_norm: tuple[np.ndarray, np.ndarray], neighbor_norm: tuple[np.ndarray, np.ndarray],
    output_norm: tuple[np.ndarray, np.ndarray], batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    adapted_means, adapted_scales, base_means, base_scales = [], [], [], []
    feature_mean, feature_scale = target_norm
    neighbor_mean, neighbor_scale = neighbor_norm
    output_mean, output_scale = output_norm
    point_count = len(target_features) // len(neighbors)
    for start in range(0, len(target_features), batch_size):
        end = min(start + batch_size, len(target_features))
        actor_index = np.arange(start, end) // point_count
        target = torch.from_numpy((target_features[start:end] - feature_mean) / feature_scale).cuda()
        neighbor = torch.from_numpy((neighbors[actor_index] - neighbor_mean) / neighbor_scale).cuda()
        valid = torch.from_numpy(mask[actor_index]).cuda()
        base_mean, base_scale = base(target)
        delta_mean, log_scale = adapter(target, neighbor, valid)
        adapted_mean = base_mean + delta_mean
        adapted_scale = base_scale * torch.exp(log_scale)
        adapted_means.append(adapted_mean.cpu().numpy() * output_scale + output_mean)
        adapted_scales.append(adapted_scale.cpu().numpy() * output_scale)
        base_means.append(base_mean.cpu().numpy() * output_scale + output_mean)
        base_scales.append(base_scale.cpu().numpy() * output_scale)
    return tuple(map(np.concatenate, (adapted_means, adapted_scales, base_means, base_scales)))


def _trajectory_score(
    arrays: dict[str, np.ndarray], base_models: list[DirectionalActorGaussian],
    adapters: list[InteractionResidualAdapter], target_norm: tuple[np.ndarray, np.ndarray],
    neighbor_norm: tuple[np.ndarray, np.ndarray], output_norm: tuple[np.ndarray, np.ndarray],
    neighbor_count: int, prediction_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_features, _, inverse, neighbors, mask = _context_entries(arrays, neighbor_count)
    point_count = int(arrays["actor_position_error_vector_ego_profile_m"].shape[1])
    adapted_member_mean, adapted_member_scale = [], []
    base_member_mean, base_member_scale = [], []
    for base, adapter in zip(base_models, adapters):
        adapted_mean, adapted_scale, base_mean, base_scale = _predict_member(
            base, adapter, target_features, neighbors, mask, target_norm, neighbor_norm,
            output_norm, prediction_batch_size,
        )
        adapted_member_mean.append(adapted_mean.reshape(-1, point_count, 2)[inverse])
        adapted_member_scale.append(adapted_scale.reshape(-1, point_count, 2)[inverse])
        base_member_mean.append(base_mean.reshape(-1, point_count, 2)[inverse])
        base_member_scale.append(base_scale.reshape(-1, point_count, 2)[inverse])

    normal = np.asarray(arrays["query_boundary_normal_ego_profile"], dtype=np.float32)
    signed = np.asarray(arrays["predicted_separation_profile_m"], dtype=np.float32) - np.asarray(
        arrays["occupancy_interaction_radius_m"], dtype=np.float32,
    )[:, None]

    def score(means: list[np.ndarray], scales: list[np.ndarray]) -> np.ndarray:
        member_mean = np.stack(means)
        member_scale = np.stack(scales)
        projected_member_mean = np.sum(normal[None] * member_mean, axis=3)
        projected_mean = projected_member_mean.mean(axis=0)
        variance = np.mean(np.sum(np.square(normal[None] * member_scale), axis=3), axis=0)
        variance += projected_member_mean.var(axis=0)
        return np.max(-(
            np.abs(signed) + np.sign(signed) * projected_mean
        ) / np.sqrt(np.maximum(variance, 1e-8)), axis=1)

    adapted_row_score = score(adapted_member_mean, adapted_member_scale)
    base_row_score = score(base_member_mean, base_member_scale)
    keys = np.stack((
        arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["query_id"],
    ), axis=1)
    identities = np.unique(keys, axis=0)
    return (
        _aligned_group_max(keys, adapted_row_score, identities),
        _aligned_group_max(keys, base_row_score, identities),
        identities[:, 0].astype(np.int32),
    )


def _evaluate(
    arrays: dict[str, np.ndarray], base_models: list[DirectionalActorGaussian],
    adapters: list[InteractionResidualAdapter], target_norm: tuple[np.ndarray, np.ndarray],
    neighbor_norm: tuple[np.ndarray, np.ndarray], output_norm: tuple[np.ndarray, np.ndarray],
    config: dict,
) -> dict[str, float | int]:
    interaction_score, p126_score, scenes = _trajectory_score(
        arrays, base_models, adapters, target_norm, neighbor_norm, output_norm,
        int(config["interaction"]["neighbor_count"]), int(config["model"]["prediction_batch_size"]),
    )
    actual_cost, cost_scenes = _continuous_cost(arrays, float(config["boundary_state_cost"]["clearance_floor_m"]))
    if not np.array_equal(scenes, cost_scenes):
        raise RuntimeError("P164 trajectory grouping is not aligned")
    coverage = float(config["selection"]["coverage_fraction"])
    interaction_selected = _select_by_scene(interaction_score, scenes, coverage)
    p126_selected = _select_by_scene(p126_score, scenes, coverage)
    interaction_rank = spearman_correlation(actual_cost, interaction_score)
    p126_rank = spearman_correlation(actual_cost, p126_score)
    return {
        "row_count": int(len(arrays["features"])), "trajectory_count": int(len(actual_cost)),
        "selected_trajectory_count": int(len(interaction_selected)),
        "interaction_selected_mean_cost": float(actual_cost[interaction_selected].mean()),
        "p126_selected_mean_cost": float(actual_cost[p126_selected].mean()),
        "interaction_minus_p126_selected_cost": float(
            actual_cost[interaction_selected].mean() - actual_cost[p126_selected].mean()
        ),
        "interaction_cost_spearman": interaction_rank, "p126_cost_spearman": p126_rank,
        "spearman_gain_over_p126": float(interaction_rank - p126_rank),
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
    target_norm = tuple(np.asarray(frozen[name], dtype=np.float32) for name in ("feature_mean", "feature_scale"))
    output_norm = tuple(np.asarray(frozen[name], dtype=np.float32) for name in ("target_mean", "target_scale"))
    base_models = []
    for state in frozen["member_state_dicts"]:
        model = DirectionalActorGaussian(20, frozen["hidden_dimensions"]).cuda()
        model.load_state_dict(state)
        base_models.append(model.eval())

    source = dict(np.load(
        args.runs_root / config["source_rows"]["run"] / config["source_rows"]["artifact"], allow_pickle=False,
    ))
    target_features_raw, target_raw, _, neighbors_raw, neighbor_mask = _context_entries(
        source, int(config["interaction"]["neighbor_count"]),
    )
    valid_neighbors = neighbors_raw[neighbor_mask]
    neighbor_mean = valid_neighbors.mean(axis=0)
    neighbor_scale = valid_neighbors.std(axis=0).clip(min=1e-4)
    neighbor_norm = (neighbor_mean, neighbor_scale)
    target_features = torch.from_numpy((target_features_raw - target_norm[0]) / target_norm[1]).cuda()
    target = torch.from_numpy((target_raw - output_norm[0]) / output_norm[1]).cuda()
    neighbors = torch.from_numpy((neighbors_raw - neighbor_mean) / neighbor_scale).cuda()
    neighbor_mask_tensor = torch.from_numpy(neighbor_mask).cuda()
    point_count = int(source["actor_position_error_vector_ego_profile_m"].shape[1])

    adapters = []
    final_losses = {}
    model_config = config["model"]
    torch.cuda.reset_peak_memory_stats()
    for member_index, seed_value in enumerate(config["member_seeds"]):
        seed_value = int(seed_value)
        torch.manual_seed(seed_value)
        adapter = InteractionResidualAdapter(
            target_features.shape[1], neighbors.shape[2], int(config["interaction"]["embedding_dimension"]),
        ).cuda()
        optimizer = torch.optim.AdamW(
            adapter.parameters(), lr=float(model_config["learning_rate"]),
            weight_decay=float(model_config["weight_decay"]),
        )
        base_mean_chunks, base_scale_chunks = [], []
        with torch.no_grad():
            for start in range(0, len(target_features), int(model_config["prediction_batch_size"])):
                mean, scale = base_models[member_index](
                    target_features[start:start + int(model_config["prediction_batch_size"])]
                )
                base_mean_chunks.append(mean)
                base_scale_chunks.append(scale)
        base_mean = torch.cat(base_mean_chunks)
        base_scale = torch.cat(base_scale_chunks)
        final_loss = 0.0
        for step in range(int(model_config["steps_per_member"])):
            index = torch.randint(len(target_features), (int(model_config["batch_size"]),), device="cuda")
            actor_index = index // point_count
            delta_mean, log_scale = adapter(
                target_features[index], neighbors[actor_index], neighbor_mask_tensor[actor_index],
            )
            mean = base_mean[index] + delta_mean
            scale = base_scale[index] * torch.exp(log_scale)
            residual = (target[index] - mean) / scale
            loss = (0.5 * residual.square() + torch.log(scale)).sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
            if step % 1000 == 0 or step + 1 == int(model_config["steps_per_member"]):
                print(f"P164 interaction seed={seed_value} step={step + 1} nll={final_loss:.6f}", flush=True)
        final_losses[str(seed_value)] = final_loss
        adapters.append(adapter.eval())
        del base_mean, base_scale, base_mean_chunks, base_scale_chunks

    torch.save({
        "neighbor_mean": neighbor_mean, "neighbor_scale": neighbor_scale,
        "neighbor_count": int(config["interaction"]["neighbor_count"]),
        "embedding_dimension": int(config["interaction"]["embedding_dimension"]),
        "member_seeds": config["member_seeds"],
        "adapter_state_dicts": [adapter.state_dict() for adapter in adapters],
    }, run_dir / config["model_artifact"])

    results = {}
    for cohort in config["decision_cohorts"]:
        arrays = dict(np.load(args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False))
        results[cohort["name"]] = _evaluate(
            arrays, base_models, adapters, target_norm, neighbor_norm, output_norm, config,
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
            _subset(diagnostic_arrays, float(horizon)), base_models, adapters,
            target_norm, neighbor_norm, output_norm, config,
        )
        print(json.dumps({f"P147_H{key}": diagnostic_results[key]}, indent=2), flush=True)

    gains = [row["spearman_gain_over_p126"] for row in results.values()]
    decisions = {
        "no_selected_cost_regression": all(
            row["interaction_selected_mean_cost"] <= row["p126_selected_mean_cost"] for row in results.values()
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
            "actor_time_tokens": int(len(target_features)), "unique_actor_states": int(len(neighbors)),
            "mean_valid_neighbors": float(neighbor_mask.sum(axis=1).mean()), "member_final_nll": final_losses,
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
