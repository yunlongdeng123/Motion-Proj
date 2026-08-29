"""Train a P126-anchored trajectory-set rank compiler on continuous boundary-state cost."""

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

from motion_proj.worldsim_v67.actor_state_reliability import spearman_correlation
from scripts.run_worldsim_v67_p86_direct_trajectory_reliability import _select_by_scene
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import (
    DirectionalActorGaussian, _actor_entries, _predict as _predict_actor,
)
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost


class TrajectorySetResidual(nn.Module):
    def __init__(
        self, feature_count: int, element_dimensions: list[int],
        decoder_dimensions: list[int], residual_bound: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = feature_count
        for hidden in element_dimensions:
            layers.extend((nn.Linear(width, int(hidden)), nn.SiLU()))
            width = int(hidden)
        self.element_encoder = nn.Sequential(*layers)
        decoder: list[nn.Module] = []
        decoder_width = width * 2
        for hidden in decoder_dimensions:
            decoder.extend((nn.Linear(decoder_width, int(hidden)), nn.SiLU()))
            decoder_width = int(hidden)
        decoder.append(nn.Linear(decoder_width, 1))
        self.decoder = nn.Sequential(*decoder)
        self.residual_bound = float(residual_bound)

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor, base_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.element_encoder(features)
        expanded = mask.unsqueeze(-1)
        mean = (encoded * expanded).sum(1) / expanded.sum(1).clamp(min=1)
        maximum = encoded.masked_fill(~expanded, -torch.inf).max(1).values
        residual = self.residual_bound * torch.tanh(
            self.decoder(torch.cat((mean, maximum), dim=1)).reshape(-1)
        )
        return base_score + residual, residual


def _p126_row_score(
    arrays: dict[str, np.ndarray], models: list[DirectionalActorGaussian],
    feature_mean: np.ndarray, feature_scale: np.ndarray,
    target_mean: np.ndarray, target_scale: np.ndarray,
) -> np.ndarray:
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
    predicted = np.asarray(arrays["predicted_separation_profile_m"], dtype=np.float32)
    radius = np.asarray(arrays["occupancy_interaction_radius_m"], dtype=np.float32)[:, None]
    signed = predicted - radius
    return np.max(-(
        np.abs(signed) + np.sign(signed) * projected_mean
    ) / np.sqrt(total_variance), axis=1).astype(np.float32)


def _row_features(arrays: dict[str, np.ndarray], row_score: np.ndarray) -> np.ndarray:
    predicted = np.asarray(arrays["predicted_separation_profile_m"], dtype=np.float32)
    radius = np.asarray(arrays["occupancy_interaction_radius_m"], dtype=np.float32)[:, None]
    signed = predicted - radius
    normal = np.asarray(arrays["query_boundary_normal_ego_profile"], dtype=np.float32)
    return np.concatenate((
        np.asarray(arrays["features"], dtype=np.float32),
        signed,
        normal.reshape(len(normal), -1),
        row_score[:, None],
    ), axis=1)


def _build_sets(
    arrays: dict[str, np.ndarray], row_score: np.ndarray, maximum_rows: int,
) -> dict[str, np.ndarray]:
    tokens = _row_features(arrays, row_score)
    keys = np.stack((
        arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["query_id"],
    ), axis=1)
    identities, inverse = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_inverse)) + 1]
    ends = np.r_[starts[1:], len(order)]
    sets = np.zeros((len(identities), maximum_rows, tokens.shape[1]), dtype=np.float32)
    mask = np.zeros((len(identities), maximum_rows), dtype=bool)
    base = np.zeros(len(identities), dtype=np.float32)
    for group, (start, end) in enumerate(zip(starts, ends)):
        members = order[start:end]
        chosen = members[np.argsort(-row_score[members], kind="stable")[:maximum_rows]]
        sets[group, :len(chosen)] = tokens[chosen]
        mask[group, :len(chosen)] = True
        base[group] = float(np.max(row_score[members]))
    return {
        "sets": sets, "mask": mask, "base_score": base,
        "scene_index": identities[:, 0].astype(np.int32), "identities": identities,
    }


def _scene_table(scene_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, inverse = np.unique(scene_ids, return_inverse=True)
    counts = np.bincount(inverse)
    table = np.zeros((len(counts), int(counts.max())), dtype=np.int64)
    order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.cumsum(counts)]
    for scene in range(len(counts)):
        selected = order[starts[scene]:starts[scene + 1]]
        table[scene, :len(selected)] = selected
    return table, counts.astype(np.int64)


@torch.no_grad()
def _predict(
    model: TrajectorySetResidual, features: torch.Tensor,
    mask: torch.Tensor, base: torch.Tensor,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(features), 4096):
        score, _ = model(
            features[start:start + 4096], mask[start:start + 4096],
            base[start:start + 4096],
        )
        outputs.append(score.cpu().numpy())
    return np.concatenate(outputs)


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
    feature_mean = np.asarray(frozen["feature_mean"], dtype=np.float32)
    feature_scale = np.asarray(frozen["feature_scale"], dtype=np.float32)
    target_mean = np.asarray(frozen["target_mean"], dtype=np.float32)
    target_scale = np.asarray(frozen["target_scale"], dtype=np.float32)
    base_models = []
    for state_dict in frozen["member_state_dicts"]:
        member = DirectionalActorGaussian(20, frozen["hidden_dimensions"]).cuda()
        member.load_state_dict(state_dict)
        base_models.append(member.eval())
    source = dict(np.load(
        args.runs_root / config["source_rows"]["run"] / config["source_rows"]["artifact"],
        allow_pickle=False,
    ))
    source_row_score = _p126_row_score(
        source, base_models, feature_mean, feature_scale, target_mean, target_scale,
    )
    model_config = config["model"]
    maximum_rows = int(model_config["maximum_actor_query_rows"])
    source_sets = _build_sets(source, source_row_score, maximum_rows)
    source_cost, source_scenes = _continuous_cost(
        source, float(config["boundary_state_cost"]["clearance_floor_m"]),
    )
    if not np.array_equal(source_scenes, source_sets["scene_index"]):
        raise RuntimeError("P144 source grouping is not aligned")
    all_row_features = _row_features(source, source_row_score)
    token_mean = all_row_features.mean(0)
    token_scale = all_row_features.std(0).clip(min=1e-4)
    normalized = (source_sets["sets"] - token_mean) / token_scale
    normalized[~source_sets["mask"]] = 0.0
    features = torch.from_numpy(normalized).cuda()
    mask = torch.from_numpy(source_sets["mask"]).cuda()
    base = torch.from_numpy(source_sets["base_score"]).cuda()
    costs = torch.from_numpy(source_cost).cuda()
    scene_table_np, scene_counts_np = _scene_table(source_scenes)
    scene_table = torch.from_numpy(scene_table_np).cuda()
    scene_counts = torch.from_numpy(scene_counts_np).cuda()
    torch.manual_seed(int(config["seed"]))
    model = TrajectorySetResidual(
        features.shape[2], model_config["element_dimensions"],
        model_config["decoder_dimensions"], float(model_config["residual_bound"]),
    ).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    steps = int(model_config["steps"])
    batch_size = int(model_config["pair_batch_size"])
    penalty_weight = float(model_config["residual_regularization"])
    final_loss = 0.0
    torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        scene = torch.randint(len(scene_table), (batch_size,), device="cuda")
        first_pos = torch.floor(torch.rand(batch_size, device="cuda") * scene_counts[scene]).long()
        second_pos = torch.floor(torch.rand(batch_size, device="cuda") * scene_counts[scene]).long()
        first = scene_table[scene, first_pos]
        second = scene_table[scene, second_pos]
        first_score, first_residual = model(features[first], mask[first], base[first])
        second_score, second_residual = model(features[second], mask[second], base[second])
        direction = torch.where(costs[first] >= costs[second], 1.0, -1.0)
        rank_loss = functional.softplus(-direction * (first_score - second_score)).mean()
        penalty = first_residual.square().mean() + second_residual.square().mean()
        loss = rank_loss + penalty_weight * penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step % 500 == 0 or step + 1 == steps:
            print(f"P144 trajectory-set step={step + 1} loss={final_loss:.6f}", flush=True)
    torch.save({
        "token_mean": token_mean, "token_scale": token_scale,
        "maximum_actor_query_rows": maximum_rows,
        "element_dimensions": model_config["element_dimensions"],
        "decoder_dimensions": model_config["decoder_dimensions"],
        "residual_bound": float(model_config["residual_bound"]),
        "model_state_dict": model.state_dict(), "frozen_base": config["frozen_p126"],
    }, run_dir / config["model_artifact"])
    coverage = float(config["selection"]["coverage_fraction"])
    floor = float(config["boundary_state_cost"]["clearance_floor_m"])
    results = {}
    for cohort in config["development_cohorts"]:
        arrays = dict(np.load(
            args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False,
        ))
        row_score = _p126_row_score(
            arrays, base_models, feature_mean, feature_scale, target_mean, target_scale,
        )
        grouped = _build_sets(arrays, row_score, maximum_rows)
        cohort_normalized = (grouped["sets"] - token_mean) / token_scale
        cohort_normalized[~grouped["mask"]] = 0.0
        learned = _predict(
            model.eval(), torch.from_numpy(cohort_normalized).cuda(),
            torch.from_numpy(grouped["mask"]).cuda(),
            torch.from_numpy(grouped["base_score"]).cuda(),
        )
        actual_cost, cost_scenes = _continuous_cost(arrays, floor)
        if not np.array_equal(grouped["scene_index"], cost_scenes):
            raise RuntimeError("P144 evaluation grouping is not aligned")
        selected = _select_by_scene(learned, cost_scenes, coverage)
        base_selected = _select_by_scene(grouped["base_score"], cost_scenes, coverage)
        reference = config["frozen_p126_comparisons"][cohort["name"]]
        learned_spearman = spearman_correlation(actual_cost, learned)
        results[cohort["name"]] = {
            "trajectory_count": int(len(actual_cost)),
            "selected_trajectory_count": int(len(selected)),
            "set_compiler_selected_mean_cost": float(actual_cost[selected].mean()),
            "recomputed_p126_selected_mean_cost": float(actual_cost[base_selected].mean()),
            "frozen_p126_selected_mean_cost": float(reference["selected_cost"]),
            "set_compiler_cost_spearman": learned_spearman,
            "p126_cost_spearman": float(reference["spearman"]),
            "spearman_gain_over_p126": float(learned_spearman - float(reference["spearman"])),
        }
        print(json.dumps({cohort["name"]: results[cohort["name"]]}, indent=2), flush=True)
    gains = [x["spearman_gain_over_p126"] for x in results.values()]
    decisions = {
        "no_selected_cost_regression": all(
            x["set_compiler_selected_mean_cost"] <= x["frozen_p126_selected_mean_cost"]
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
        "training": {"source_trajectory_count": int(len(source_cost)),
                     "source_actor_query_rows": int(len(source_row_score)),
                     "final_pairwise_residual_loss": final_loss},
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
