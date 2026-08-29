"""Train a coherent mixture distribution over complete Actor residual trajectories."""

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
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost


class CoherentTrajectoryMixture(torch.nn.Module):
    def __init__(
        self, feature_count: int, hidden_dimensions: list[int],
        component_count: int, point_count: int,
    ) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        width = feature_count
        for hidden in hidden_dimensions:
            layers.extend((torch.nn.Linear(width, int(hidden)), torch.nn.SiLU()))
            width = int(hidden)
        self.backbone = torch.nn.Sequential(*layers)
        self.logit_head = torch.nn.Linear(width, component_count)
        self.distribution_head = torch.nn.Linear(width, component_count * point_count * 4)
        self.component_count = component_count
        self.point_count = point_count

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(features)
        output = self.distribution_head(hidden).reshape(
            -1, self.component_count, self.point_count, 4,
        )
        return (
            self.logit_head(hidden), output[:, :, :, :2],
            torch.nn.functional.softplus(output[:, :, :, 2:]) + 0.02,
        )


def _actor_sequences(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = np.stack((
        arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["actor_id"],
    ), axis=1)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    actor = np.asarray(arrays["features"], dtype=np.float32)[first, :len(ACTOR_FEATURE_NAMES)]
    horizon = np.asarray(arrays["horizon_seconds"], dtype=np.float32)[first, None]
    residual = np.asarray(arrays["actor_position_error_vector_ego_profile_m"], dtype=np.float32)[first]
    return np.concatenate((actor, horizon), axis=1), residual, inverse


@torch.no_grad()
def _predict(
    model: CoherentTrajectoryMixture, features: np.ndarray,
    feature_mean: np.ndarray, feature_scale: np.ndarray,
    target_mean: np.ndarray, target_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights, means, scales = [], [], []
    for start in range(0, len(features), 65536):
        batch = torch.from_numpy((features[start:start + 65536] - feature_mean) / feature_scale).cuda()
        logits, mean, scale = model(batch)
        weights.append(torch.softmax(logits, dim=1).cpu().numpy())
        means.append(mean.cpu().numpy() * target_scale[None] + target_mean[None])
        scales.append(scale.cpu().numpy() * target_scale[None])
    return np.concatenate(weights), np.concatenate(means), np.concatenate(scales)


def _trajectory_score(
    arrays: dict[str, np.ndarray], model: CoherentTrajectoryMixture,
    feature_mean: np.ndarray, feature_scale: np.ndarray,
    target_mean: np.ndarray, target_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    actor_features, _, inverse = _actor_sequences(arrays)
    weights, means, scales = _predict(
        model.eval(), actor_features, feature_mean, feature_scale, target_mean, target_scale,
    )
    row_weights = weights[inverse]
    row_means = means[inverse]
    row_scales = scales[inverse]
    normal = np.asarray(arrays["query_boundary_normal_ego_profile"], dtype=np.float32)
    projected_mean = np.sum(normal[:, None] * row_means, axis=3)
    projected_scale = np.sqrt(np.sum(np.square(normal[:, None] * row_scales), axis=3)).clip(min=1e-4)
    predicted = np.asarray(arrays["predicted_separation_profile_m"], dtype=np.float32)
    radius = np.asarray(arrays["occupancy_interaction_radius_m"], dtype=np.float32)[:, None]
    signed = predicted - radius
    margin = (
        np.abs(signed)[:, None] + np.sign(signed)[:, None] * projected_mean
    ) / projected_scale
    crossing = torch.special.ndtr(torch.from_numpy(-margin).cuda()).cpu().numpy()
    component_any = 1.0 - np.prod(1.0 - np.clip(crossing, 0.0, 1.0 - 1e-7), axis=2)
    row_score = np.sum(row_weights * component_any, axis=1)
    keys = np.stack((
        arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["query_id"],
    ), axis=1)
    identities = np.unique(keys, axis=0)
    entropy = -np.sum(weights * np.log(np.clip(weights, 1e-8, 1.0)), axis=1)
    return (
        _aligned_group_max(keys, row_score, identities),
        identities[:, 0].astype(np.int32),
        {"mean_mixture_entropy": float(entropy.mean()),
         "mean_max_component_weight": float(weights.max(axis=1).mean())},
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
    torch.manual_seed(int(config["seed"]))
    source = dict(np.load(
        args.runs_root / config["source_rows"]["run"] / config["source_rows"]["artifact"],
        allow_pickle=False,
    ))
    raw_features, raw_target, _ = _actor_sequences(source)
    feature_mean = raw_features.mean(0)
    feature_scale = raw_features.std(0).clip(min=1e-4)
    target_mean = raw_target.mean(0)
    target_scale = raw_target.std(0).clip(min=0.05)
    features = torch.from_numpy((raw_features - feature_mean) / feature_scale).cuda()
    targets = torch.from_numpy((raw_target - target_mean) / target_scale).cuda()
    model_config = config["model"]
    model = CoherentTrajectoryMixture(
        features.shape[1], model_config["hidden_dimensions"],
        int(model_config["component_count"]), raw_target.shape[1],
    ).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    final_loss = 0.0
    torch.cuda.reset_peak_memory_stats()
    for step in range(int(model_config["steps"])):
        index = torch.randint(len(features), (int(model_config["batch_size"]),), device="cuda")
        logits, mean, scale = model(features[index])
        normalized = (targets[index, None] - mean) / scale
        component_log_likelihood = -(
            0.5 * normalized.square() + torch.log(scale)
        ).sum(dim=(2, 3))
        loss = -torch.logsumexp(
            torch.log_softmax(logits, dim=1) + component_log_likelihood, dim=1,
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step % 500 == 0 or step + 1 == int(model_config["steps"]):
            print(f"P149 coherent-mixture step={step + 1} nll={final_loss:.6f}", flush=True)
    torch.save({
        "feature_mean": feature_mean, "feature_scale": feature_scale,
        "target_mean": target_mean, "target_scale": target_scale,
        "hidden_dimensions": model_config["hidden_dimensions"],
        "component_count": int(model_config["component_count"]),
        "point_count": int(raw_target.shape[1]), "model_state_dict": model.state_dict(),
    }, run_dir / config["model_artifact"])
    floor = float(config["boundary_state_cost"]["clearance_floor_m"])
    coverage = float(config["selection"]["coverage_fraction"])
    results = {}
    for cohort in config["development_cohorts"]:
        arrays = dict(np.load(args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False))
        score, scenes, diagnostics = _trajectory_score(
            arrays, model, feature_mean, feature_scale, target_mean, target_scale,
        )
        actual_cost, cost_scenes = _continuous_cost(arrays, floor)
        if not np.array_equal(scenes, cost_scenes):
            raise RuntimeError("P149 trajectory grouping is not aligned")
        selected = _select_by_scene(score, scenes, coverage)
        reference = config["frozen_p126_comparisons"][cohort["name"]]
        model_spearman = spearman_correlation(actual_cost, score)
        results[cohort["name"]] = {
            "trajectory_count": int(len(actual_cost)), "selected_trajectory_count": int(len(selected)),
            "coherent_mixture_selected_mean_cost": float(actual_cost[selected].mean()),
            "p126_selected_mean_cost": float(reference["selected_cost"]),
            "coherent_mixture_cost_spearman": model_spearman,
            "p126_cost_spearman": float(reference["spearman"]),
            "spearman_gain_over_p126": float(model_spearman - float(reference["spearman"])),
            **diagnostics,
        }
        print(json.dumps({cohort["name"]: results[cohort["name"]]}, indent=2), flush=True)
    gains = [row["spearman_gain_over_p126"] for row in results.values()]
    decisions = {
        "no_selected_cost_regression": all(row["coherent_mixture_selected_mean_cost"] <= row["p126_selected_mean_cost"] for row in results.values()),
        "minimum_mean_spearman_gain": float(np.mean(gains)) >= float(config["decision"]["minimum_mean_spearman_gain"]),
    }
    verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "training": {"actor_sequence_count": int(len(features)), "point_count": int(raw_target.shape[1]),
                     "component_count": int(model_config["component_count"]),
                     "source_horizons_seconds": sorted(np.unique(source["horizon_seconds"]).tolist()),
                     "final_mixture_nll": final_loss},
        "development_evaluations": results, "decision_checks": decisions,
        "mean_spearman_gain": float(np.mean(gains)),
        "resources": {"gpu": torch.cuda.get_device_name(0),
                      "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                      "wall_seconds": time.monotonic() - started},
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict, "decision_checks": decisions}, indent=2), flush=True)


if __name__ == "__main__":
    main()
