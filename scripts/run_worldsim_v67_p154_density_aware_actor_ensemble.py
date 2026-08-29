"""Fit a hidden-feature density flow and make frozen P126 variance distance aware."""

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
import yaml

from motion_proj.worldsim_v67.actor_state_reliability import spearman_correlation
from scripts.run_worldsim_v67_p86_direct_trajectory_reliability import _select_by_scene
from scripts.run_worldsim_v67_p104_temporal_flip_supervision import _aligned_group_max
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian, _actor_entries, _predict
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost


class AffineCoupling(torch.nn.Module):
    def __init__(self, dimension: int, hidden: int, mask: torch.Tensor, maximum_log_scale: float) -> None:
        super().__init__()
        self.register_buffer("mask", mask)
        self.network = torch.nn.Sequential(
            torch.nn.Linear(dimension, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, dimension * 2),
        )
        self.maximum_log_scale = maximum_log_scale

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masked = inputs * self.mask
        shift, raw_scale = self.network(masked).chunk(2, dim=1)
        inverse_mask = 1.0 - self.mask
        log_scale = torch.tanh(raw_scale) * self.maximum_log_scale * inverse_mask
        outputs = masked + inverse_mask * (inputs * torch.exp(log_scale) + shift)
        return outputs, log_scale.sum(1)


class HiddenDensityFlow(torch.nn.Module):
    def __init__(self, dimension: int, hidden: int, layer_count: int, maximum_log_scale: float) -> None:
        super().__init__()
        layers = []
        for index in range(layer_count):
            mask = ((torch.arange(dimension) + index) % 2).float()[None]
            layers.append(AffineCoupling(dimension, hidden, mask, maximum_log_scale))
        self.layers = torch.nn.ModuleList(layers)

    def negative_log_density(self, inputs: torch.Tensor) -> torch.Tensor:
        values = inputs
        log_determinant = torch.zeros(len(inputs), device=inputs.device)
        for layer in self.layers:
            values, increment = layer(values)
            log_determinant += increment
        return 0.5 * values.square().sum(1) + 0.5 * values.shape[1] * math.log(2.0 * math.pi) - log_determinant


@torch.no_grad()
def _hidden(model: DirectionalActorGaussian, normalized_features: torch.Tensor) -> torch.Tensor:
    return model.network[:-1](normalized_features)


@torch.no_grad()
def _density_nll(
    flow: HiddenDensityFlow, raw_features: np.ndarray,
    input_mean: np.ndarray, input_scale: np.ndarray,
    p109: DirectionalActorGaussian, hidden_mean: np.ndarray, hidden_scale: np.ndarray,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(raw_features), 65536):
        features = torch.from_numpy((raw_features[start:start + 65536] - input_mean) / input_scale).cuda()
        hidden = (_hidden(p109, features) - torch.from_numpy(hidden_mean).cuda()) / torch.from_numpy(hidden_scale).cuda()
        outputs.append(flow.negative_log_density(hidden).cpu().numpy())
    return np.concatenate(outputs)


def _trajectory_score(
    arrays: dict[str, np.ndarray], models: list[DirectionalActorGaussian],
    feature_mean: np.ndarray, feature_scale: np.ndarray,
    target_mean: np.ndarray, target_scale: np.ndarray,
    p109: DirectionalActorGaussian, flow: HiddenDensityFlow,
    hidden_mean: np.ndarray, hidden_scale: np.ndarray, nll_mean: float, nll_scale: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    actor_features, _, inverse = _actor_entries(arrays)
    point_count = int(arrays["actor_position_error_vector_ego_profile_m"].shape[1])
    member_means, member_scales = [], []
    for model in models:
        mean, scale = _predict(model.eval(), actor_features, feature_mean, feature_scale, target_mean, target_scale)
        member_means.append(mean.reshape(-1, point_count, 2)[inverse])
        member_scales.append(scale.reshape(-1, point_count, 2)[inverse])
    means = np.stack(member_means, axis=0)
    scales = np.stack(member_scales, axis=0)
    density_nll = _density_nll(
        flow, actor_features, feature_mean, feature_scale, p109, hidden_mean, hidden_scale,
    ).reshape(-1, point_count)[inverse]
    standardized_nll = (density_nll - nll_mean) / nll_scale
    inflation = 1.0 + np.maximum(standardized_nll, 0.0)
    normal = np.asarray(arrays["query_boundary_normal_ego_profile"], dtype=np.float32)
    projected_member_mean = np.sum(normal[None] * means, axis=3)
    projected_mean = projected_member_mean.mean(axis=0)
    base_variance = (
        np.mean(np.sum(np.square(normal[None] * scales), axis=3), axis=0)
        + projected_member_mean.var(axis=0)
    )
    total_variance = np.maximum(base_variance * inflation, 1e-8)
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
        {"mean_density_variance_inflation": float(inflation.mean()),
         "p95_density_variance_inflation": float(np.quantile(inflation, 0.95))},
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
    source = dict(np.load(args.runs_root / config["source_rows"]["run"] / config["source_rows"]["artifact"], allow_pickle=False))
    raw_features, _, _ = _actor_entries(source)
    p109_checkpoint = torch.load(args.runs_root / config["frozen_p109"]["run"] / config["frozen_p109"]["artifact"], map_location="cuda")
    feature_mean = np.asarray(p109_checkpoint["feature_mean"], dtype=np.float32)
    feature_scale = np.asarray(p109_checkpoint["feature_scale"], dtype=np.float32)
    p109 = DirectionalActorGaussian(20, p109_checkpoint["hidden_dimensions"]).cuda()
    p109.load_state_dict(p109_checkpoint["model_state_dict"])
    p109.eval()
    normalized = torch.from_numpy((raw_features - feature_mean) / feature_scale).cuda()
    with torch.no_grad():
        source_hidden = _hidden(p109, normalized)
    hidden_mean_tensor = source_hidden.mean(0)
    hidden_scale_tensor = source_hidden.std(0).clamp(min=1e-4)
    source_hidden = (source_hidden - hidden_mean_tensor) / hidden_scale_tensor
    density = config["density_model"]
    flow = HiddenDensityFlow(
        source_hidden.shape[1], int(density["hidden_dimension"]),
        int(density["coupling_layers"]), float(density["maximum_log_scale"]),
    ).cuda()
    optimizer = torch.optim.AdamW(flow.parameters(), lr=float(density["learning_rate"]), weight_decay=float(density["weight_decay"]))
    final_nll = 0.0
    torch.cuda.reset_peak_memory_stats()
    for step in range(int(density["steps"])):
        index = torch.randint(len(source_hidden), (int(density["batch_size"]),), device="cuda")
        loss = flow.negative_log_density(source_hidden[index]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_nll = float(loss.detach().cpu())
        if step % 500 == 0 or step + 1 == int(density["steps"]):
            print(f"P154 density-flow step={step + 1} nll={final_nll:.6f}", flush=True)
    with torch.no_grad():
        source_nll = []
        for start in range(0, len(source_hidden), 65536):
            source_nll.append(flow.negative_log_density(source_hidden[start:start + 65536]).cpu().numpy())
    source_nll = np.concatenate(source_nll)
    nll_mean, nll_scale = float(source_nll.mean()), float(max(source_nll.std(), 1e-4))
    hidden_mean = hidden_mean_tensor.cpu().numpy()
    hidden_scale = hidden_scale_tensor.cpu().numpy()
    torch.save({
        "flow_state_dict": flow.state_dict(), "hidden_dimension": int(source_hidden.shape[1]),
        "hidden_mean": hidden_mean, "hidden_scale": hidden_scale,
        "source_nll_mean": nll_mean, "source_nll_scale": nll_scale,
    }, run_dir / config["model_artifact"])
    ensemble = torch.load(args.runs_root / config["frozen_p126"]["run"] / config["frozen_p126"]["artifact"], map_location="cuda")
    models = []
    for state in ensemble["member_state_dicts"]:
        model = DirectionalActorGaussian(20, ensemble["hidden_dimensions"]).cuda()
        model.load_state_dict(state)
        models.append(model.eval())
    target_mean = np.asarray(ensemble["target_mean"], dtype=np.float32)
    target_scale = np.asarray(ensemble["target_scale"], dtype=np.float32)
    floor = float(config["boundary_state_cost"]["clearance_floor_m"])
    coverage = float(config["selection"]["coverage_fraction"])
    results = {}
    for cohort in config["development_cohorts"]:
        arrays = dict(np.load(args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False))
        score, scenes, diagnostics = _trajectory_score(
            arrays, models, feature_mean, feature_scale, target_mean, target_scale,
            p109, flow, hidden_mean, hidden_scale, nll_mean, nll_scale,
        )
        actual_cost, cost_scenes = _continuous_cost(arrays, floor)
        if not np.array_equal(scenes, cost_scenes):
            raise RuntimeError("P154 trajectory grouping is not aligned")
        selected = _select_by_scene(score, scenes, coverage)
        reference = config["frozen_p126_comparisons"][cohort["name"]]
        model_spearman = spearman_correlation(actual_cost, score)
        results[cohort["name"]] = {
            "trajectory_count": int(len(actual_cost)), "selected_trajectory_count": int(len(selected)),
            "density_aware_selected_mean_cost": float(actual_cost[selected].mean()),
            "p126_selected_mean_cost": float(reference["selected_cost"]),
            "density_aware_cost_spearman": model_spearman,
            "p126_cost_spearman": float(reference["spearman"]),
            "spearman_gain_over_p126": float(model_spearman - float(reference["spearman"])), **diagnostics,
        }
        print(json.dumps({cohort["name"]: results[cohort["name"]]}, indent=2), flush=True)
    gains = [row["spearman_gain_over_p126"] for row in results.values()]
    decisions = {
        "no_selected_cost_regression": all(row["density_aware_selected_mean_cost"] <= row["p126_selected_mean_cost"] for row in results.values()),
        "minimum_mean_spearman_gain": float(np.mean(gains)) >= float(config["decision"]["minimum_mean_spearman_gain"]),
    }
    verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "density_training": {"actor_time_tokens": int(len(source_hidden)), "hidden_dimension": int(source_hidden.shape[1]),
                             "final_flow_nll": final_nll, "source_nll_mean": nll_mean, "source_nll_scale": nll_scale},
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
