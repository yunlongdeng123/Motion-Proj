"""Train a dense task-conditioned distribution over normalized boundary-state cost."""

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
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost
from scripts.run_worldsim_v67_p142_task_conditioned_projected_ensemble import ConditionalProjectedGaussian


def _cost_tokens(
    arrays: dict[str, np.ndarray], floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    residual = np.asarray(arrays["actor_position_error_vector_ego_profile_m"], dtype=np.float32)
    normal = np.asarray(arrays["query_boundary_normal_ego_profile"], dtype=np.float32)
    point_count = residual.shape[1]
    base = np.asarray(arrays["features"], dtype=np.float32)
    fractions = np.linspace(0.0, 1.0, point_count, dtype=np.float32)
    predicted = np.asarray(arrays["predicted_separation_profile_m"], dtype=np.float32)
    radius = np.asarray(arrays["occupancy_interaction_radius_m"], dtype=np.float32)[:, None]
    clearance = np.maximum(np.abs(predicted - radius), floor)
    features = np.concatenate((
        np.broadcast_to(base[:, None, :], (len(base), point_count, base.shape[1])),
        np.broadcast_to(fractions[None, :, None], (len(base), point_count, 1)),
        normal, np.log(clearance)[:, :, None],
    ), axis=2)
    projected = np.abs(np.sum(residual * normal, axis=2))
    target = np.log1p(projected / clearance)
    return features.reshape(-1, features.shape[-1]), target.reshape(-1)


@torch.no_grad()
def _predict(
    model: ConditionalProjectedGaussian, raw_features: np.ndarray,
    feature_mean: np.ndarray, feature_scale: np.ndarray,
    target_mean: float, target_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    means, scales = [], []
    for start in range(0, len(raw_features), 262144):
        batch = torch.from_numpy((raw_features[start:start + 262144] - feature_mean) / feature_scale).cuda()
        mean, scale = model(batch)
        means.append((mean * target_scale + target_mean).cpu().numpy())
        scales.append((scale * target_scale).cpu().numpy())
    return np.concatenate(means), np.concatenate(scales)


def _trajectory_score(
    arrays: dict[str, np.ndarray], models: list[ConditionalProjectedGaussian],
    feature_mean: np.ndarray, feature_scale: np.ndarray,
    target_mean: float, target_scale: float, floor: float, upper_sigma: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    raw_features, _ = _cost_tokens(arrays, floor)
    point_count = int(arrays["actor_position_error_vector_ego_profile_m"].shape[1])
    member_means, member_scales = [], []
    for model in models:
        mean, scale = _predict(model.eval(), raw_features, feature_mean, feature_scale, target_mean, target_scale)
        member_means.append(mean.reshape(-1, point_count))
        member_scales.append(scale.reshape(-1, point_count))
    means = np.stack(member_means, axis=0)
    scales = np.stack(member_scales, axis=0)
    mean = means.mean(axis=0)
    epistemic_variance = means.var(axis=0)
    total_variance = np.maximum(epistemic_variance + np.square(scales).mean(axis=0), 1e-8)
    row_score = np.max(np.expm1(mean + upper_sigma * np.sqrt(total_variance)), axis=1)
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
    floor = float(config["boundary_state_cost"]["clearance_floor_m"])
    source = dict(np.load(args.runs_root / config["source_rows"]["run"] / config["source_rows"]["artifact"], allow_pickle=False))
    raw_features, raw_target = _cost_tokens(source, floor)
    feature_mean = raw_features.mean(0)
    feature_scale = raw_features.std(0).clip(min=1e-4)
    target_mean = float(raw_target.mean())
    target_scale = float(max(raw_target.std(), 0.05))
    features = torch.from_numpy((raw_features - feature_mean) / feature_scale).cuda()
    targets = torch.from_numpy((raw_target - target_mean) / target_scale).cuda()
    model_config = config["model"]
    models = []
    final_losses = {}
    torch.cuda.reset_peak_memory_stats()
    for seed_value in config["member_seeds"]:
        seed_value = int(seed_value)
        torch.manual_seed(seed_value)
        model = ConditionalProjectedGaussian(features.shape[1], model_config["hidden_dimensions"]).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"]))
        final_loss = 0.0
        for step in range(int(model_config["steps"])):
            index = torch.randint(len(features), (int(model_config["batch_size"]),), device="cuda")
            mean, scale = model(features[index])
            normalized = (targets[index] - mean) / scale
            loss = (0.5 * normalized.square() + torch.log(scale)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
            if step % 500 == 0 or step + 1 == int(model_config["steps"]):
                print(f"P150 dense-cost seed={seed_value} step={step + 1} nll={final_loss:.6f}", flush=True)
        final_losses[str(seed_value)] = final_loss
        models.append(model.eval())
    torch.save({
        "feature_mean": feature_mean, "feature_scale": feature_scale,
        "target_mean": target_mean, "target_scale": target_scale,
        "hidden_dimensions": model_config["hidden_dimensions"],
        "member_state_dicts": [model.state_dict() for model in models],
    }, run_dir / config["model_artifact"])
    coverage = float(config["selection"]["coverage_fraction"])
    upper_sigma = float(model_config["upper_standard_deviations"])
    results = {}
    for cohort in config["development_cohorts"]:
        arrays = dict(np.load(args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False))
        score, scenes, epistemic_fraction = _trajectory_score(
            arrays, models, feature_mean, feature_scale, target_mean, target_scale, floor, upper_sigma,
        )
        actual_cost, cost_scenes = _continuous_cost(arrays, floor)
        if not np.array_equal(scenes, cost_scenes):
            raise RuntimeError("P150 trajectory grouping is not aligned")
        selected = _select_by_scene(score, scenes, coverage)
        reference = config["frozen_p126_comparisons"][cohort["name"]]
        model_spearman = spearman_correlation(actual_cost, score)
        results[cohort["name"]] = {
            "trajectory_count": int(len(actual_cost)), "selected_trajectory_count": int(len(selected)),
            "dense_cost_selected_mean_cost": float(actual_cost[selected].mean()),
            "p126_selected_mean_cost": float(reference["selected_cost"]),
            "dense_cost_spearman": model_spearman, "p126_cost_spearman": float(reference["spearman"]),
            "spearman_gain_over_p126": float(model_spearman - float(reference["spearman"])),
            "mean_projected_epistemic_fraction": epistemic_fraction,
        }
        print(json.dumps({cohort["name"]: results[cohort["name"]]}, indent=2), flush=True)
    gains = [row["spearman_gain_over_p126"] for row in results.values()]
    decisions = {
        "no_selected_cost_regression": all(row["dense_cost_selected_mean_cost"] <= row["p126_selected_mean_cost"] for row in results.values()),
        "minimum_mean_spearman_gain": float(np.mean(gains)) >= float(config["decision"]["minimum_mean_spearman_gain"]),
    }
    verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "training": {"conditional_time_tokens": int(len(features)), "member_final_nll": final_losses},
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
