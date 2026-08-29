"""Train a P126-shaped Actor ensemble with same-time-fraction RegMixup."""

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
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian, _actor_entries
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost
from scripts.run_worldsim_v67_p127_ensemble_continuous_selection import _ensemble_trajectory_score


def _gaussian_nll(
    model: DirectionalActorGaussian, features: torch.Tensor, targets: torch.Tensor,
) -> torch.Tensor:
    mean, scale = model(features)
    normalized = (targets - mean) / scale
    return (0.5 * normalized.square() + torch.log(scale)).sum(1).mean()


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
    source = dict(np.load(args.runs_root / config["source_rows"]["run"] / config["source_rows"]["artifact"], allow_pickle=False))
    raw_features, raw_target, _ = _actor_entries(source)
    frozen = torch.load(args.runs_root / config["frozen_p109"]["run"] / config["frozen_p109"]["artifact"], map_location="cuda")
    feature_mean = np.asarray(frozen["feature_mean"], dtype=np.float32)
    feature_scale = np.asarray(frozen["feature_scale"], dtype=np.float32)
    target_mean = np.asarray(frozen["target_mean"], dtype=np.float32)
    target_scale = np.asarray(frozen["target_scale"], dtype=np.float32)
    features = torch.from_numpy((raw_features - feature_mean) / feature_scale).cuda()
    targets = torch.from_numpy((raw_target - target_mean) / target_scale).cuda()
    fractions = np.rint(raw_features[:, -1] * 1000).astype(np.int32)
    unique_fractions = np.unique(fractions)
    buckets = [np.flatnonzero(fractions == value).astype(np.int64) for value in unique_fractions]
    bucket_length = min(len(row) for row in buckets)
    bucket_matrix = torch.from_numpy(np.stack([row[:bucket_length] for row in buckets])).cuda()
    model_config = config["model"]
    original_batch_size = int(model_config["original_batch_size"])
    mixup_batch_size = int(model_config["mixup_batch_size"])
    beta = torch.distributions.Beta(
        torch.tensor(float(model_config["mixup_beta_alpha"]), device="cuda"),
        torch.tensor(float(model_config["mixup_beta_alpha"]), device="cuda"),
    )
    models = []
    final_losses = {}
    torch.cuda.reset_peak_memory_stats()
    for seed_value in config["member_seeds"]:
        seed_value = int(seed_value)
        torch.manual_seed(seed_value)
        model = DirectionalActorGaussian(features.shape[1], model_config["hidden_dimensions"]).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"]))
        final_loss = 0.0
        for step in range(int(model_config["steps"])):
            original_index = torch.randint(len(features), (original_batch_size,), device="cuda")
            fraction_index = torch.randint(len(unique_fractions), (mixup_batch_size,), device="cuda")
            left_position = torch.randint(bucket_length, (mixup_batch_size,), device="cuda")
            right_position = torch.randint(bucket_length, (mixup_batch_size,), device="cuda")
            left = bucket_matrix[fraction_index, left_position]
            right = bucket_matrix[fraction_index, right_position]
            weight = beta.sample((mixup_batch_size, 1))
            mixed_features = weight * features[left] + (1.0 - weight) * features[right]
            mixed_targets = weight * targets[left] + (1.0 - weight) * targets[right]
            original_loss = _gaussian_nll(model, features[original_index], targets[original_index])
            mixup_loss = _gaussian_nll(model, mixed_features, mixed_targets)
            loss = float(model_config["original_loss_weight"]) * original_loss + float(model_config["mixup_loss_weight"]) * mixup_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
            if step % 500 == 0 or step + 1 == int(model_config["steps"]):
                print(f"P155 RegMixup seed={seed_value} step={step + 1} nll={final_loss:.6f}", flush=True)
        final_losses[str(seed_value)] = final_loss
        models.append(model.eval())
    torch.save({
        "feature_mean": feature_mean, "feature_scale": feature_scale, "target_mean": target_mean,
        "target_scale": target_scale, "hidden_dimensions": model_config["hidden_dimensions"],
        "member_state_dicts": [model.state_dict() for model in models],
    }, run_dir / config["model_artifact"])
    floor = float(config["boundary_state_cost"]["clearance_floor_m"])
    coverage = float(config["selection"]["coverage_fraction"])
    results = {}
    for cohort in config["development_cohorts"]:
        arrays = dict(np.load(args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False))
        score, scenes = _ensemble_trajectory_score(arrays, models, feature_mean, feature_scale, target_mean, target_scale)
        actual_cost, cost_scenes = _continuous_cost(arrays, floor)
        if not np.array_equal(scenes, cost_scenes):
            raise RuntimeError("P155 trajectory grouping is not aligned")
        selected = _select_by_scene(score, scenes, coverage)
        reference = config["frozen_p126_comparisons"][cohort["name"]]
        model_spearman = spearman_correlation(actual_cost, score)
        results[cohort["name"]] = {
            "trajectory_count": int(len(actual_cost)), "selected_trajectory_count": int(len(selected)),
            "RegMixup_selected_mean_cost": float(actual_cost[selected].mean()),
            "p126_selected_mean_cost": float(reference["selected_cost"]),
            "RegMixup_cost_spearman": model_spearman, "p126_cost_spearman": float(reference["spearman"]),
            "spearman_gain_over_p126": float(model_spearman - float(reference["spearman"])),
        }
        print(json.dumps({cohort["name"]: results[cohort["name"]]}, indent=2), flush=True)
    gains = [row["spearman_gain_over_p126"] for row in results.values()]
    decisions = {
        "no_selected_cost_regression": all(row["RegMixup_selected_mean_cost"] <= row["p126_selected_mean_cost"] for row in results.values()),
        "minimum_mean_spearman_gain": float(np.mean(gains)) >= float(config["decision"]["minimum_mean_spearman_gain"]),
    }
    verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "training": {"actor_time_tokens": int(len(features)), "time_fraction_bucket_count": int(len(unique_fractions)),
                     "member_final_RegMixup_nll": final_losses},
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
