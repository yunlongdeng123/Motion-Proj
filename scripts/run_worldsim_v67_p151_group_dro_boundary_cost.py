"""Train the dense boundary-cost ensemble with scene-horizon group DRO."""

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
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost
from scripts.run_worldsim_v67_p142_task_conditioned_projected_ensemble import ConditionalProjectedGaussian
from scripts.run_worldsim_v67_p150_dense_boundary_cost_ensemble import _cost_tokens, _trajectory_score


def _group_matrix(arrays: dict[str, np.ndarray], point_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    row_keys = np.stack((
        arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
    ), axis=1)
    _, row_group = np.unique(row_keys, axis=0, return_inverse=True)
    token_group = np.repeat(row_group, point_count)
    members = [np.flatnonzero(token_group == group).astype(np.int64) for group in range(int(token_group.max()) + 1)]
    lengths = np.asarray([len(row) for row in members], dtype=np.int64)
    matrix = np.zeros((len(members), int(lengths.max())), dtype=np.int64)
    for group, row in enumerate(members):
        matrix[group, :len(row)] = row
    return torch.from_numpy(matrix).cuda(), torch.from_numpy(lengths).cuda()


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
    point_count = int(source["actor_position_error_vector_ego_profile_m"].shape[1])
    group_matrix, group_lengths = _group_matrix(source, point_count)
    model_config = config["model"]
    groups_per_batch = min(int(model_config["groups_per_batch"]), len(group_lengths))
    tokens_per_group = int(model_config["tokens_per_group"])
    worst_group_count = max(1, int(math.ceil(groups_per_batch * float(model_config["worst_group_fraction"]))))
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
            groups = torch.randperm(len(group_lengths), device="cuda")[:groups_per_batch]
            positions = torch.randint(int(group_lengths.max()), (groups_per_batch, tokens_per_group), device="cuda")
            positions %= group_lengths[groups, None]
            index = group_matrix[groups[:, None], positions]
            mean, scale = model(features[index.reshape(-1)])
            normalized = (targets[index.reshape(-1)] - mean) / scale
            token_loss = 0.5 * normalized.square() + torch.log(scale)
            group_loss = token_loss.reshape(groups_per_batch, tokens_per_group).mean(1)
            loss = torch.topk(group_loss, worst_group_count, largest=True).values.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
            if step % 500 == 0 or step + 1 == int(model_config["steps"]):
                print(f"P151 group-DRO seed={seed_value} step={step + 1} worst-q-nll={final_loss:.6f}", flush=True)
        final_losses[str(seed_value)] = final_loss
        models.append(model.eval())
    torch.save({
        "feature_mean": feature_mean, "feature_scale": feature_scale, "target_mean": target_mean,
        "target_scale": target_scale, "hidden_dimensions": model_config["hidden_dimensions"],
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
            raise RuntimeError("P151 trajectory grouping is not aligned")
        selected = _select_by_scene(score, scenes, coverage)
        reference = config["frozen_p126_comparisons"][cohort["name"]]
        model_spearman = spearman_correlation(actual_cost, score)
        results[cohort["name"]] = {
            "trajectory_count": int(len(actual_cost)), "selected_trajectory_count": int(len(selected)),
            "group_dro_selected_mean_cost": float(actual_cost[selected].mean()),
            "p126_selected_mean_cost": float(reference["selected_cost"]),
            "group_dro_cost_spearman": model_spearman, "p126_cost_spearman": float(reference["spearman"]),
            "spearman_gain_over_p126": float(model_spearman - float(reference["spearman"])),
            "mean_projected_epistemic_fraction": epistemic_fraction,
        }
        print(json.dumps({cohort["name"]: results[cohort["name"]]}, indent=2), flush=True)
    gains = [row["spearman_gain_over_p126"] for row in results.values()]
    decisions = {
        "no_selected_cost_regression": all(row["group_dro_selected_mean_cost"] <= row["p126_selected_mean_cost"] for row in results.values()),
        "minimum_mean_spearman_gain": float(np.mean(gains)) >= float(config["decision"]["minimum_mean_spearman_gain"]),
    }
    verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "training": {"conditional_time_tokens": int(len(features)), "environment_count": int(len(group_lengths)),
                     "groups_per_batch": groups_per_batch, "worst_group_count": worst_group_count,
                     "member_final_worst_quartile_nll": final_losses},
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
