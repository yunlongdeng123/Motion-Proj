"""Train a three-snapshot Actor ensemble along one cyclic optimization path."""

from __future__ import annotations

import argparse
import copy
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
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import (
    DirectionalActorGaussian, _actor_entries,
)
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost
from scripts.run_worldsim_v67_p127_ensemble_continuous_selection import _ensemble_trajectory_score


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
    raw_features, raw_target, _ = _actor_entries(source)
    feature_mean = raw_features.mean(0)
    feature_scale = raw_features.std(0).clip(min=1e-4)
    target_mean = raw_target.mean(0)
    target_scale = raw_target.std(0).clip(min=0.05)
    features = torch.from_numpy((raw_features - feature_mean) / feature_scale).cuda()
    targets = torch.from_numpy((raw_target - target_mean) / target_scale).cuda()
    model_config = config["model"]
    model = DirectionalActorGaussian(
        features.shape[1], model_config["hidden_dimensions"],
    ).cuda()
    maximum_lr = float(model_config["maximum_learning_rate"])
    minimum_lr = float(model_config["minimum_learning_rate"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=maximum_lr, weight_decay=float(model_config["weight_decay"]),
    )
    cycles = int(model_config["cycles"])
    steps = int(model_config["steps"])
    cycle_steps = steps // cycles
    snapshots: list[dict[str, torch.Tensor]] = []
    snapshot_losses: list[float] = []
    final_loss = 0.0
    torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        cycle_position = step % cycle_steps
        fraction = cycle_position / max(cycle_steps - 1, 1)
        learning_rate = minimum_lr + 0.5 * (maximum_lr - minimum_lr) * (
            1.0 + math.cos(math.pi * fraction)
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        index = torch.randint(
            len(features), (int(model_config["batch_size"]),), device="cuda",
        )
        mean, scale = model(features[index])
        residual = (targets[index] - mean) / scale
        loss = (0.5 * residual.square() + torch.log(scale)).sum(dim=1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if (step + 1) % cycle_steps == 0:
            snapshots.append(copy.deepcopy(model.state_dict()))
            snapshot_losses.append(final_loss)
        if step % 250 == 0 or step + 1 == steps:
            print(
                f"P136 snapshot step={step + 1} lr={learning_rate:.8f} nll={final_loss:.6f}",
                flush=True,
            )
    models = []
    for state in snapshots:
        member = DirectionalActorGaussian(
            features.shape[1], model_config["hidden_dimensions"],
        ).cuda()
        member.load_state_dict(state)
        models.append(member.eval())
    torch.save({
        "feature_mean": feature_mean, "feature_scale": feature_scale,
        "target_mean": target_mean, "target_scale": target_scale,
        "hidden_dimensions": model_config["hidden_dimensions"],
        "snapshot_steps": [cycle_steps * (index + 1) for index in range(cycles)],
        "snapshot_state_dicts": snapshots,
    }, run_dir / config["model_artifact"])
    floor = float(config["boundary_state_cost"]["clearance_floor_m"])
    coverage = float(config["selection"]["coverage_fraction"])
    results = {}
    for cohort in config["development_cohorts"]:
        arrays = dict(np.load(
            args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False,
        ))
        score, scenes = _ensemble_trajectory_score(
            arrays, models, feature_mean, feature_scale, target_mean, target_scale,
        )
        actual_cost, cost_scenes = _continuous_cost(arrays, floor)
        if not np.array_equal(scenes, cost_scenes):
            raise RuntimeError("P136 trajectory grouping is not aligned")
        selected = _select_by_scene(score, scenes, coverage)
        reference = config["frozen_p126_comparisons"][cohort["name"]]
        model_spearman = spearman_correlation(actual_cost, score)
        results[cohort["name"]] = {
            "trajectory_count": int(len(actual_cost)),
            "selected_trajectory_count": int(len(selected)),
            "snapshot_selected_mean_cost": float(actual_cost[selected].mean()),
            "deep_ensemble_selected_mean_cost": float(reference["selected_cost"]),
            "snapshot_cost_spearman": model_spearman,
            "deep_ensemble_cost_spearman": float(reference["spearman"]),
            "spearman_difference_from_deep_ensemble": float(
                model_spearman - float(reference["spearman"])
            ),
        }
        print(json.dumps({cohort["name"]: results[cohort["name"]]}, indent=2), flush=True)
    differences = [x["spearman_difference_from_deep_ensemble"] for x in results.values()]
    decisions = {
        "no_selected_cost_regression_from_deep_ensemble": all(
            x["snapshot_selected_mean_cost"] <= x["deep_ensemble_selected_mean_cost"]
            for x in results.values()
        ),
        "mean_spearman_retention": float(np.mean(differences))
        >= float(config["decision"]["minimum_mean_spearman_difference_from_deep_ensemble"]),
    }
    verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "training": {"actor_time_tokens": int(len(features)), "cycles": cycles,
                     "snapshot_final_nll": snapshot_losses},
        "development_evaluations": results, "decision_checks": decisions,
        "mean_spearman_difference_from_deep_ensemble": float(np.mean(differences)),
        "resources": {"gpu": torch.cuda.get_device_name(0),
                      "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                      "wall_seconds": time.monotonic() - started},
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({
        "status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict,
                      "decision_checks": decisions}, indent=2), flush=True)


if __name__ == "__main__":
    main()
