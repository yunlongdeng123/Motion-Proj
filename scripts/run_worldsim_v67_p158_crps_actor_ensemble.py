"""Train the shared P126 ensemble with Gaussian CRPS instead of Gaussian NLL."""

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
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import (
    DirectionalActorGaussian, _actor_entries,
)
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost
from scripts.run_worldsim_v67_p127_ensemble_continuous_selection import _ensemble_trajectory_score


def _gaussian_crps(target: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    z = (target - mean) / scale
    phi = torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    return (scale * (z * (2.0 * cdf - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))).sum(1).mean()


def _subset(arrays: dict[str, np.ndarray], horizon: float) -> dict[str, np.ndarray]:
    mask = np.isclose(np.asarray(arrays["horizon_seconds"], dtype=np.float32), horizon)
    return {name: value[mask] for name, value in arrays.items()}


def _evaluate(
    arrays: dict[str, np.ndarray], candidate: list[DirectionalActorGaussian], baseline: list[DirectionalActorGaussian],
    feature_mean: np.ndarray, feature_scale: np.ndarray, target_mean: np.ndarray, target_scale: np.ndarray,
    coverage: float, floor: float,
) -> dict[str, float | int]:
    candidate_score, scenes = _ensemble_trajectory_score(
        arrays, candidate, feature_mean, feature_scale, target_mean, target_scale,
    )
    baseline_score, baseline_scenes = _ensemble_trajectory_score(
        arrays, baseline, feature_mean, feature_scale, target_mean, target_scale,
    )
    actual_cost, cost_scenes = _continuous_cost(arrays, floor)
    if not (np.array_equal(scenes, baseline_scenes) and np.array_equal(scenes, cost_scenes)):
        raise RuntimeError("P158 trajectory grouping is not aligned")
    candidate_selected = _select_by_scene(candidate_score, scenes, coverage)
    baseline_selected = _select_by_scene(baseline_score, scenes, coverage)
    candidate_rank = spearman_correlation(actual_cost, candidate_score)
    baseline_rank = spearman_correlation(actual_cost, baseline_score)
    return {
        "row_count": int(len(arrays["features"])),
        "trajectory_count": int(len(actual_cost)),
        "selected_trajectory_count": int(len(candidate_selected)),
        "crps_selected_mean_cost": float(actual_cost[candidate_selected].mean()),
        "p126_selected_mean_cost": float(actual_cost[baseline_selected].mean()),
        "crps_minus_p126_selected_cost": float(actual_cost[candidate_selected].mean() - actual_cost[baseline_selected].mean()),
        "crps_cost_spearman": candidate_rank,
        "p126_cost_spearman": baseline_rank,
        "spearman_gain_over_p126": float(candidate_rank - baseline_rank),
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
    source = dict(np.load(
        args.runs_root / config["source_rows"]["run"] / config["source_rows"]["artifact"], allow_pickle=False,
    ))
    raw_features, raw_target, _ = _actor_entries(source)
    frozen = torch.load(
        args.runs_root / config["frozen_p126"]["run"] / config["frozen_p126"]["artifact"], map_location="cuda",
    )
    feature_mean = np.asarray(frozen["feature_mean"], dtype=np.float32)
    feature_scale = np.asarray(frozen["feature_scale"], dtype=np.float32)
    target_mean = np.asarray(frozen["target_mean"], dtype=np.float32)
    target_scale = np.asarray(frozen["target_scale"], dtype=np.float32)
    features = torch.from_numpy((raw_features - feature_mean) / feature_scale).cuda()
    target = torch.from_numpy((raw_target - target_mean) / target_scale).cuda()
    model_config = config["model"]
    baseline = []
    for state in frozen["member_state_dicts"]:
        model = DirectionalActorGaussian(features.shape[1], frozen["hidden_dimensions"]).cuda()
        model.load_state_dict(state)
        baseline.append(model.eval())
    candidate = []
    final_crps = {}
    torch.cuda.reset_peak_memory_stats()
    for seed_value in config["member_seeds"]:
        seed_value = int(seed_value)
        torch.manual_seed(seed_value)
        model = DirectionalActorGaussian(features.shape[1], model_config["hidden_dimensions"]).cuda()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"]),
        )
        final_loss = 0.0
        for step in range(int(model_config["steps_per_member"])):
            index = torch.randint(len(features), (int(model_config["batch_size"]),), device="cuda")
            mean, scale = model(features[index])
            loss = _gaussian_crps(target[index], mean, scale)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
            if step % 1000 == 0 or step + 1 == int(model_config["steps_per_member"]):
                print(f"P158 seed={seed_value} step={step + 1} crps={final_loss:.6f}", flush=True)
        final_crps[str(seed_value)] = final_loss
        candidate.append(model.eval())
    torch.save({
        "feature_mean": feature_mean, "feature_scale": feature_scale,
        "target_mean": target_mean, "target_scale": target_scale,
        "hidden_dimensions": model_config["hidden_dimensions"], "member_seeds": config["member_seeds"],
        "member_state_dicts": [model.state_dict() for model in candidate],
    }, run_dir / config["model_artifact"])
    coverage = float(config["selection"]["coverage_fraction"])
    floor = float(config["boundary_state_cost"]["clearance_floor_m"])
    decision_results = {}
    for cohort in config["decision_cohorts"]:
        arrays = dict(np.load(args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False))
        decision_results[cohort["name"]] = _evaluate(
            arrays, candidate, baseline, feature_mean, feature_scale, target_mean, target_scale, coverage, floor,
        )
        print(json.dumps({cohort["name"]: decision_results[cohort["name"]]}, indent=2), flush=True)
    diagnostic_spec = config["post_confirmation_diagnostic"]
    diagnostic_arrays = dict(np.load(
        args.runs_root / diagnostic_spec["run"] / diagnostic_spec["artifact"], allow_pickle=False,
    ))
    diagnostic_results = {}
    for horizon in diagnostic_spec["horizons_seconds"]:
        key = str(float(horizon))
        diagnostic_results[key] = _evaluate(
            _subset(diagnostic_arrays, float(horizon)), candidate, baseline,
            feature_mean, feature_scale, target_mean, target_scale, coverage, floor,
        )
        print(json.dumps({f"P147_H{key}": diagnostic_results[key]}, indent=2), flush=True)
    gains = [row["spearman_gain_over_p126"] for row in decision_results.values()]
    decisions = {
        "no_selected_cost_regression": all(row["crps_selected_mean_cost"] <= row["p126_selected_mean_cost"] for row in decision_results.values()),
        "minimum_mean_spearman_gain": float(np.mean(gains)) >= float(config["decision"]["minimum_mean_spearman_gain_over_p126"]),
    }
    verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict, "role": config["role"],
        "training": {"actor_time_tokens": int(len(features)), "member_count": len(candidate), "member_final_crps": final_crps},
        "consumed_development_evaluations": decision_results,
        "post_confirmation_consumed_p147_diagnostic": diagnostic_results,
        "decision_checks": decisions, "mean_spearman_gain": float(np.mean(gains)),
        "resources": {"gpu": torch.cuda.get_device_name(0), "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                      "wall_seconds": time.monotonic() - started},
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({
        "status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict, "decision_checks": decisions}, indent=2), flush=True)


if __name__ == "__main__":
    main()
