"""Train a source-only continuous-cost rank residual near fixed 50% coverage."""

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
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p114_monotone_tail_risk import (
    _crossing_probability, _trajectory_tail_features,
)
from scripts.run_worldsim_v67_p119_ranked_range_tail import (
    RankedRangeResidual, _head_features, _score, _within_scene_percentile,
)
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost


def _within_scene_pairs(
    scenes: np.ndarray, base_percentile: np.ndarray, cost_percentile: np.ndarray,
    lower_base: float, upper_base: float, safe_max: float, unsafe_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    safe_pairs: list[np.ndarray] = []
    unsafe_pairs: list[np.ndarray] = []
    generator = np.random.default_rng(0)
    in_band = (base_percentile >= lower_base) & (base_percentile <= upper_base)
    for scene in np.unique(scenes):
        members = scenes == scene
        safe = np.flatnonzero(members & in_band & (cost_percentile <= safe_max))
        unsafe = np.flatnonzero(members & in_band & (cost_percentile >= unsafe_min))
        if not len(safe) or not len(unsafe):
            continue
        count = max(len(safe), len(unsafe))
        safe_pairs.append(generator.choice(safe, count, replace=len(safe) < count))
        unsafe_pairs.append(generator.choice(unsafe, count, replace=len(unsafe) < count))
    if not safe_pairs:
        raise RuntimeError("continuous ranked range produced no within-scene pairs")
    return np.concatenate(safe_pairs), np.concatenate(unsafe_pairs)


def _evaluate(
    model: RankedRangeResidual, arrays: dict[str, np.ndarray], probability: np.ndarray,
    top_k: int, feature_mean: np.ndarray, feature_scale: np.ndarray,
    coverage: float, floor: float,
) -> dict[str, float | int]:
    grouped = _trajectory_tail_features(arrays, probability, top_k)
    raw_features, base_logit = _head_features(grouped)
    actual_cost, scenes = _continuous_cost(arrays, floor)
    if len(actual_cost) != len(raw_features) or not np.array_equal(scenes, grouped["scene_index"]):
        raise RuntimeError("continuous cost is not aligned with trajectory groups")
    learned = _score(model.eval(), raw_features, base_logit, feature_mean, feature_scale)
    clearance = grouped["clearance_score"]
    learned_selected = _select_by_scene(learned, scenes, coverage)
    base_selected = _select_by_scene(base_logit, scenes, coverage)
    clearance_selected = _select_by_scene(clearance, scenes, coverage)
    mean_cost = float(actual_cost.mean())
    learned_cost = float(actual_cost[learned_selected].mean())
    base_cost = float(actual_cost[base_selected].mean())
    clearance_cost = float(actual_cost[clearance_selected].mean())
    learned_spearman = spearman_correlation(actual_cost, learned)
    base_spearman = spearman_correlation(actual_cost, base_logit)
    return {
        "trajectory_count": int(len(actual_cost)),
        "selected_trajectory_count": int(len(learned_selected)),
        "all_mean_boundary_state_cost": mean_cost,
        "rank_residual_selected_mean_cost": learned_cost,
        "p109_selected_mean_cost": base_cost,
        "clearance_selected_mean_cost": clearance_cost,
        "rank_residual_selected_cost_reduction": float((mean_cost - learned_cost) / max(mean_cost, 1e-12)),
        "rank_residual_cost_spearman": learned_spearman,
        "p109_cost_spearman": base_spearman,
        "clearance_cost_spearman": spearman_correlation(actual_cost, clearance),
        "rank_residual_spearman_gain_over_p109": learned_spearman - base_spearman,
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
    seed = int(config["seed"])
    torch.manual_seed(seed)
    checkpoint = torch.load(
        args.runs_root / config["frozen_p109"]["run"] / config["frozen_p109"]["artifact"],
        map_location="cuda",
    )
    actor_model = DirectionalActorGaussian(20, checkpoint["hidden_dimensions"]).cuda()
    actor_model.load_state_dict(checkpoint["model_state_dict"])
    actor_model.eval()
    actor_feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    actor_feature_scale = np.asarray(checkpoint["feature_scale"], dtype=np.float32)
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
    target_scale = np.asarray(checkpoint["target_scale"], dtype=np.float32)
    rows_root = args.runs_root / config["rows"]["run"]
    source_raw = dict(np.load(rows_root / config["rows"]["source_artifact"], allow_pickle=False))
    source_probability, _ = _crossing_probability(
        source_raw, actor_model, actor_feature_mean, actor_feature_scale, target_mean, target_scale,
    )
    model_config = config["model"]
    top_k = int(model_config["top_k_crossing_probabilities"])
    grouped = _trajectory_tail_features(source_raw, source_probability, top_k)
    raw_features, base_logit = _head_features(grouped)
    floor = float(config["boundary_state_cost"]["clearance_floor_m"])
    source_cost, source_scenes = _continuous_cost(source_raw, floor)
    if len(source_cost) != len(raw_features) or not np.array_equal(source_scenes, grouped["scene_index"]):
        raise RuntimeError("source continuous cost is not aligned with trajectory features")
    base_percentile = _within_scene_percentile(base_logit, source_scenes)
    cost_percentile = _within_scene_percentile(source_cost, source_scenes)
    ranked_range = config["continuous_ranked_range"]
    safe_pair, unsafe_pair = _within_scene_pairs(
        source_scenes, base_percentile, cost_percentile,
        float(ranked_range["base_percentile_min"]), float(ranked_range["base_percentile_max"]),
        float(ranked_range["safe_cost_percentile_max"]),
        float(ranked_range["unsafe_cost_percentile_min"]),
    )
    feature_mean = raw_features.mean(0)
    feature_scale = raw_features.std(0).clip(min=1e-4)
    features = torch.from_numpy((raw_features - feature_mean) / feature_scale).cuda()
    base = torch.from_numpy(base_logit).cuda()
    safe_pair_tensor = torch.from_numpy(safe_pair).cuda()
    unsafe_pair_tensor = torch.from_numpy(unsafe_pair).cuda()
    model = RankedRangeResidual(
        raw_features.shape[1], int(model_config["hidden_dimension"]),
        float(model_config["residual_bound"]),
    ).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    steps = int(model_config["steps"])
    batch_size = int(model_config["pair_batch_size"])
    regularization = float(model_config["residual_regularization"])
    final_loss = 0.0
    torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        sampled = torch.randint(len(safe_pair_tensor), (batch_size,), device="cuda")
        safe_index = safe_pair_tensor[sampled]
        unsafe_index = unsafe_pair_tensor[sampled]
        safe_score, safe_residual = model(features[safe_index], base[safe_index])
        unsafe_score, unsafe_residual = model(features[unsafe_index], base[unsafe_index])
        rank_loss = torch.nn.functional.softplus(-(unsafe_score - safe_score)).mean()
        penalty = safe_residual.square().mean() + unsafe_residual.square().mean()
        loss = rank_loss + regularization * penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step % 250 == 0 or step + 1 == steps:
            print(f"P123 continuous-rank step={step + 1} loss={final_loss:.6f}", flush=True)
    torch.save({
        "feature_mean": feature_mean, "feature_scale": feature_scale,
        "hidden_dimension": int(model_config["hidden_dimension"]),
        "residual_bound": float(model_config["residual_bound"]),
        "model_state_dict": model.state_dict(),
    }, run_dir / config["model_artifact"])
    results = {}
    for cohort in config["development_cohorts"]:
        cohort_root = args.runs_root / cohort.get("run", config["rows"]["run"])
        raw = dict(np.load(cohort_root / cohort["artifact"], allow_pickle=False))
        probability, _ = _crossing_probability(
            raw, actor_model, actor_feature_mean, actor_feature_scale, target_mean, target_scale,
        )
        results[cohort["name"]] = _evaluate(
            model, raw, probability, top_k, feature_mean, feature_scale,
            float(config["selection"]["coverage_fraction"]), floor,
        )
        print(json.dumps({cohort["name"]: results[cohort["name"]]}, indent=2), flush=True)
    gains = [value["rank_residual_spearman_gain_over_p109"] for value in results.values()]
    decisions = {
        "no_selected_cost_regression": all(
            value["rank_residual_selected_mean_cost"] <= value["p109_selected_mean_cost"]
            for value in results.values()
        ),
        "minimum_mean_spearman_gain": float(np.mean(gains))
        >= float(config["decision"]["minimum_mean_spearman_gain_over_p109"]),
    }
    verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "training": {"source_trajectory_count": int(len(source_cost)),
                     "source_pair_count": int(len(safe_pair)),
                     "source_mean_boundary_state_cost": float(source_cost.mean()),
                     "final_continuous_rank_loss": final_loss},
        "development_evaluations": results, "decision_checks": decisions,
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
