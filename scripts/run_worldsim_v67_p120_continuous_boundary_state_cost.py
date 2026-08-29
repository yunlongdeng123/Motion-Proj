"""Regress a continuous trajectory-conditioned boundary-state reliability cost."""

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
from scripts.run_worldsim_v67_p119_ranked_range_tail import _head_features


class BoundaryStateCostRegressor(torch.nn.Module):
    def __init__(self, feature_count: int, hidden_dimensions: list[int]) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        width = feature_count
        for hidden in hidden_dimensions:
            layers.extend((torch.nn.Linear(width, int(hidden)), torch.nn.SiLU()))
            width = int(hidden)
        layers.append(torch.nn.Linear(width, 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).reshape(-1)


def _continuous_cost(
    arrays: dict[str, np.ndarray], floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    keys = np.stack((
        arrays["scene_index"],
        np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["query_id"],
    ), axis=1)
    identities, inverse = np.unique(keys, axis=0, return_inverse=True)
    residual = np.asarray(arrays["actor_position_error_vector_ego_profile_m"], dtype=np.float32)
    normal = np.asarray(arrays["query_boundary_normal_ego_profile"], dtype=np.float32)
    projected_error = np.abs(np.sum(normal * residual, axis=2))
    predicted = np.asarray(arrays["predicted_separation_profile_m"], dtype=np.float32)
    radius = np.asarray(arrays["occupancy_interaction_radius_m"], dtype=np.float32)[:, None]
    signed_clearance = predicted - radius
    row_cost = np.max(projected_error / np.maximum(np.abs(signed_clearance), floor), axis=1)
    trajectory_cost = np.zeros(len(identities), dtype=np.float32)
    np.maximum.at(trajectory_cost, inverse, row_cost)
    return trajectory_cost, identities[:, 0].astype(np.int32)


@torch.no_grad()
def _predict(
    model: BoundaryStateCostRegressor, raw_features: np.ndarray,
    feature_mean: np.ndarray, feature_scale: np.ndarray,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(raw_features), 65536):
        batch = torch.from_numpy(
            (raw_features[start:start + 65536] - feature_mean) / feature_scale,
        ).cuda()
        outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs)


def _evaluate(
    model: BoundaryStateCostRegressor, arrays: dict[str, np.ndarray],
    probability: np.ndarray, top_k: int, feature_mean: np.ndarray,
    feature_scale: np.ndarray, coverage: float, floor: float,
) -> dict[str, float | int]:
    grouped = _trajectory_tail_features(arrays, probability, top_k)
    raw_features, base_logit = _head_features(grouped)
    actual_cost, scenes = _continuous_cost(arrays, floor)
    if len(actual_cost) != len(raw_features) or not np.array_equal(scenes, grouped["scene_index"]):
        raise RuntimeError("continuous boundary-cost grouping is not aligned with P109 trajectory groups")
    learned = _predict(model.eval(), raw_features, feature_mean, feature_scale)
    clearance = grouped["clearance_score"]
    selected = _select_by_scene(learned, scenes, coverage)
    base_selected = _select_by_scene(base_logit, scenes, coverage)
    clearance_selected = _select_by_scene(clearance, scenes, coverage)
    mean_cost = float(actual_cost.mean())
    learned_cost = float(actual_cost[selected].mean())
    base_cost = float(actual_cost[base_selected].mean())
    clearance_cost = float(actual_cost[clearance_selected].mean())
    learned_spearman = spearman_correlation(actual_cost, learned)
    base_spearman = spearman_correlation(actual_cost, base_logit)
    return {
        "trajectory_count": int(len(actual_cost)),
        "selected_trajectory_count": int(len(selected)),
        "all_mean_boundary_state_cost": mean_cost,
        "learned_selected_mean_boundary_state_cost": learned_cost,
        "p109_selected_mean_boundary_state_cost": base_cost,
        "clearance_selected_mean_boundary_state_cost": clearance_cost,
        "learned_selected_cost_reduction": float((mean_cost - learned_cost) / max(mean_cost, 1e-12)),
        "p109_selected_cost_reduction": float((mean_cost - base_cost) / max(mean_cost, 1e-12)),
        "learned_cost_spearman": learned_spearman,
        "p109_cost_spearman": base_spearman,
        "clearance_cost_spearman": spearman_correlation(actual_cost, clearance),
        "learned_spearman_gain_over_p109": learned_spearman - base_spearman,
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
    torch.manual_seed(int(config["seed"]))
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
    top_k = int(config["model"]["top_k_crossing_probabilities"])
    grouped = _trajectory_tail_features(source_raw, source_probability, top_k)
    raw_features, _ = _head_features(grouped)
    floor = float(config["boundary_state_cost"]["clearance_floor_m"])
    source_cost, source_scenes = _continuous_cost(source_raw, floor)
    if len(source_cost) != len(raw_features) or not np.array_equal(source_scenes, grouped["scene_index"]):
        raise RuntimeError("source continuous cost is not aligned with trajectory features")
    feature_mean = raw_features.mean(0)
    feature_scale = raw_features.std(0).clip(min=1e-4)
    log_cost = np.log1p(source_cost)
    target_mean_log = float(log_cost.mean())
    target_scale_log = float(max(log_cost.std(), 1e-4))
    features = torch.from_numpy((raw_features - feature_mean) / feature_scale).cuda()
    targets = torch.from_numpy((log_cost - target_mean_log) / target_scale_log).cuda()
    model_config = config["model"]
    model = BoundaryStateCostRegressor(
        raw_features.shape[1], model_config["hidden_dimensions"],
    ).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    final_loss = 0.0
    torch.cuda.reset_peak_memory_stats()
    for step in range(int(model_config["steps"])):
        indices = torch.randint(len(features), (int(model_config["batch_size"]),), device="cuda")
        prediction = model(features[indices])
        loss = torch.nn.functional.smooth_l1_loss(prediction, targets[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step % 250 == 0 or step + 1 == int(model_config["steps"]):
            print(f"P120 continuous-boundary-cost step={step + 1} huber={final_loss:.6f}", flush=True)
    torch.save({
        "feature_mean": feature_mean, "feature_scale": feature_scale,
        "target_mean_log": target_mean_log, "target_scale_log": target_scale_log,
        "hidden_dimensions": model_config["hidden_dimensions"],
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
    gains = [float(value["learned_spearman_gain_over_p109"]) for value in results.values()]
    decisions = {
        "no_selected_mean_cost_regression": all(
            value["learned_selected_mean_boundary_state_cost"]
            <= value["p109_selected_mean_boundary_state_cost"] for value in results.values()
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
                     "source_mean_boundary_state_cost": float(source_cost.mean()),
                     "final_huber": final_loss},
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
