"""Train three-member Actor experts independently for each source horizon."""

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
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost
from scripts.run_worldsim_v67_p127_ensemble_continuous_selection import _ensemble_trajectory_score


def _entries(
    arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keys = np.stack((
        arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["actor_id"],
    ), axis=1)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    actor = np.asarray(arrays["features"], dtype=np.float32)[first, :len(ACTOR_FEATURE_NAMES)]
    residual = np.asarray(arrays["actor_position_error_vector_ego_profile_m"], dtype=np.float32)[first]
    horizons = np.asarray(arrays["horizon_seconds"], dtype=np.float32)[first]
    fractions = np.linspace(0.0, 1.0, residual.shape[1], dtype=np.float32)
    features = np.concatenate((
        np.broadcast_to(actor[:, None, :], (len(actor), residual.shape[1], actor.shape[1])),
        np.broadcast_to(fractions[None, :, None], (len(actor), residual.shape[1], 1)),
    ), axis=2)
    return features.reshape(-1, features.shape[-1]), residual.reshape(-1, 2), inverse, np.repeat(horizons, residual.shape[1])


def _route_horizon(target: float, source_horizons: list[float]) -> float:
    eligible = [value for value in source_horizons if value <= target + 1e-6]
    return max(eligible) if eligible else min(source_horizons)


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
    raw_features, raw_target, _, token_horizons = _entries(source)
    model_config = config["model"]
    source_horizons = [float(value) for value in config["source_horizons_seconds"]]
    experts = {}
    final_losses = {}
    torch.cuda.reset_peak_memory_stats()
    for horizon in source_horizons:
        mask = np.isclose(token_horizons, horizon)
        horizon_features = raw_features[mask]
        horizon_target = raw_target[mask]
        feature_mean = horizon_features.mean(0)
        feature_scale = horizon_features.std(0).clip(min=1e-4)
        target_mean = horizon_target.mean(0)
        target_scale = horizon_target.std(0).clip(min=0.05)
        features = torch.from_numpy((horizon_features - feature_mean) / feature_scale).cuda()
        targets = torch.from_numpy((horizon_target - target_mean) / target_scale).cuda()
        models = []
        horizon_losses = {}
        for seed_value in config["member_seeds"]:
            seed_value = int(seed_value)
            torch.manual_seed(seed_value + int(round(horizon * 100)))
            model = DirectionalActorGaussian(features.shape[1], model_config["hidden_dimensions"]).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"]))
            final_loss = 0.0
            for step in range(int(model_config["steps_per_member"])):
                index = torch.randint(len(features), (int(model_config["batch_size"]),), device="cuda")
                mean, scale = model(features[index])
                normalized = (targets[index] - mean) / scale
                loss = (0.5 * normalized.square() + torch.log(scale)).sum(1).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                final_loss = float(loss.detach().cpu())
                if step % 1000 == 0 or step + 1 == int(model_config["steps_per_member"]):
                    print(f"P157 H={horizon} seed={seed_value} step={step + 1} nll={final_loss:.6f}", flush=True)
            horizon_losses[str(seed_value)] = final_loss
            models.append(model.eval())
        key = str(horizon)
        final_losses[key] = horizon_losses
        experts[key] = {
            "feature_mean": feature_mean, "feature_scale": feature_scale,
            "target_mean": target_mean, "target_scale": target_scale,
            "models": models, "token_count": int(len(features)),
        }
    torch.save({
        "hidden_dimensions": model_config["hidden_dimensions"],
        "source_horizons_seconds": source_horizons,
        "experts": {
            key: {name: value for name, value in expert.items() if name != "models"}
            | {"member_state_dicts": [model.state_dict() for model in expert["models"]]}
            for key, expert in experts.items()
        },
    }, run_dir / config["model_artifact"])
    floor = float(config["boundary_state_cost"]["clearance_floor_m"])
    coverage = float(config["selection"]["coverage_fraction"])
    results = {}
    for cohort in config["development_cohorts"]:
        arrays = dict(np.load(args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False))
        unique_horizons = np.unique(np.asarray(arrays["horizon_seconds"], dtype=np.float32))
        if len(unique_horizons) != 1:
            raise RuntimeError("P157 development cohort must contain one horizon")
        requested = float(unique_horizons[0])
        routed = _route_horizon(requested, source_horizons)
        expert = experts[str(routed)]
        score, scenes = _ensemble_trajectory_score(
            arrays, expert["models"], expert["feature_mean"], expert["feature_scale"],
            expert["target_mean"], expert["target_scale"],
        )
        actual_cost, cost_scenes = _continuous_cost(arrays, floor)
        if not np.array_equal(scenes, cost_scenes):
            raise RuntimeError("P157 trajectory grouping is not aligned")
        selected = _select_by_scene(score, scenes, coverage)
        reference = config["frozen_p126_comparisons"][cohort["name"]]
        model_spearman = spearman_correlation(actual_cost, score)
        results[cohort["name"]] = {
            "requested_horizon_seconds": requested, "routed_expert_horizon_seconds": routed,
            "trajectory_count": int(len(actual_cost)), "selected_trajectory_count": int(len(selected)),
            "specialist_selected_mean_cost": float(actual_cost[selected].mean()),
            "p126_selected_mean_cost": float(reference["selected_cost"]),
            "specialist_cost_spearman": model_spearman, "p126_cost_spearman": float(reference["spearman"]),
            "spearman_gain_over_p126": float(model_spearman - float(reference["spearman"])),
        }
        print(json.dumps({cohort["name"]: results[cohort["name"]]}, indent=2), flush=True)
    gains = [row["spearman_gain_over_p126"] for row in results.values()]
    decisions = {
        "no_selected_cost_regression": all(row["specialist_selected_mean_cost"] <= row["p126_selected_mean_cost"] for row in results.values()),
        "minimum_mean_spearman_gain": float(np.mean(gains)) >= float(config["decision"]["minimum_mean_spearman_gain"]),
    }
    verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "training": {"expert_count": len(experts), "members_per_expert": len(config["member_seeds"]),
                     "tokens_per_expert": {key: expert["token_count"] for key, expert in experts.items()},
                     "member_final_nll": final_losses},
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
