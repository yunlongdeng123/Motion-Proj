"""Compile frozen P165 joint samples with an upper-tail coherent risk mean."""

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
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost
from scripts.run_worldsim_v67_p127_ensemble_continuous_selection import _ensemble_trajectory_score
from scripts.run_worldsim_v67_p158_crps_actor_ensemble import _subset
from scripts.run_worldsim_v67_p165_joint_multi_actor_diffusion import (
    JointActorDiffusion, _base_actor_distribution, _pack_groups, _sample_joint_innovation,
)


def _tail_mean_score(
    arrays: dict[str, np.ndarray], actor: dict[str, np.ndarray], packed: dict[str, np.ndarray],
    model: JointActorDiffusion, condition_norm: tuple[np.ndarray, np.ndarray],
    alpha_bar: torch.Tensor, config: dict,
) -> tuple[np.ndarray, np.ndarray, float]:
    sampling = config["sampling"]
    innovation = _sample_joint_innovation(
        model, packed["condition"], packed["mask"], *condition_norm, alpha_bar,
        int(sampling["sample_count"]), int(sampling["inference_steps"]),
        int(sampling["group_batch_size"]), int(sampling["seed"]),
    )
    residual = packed["base_mean"][None] + packed["base_scale"][None] * innovation
    residual = residual.reshape(*residual.shape[:3], -1, 2)
    row_actor = actor["row_to_actor"]
    row_residual = residual[
        :, packed["actor_to_group"][row_actor], packed["actor_slot"][row_actor], :, :,
    ]
    normal = np.asarray(arrays["query_boundary_normal_ego_profile"], dtype=np.float32)
    projected_error = np.abs(np.sum(normal[None] * row_residual, axis=3))
    signed = np.asarray(arrays["predicted_separation_profile_m"], dtype=np.float32) - np.asarray(
        arrays["occupancy_interaction_radius_m"], dtype=np.float32,
    )[:, None]
    floor = float(config["boundary_state_cost"]["clearance_floor_m"])
    row_cost = np.max(projected_error / np.maximum(np.abs(signed), floor)[None], axis=2)
    keys = np.stack((
        arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["query_id"],
    ), axis=1)
    identities, inverse = np.unique(keys, axis=0, return_inverse=True)
    samples = np.zeros((len(row_cost), len(identities)), dtype=np.float32)
    for sample_index in range(len(row_cost)):
        np.maximum.at(samples[sample_index], inverse, row_cost[sample_index])
    alpha = float(sampling["tail_probability_level"])
    tail_count = max(1, int(np.ceil((1.0 - alpha) * len(samples))))
    tail = np.sort(samples, axis=0)[-tail_count:]
    return tail.mean(axis=0).astype(np.float32), identities[:, 0].astype(np.int32), float(tail.mean())


def _evaluate(
    arrays: dict[str, np.ndarray], base_models: list[DirectionalActorGaussian], frozen: dict,
    model: JointActorDiffusion, condition_norm: tuple[np.ndarray, np.ndarray],
    alpha_bar: torch.Tensor, config: dict,
) -> dict[str, float | int]:
    feature_mean = np.asarray(frozen["feature_mean"], dtype=np.float32)
    feature_scale = np.asarray(frozen["feature_scale"], dtype=np.float32)
    target_mean = np.asarray(frozen["target_mean"], dtype=np.float32)
    target_scale = np.asarray(frozen["target_scale"], dtype=np.float32)
    actor = _base_actor_distribution(
        arrays, base_models, feature_mean, feature_scale, target_mean, target_scale,
    )
    packed = _pack_groups(actor, int(config["model"]["maximum_actors_per_group"]))
    score, scenes, mean_tail_cost = _tail_mean_score(
        arrays, actor, packed, model, condition_norm, alpha_bar, config,
    )
    p126_score, p126_scenes = _ensemble_trajectory_score(
        arrays, base_models, feature_mean, feature_scale, target_mean, target_scale,
    )
    actual_cost, cost_scenes = _continuous_cost(
        arrays, float(config["boundary_state_cost"]["clearance_floor_m"]),
    )
    if not (np.array_equal(scenes, p126_scenes) and np.array_equal(scenes, cost_scenes)):
        raise RuntimeError("P168 trajectory grouping is not aligned")
    coverage = float(config["selection"]["coverage_fraction"])
    selected = _select_by_scene(score, scenes, coverage)
    p126_selected = _select_by_scene(p126_score, scenes, coverage)
    rank = spearman_correlation(actual_cost, score)
    p126_rank = spearman_correlation(actual_cost, p126_score)
    cost = float(actual_cost[selected].mean())
    p126_cost = float(actual_cost[p126_selected].mean())
    return {
        "row_count": int(len(arrays["features"])), "actor_group_count": int(len(packed["group_keys"])),
        "trajectory_count": int(len(actual_cost)), "selected_trajectory_count": int(len(selected)),
        "tail_mean_selected_cost": cost, "p126_selected_mean_cost": p126_cost,
        "tail_mean_minus_p126_selected_cost": cost - p126_cost,
        "tail_mean_cost_spearman": rank, "p126_cost_spearman": p126_rank,
        "spearman_gain_over_p126": rank - p126_rank, "mean_upper_tail_sample_cost": mean_tail_cost,
    }


def _decisions(results: dict[str, dict[str, float | int]], threshold: float) -> dict[str, bool]:
    return {
        "no_selected_cost_regression": all(
            float(row["tail_mean_minus_p126_selected_cost"]) <= 0.0 for row in results.values()
        ),
        "minimum_mean_spearman_gain": float(np.mean([
            float(row["spearman_gain_over_p126"]) for row in results.values()
        ])) >= threshold,
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
    frozen = torch.load(
        args.runs_root / config["frozen_p126"]["run"] / config["frozen_p126"]["artifact"],
        map_location="cuda",
    )
    base_models = []
    for state in frozen["member_state_dicts"]:
        base = DirectionalActorGaussian(20, frozen["hidden_dimensions"]).cuda()
        base.load_state_dict(state)
        base_models.append(base.eval())
    checkpoint = torch.load(
        args.runs_root / config["frozen_p165"]["run"] / config["frozen_p165"]["artifact"],
        map_location="cuda",
    )
    model_config = config["model"]
    model = JointActorDiffusion(
        int(model_config["state_dimension"]), int(model_config["condition_dimension"]),
        int(model_config["hidden_dimension"]), int(model_config["layer_count"]),
        int(model_config["head_count"]), len(checkpoint["beta"]),
    ).cuda()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    condition_norm = (
        np.asarray(checkpoint["condition_mean"], dtype=np.float32),
        np.asarray(checkpoint["condition_scale"], dtype=np.float32),
    )
    beta = torch.from_numpy(np.asarray(checkpoint["beta"], dtype=np.float32)).cuda()
    alpha_bar = torch.cumprod(1.0 - beta, dim=0)
    torch.cuda.reset_peak_memory_stats()
    development = {}
    for cohort in config["decision_cohorts"]:
        arrays = dict(np.load(args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False))
        development[cohort["name"]] = _evaluate(
            arrays, base_models, frozen, model, condition_norm, alpha_bar, config,
        )
        print(json.dumps({cohort["name"]: development[cohort["name"]]}, indent=2), flush=True)
    threshold = float(config["decision"]["minimum_mean_spearman_gain_over_p126"])
    development_decisions = _decisions(development, threshold)
    (run_dir / "development.json").write_text(json.dumps({
        "evaluations": development, "decision_checks": development_decisions,
    }, indent=2) + "\n", encoding="utf-8")
    prospective = {}
    prospective_decisions: dict[str, bool] = {}
    if all(development_decisions.values()):
        spec = config["prospective_p167"]
        rows_path = args.runs_root / spec["run"] / spec["artifact"]
        deadline = time.monotonic() + float(spec["readiness_timeout_seconds"])
        while not rows_path.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"P168 prospective rows not ready: {rows_path}")
            time.sleep(5.0)
        arrays = dict(np.load(rows_path, allow_pickle=False))
        for horizon in spec["horizons_seconds"]:
            key = str(float(horizon))
            prospective[key] = _evaluate(
                _subset(arrays, float(horizon)), base_models, frozen, model, condition_norm, alpha_bar, config,
            )
            print(json.dumps({f"P167_H{key}": prospective[key]}, indent=2), flush=True)
        prospective_decisions = _decisions(prospective, threshold)
    passed = all(development_decisions.values()) and bool(prospective) and all(prospective_decisions.values())
    verdict = config["verdict_on_pass"] if passed else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"], "consumed_development_evaluations": development,
        "development_decision_checks": development_decisions,
        "prospective_p167_evaluations": prospective,
        "prospective_decision_checks": prospective_decisions,
        "resources": {
            "gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "wall_seconds": time.monotonic() - started,
        },
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({
        "status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict}, indent=2), flush=True)


if __name__ == "__main__":
    main()
