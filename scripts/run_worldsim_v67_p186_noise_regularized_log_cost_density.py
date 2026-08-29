"""Smooth the P182 conditional density with fixed source condition/target noise."""

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

from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost
from scripts.run_worldsim_v67_p127_ensemble_continuous_selection import _ensemble_trajectory_score
from scripts.run_worldsim_v67_p166_monotone_expected_cost_calibration import _trajectory_horizon
from scripts.run_worldsim_v67_p178_clearance_conditioned_reliability_cdf import _trajectory_clearance
from scripts.run_worldsim_v67_p182_log_cost_mixture_density import (
    LogCostMixtureDensity,
    _mixture_nll,
    _predict_cdf,
)


def _load_density(frozen: dict) -> LogCostMixtureDensity:
    model = LogCostMixtureDensity(
        int(frozen["component_count"]), [int(value) for value in frozen["hidden_dimensions"]],
    ).cuda()
    model.load_state_dict(frozen["model_state_dict"])
    return model.eval()


def _evaluate(
    arrays: dict[str, np.ndarray],
    actor_models: list[DirectionalActorGaussian],
    actor_ensemble: dict,
    model: LogCostMixtureDensity,
    p182_model: LogCostMixtureDensity,
    frozen_p182: dict,
    config: dict,
) -> dict:
    score, scenes = _ensemble_trajectory_score(
        arrays, actor_models,
        np.asarray(actor_ensemble["feature_mean"], dtype=np.float32),
        np.asarray(actor_ensemble["feature_scale"], dtype=np.float32),
        np.asarray(actor_ensemble["target_mean"], dtype=np.float32),
        np.asarray(actor_ensemble["target_scale"], dtype=np.float32),
    )
    actual_cost, cost_scenes = _continuous_cost(
        arrays, float(config["boundary_state_cost"]["clearance_floor_m"]),
    )
    if not np.array_equal(scenes, cost_scenes):
        raise RuntimeError("P186 trajectory grouping is not aligned")
    horizon = _trajectory_horizon(arrays)
    clearance = _trajectory_clearance(arrays)
    budgets = np.asarray(config["reliability_budgets"], dtype=np.float32)
    predicted = _predict_cdf(
        model, score, horizon, clearance, budgets, tuple(frozen_p182["norms"]),
    )
    p182 = _predict_cdf(
        p182_model, score, horizon, clearance, budgets, tuple(frozen_p182["norms"]),
    )
    target = actual_cost[:, None] <= budgets[None]
    brier = float(np.mean(np.square(predicted - target)))
    p182_brier = float(np.mean(np.square(p182 - target)))
    error = float(np.mean(np.abs(predicted.mean(axis=0) - target.mean(axis=0))))
    p182_error = float(np.mean(np.abs(p182.mean(axis=0) - target.mean(axis=0))))
    return {
        "row_count": int(len(arrays["features"])), "trajectory_count": int(len(actual_cost)),
        "noise_regularized_integrated_brier": brier, "p182_integrated_brier": p182_brier,
        "brier_change_vs_p182": float((brier - p182_brier) / max(p182_brier, 1e-12)),
        "noise_regularized_mean_absolute_reliability_error": error,
        "p182_mean_absolute_reliability_error": p182_error,
        "calibration_error_reduction_vs_p182": float(
            (p182_error - error) / max(p182_error, 1e-12)
        ),
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

    actor_ensemble = torch.load(
        args.runs_root / config["frozen_p126"]["run"] / config["frozen_p126"]["artifact"],
        map_location="cuda",
    )
    actor_models = []
    for state in actor_ensemble["member_state_dicts"]:
        member = DirectionalActorGaussian(20, actor_ensemble["hidden_dimensions"]).cuda()
        member.load_state_dict(state)
        actor_models.append(member.eval())
    frozen_p182 = torch.load(
        args.runs_root / config["frozen_p182"]["run"] / config["frozen_p182"]["artifact"],
        map_location="cuda",
    )
    p182_model = _load_density(frozen_p182)
    source = dict(np.load(
        args.runs_root / config["source_rows"]["run"] / config["source_rows"]["artifact"],
        allow_pickle=False,
    ))
    source_score, source_scenes = _ensemble_trajectory_score(
        source, actor_models,
        np.asarray(actor_ensemble["feature_mean"], dtype=np.float32),
        np.asarray(actor_ensemble["feature_scale"], dtype=np.float32),
        np.asarray(actor_ensemble["target_mean"], dtype=np.float32),
        np.asarray(actor_ensemble["target_scale"], dtype=np.float32),
    )
    source_cost, cost_scenes = _continuous_cost(
        source, float(config["boundary_state_cost"]["clearance_floor_m"]),
    )
    if not np.array_equal(source_scenes, cost_scenes):
        raise RuntimeError("P186 source grouping is not aligned")
    source_horizon = _trajectory_horizon(source)
    source_clearance = _trajectory_clearance(source)
    norms = tuple(float(value) for value in frozen_p182["norms"])
    condition = torch.from_numpy(np.stack((
        (source_score - norms[0]) / norms[1],
        (source_horizon - norms[2]) / norms[3],
        (source_clearance - norms[4]) / norms[5],
    ), axis=1).astype(np.float32)).cuda()
    target = torch.from_numpy(np.log1p(source_cost).astype(np.float32)).cuda()

    model_config = config["model"]
    torch.manual_seed(int(config["seed"]))
    model = LogCostMixtureDensity(
        int(model_config["component_count"]),
        [int(value) for value in model_config["hidden_dimensions"]],
    ).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    condition_noise = float(model_config["condition_noise_standard_deviation"])
    target_noise = float(model_config["log_cost_noise_standard_deviation"])
    final_nll = 0.0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    for step in range(int(model_config["training_steps"])):
        index = torch.randint(len(condition), (int(model_config["batch_size"]),), device="cuda")
        noisy_condition = condition[index] + condition_noise * torch.randn_like(condition[index])
        noisy_target = target[index] + target_noise * torch.randn_like(target[index])
        logits, means, scales = model(noisy_condition)
        loss = _mixture_nll(logits, means, scales, noisy_target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_nll = float(loss.detach().cpu())
        if step % 500 == 0 or step + 1 == int(model_config["training_steps"]):
            print(f"P186 noise-regularized density step={step + 1} nll={final_nll:.6f}", flush=True)

    torch.save({
        "model_state_dict": model.state_dict(), "norms": norms,
        "component_count": model_config["component_count"],
        "hidden_dimensions": model_config["hidden_dimensions"],
        "reliability_budgets": config["reliability_budgets"],
    }, run_dir / config["model_artifact"])
    evaluations = {}
    for cohort in config["decision_cohorts"]:
        arrays = dict(np.load(args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False))
        evaluations[cohort["name"]] = _evaluate(
            arrays, actor_models, actor_ensemble, model.eval(), p182_model, frozen_p182, config,
        )
        print(json.dumps({cohort["name"]: evaluations[cohort["name"]]}, indent=2), flush=True)
    calibration_reductions = [
        float(row["calibration_error_reduction_vs_p182"]) for row in evaluations.values()
    ]
    checks = {
        "brier_noninferior_to_p182_every_cohort": all(
            float(row["noise_regularized_integrated_brier"])
            <= float(row["p182_integrated_brier"]) for row in evaluations.values()
        ),
        "minimum_mean_calibration_error_reduction_vs_p182": float(np.mean(calibration_reductions))
        >= float(config["decision"]["minimum_mean_calibration_error_reduction_vs_p182"]),
    }
    verdict = config["verdict_on_pass"] if all(checks.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "training": {
            "trajectory_count": int(len(source_score)),
            "source_scene_count": int(len(np.unique(source_scenes))), "final_nll": final_nll,
        },
        "consumed_development_evaluations": evaluations, "decision_checks": checks,
        "mean_calibration_error_reduction_vs_p182": float(np.mean(calibration_reductions)),
        "resources": {
            "gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "wall_seconds": time.monotonic() - started,
        },
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    (run_dir / "status.json").write_text(json.dumps({
        "status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict}, indent=2), flush=True)


if __name__ == "__main__":
    main()
