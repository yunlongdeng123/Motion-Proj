"""Ensemble conditional log-cost densities across scene-bootstrap environments."""

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
from scripts.run_worldsim_v67_p173_monotone_visit_reliability_cdf import (
    HorizonOnlyReliabilityCDF,
    MonotoneReliabilityCDF,
)
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
    models: list[LogCostMixtureDensity],
    frozen_p182_model: LogCostMixtureDensity,
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
        raise RuntimeError("P184 trajectory grouping is not aligned")
    horizon = _trajectory_horizon(arrays)
    clearance = _trajectory_clearance(arrays)
    budgets = np.asarray(config["reliability_budgets"], dtype=np.float32)
    member_probability = np.stack([
        _predict_cdf(model, score, horizon, clearance, budgets, tuple(frozen_p182["norms"]))
        for model in models
    ])
    predicted = member_probability.mean(axis=0)
    p182 = _predict_cdf(
        frozen_p182_model, score, horizon, clearance, budgets, tuple(frozen_p182["norms"]),
    )
    target = actual_cost[:, None] <= budgets[None]
    brier = float(np.mean(np.square(predicted - target)))
    p182_brier = float(np.mean(np.square(p182 - target)))
    error = float(np.mean(np.abs(predicted.mean(axis=0) - target.mean(axis=0))))
    p182_error = float(np.mean(np.abs(p182.mean(axis=0) - target.mean(axis=0))))
    return {
        "row_count": int(len(arrays["features"])), "trajectory_count": int(len(actual_cost)),
        "bootstrap_density_ensemble_integrated_brier": brier,
        "p182_integrated_brier": p182_brier,
        "brier_change_vs_p182": float((brier - p182_brier) / max(p182_brier, 1e-12)),
        "bootstrap_density_ensemble_mean_absolute_reliability_error": error,
        "p182_mean_absolute_reliability_error": p182_error,
        "calibration_error_reduction_vs_p182": float(
            (p182_error - error) / max(p182_error, 1e-12)
        ),
        "mean_member_probability_deviation": float(member_probability.std(axis=0).mean()),
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
    frozen_p182_model = _load_density(frozen_p182)

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
        raise RuntimeError("P184 source grouping is not aligned")
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
    member_count = int(model_config["member_count"])
    models = []
    for member_index in range(member_count):
        torch.manual_seed(int(config["seed"]) + 1009 * member_index)
        models.append(LogCostMixtureDensity(
            int(model_config["component_count"]),
            [int(value) for value in model_config["hidden_dimensions"]],
        ).cuda())
    unique_scenes = np.unique(source_scenes)
    scene_rows = {scene: np.flatnonzero(source_scenes == scene) for scene in unique_scenes}
    rng = np.random.default_rng(int(config["seed"]))
    member_pools = []
    unique_scene_counts = []
    for _ in range(member_count):
        sampled = rng.choice(unique_scenes, size=len(unique_scenes), replace=True)
        unique_scene_counts.append(int(len(np.unique(sampled))))
        member_pools.append(torch.from_numpy(np.concatenate([
            scene_rows[scene] for scene in sampled
        ]).astype(np.int64)).cuda())
    optimizer = torch.optim.AdamW(
        [parameter for model in models for parameter in model.parameters()],
        lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"]),
    )
    final_losses = [0.0] * member_count
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    for step in range(int(model_config["training_steps"])):
        losses = []
        for model, pool in zip(models, member_pools):
            pool_index = torch.randint(
                len(pool), (int(model_config["batch_size_per_member"]),), device="cuda",
            )
            index = pool[pool_index]
            logits, means, scales = model(condition[index])
            losses.append(_mixture_nll(logits, means, scales, target[index]))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_losses = [float(value.detach().cpu()) for value in losses]
        if step % 500 == 0 or step + 1 == int(model_config["training_steps"]):
            formatted = "/".join(f"{value:.5f}" for value in final_losses)
            print(f"P184 bootstrap density step={step + 1} member_nll={formatted}", flush=True)

    torch.save({
        "member_state_dicts": [model.state_dict() for model in models], "norms": norms,
        "component_count": model_config["component_count"],
        "hidden_dimensions": model_config["hidden_dimensions"],
        "reliability_budgets": config["reliability_budgets"],
    }, run_dir / config["model_artifact"])
    evaluations = {}
    for cohort in config["decision_cohorts"]:
        arrays = dict(np.load(args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False))
        evaluations[cohort["name"]] = _evaluate(
            arrays, actor_models, actor_ensemble, [model.eval() for model in models],
            frozen_p182_model, frozen_p182, config,
        )
        print(json.dumps({cohort["name"]: evaluations[cohort["name"]]}, indent=2), flush=True)
    calibration_reductions = [
        float(row["calibration_error_reduction_vs_p182"]) for row in evaluations.values()
    ]
    checks = {
        "brier_noninferior_to_p182_every_cohort": all(
            float(row["bootstrap_density_ensemble_integrated_brier"])
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
            "trajectory_count": int(len(source_score)), "source_scene_count": int(len(unique_scenes)),
            "member_count": member_count, "bootstrap_unique_scene_counts": unique_scene_counts,
            "final_member_nll": final_losses,
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
