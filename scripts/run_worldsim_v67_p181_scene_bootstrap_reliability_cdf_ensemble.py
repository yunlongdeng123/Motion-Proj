"""Marginalize monotone reliability CDFs trained on scene-bootstrap environments."""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
import yaml

from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost
from scripts.run_worldsim_v67_p127_ensemble_continuous_selection import _ensemble_trajectory_score
from scripts.run_worldsim_v67_p166_monotone_expected_cost_calibration import _trajectory_horizon
from scripts.run_worldsim_v67_p173_monotone_visit_reliability_cdf import (
    HorizonOnlyReliabilityCDF,
    MonotoneReliabilityCDF,
    _predict_surface,
)


@torch.no_grad()
def _predict_ensemble(
    models: list[MonotoneReliabilityCDF],
    score: np.ndarray,
    horizon: np.ndarray,
    budgets: np.ndarray,
    norms: tuple[float, float, float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    score_mean, score_scale, horizon_mean, horizon_scale, budget_mean, budget_scale = norms
    trajectory_count = len(score)
    budget_count = len(budgets)
    flat_score = np.repeat(((score - score_mean) / score_scale).astype(np.float32), budget_count)
    flat_horizon = np.repeat(
        ((horizon - horizon_mean) / horizon_scale).astype(np.float32), budget_count,
    )
    flat_budget = np.tile(
        ((np.log1p(budgets) - budget_mean) / budget_scale).astype(np.float32), trajectory_count,
    )
    means, deviations = [], []
    for start in range(0, len(flat_score), 131072):
        end = start + 131072
        score_tensor = torch.from_numpy(flat_score[start:end]).cuda()
        horizon_tensor = torch.from_numpy(flat_horizon[start:end]).cuda()[:, None]
        budget_tensor = torch.from_numpy(flat_budget[start:end]).cuda()
        member_probability = torch.stack([
            torch.sigmoid(model(score_tensor, horizon_tensor, budget_tensor)) for model in models
        ])
        means.append(member_probability.mean(dim=0).cpu().numpy())
        deviations.append(member_probability.std(dim=0, unbiased=False).cpu().numpy())
    shape = (trajectory_count, budget_count)
    return np.concatenate(means).reshape(shape), np.concatenate(deviations).reshape(shape)


def _evaluate(
    arrays: dict[str, np.ndarray],
    actor_models: list[DirectionalActorGaussian],
    actor_ensemble: dict,
    cdf_models: list[MonotoneReliabilityCDF],
    p173_model: MonotoneReliabilityCDF,
    p173_baseline: HorizonOnlyReliabilityCDF,
    frozen_p173: dict,
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
        raise RuntimeError("P181 trajectory grouping is not aligned")
    horizon = _trajectory_horizon(arrays)
    budgets = np.asarray(config["reliability_budgets"], dtype=np.float32)
    ensemble_probability, member_deviation = _predict_ensemble(
        cdf_models, score, horizon, budgets, tuple(frozen_p173["norms"]),
    )
    p173, horizon_only = _predict_surface(
        p173_model, p173_baseline, score, horizon, budgets, tuple(frozen_p173["norms"]),
    )
    target = actual_cost[:, None] <= budgets[None]
    brier = float(np.mean(np.square(ensemble_probability - target)))
    p173_brier = float(np.mean(np.square(p173 - target)))
    horizon_brier = float(np.mean(np.square(horizon_only - target)))
    error = float(np.mean(np.abs(ensemble_probability.mean(axis=0) - target.mean(axis=0))))
    p173_error = float(np.mean(np.abs(p173.mean(axis=0) - target.mean(axis=0))))
    return {
        "row_count": int(len(arrays["features"])),
        "trajectory_count": int(len(actual_cost)),
        "scene_bootstrap_ensemble_integrated_brier": brier,
        "p173_integrated_brier": p173_brier,
        "horizon_only_integrated_brier": horizon_brier,
        "brier_change_vs_p173": float((brier - p173_brier) / max(p173_brier, 1e-12)),
        "scene_bootstrap_ensemble_mean_absolute_reliability_error": error,
        "p173_mean_absolute_reliability_error": p173_error,
        "calibration_error_reduction_vs_p173": float(
            (p173_error - error) / max(p173_error, 1e-12)
        ),
        "mean_member_probability_deviation": float(member_deviation.mean()),
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
    frozen_p173 = torch.load(
        args.runs_root / config["frozen_p173"]["run"] / config["frozen_p173"]["artifact"],
        map_location="cuda",
    )
    p173_model = MonotoneReliabilityCDF(
        list(frozen_p173["score_knots"]), list(frozen_p173["budget_knots"]),
    ).cuda()
    p173_model.load_state_dict(frozen_p173["model_state_dict"])
    p173_model.eval()
    p173_baseline = HorizonOnlyReliabilityCDF(list(frozen_p173["budget_knots"])).cuda()
    p173_baseline.load_state_dict(frozen_p173["baseline_state_dict"])
    p173_baseline.eval()

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
        raise RuntimeError("P181 source grouping is not aligned")
    source_horizon = _trajectory_horizon(source)
    budgets = np.asarray(config["reliability_budgets"], dtype=np.float32)
    norms = tuple(float(value) for value in frozen_p173["norms"])
    score = torch.from_numpy(((source_score - norms[0]) / norms[1]).astype(np.float32)).cuda()
    horizon = torch.from_numpy(
        ((source_horizon - norms[2]) / norms[3]).astype(np.float32)
    ).cuda()[:, None]
    target_cost = torch.from_numpy(source_cost.astype(np.float32)).cuda()
    normalized_budgets = torch.from_numpy(
        ((np.log1p(budgets) - norms[4]) / norms[5]).astype(np.float32)
    ).cuda()
    raw_budgets = torch.from_numpy(budgets).cuda()

    model_config = config["model"]
    member_count = int(model_config["member_count"])
    cdf_models = []
    for member_index in range(member_count):
        torch.manual_seed(int(config["seed"]) + 1009 * member_index)
        cdf_models.append(MonotoneReliabilityCDF(
            [float(value) for value in model_config["score_knots"]],
            [float(value) for value in model_config["budget_knots"]],
        ).cuda())

    unique_scenes = np.unique(source_scenes)
    scene_rows = {scene: np.flatnonzero(source_scenes == scene) for scene in unique_scenes}
    rng = np.random.default_rng(int(config["seed"]))
    member_pools = []
    unique_scene_counts = []
    for _ in range(member_count):
        sampled_scenes = rng.choice(unique_scenes, size=len(unique_scenes), replace=True)
        unique_scene_counts.append(int(len(np.unique(sampled_scenes))))
        member_pools.append(torch.from_numpy(np.concatenate([
            scene_rows[scene] for scene in sampled_scenes
        ]).astype(np.int64)).cuda())

    optimizer = torch.optim.AdamW(
        [parameter for model in cdf_models for parameter in model.parameters()],
        lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"]),
    )
    final_member_losses = [0.0] * member_count
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    for step in range(int(model_config["training_steps"])):
        member_losses = []
        for model, pool in zip(cdf_models, member_pools):
            pool_index = torch.randint(
                len(pool), (int(model_config["batch_size_per_member"]),), device="cuda",
            )
            index = pool[pool_index]
            budget_index = torch.randint(len(budgets), (len(index),), device="cuda")
            target = (target_cost[index] <= raw_budgets[budget_index]).float()
            prediction = model(score[index], horizon[index], normalized_budgets[budget_index])
            member_losses.append(functional.binary_cross_entropy_with_logits(prediction, target))
        loss = torch.stack(member_losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_member_losses = [float(value.detach().cpu()) for value in member_losses]
        if step % 500 == 0 or step + 1 == int(model_config["training_steps"]):
            formatted = "/".join(f"{value:.5f}" for value in final_member_losses)
            print(f"P181 scene-bootstrap ensemble step={step + 1} member_bce={formatted}", flush=True)

    torch.save({
        "member_state_dicts": [model.state_dict() for model in cdf_models],
        "norms": norms, "score_knots": model_config["score_knots"],
        "budget_knots": model_config["budget_knots"],
        "reliability_budgets": config["reliability_budgets"],
    }, run_dir / config["model_artifact"])
    evaluations = {}
    for cohort in config["decision_cohorts"]:
        arrays = dict(np.load(args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False))
        evaluations[cohort["name"]] = _evaluate(
            arrays, actor_models, actor_ensemble, [model.eval() for model in cdf_models],
            p173_model, p173_baseline, frozen_p173, config,
        )
        print(json.dumps({cohort["name"]: evaluations[cohort["name"]]}, indent=2), flush=True)
    calibration_reductions = [
        float(row["calibration_error_reduction_vs_p173"]) for row in evaluations.values()
    ]
    checks = {
        "brier_noninferior_to_p173_every_cohort": all(
            float(row["scene_bootstrap_ensemble_integrated_brier"])
            <= float(row["p173_integrated_brier"]) for row in evaluations.values()
        ),
        "minimum_mean_calibration_error_reduction_vs_p173": float(np.mean(calibration_reductions))
        >= float(config["decision"]["minimum_mean_calibration_error_reduction_vs_p173"]),
    }
    verdict = config["verdict_on_pass"] if all(checks.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "training": {
            "trajectory_count": int(len(source_score)), "source_scene_count": int(len(unique_scenes)),
            "member_count": member_count, "bootstrap_unique_scene_counts": unique_scene_counts,
            "final_member_bce": final_member_losses,
        },
        "consumed_development_evaluations": evaluations,
        "decision_checks": checks,
        "mean_calibration_error_reduction_vs_p173": float(np.mean(calibration_reductions)),
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
