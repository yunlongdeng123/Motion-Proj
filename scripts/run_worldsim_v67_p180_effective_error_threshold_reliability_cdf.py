"""Learn visited-state reliability through the physical budget-times-clearance threshold."""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
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


class EffectiveErrorThresholdCDF(nn.Module):
    """CDF anti-monotone in uncertainty and monotone in budget times minimum clearance."""

    def __init__(self, score_knots: list[float], threshold_knots: list[float]) -> None:
        super().__init__()
        self.register_buffer("score_knots", torch.tensor(score_knots, dtype=torch.float32))
        self.register_buffer("threshold_knots", torch.tensor(threshold_knots, dtype=torch.float32))
        parameter_count = 2 + len(score_knots) + 1 + len(threshold_knots)
        self.horizon_parameters = nn.Sequential(
            nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, parameter_count),
        )

    def forward(
        self, score: torch.Tensor, horizon: torch.Tensor, threshold: torch.Tensor,
    ) -> torch.Tensor:
        parameters = self.horizon_parameters(horizon)
        cursor = 0
        base = parameters[:, cursor]
        cursor += 1
        score_slope = functional.softplus(parameters[:, cursor])
        cursor += 1
        score_increments = functional.softplus(
            parameters[:, cursor:cursor + len(self.score_knots)]
        )
        cursor += len(self.score_knots)
        threshold_slope = functional.softplus(parameters[:, cursor])
        cursor += 1
        threshold_increments = functional.softplus(parameters[:, cursor:])
        score_risk = score_slope * score + torch.sum(
            score_increments * functional.relu(score[:, None] - self.score_knots[None]), dim=1,
        )
        threshold_relief = threshold_slope * threshold + torch.sum(
            threshold_increments
            * functional.relu(threshold[:, None] - self.threshold_knots[None]), dim=1,
        )
        return base - score_risk + threshold_relief


def _trajectory_minimum_clearance(
    arrays: dict[str, np.ndarray], floor: float,
) -> np.ndarray:
    keys = np.stack((
        arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["query_id"],
    ), axis=1)
    identities, inverse = np.unique(keys, axis=0, return_inverse=True)
    predicted = np.asarray(arrays["predicted_separation_profile_m"], dtype=np.float32)
    radius = np.asarray(arrays["occupancy_interaction_radius_m"], dtype=np.float32)[:, None]
    row_clearance = np.min(np.maximum(np.abs(predicted - radius), floor), axis=1)
    clearance = np.full(len(identities), np.inf, dtype=np.float32)
    np.minimum.at(clearance, inverse, row_clearance)
    return clearance


@torch.no_grad()
def _predict_effective(
    model: EffectiveErrorThresholdCDF,
    score: np.ndarray,
    clearance: np.ndarray,
    horizon: np.ndarray,
    budgets: np.ndarray,
    norms: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    score_mean, score_scale, horizon_mean, horizon_scale, threshold_mean, threshold_scale = norms
    trajectory_count = len(score)
    budget_count = len(budgets)
    flat_score = np.repeat(((score - score_mean) / score_scale).astype(np.float32), budget_count)
    flat_horizon = np.repeat(
        ((horizon - horizon_mean) / horizon_scale).astype(np.float32), budget_count,
    )
    raw_threshold = np.log1p(clearance[:, None] * budgets[None]).astype(np.float32)
    flat_threshold = ((raw_threshold - threshold_mean) / threshold_scale).reshape(-1)
    outputs = []
    for start in range(0, len(flat_score), 131072):
        end = start + 131072
        outputs.append(torch.sigmoid(model(
            torch.from_numpy(flat_score[start:end]).cuda(),
            torch.from_numpy(flat_horizon[start:end]).cuda()[:, None],
            torch.from_numpy(flat_threshold[start:end]).cuda(),
        )).cpu().numpy())
    return np.concatenate(outputs).reshape(trajectory_count, budget_count)


def _evaluate(
    arrays: dict[str, np.ndarray],
    actor_models: list[DirectionalActorGaussian],
    ensemble: dict,
    model: EffectiveErrorThresholdCDF,
    p173_model: MonotoneReliabilityCDF,
    p173_baseline: HorizonOnlyReliabilityCDF,
    frozen_p173: dict,
    norms: tuple[float, float, float, float, float, float],
    config: dict,
) -> dict:
    score, scenes = _ensemble_trajectory_score(
        arrays, actor_models,
        np.asarray(ensemble["feature_mean"], dtype=np.float32),
        np.asarray(ensemble["feature_scale"], dtype=np.float32),
        np.asarray(ensemble["target_mean"], dtype=np.float32),
        np.asarray(ensemble["target_scale"], dtype=np.float32),
    )
    floor = float(config["boundary_state_cost"]["clearance_floor_m"])
    actual_cost, cost_scenes = _continuous_cost(arrays, floor)
    if not np.array_equal(scenes, cost_scenes):
        raise RuntimeError("P180 trajectory grouping is not aligned")
    clearance = _trajectory_minimum_clearance(arrays, floor)
    horizon = _trajectory_horizon(arrays)
    budgets = np.asarray(config["reliability_budgets"], dtype=np.float32)
    predicted = _predict_effective(model, score, clearance, horizon, budgets, norms)
    p173, horizon_only = _predict_surface(
        p173_model, p173_baseline, score, horizon, budgets, tuple(frozen_p173["norms"]),
    )
    target = actual_cost[:, None] <= budgets[None]
    brier = float(np.mean(np.square(predicted - target)))
    p173_brier = float(np.mean(np.square(p173 - target)))
    horizon_brier = float(np.mean(np.square(horizon_only - target)))
    error = float(np.mean(np.abs(predicted.mean(axis=0) - target.mean(axis=0))))
    p173_error = float(np.mean(np.abs(p173.mean(axis=0) - target.mean(axis=0))))
    return {
        "row_count": int(len(arrays["features"])),
        "trajectory_count": int(len(actual_cost)),
        "effective_threshold_integrated_brier": brier,
        "p173_integrated_brier": p173_brier,
        "horizon_only_integrated_brier": horizon_brier,
        "brier_change_vs_p173": float((brier - p173_brier) / max(p173_brier, 1e-12)),
        "effective_threshold_mean_absolute_reliability_error": error,
        "p173_mean_absolute_reliability_error": p173_error,
        "calibration_error_reduction_vs_p173": float(
            (p173_error - error) / max(p173_error, 1e-12)
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

    ensemble = torch.load(
        args.runs_root / config["frozen_p126"]["run"] / config["frozen_p126"]["artifact"],
        map_location="cuda",
    )
    actor_models = []
    for state in ensemble["member_state_dicts"]:
        member = DirectionalActorGaussian(20, ensemble["hidden_dimensions"]).cuda()
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
        np.asarray(ensemble["feature_mean"], dtype=np.float32),
        np.asarray(ensemble["feature_scale"], dtype=np.float32),
        np.asarray(ensemble["target_mean"], dtype=np.float32),
        np.asarray(ensemble["target_scale"], dtype=np.float32),
    )
    floor = float(config["boundary_state_cost"]["clearance_floor_m"])
    source_cost, cost_scenes = _continuous_cost(source, floor)
    if not np.array_equal(source_scenes, cost_scenes):
        raise RuntimeError("P180 source grouping is not aligned")
    source_clearance = _trajectory_minimum_clearance(source, floor)
    source_horizon = _trajectory_horizon(source)
    budgets = np.asarray(config["reliability_budgets"], dtype=np.float32)
    source_thresholds = np.log1p(source_clearance[:, None] * budgets[None])
    norms = (
        float(source_score.mean()), float(max(source_score.std(), 1e-4)),
        float(source_horizon.mean()), float(max(source_horizon.std(), 1e-4)),
        float(source_thresholds.mean()), float(max(source_thresholds.std(), 1e-4)),
    )
    score = torch.from_numpy(((source_score - norms[0]) / norms[1]).astype(np.float32)).cuda()
    horizon = torch.from_numpy(
        ((source_horizon - norms[2]) / norms[3]).astype(np.float32)
    ).cuda()[:, None]
    clearance = torch.from_numpy(source_clearance.astype(np.float32)).cuda()
    target_cost = torch.from_numpy(source_cost.astype(np.float32)).cuda()
    raw_budgets = torch.from_numpy(budgets).cuda()

    torch.manual_seed(int(config["seed"]))
    model_config = config["model"]
    model = EffectiveErrorThresholdCDF(
        [float(value) for value in model_config["score_knots"]],
        [float(value) for value in model_config["threshold_knots"]],
    ).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    final_loss = 0.0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    for step in range(int(model_config["training_steps"])):
        index = torch.randint(len(score), (int(model_config["batch_size"]),), device="cuda")
        budget_index = torch.randint(len(budgets), (len(index),), device="cuda")
        threshold = torch.log1p(clearance[index] * raw_budgets[budget_index])
        threshold = (threshold - norms[4]) / norms[5]
        target = (target_cost[index] <= raw_budgets[budget_index]).float()
        prediction = model(score[index], horizon[index], threshold)
        loss = functional.binary_cross_entropy_with_logits(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step % 1000 == 0 or step + 1 == int(model_config["training_steps"]):
            print(f"P180 effective-threshold-CDF step={step + 1} bce={final_loss:.6f}", flush=True)

    torch.save({
        "model_state_dict": model.state_dict(), "norms": norms,
        "score_knots": model_config["score_knots"],
        "threshold_knots": model_config["threshold_knots"],
        "reliability_budgets": config["reliability_budgets"],
    }, run_dir / config["model_artifact"])
    evaluations = {}
    for cohort in config["decision_cohorts"]:
        arrays = dict(np.load(args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False))
        evaluations[cohort["name"]] = _evaluate(
            arrays, actor_models, ensemble, model.eval(), p173_model, p173_baseline,
            frozen_p173, norms, config,
        )
        print(json.dumps({cohort["name"]: evaluations[cohort["name"]]}, indent=2), flush=True)
    calibration_reductions = [
        float(row["calibration_error_reduction_vs_p173"]) for row in evaluations.values()
    ]
    checks = {
        "brier_noninferior_to_p173_every_cohort": all(
            float(row["effective_threshold_integrated_brier"])
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
            "trajectory_count": int(len(source_score)),
            "source_scene_count": int(len(np.unique(source_scenes))),
            "final_reliability_bce": final_loss,
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
