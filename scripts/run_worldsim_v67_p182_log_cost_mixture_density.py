"""Fit a conditional log-cost density and derive the visited-state reliability CDF."""

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
from scripts.run_worldsim_v67_p178_clearance_conditioned_reliability_cdf import _trajectory_clearance


class LogCostMixtureDensity(nn.Module):
    """Low-capacity Gaussian mixture for log1p continuous boundary-state cost."""

    def __init__(self, component_count: int, hidden_dimensions: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = 3
        for hidden in hidden_dimensions:
            layers.extend((nn.Linear(width, hidden), nn.SiLU()))
            width = hidden
        layers.append(nn.Linear(width, 3 * component_count))
        self.network = nn.Sequential(*layers)
        self.component_count = component_count

    def forward(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.network(condition)
        logits, means, raw_scales = output.chunk(3, dim=1)
        scales = 0.05 + functional.softplus(raw_scales)
        return logits, means, scales


def _mixture_nll(
    logits: torch.Tensor, means: torch.Tensor, scales: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    standardized = (target[:, None] - means) / scales
    log_density = -0.5 * standardized.square() - torch.log(scales) - 0.5 * math.log(2.0 * math.pi)
    return -torch.logsumexp(functional.log_softmax(logits, dim=1) + log_density, dim=1).mean()


@torch.no_grad()
def _predict_cdf(
    model: LogCostMixtureDensity,
    score: np.ndarray,
    horizon: np.ndarray,
    clearance: np.ndarray,
    budgets: np.ndarray,
    norms: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    conditions = np.stack((
        (score - norms[0]) / norms[1],
        (horizon - norms[2]) / norms[3],
        (clearance - norms[4]) / norms[5],
    ), axis=1).astype(np.float32)
    outputs = []
    budget_tensor = torch.from_numpy(np.log1p(budgets).astype(np.float32)).cuda()
    for start in range(0, len(conditions), 131072):
        condition = torch.from_numpy(conditions[start:start + 131072]).cuda()
        logits, means, scales = model(condition)
        standardized = (budget_tensor[None, :, None] - means[:, None]) / scales[:, None]
        component_cdf = 0.5 * (1.0 + torch.erf(standardized / math.sqrt(2.0)))
        probability = torch.sum(
            functional.softmax(logits, dim=1)[:, None] * component_cdf, dim=2,
        )
        outputs.append(probability.cpu().numpy())
    return np.concatenate(outputs)


def _evaluate(
    arrays: dict[str, np.ndarray],
    actor_models: list[DirectionalActorGaussian],
    actor_ensemble: dict,
    density: LogCostMixtureDensity,
    p173_model: MonotoneReliabilityCDF,
    p173_baseline: HorizonOnlyReliabilityCDF,
    frozen_p173: dict,
    norms: tuple[float, float, float, float, float, float],
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
        raise RuntimeError("P182 trajectory grouping is not aligned")
    horizon = _trajectory_horizon(arrays)
    clearance = _trajectory_clearance(arrays)
    budgets = np.asarray(config["reliability_budgets"], dtype=np.float32)
    predicted = _predict_cdf(density, score, horizon, clearance, budgets, norms)
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
        "row_count": int(len(arrays["features"])), "trajectory_count": int(len(actual_cost)),
        "log_cost_mixture_integrated_brier": brier, "p173_integrated_brier": p173_brier,
        "horizon_only_integrated_brier": horizon_brier,
        "brier_change_vs_p173": float((brier - p173_brier) / max(p173_brier, 1e-12)),
        "log_cost_mixture_mean_absolute_reliability_error": error,
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
        raise RuntimeError("P182 source grouping is not aligned")
    source_horizon = _trajectory_horizon(source)
    source_clearance = _trajectory_clearance(source)
    norms = (
        float(source_score.mean()), float(max(source_score.std(), 1e-4)),
        float(source_horizon.mean()), float(max(source_horizon.std(), 1e-4)),
        float(source_clearance.mean()), float(max(source_clearance.std(), 1e-4)),
    )
    condition = torch.from_numpy(np.stack((
        (source_score - norms[0]) / norms[1],
        (source_horizon - norms[2]) / norms[3],
        (source_clearance - norms[4]) / norms[5],
    ), axis=1).astype(np.float32)).cuda()
    target = torch.from_numpy(np.log1p(source_cost).astype(np.float32)).cuda()

    torch.manual_seed(int(config["seed"]))
    model_config = config["model"]
    density = LogCostMixtureDensity(
        int(model_config["component_count"]),
        [int(value) for value in model_config["hidden_dimensions"]],
    ).cuda()
    optimizer = torch.optim.AdamW(
        density.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    final_nll = 0.0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    for step in range(int(model_config["training_steps"])):
        index = torch.randint(len(condition), (int(model_config["batch_size"]),), device="cuda")
        logits, means, scales = density(condition[index])
        loss = _mixture_nll(logits, means, scales, target[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_nll = float(loss.detach().cpu())
        if step % 500 == 0 or step + 1 == int(model_config["training_steps"]):
            print(f"P182 log-cost mixture step={step + 1} nll={final_nll:.6f}", flush=True)

    torch.save({
        "model_state_dict": density.state_dict(), "norms": norms,
        "component_count": model_config["component_count"],
        "hidden_dimensions": model_config["hidden_dimensions"],
        "reliability_budgets": config["reliability_budgets"],
    }, run_dir / config["model_artifact"])
    evaluations = {}
    for cohort in config["decision_cohorts"]:
        arrays = dict(np.load(args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False))
        evaluations[cohort["name"]] = _evaluate(
            arrays, actor_models, actor_ensemble, density.eval(), p173_model,
            p173_baseline, frozen_p173, norms, config,
        )
        print(json.dumps({cohort["name"]: evaluations[cohort["name"]]}, indent=2), flush=True)
    calibration_reductions = [
        float(row["calibration_error_reduction_vs_p173"]) for row in evaluations.values()
    ]
    checks = {
        "brier_noninferior_to_p173_every_cohort": all(
            float(row["log_cost_mixture_integrated_brier"])
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
            "source_scene_count": int(len(np.unique(source_scenes))), "final_log_cost_nll": final_nll,
        },
        "consumed_development_evaluations": evaluations, "decision_checks": checks,
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
