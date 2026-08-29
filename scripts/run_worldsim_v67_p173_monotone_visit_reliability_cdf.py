"""Learn a monotone reliability CDF for actor states visited by an ego trajectory."""

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


class MonotoneReliabilityCDF(nn.Module):
    """CDF that decreases with uncertainty score and increases with cost budget."""

    def __init__(self, score_knots: list[float], budget_knots: list[float]) -> None:
        super().__init__()
        self.register_buffer("score_knots", torch.tensor(score_knots, dtype=torch.float32))
        self.register_buffer("budget_knots", torch.tensor(budget_knots, dtype=torch.float32))
        parameter_count = 2 + len(score_knots) + 1 + len(budget_knots)
        self.horizon_parameters = nn.Sequential(
            nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, parameter_count),
        )

    def forward(
        self, score: torch.Tensor, horizon: torch.Tensor, budget: torch.Tensor,
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
        budget_slope = functional.softplus(parameters[:, cursor])
        cursor += 1
        budget_increments = functional.softplus(parameters[:, cursor:])
        score_risk = score_slope * score + torch.sum(
            score_increments * functional.relu(score[:, None] - self.score_knots[None]), dim=1,
        )
        budget_relief = budget_slope * budget + torch.sum(
            budget_increments * functional.relu(budget[:, None] - self.budget_knots[None]), dim=1,
        )
        return base - score_risk + budget_relief


class HorizonOnlyReliabilityCDF(nn.Module):
    """Matched monotone CDF control without the P126 trajectory score."""

    def __init__(self, budget_knots: list[float]) -> None:
        super().__init__()
        self.register_buffer("budget_knots", torch.tensor(budget_knots, dtype=torch.float32))
        self.horizon_parameters = nn.Sequential(
            nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 2 + len(budget_knots)),
        )

    def forward(self, horizon: torch.Tensor, budget: torch.Tensor) -> torch.Tensor:
        parameters = self.horizon_parameters(horizon)
        base = parameters[:, 0]
        slope = functional.softplus(parameters[:, 1])
        increments = functional.softplus(parameters[:, 2:])
        budget_relief = slope * budget + torch.sum(
            increments * functional.relu(budget[:, None] - self.budget_knots[None]), dim=1,
        )
        return base + budget_relief


@torch.no_grad()
def _predict_surface(
    model: MonotoneReliabilityCDF,
    baseline: HorizonOnlyReliabilityCDF,
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
    outputs, controls = [], []
    for start in range(0, len(flat_score), 131072):
        end = start + 131072
        score_tensor = torch.from_numpy(flat_score[start:end]).cuda()
        horizon_tensor = torch.from_numpy(flat_horizon[start:end]).cuda()[:, None]
        budget_tensor = torch.from_numpy(flat_budget[start:end]).cuda()
        outputs.append(torch.sigmoid(model(score_tensor, horizon_tensor, budget_tensor)).cpu().numpy())
        controls.append(torch.sigmoid(baseline(horizon_tensor, budget_tensor)).cpu().numpy())
    shape = (trajectory_count, budget_count)
    return np.concatenate(outputs).reshape(shape), np.concatenate(controls).reshape(shape)


def _evaluate(
    arrays: dict[str, np.ndarray],
    models: list[DirectionalActorGaussian],
    frozen: dict,
    model: MonotoneReliabilityCDF,
    baseline: HorizonOnlyReliabilityCDF,
    norms: tuple[float, float, float, float, float, float],
    config: dict,
) -> dict:
    score, scenes = _ensemble_trajectory_score(
        arrays, models,
        np.asarray(frozen["feature_mean"], dtype=np.float32),
        np.asarray(frozen["feature_scale"], dtype=np.float32),
        np.asarray(frozen["target_mean"], dtype=np.float32),
        np.asarray(frozen["target_scale"], dtype=np.float32),
    )
    actual_cost, cost_scenes = _continuous_cost(
        arrays, float(config["boundary_state_cost"]["clearance_floor_m"]),
    )
    if not np.array_equal(scenes, cost_scenes):
        raise RuntimeError("P173 trajectory grouping is not aligned")
    horizon = _trajectory_horizon(arrays)
    budgets = np.asarray(config["reliability_budgets"], dtype=np.float32)
    predicted, control = _predict_surface(model, baseline, score, horizon, budgets, norms)
    target = actual_cost[:, None] <= budgets[None]
    model_brier = np.mean(np.square(predicted - target), axis=0)
    control_brier = np.mean(np.square(control - target), axis=0)
    integrated = float(model_brier.mean())
    integrated_control = float(control_brier.mean())
    return {
        "row_count": int(len(arrays["features"])),
        "trajectory_count": int(len(actual_cost)),
        "integrated_brier": integrated,
        "horizon_only_integrated_brier": integrated_control,
        "integrated_brier_reduction": float(
            (integrated_control - integrated) / max(integrated_control, 1e-12)
        ),
        "mean_absolute_reliability_error": float(
            np.mean(np.abs(predicted.mean(axis=0) - target.mean(axis=0)))
        ),
        "horizon_only_mean_absolute_reliability_error": float(
            np.mean(np.abs(control.mean(axis=0) - target.mean(axis=0)))
        ),
        "per_budget": {
            str(float(budget)): {
                "empirical_reliability": float(target[:, index].mean()),
                "predicted_reliability": float(predicted[:, index].mean()),
                "horizon_only_predicted_reliability": float(control[:, index].mean()),
                "brier": float(model_brier[index]),
                "horizon_only_brier": float(control_brier[index]),
            }
            for index, budget in enumerate(budgets)
        },
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
    models = []
    for state in frozen["member_state_dicts"]:
        member = DirectionalActorGaussian(20, frozen["hidden_dimensions"]).cuda()
        member.load_state_dict(state)
        models.append(member.eval())
    source = dict(np.load(
        args.runs_root / config["source_rows"]["run"] / config["source_rows"]["artifact"],
        allow_pickle=False,
    ))
    source_score, source_scenes = _ensemble_trajectory_score(
        source, models,
        np.asarray(frozen["feature_mean"], dtype=np.float32),
        np.asarray(frozen["feature_scale"], dtype=np.float32),
        np.asarray(frozen["target_mean"], dtype=np.float32),
        np.asarray(frozen["target_scale"], dtype=np.float32),
    )
    source_cost, cost_scenes = _continuous_cost(
        source, float(config["boundary_state_cost"]["clearance_floor_m"]),
    )
    if not np.array_equal(source_scenes, cost_scenes):
        raise RuntimeError("P173 source grouping is not aligned")
    source_horizon = _trajectory_horizon(source)
    budgets = np.asarray(config["reliability_budgets"], dtype=np.float32)
    log_budgets = np.log1p(budgets)
    norms = (
        float(source_score.mean()), float(max(source_score.std(), 1e-4)),
        float(source_horizon.mean()), float(max(source_horizon.std(), 1e-4)),
        float(log_budgets.mean()), float(max(log_budgets.std(), 1e-4)),
    )
    score = torch.from_numpy(((source_score - norms[0]) / norms[1]).astype(np.float32)).cuda()
    horizon = torch.from_numpy(((source_horizon - norms[2]) / norms[3]).astype(np.float32)).cuda()[:, None]
    target_cost = torch.from_numpy(source_cost.astype(np.float32)).cuda()
    normalized_budgets = torch.from_numpy(((log_budgets - norms[4]) / norms[5]).astype(np.float32)).cuda()
    raw_budgets = torch.from_numpy(budgets).cuda()

    torch.manual_seed(int(config["seed"]))
    model_config = config["model"]
    model = MonotoneReliabilityCDF(
        [float(value) for value in model_config["score_knots"]],
        [float(value) for value in model_config["budget_knots"]],
    ).cuda()
    baseline = HorizonOnlyReliabilityCDF(
        [float(value) for value in model_config["budget_knots"]],
    ).cuda()
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(baseline.parameters()),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    final_model_loss = 0.0
    final_baseline_loss = 0.0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    for step in range(int(model_config["training_steps"])):
        index = torch.randint(len(score), (int(model_config["batch_size"]),), device="cuda")
        budget_index = torch.randint(len(budgets), (len(index),), device="cuda")
        sampled_budget = normalized_budgets[budget_index]
        target = (target_cost[index] <= raw_budgets[budget_index]).float()
        prediction = model(score[index], horizon[index], sampled_budget)
        control = baseline(horizon[index], sampled_budget)
        model_loss = functional.binary_cross_entropy_with_logits(prediction, target)
        baseline_loss = functional.binary_cross_entropy_with_logits(control, target)
        loss = model_loss + baseline_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_model_loss = float(model_loss.detach().cpu())
        final_baseline_loss = float(baseline_loss.detach().cpu())
        if step % 1000 == 0 or step + 1 == int(model_config["training_steps"]):
            print(
                f"P173 reliability-CDF step={step + 1} model={final_model_loss:.6f} "
                f"horizon={final_baseline_loss:.6f}", flush=True,
            )

    torch.save({
        "model_state_dict": model.state_dict(),
        "baseline_state_dict": baseline.state_dict(),
        "norms": norms,
        "score_knots": model_config["score_knots"],
        "budget_knots": model_config["budget_knots"],
        "reliability_budgets": config["reliability_budgets"],
    }, run_dir / config["model_artifact"])
    evaluations = {}
    for cohort in config["decision_cohorts"]:
        arrays = dict(np.load(args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False))
        evaluations[cohort["name"]] = _evaluate(
            arrays, models, frozen, model.eval(), baseline.eval(), norms, config,
        )
        print(json.dumps({cohort["name"]: evaluations[cohort["name"]]}, indent=2), flush=True)
    reductions = [float(row["integrated_brier_reduction"]) for row in evaluations.values()]
    checks = {
        "brier_noninferior_every_cohort": all(
            float(row["integrated_brier"]) <= float(row["horizon_only_integrated_brier"])
            for row in evaluations.values()
        ),
        "minimum_mean_integrated_brier_reduction": float(np.mean(reductions))
        >= float(config["decision"]["minimum_mean_integrated_brier_reduction"]),
    }
    passed = all(checks.values())
    verdict = config["verdict_on_pass"] if passed else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"],
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "role": config["role"],
        "training": {
            "trajectory_count": int(len(source_score)),
            "source_scene_count": int(len(np.unique(source_scenes))),
            "final_reliability_bce": final_model_loss,
            "final_horizon_only_bce": final_baseline_loss,
        },
        "consumed_development_evaluations": evaluations,
        "decision_checks": checks,
        "mean_integrated_brier_reduction": float(np.mean(reductions)),
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
