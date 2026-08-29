"""Group-split beta calibration for the monotone visited-state reliability CDF."""

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


class MonotoneBetaCalibration(nn.Module):
    """A monotone beta calibration map initialized to the identity."""

    def __init__(self) -> None:
        super().__init__()
        identity_raw = math.log(math.expm1(1.0))
        self.raw_a = nn.Parameter(torch.tensor(identity_raw, dtype=torch.float32))
        self.raw_b = nn.Parameter(torch.tensor(identity_raw, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, probability: torch.Tensor) -> torch.Tensor:
        probability = probability.clamp(1e-6, 1.0 - 1e-6)
        a = functional.softplus(self.raw_a)
        b = functional.softplus(self.raw_b)
        return a * torch.log(probability) - b * torch.log1p(-probability) + self.bias

    def coefficients(self) -> tuple[float, float, float]:
        return (
            float(functional.softplus(self.raw_a).detach().cpu()),
            float(functional.softplus(self.raw_b).detach().cpu()),
            float(self.bias.detach().cpu()),
        )


def _apply_beta(probability: np.ndarray, coefficients: tuple[float, float, float]) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    a, b, bias = coefficients
    logit = a * np.log(clipped) - b * np.log1p(-clipped) + bias
    return 1.0 / (1.0 + np.exp(-logit))


def _evaluate(
    arrays: dict[str, np.ndarray],
    models: list[DirectionalActorGaussian],
    frozen: dict,
    model: MonotoneReliabilityCDF,
    baseline: HorizonOnlyReliabilityCDF,
    calibration: MonotoneBetaCalibration,
    baseline_calibration: MonotoneBetaCalibration,
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
        raise RuntimeError("P174 trajectory grouping is not aligned")
    horizon = _trajectory_horizon(arrays)
    budgets = np.asarray(config["reliability_budgets"], dtype=np.float32)
    raw, raw_control = _predict_surface(model, baseline, score, horizon, budgets, norms)
    calibrated = _apply_beta(raw, calibration.coefficients())
    calibrated_control = _apply_beta(raw_control, baseline_calibration.coefficients())
    target = actual_cost[:, None] <= budgets[None]
    raw_brier = float(np.mean(np.square(raw - target)))
    calibrated_brier = float(np.mean(np.square(calibrated - target)))
    control_brier = float(np.mean(np.square(calibrated_control - target)))
    raw_error = float(np.mean(np.abs(raw.mean(axis=0) - target.mean(axis=0))))
    calibrated_error = float(np.mean(np.abs(calibrated.mean(axis=0) - target.mean(axis=0))))
    control_error = float(np.mean(np.abs(calibrated_control.mean(axis=0) - target.mean(axis=0))))
    return {
        "row_count": int(len(arrays["features"])),
        "trajectory_count": int(len(actual_cost)),
        "raw_integrated_brier": raw_brier,
        "calibrated_integrated_brier": calibrated_brier,
        "calibrated_horizon_only_integrated_brier": control_brier,
        "calibrated_brier_reduction_over_horizon_only": float(
            (control_brier - calibrated_brier) / max(control_brier, 1e-12)
        ),
        "raw_mean_absolute_reliability_error": raw_error,
        "calibrated_mean_absolute_reliability_error": calibrated_error,
        "calibrated_horizon_only_mean_absolute_reliability_error": control_error,
        "calibration_error_reduction_vs_raw": float(
            (raw_error - calibrated_error) / max(raw_error, 1e-12)
        ),
        "per_budget": {
            str(float(budget)): {
                "empirical_reliability": float(target[:, index].mean()),
                "raw_reliability": float(raw[:, index].mean()),
                "calibrated_reliability": float(calibrated[:, index].mean()),
                "calibrated_horizon_only_reliability": float(calibrated_control[:, index].mean()),
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
        raise RuntimeError("P174 source grouping is not aligned")
    source_horizon = _trajectory_horizon(source)
    ordered_scenes = np.unique(source_scenes)
    modulus = int(config["split"]["ordered_scene_modulus"])
    calibration_remainder = int(config["split"]["calibration_remainder"])
    calibration_scenes = ordered_scenes[
        np.mod(np.arange(len(ordered_scenes)), modulus) == calibration_remainder
    ]
    calibration_mask = np.isin(source_scenes, calibration_scenes)
    train_mask = ~calibration_mask
    budgets = np.asarray(config["reliability_budgets"], dtype=np.float32)
    log_budgets = np.log1p(budgets)
    norms = (
        float(source_score[train_mask].mean()), float(max(source_score[train_mask].std(), 1e-4)),
        float(source_horizon[train_mask].mean()), float(max(source_horizon[train_mask].std(), 1e-4)),
        float(log_budgets.mean()), float(max(log_budgets.std(), 1e-4)),
    )
    score = torch.from_numpy(((source_score - norms[0]) / norms[1]).astype(np.float32)).cuda()
    horizon = torch.from_numpy(((source_horizon - norms[2]) / norms[3]).astype(np.float32)).cuda()[:, None]
    target_cost = torch.from_numpy(source_cost.astype(np.float32)).cuda()
    normalized_budgets = torch.from_numpy(((log_budgets - norms[4]) / norms[5]).astype(np.float32)).cuda()
    raw_budgets = torch.from_numpy(budgets).cuda()
    train_indices = torch.from_numpy(np.flatnonzero(train_mask).astype(np.int64)).cuda()
    calibration_indices = torch.from_numpy(np.flatnonzero(calibration_mask).astype(np.int64)).cuda()

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
        draw = torch.randint(len(train_indices), (int(model_config["batch_size"]),), device="cuda")
        index = train_indices[draw]
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
                f"P174 base step={step + 1} model={final_model_loss:.6f} "
                f"horizon={final_baseline_loss:.6f}", flush=True,
            )

    calibration = MonotoneBetaCalibration().cuda()
    baseline_calibration = MonotoneBetaCalibration().cuda()
    calibration_config = config["calibration"]
    calibration_optimizer = torch.optim.AdamW(
        list(calibration.parameters()) + list(baseline_calibration.parameters()),
        lr=float(calibration_config["learning_rate"]), weight_decay=0.0,
    )
    final_calibration_loss = 0.0
    final_baseline_calibration_loss = 0.0
    model.eval()
    baseline.eval()
    for step in range(int(calibration_config["training_steps"])):
        draw = torch.randint(
            len(calibration_indices), (int(calibration_config["batch_size"]),), device="cuda",
        )
        index = calibration_indices[draw]
        budget_index = torch.randint(len(budgets), (len(index),), device="cuda")
        sampled_budget = normalized_budgets[budget_index]
        target = (target_cost[index] <= raw_budgets[budget_index]).float()
        with torch.no_grad():
            raw_probability = torch.sigmoid(model(score[index], horizon[index], sampled_budget))
            raw_control_probability = torch.sigmoid(baseline(horizon[index], sampled_budget))
        calibrated_logit = calibration(raw_probability)
        calibrated_control_logit = baseline_calibration(raw_control_probability)
        calibration_loss = functional.binary_cross_entropy_with_logits(calibrated_logit, target)
        baseline_calibration_loss = functional.binary_cross_entropy_with_logits(
            calibrated_control_logit, target,
        )
        loss = calibration_loss + baseline_calibration_loss
        calibration_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        calibration_optimizer.step()
        final_calibration_loss = float(calibration_loss.detach().cpu())
        final_baseline_calibration_loss = float(baseline_calibration_loss.detach().cpu())
        if step % 500 == 0 or step + 1 == int(calibration_config["training_steps"]):
            print(
                f"P174 beta step={step + 1} model={final_calibration_loss:.6f} "
                f"horizon={final_baseline_calibration_loss:.6f}", flush=True,
            )

    torch.save({
        "model_state_dict": model.state_dict(),
        "baseline_state_dict": baseline.state_dict(),
        "calibration_state_dict": calibration.state_dict(),
        "baseline_calibration_state_dict": baseline_calibration.state_dict(),
        "calibration_coefficients": calibration.coefficients(),
        "baseline_calibration_coefficients": baseline_calibration.coefficients(),
        "norms": norms,
        "score_knots": model_config["score_knots"],
        "budget_knots": model_config["budget_knots"],
        "reliability_budgets": config["reliability_budgets"],
    }, run_dir / config["model_artifact"])
    evaluations = {}
    for cohort in config["decision_cohorts"]:
        arrays = dict(np.load(args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False))
        evaluations[cohort["name"]] = _evaluate(
            arrays, models, frozen, model, baseline, calibration, baseline_calibration, norms, config,
        )
        print(json.dumps({cohort["name"]: evaluations[cohort["name"]]}, indent=2), flush=True)
    calibration_reductions = [
        float(row["calibration_error_reduction_vs_raw"]) for row in evaluations.values()
    ]
    checks = {
        "calibrated_brier_better_than_horizon_only_every_cohort": all(
            float(row["calibrated_integrated_brier"])
            <= float(row["calibrated_horizon_only_integrated_brier"])
            for row in evaluations.values()
        ),
        "minimum_mean_calibration_error_reduction": float(np.mean(calibration_reductions))
        >= float(config["decision"]["minimum_mean_calibration_error_reduction"]),
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
            "train_trajectory_count": int(train_mask.sum()),
            "calibration_trajectory_count": int(calibration_mask.sum()),
            "train_scene_count": int(len(np.unique(source_scenes[train_mask]))),
            "calibration_scene_count": int(len(calibration_scenes)),
            "final_reliability_bce": final_model_loss,
            "final_horizon_only_bce": final_baseline_loss,
            "final_beta_calibration_bce": final_calibration_loss,
            "final_horizon_only_beta_calibration_bce": final_baseline_calibration_loss,
            "beta_coefficients": calibration.coefficients(),
            "horizon_only_beta_coefficients": baseline_calibration.coefficients(),
        },
        "consumed_development_evaluations": evaluations,
        "decision_checks": checks,
        "mean_calibration_error_reduction": float(np.mean(calibration_reductions)),
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
