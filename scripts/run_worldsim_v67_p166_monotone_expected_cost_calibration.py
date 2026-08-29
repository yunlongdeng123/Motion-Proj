"""Calibrate frozen P126 trajectory scores to expected continuous boundary-state cost."""

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
from scripts.run_worldsim_v67_p158_crps_actor_ensemble import _subset


class HorizonOnlyCalibration(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, horizon: torch.Tensor) -> torch.Tensor:
        return functional.softplus(self.linear(horizon).squeeze(1))


class MonotoneScoreCalibration(nn.Module):
    def __init__(self, knots: list[float]) -> None:
        super().__init__()
        self.register_buffer("knots", torch.tensor(knots, dtype=torch.float32))
        self.horizon_parameters = nn.Linear(1, len(knots) + 2)

    def forward(self, score: torch.Tensor, horizon: torch.Tensor) -> torch.Tensor:
        parameters = self.horizon_parameters(horizon)
        base = parameters[:, 0]
        slope = functional.softplus(parameters[:, 1])
        increments = functional.softplus(parameters[:, 2:])
        hinge = functional.relu(score[:, None] - self.knots[None])
        raw = base + slope * score + torch.sum(increments * hinge, dim=1)
        return functional.softplus(raw)


def _trajectory_horizon(arrays: dict[str, np.ndarray]) -> np.ndarray:
    keys = np.stack((
        arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["query_id"],
    ), axis=1)
    identities = np.unique(keys, axis=0)
    return identities[:, 1].astype(np.float32) / 10.0


def _calibration_error(actual: np.ndarray, predicted: np.ndarray, bins: int) -> float:
    order = np.argsort(predicted)
    error = 0.0
    for indices in np.array_split(order, bins):
        if len(indices):
            error += len(indices) / len(order) * abs(float(actual[indices].mean() - predicted[indices].mean()))
    return float(error)


@torch.no_grad()
def _predict(
    calibration: MonotoneScoreCalibration, baseline: HorizonOnlyCalibration,
    score: np.ndarray, horizon: np.ndarray, score_mean: float, score_scale: float,
    horizon_mean: float, horizon_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    calibrated, horizon_only = [], []
    for start in range(0, len(score), 65536):
        normalized_score = torch.from_numpy(
            ((score[start:start + 65536] - score_mean) / score_scale).astype(np.float32)
        ).cuda()
        normalized_horizon = torch.from_numpy(
            ((horizon[start:start + 65536] - horizon_mean) / horizon_scale).astype(np.float32)
        ).cuda()[:, None]
        calibrated.append(torch.expm1(calibration(normalized_score, normalized_horizon)).cpu().numpy())
        horizon_only.append(torch.expm1(baseline(normalized_horizon)).cpu().numpy())
    return np.concatenate(calibrated), np.concatenate(horizon_only)


def _evaluate(
    arrays: dict[str, np.ndarray], models: list[DirectionalActorGaussian],
    feature_mean: np.ndarray, feature_scale: np.ndarray,
    target_mean: np.ndarray, target_scale: np.ndarray,
    calibration: MonotoneScoreCalibration, baseline: HorizonOnlyCalibration,
    norms: tuple[float, float, float, float], config: dict,
) -> dict[str, float | int]:
    score, scenes = _ensemble_trajectory_score(
        arrays, models, feature_mean, feature_scale, target_mean, target_scale,
    )
    actual, cost_scenes = _continuous_cost(
        arrays, float(config["boundary_state_cost"]["clearance_floor_m"]),
    )
    horizon = _trajectory_horizon(arrays)
    if not np.array_equal(scenes, cost_scenes):
        raise RuntimeError("P166 trajectory grouping is not aligned")
    predicted, horizon_only = _predict(calibration, baseline, score, horizon, *norms)
    calibrated_mse = float(np.mean(np.square(predicted - actual)))
    baseline_mse = float(np.mean(np.square(horizon_only - actual)))
    calibrated_mae = float(np.mean(np.abs(predicted - actual)))
    baseline_mae = float(np.mean(np.abs(horizon_only - actual)))
    bins = int(config["evaluation"]["calibration_bins"])
    return {
        "row_count": int(len(arrays["features"])), "trajectory_count": int(len(actual)),
        "calibrated_mse": calibrated_mse, "horizon_only_mse": baseline_mse,
        "mse_reduction": float((baseline_mse - calibrated_mse) / max(baseline_mse, 1e-12)),
        "calibrated_mae": calibrated_mae, "horizon_only_mae": baseline_mae,
        "mae_reduction": float((baseline_mae - calibrated_mae) / max(baseline_mae, 1e-12)),
        "calibrated_expected_cost_error": _calibration_error(actual, predicted, bins),
        "horizon_only_expected_cost_error": _calibration_error(actual, horizon_only, bins),
        "actual_mean_cost": float(actual.mean()), "predicted_mean_cost": float(predicted.mean()),
        "horizon_only_predicted_mean_cost": float(horizon_only.mean()),
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
        args.runs_root / config["frozen_p126"]["run"] / config["frozen_p126"]["artifact"], map_location="cuda",
    )
    feature_mean = np.asarray(frozen["feature_mean"], dtype=np.float32)
    feature_scale = np.asarray(frozen["feature_scale"], dtype=np.float32)
    target_mean = np.asarray(frozen["target_mean"], dtype=np.float32)
    target_scale = np.asarray(frozen["target_scale"], dtype=np.float32)
    models = []
    for state in frozen["member_state_dicts"]:
        model = DirectionalActorGaussian(20, frozen["hidden_dimensions"]).cuda()
        model.load_state_dict(state)
        models.append(model.eval())

    source = dict(np.load(
        args.runs_root / config["source_rows"]["run"] / config["source_rows"]["artifact"], allow_pickle=False,
    ))
    source_score, _ = _ensemble_trajectory_score(
        source, models, feature_mean, feature_scale, target_mean, target_scale,
    )
    source_cost, _ = _continuous_cost(
        source, float(config["boundary_state_cost"]["clearance_floor_m"]),
    )
    source_horizon = _trajectory_horizon(source)
    score_mean = float(source_score.mean())
    score_scale = float(max(source_score.std(), 1e-4))
    horizon_mean = float(source_horizon.mean())
    horizon_scale = float(max(source_horizon.std(), 1e-4))
    score = torch.from_numpy(((source_score - score_mean) / score_scale).astype(np.float32)).cuda()
    horizon = torch.from_numpy(((source_horizon - horizon_mean) / horizon_scale).astype(np.float32)).cuda()[:, None]
    target = torch.from_numpy(np.log1p(source_cost).astype(np.float32)).cuda()

    torch.manual_seed(int(config["seed"]))
    calibration = MonotoneScoreCalibration([float(value) for value in config["model"]["score_knots"]]).cuda()
    baseline = HorizonOnlyCalibration().cuda()
    optimizer = torch.optim.AdamW(
        list(calibration.parameters()) + list(baseline.parameters()),
        lr=float(config["model"]["learning_rate"]), weight_decay=float(config["model"]["weight_decay"]),
    )
    final_calibration_loss = 0.0
    final_baseline_loss = 0.0
    torch.cuda.reset_peak_memory_stats()
    for step in range(int(config["model"]["training_steps"])):
        index = torch.randint(len(score), (int(config["model"]["batch_size"]),), device="cuda")
        calibrated = calibration(score[index], horizon[index])
        horizon_only = baseline(horizon[index])
        calibration_loss = functional.mse_loss(calibrated, target[index])
        baseline_loss = functional.mse_loss(horizon_only, target[index])
        loss = calibration_loss + baseline_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_calibration_loss = float(calibration_loss.detach().cpu())
        final_baseline_loss = float(baseline_loss.detach().cpu())
        if step % 1000 == 0 or step + 1 == int(config["model"]["training_steps"]):
            print(
                f"P166 calibration step={step + 1} score_mse={final_calibration_loss:.6f} "
                f"horizon_mse={final_baseline_loss:.6f}", flush=True,
            )

    torch.save({
        "calibration_state_dict": calibration.state_dict(), "baseline_state_dict": baseline.state_dict(),
        "score_mean": score_mean, "score_scale": score_scale,
        "horizon_mean": horizon_mean, "horizon_scale": horizon_scale,
        "score_knots": config["model"]["score_knots"],
    }, run_dir / config["model_artifact"])
    norms = (score_mean, score_scale, horizon_mean, horizon_scale)
    results = {}
    for cohort in config["decision_cohorts"]:
        arrays = dict(np.load(args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False))
        results[cohort["name"]] = _evaluate(
            arrays, models, feature_mean, feature_scale, target_mean, target_scale,
            calibration.eval(), baseline.eval(), norms, config,
        )
        print(json.dumps({cohort["name"]: results[cohort["name"]]}, indent=2), flush=True)

    diagnostic_spec = config["post_confirmation_diagnostic"]
    diagnostic_arrays = dict(np.load(
        args.runs_root / diagnostic_spec["run"] / diagnostic_spec["artifact"], allow_pickle=False,
    ))
    diagnostic_results = {}
    for horizon_value in diagnostic_spec["horizons_seconds"]:
        key = str(float(horizon_value))
        diagnostic_results[key] = _evaluate(
            _subset(diagnostic_arrays, float(horizon_value)), models, feature_mean, feature_scale,
            target_mean, target_scale, calibration.eval(), baseline.eval(), norms, config,
        )
        print(json.dumps({f"P147_H{key}": diagnostic_results[key]}, indent=2), flush=True)

    reductions = [row["mse_reduction"] for row in results.values()]
    decisions = {
        "mse_noninferior_every_cohort": all(row["calibrated_mse"] <= row["horizon_only_mse"] for row in results.values()),
        "minimum_mean_mse_reduction": float(np.mean(reductions))
        >= float(config["decision"]["minimum_mean_mse_reduction"]),
    }
    verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "training": {
            "trajectory_count": int(len(source_score)), "final_calibration_log_mse": final_calibration_loss,
            "final_horizon_only_log_mse": final_baseline_loss,
        },
        "consumed_development_evaluations": results,
        "post_confirmation_consumed_p147_diagnostic": diagnostic_results,
        "decision_checks": decisions, "mean_mse_reduction": float(np.mean(reductions)),
        "resources": {
            "gpu": torch.cuda.get_device_name(0), "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "wall_seconds": time.monotonic() - started,
        },
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({
        "status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict, "decision_checks": decisions}, indent=2), flush=True)


if __name__ == "__main__":
    main()
