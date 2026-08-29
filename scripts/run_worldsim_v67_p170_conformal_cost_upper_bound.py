"""Train and split-conformalize a one-sided trajectory cost upper bound."""

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
import torch.nn.functional as functional
import yaml

from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost
from scripts.run_worldsim_v67_p127_ensemble_continuous_selection import _ensemble_trajectory_score
from scripts.run_worldsim_v67_p158_crps_actor_ensemble import _subset
from scripts.run_worldsim_v67_p166_monotone_expected_cost_calibration import (
    HorizonOnlyCalibration, MonotoneScoreCalibration, _trajectory_horizon,
)


def _pinball(prediction: torch.Tensor, target: torch.Tensor, quantile: float) -> torch.Tensor:
    error = target - prediction
    return torch.maximum(quantile * error, (quantile - 1.0) * error).mean()


@torch.no_grad()
def _predict_log(
    model: MonotoneScoreCalibration, baseline: HorizonOnlyCalibration,
    score: np.ndarray, horizon: np.ndarray, norms: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    score_mean, score_scale, horizon_mean, horizon_scale = norms
    outputs, controls = [], []
    for start in range(0, len(score), 65536):
        normalized_score = torch.from_numpy(
            ((score[start:start + 65536] - score_mean) / score_scale).astype(np.float32)
        ).cuda()
        normalized_horizon = torch.from_numpy(
            ((horizon[start:start + 65536] - horizon_mean) / horizon_scale).astype(np.float32)
        ).cuda()[:, None]
        outputs.append(model(normalized_score, normalized_horizon).cpu().numpy())
        controls.append(baseline(normalized_horizon).cpu().numpy())
    return np.concatenate(outputs), np.concatenate(controls)


def _conformal_offset(residual: np.ndarray, coverage: float) -> float:
    level = min(1.0, math.ceil((len(residual) + 1) * coverage) / len(residual))
    return float(np.quantile(residual, level, method="higher"))


def _evaluate(
    arrays: dict[str, np.ndarray], models: list[DirectionalActorGaussian], frozen: dict,
    model: MonotoneScoreCalibration, baseline: HorizonOnlyCalibration,
    norms: tuple[float, float, float, float], offsets: tuple[float, float], config: dict,
) -> dict[str, float | int]:
    score, scenes = _ensemble_trajectory_score(
        arrays, models,
        np.asarray(frozen["feature_mean"], dtype=np.float32),
        np.asarray(frozen["feature_scale"], dtype=np.float32),
        np.asarray(frozen["target_mean"], dtype=np.float32),
        np.asarray(frozen["target_scale"], dtype=np.float32),
    )
    actual, cost_scenes = _continuous_cost(
        arrays, float(config["boundary_state_cost"]["clearance_floor_m"]),
    )
    if not np.array_equal(scenes, cost_scenes):
        raise RuntimeError("P170 trajectory grouping is not aligned")
    horizon = _trajectory_horizon(arrays)
    predicted_log, baseline_log = _predict_log(model, baseline, score, horizon, norms)
    upper = np.expm1(np.maximum(predicted_log + offsets[0], 0.0))
    baseline_upper = np.expm1(np.maximum(baseline_log + offsets[1], 0.0))
    mean_upper = float(upper.mean())
    mean_baseline_upper = float(baseline_upper.mean())
    return {
        "row_count": int(len(arrays["features"])), "trajectory_count": int(len(actual)),
        "empirical_upper_coverage": float(np.mean(actual <= upper)),
        "horizon_only_empirical_upper_coverage": float(np.mean(actual <= baseline_upper)),
        "mean_cost_upper_bound": mean_upper, "horizon_only_mean_cost_upper_bound": mean_baseline_upper,
        "upper_bound_sharpness_reduction": (mean_baseline_upper - mean_upper) / max(mean_baseline_upper, 1e-12),
        "actual_mean_cost": float(actual.mean()),
    }


def _checks(results: dict[str, dict[str, float | int]], config: dict) -> dict[str, bool]:
    decision = config["decision"]
    return {
        "minimum_empirical_coverage_every_slice": all(
            float(row["empirical_upper_coverage"]) >= float(decision["minimum_empirical_coverage"])
            for row in results.values()
        ),
        "minimum_mean_sharpness_reduction": float(np.mean([
            float(row["upper_bound_sharpness_reduction"]) for row in results.values()
        ])) >= float(decision["minimum_mean_sharpness_reduction"]),
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
        model = DirectionalActorGaussian(20, frozen["hidden_dimensions"]).cuda()
        model.load_state_dict(state)
        models.append(model.eval())
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
        raise RuntimeError("P170 source grouping is not aligned")
    source_horizon = _trajectory_horizon(source)
    modulus = int(config["split"]["scene_modulus"])
    calibration_remainder = int(config["split"]["calibration_remainder"])
    ordered_scenes = np.unique(source_scenes)
    calibration_scenes = ordered_scenes[
        np.mod(np.arange(len(ordered_scenes)), modulus) == calibration_remainder
    ]
    calibration_mask = np.isin(source_scenes, calibration_scenes)
    train_mask = ~calibration_mask
    score_mean = float(source_score[train_mask].mean())
    score_scale = float(max(source_score[train_mask].std(), 1e-4))
    horizon_mean = float(source_horizon[train_mask].mean())
    horizon_scale = float(max(source_horizon[train_mask].std(), 1e-4))
    norms = (score_mean, score_scale, horizon_mean, horizon_scale)
    score = torch.from_numpy(((source_score - score_mean) / score_scale).astype(np.float32)).cuda()
    horizon = torch.from_numpy(((source_horizon - horizon_mean) / horizon_scale).astype(np.float32)).cuda()[:, None]
    target = torch.from_numpy(np.log1p(source_cost).astype(np.float32)).cuda()
    train_indices = torch.from_numpy(np.flatnonzero(train_mask).astype(np.int64)).cuda()
    model_config = config["model"]
    torch.manual_seed(int(config["seed"]))
    quantile_model = MonotoneScoreCalibration(
        [float(value) for value in model_config["score_knots"]]
    ).cuda()
    baseline = HorizonOnlyCalibration().cuda()
    optimizer = torch.optim.AdamW(
        list(quantile_model.parameters()) + list(baseline.parameters()),
        lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"]),
    )
    quantile = float(config["upper_bound"]["quantile"])
    final_model_loss = 0.0
    final_baseline_loss = 0.0
    torch.cuda.reset_peak_memory_stats()
    for step in range(int(model_config["training_steps"])):
        draw = torch.randint(len(train_indices), (int(model_config["batch_size"]),), device="cuda")
        index = train_indices[draw]
        prediction = quantile_model(score[index], horizon[index])
        control = baseline(horizon[index])
        model_loss = _pinball(prediction, target[index], quantile)
        baseline_loss = _pinball(control, target[index], quantile)
        loss = model_loss + baseline_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_model_loss = float(model_loss.detach().cpu())
        final_baseline_loss = float(baseline_loss.detach().cpu())
        if step % 1000 == 0 or step + 1 == int(model_config["training_steps"]):
            print(
                f"P170 q90 step={step + 1} model={final_model_loss:.6f} "
                f"horizon={final_baseline_loss:.6f}", flush=True,
            )
    predicted_log, baseline_log = _predict_log(
        quantile_model.eval(), baseline.eval(), source_score, source_horizon, norms,
    )
    actual_log = np.log1p(source_cost)
    coverage = float(config["upper_bound"]["coverage"])
    offsets = (
        _conformal_offset(actual_log[calibration_mask] - predicted_log[calibration_mask], coverage),
        _conformal_offset(actual_log[calibration_mask] - baseline_log[calibration_mask], coverage),
    )
    torch.save({
        "model_state_dict": quantile_model.state_dict(), "baseline_state_dict": baseline.state_dict(),
        "norms": norms, "score_knots": model_config["score_knots"], "conformal_offsets": offsets,
    }, run_dir / config["model_artifact"])
    development = {}
    for cohort in config["decision_cohorts"]:
        arrays = dict(np.load(args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False))
        development[cohort["name"]] = _evaluate(
            arrays, models, frozen, quantile_model.eval(), baseline.eval(), norms, offsets, config,
        )
        print(json.dumps({cohort["name"]: development[cohort["name"]]}, indent=2), flush=True)
    development_checks = _checks(development, config)
    (run_dir / "development.json").write_text(json.dumps({
        "evaluations": development, "decision_checks": development_checks,
    }, indent=2) + "\n", encoding="utf-8")
    prospective = {}
    prospective_checks: dict[str, bool] = {}
    if all(development_checks.values()):
        spec = config["prospective_p167"]
        rows_path = args.runs_root / spec["run"] / spec["artifact"]
        deadline = time.monotonic() + float(spec["readiness_timeout_seconds"])
        while not rows_path.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"P170 prospective rows not ready: {rows_path}")
            time.sleep(5.0)
        arrays = dict(np.load(rows_path, allow_pickle=False))
        for horizon_value in spec["horizons_seconds"]:
            key = str(float(horizon_value))
            prospective[key] = _evaluate(
                _subset(arrays, float(horizon_value)), models, frozen, quantile_model.eval(),
                baseline.eval(), norms, offsets, config,
            )
            print(json.dumps({f"P167_H{key}": prospective[key]}, indent=2), flush=True)
        prospective_checks = _checks(prospective, config)
    passed = all(development_checks.values()) and bool(prospective) and all(prospective_checks.values())
    verdict = config["verdict_on_pass"] if passed else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "training": {
            "train_trajectory_count": int(train_mask.sum()),
            "calibration_trajectory_count": int(calibration_mask.sum()),
            "train_scene_count": int(len(np.unique(source_scenes[train_mask]))),
            "calibration_scene_count": int(len(calibration_scenes)),
            "final_quantile_loss": final_model_loss, "final_horizon_only_quantile_loss": final_baseline_loss,
            "model_conformal_offset_log_cost": offsets[0],
            "horizon_only_conformal_offset_log_cost": offsets[1],
        },
        "consumed_development_evaluations": development,
        "development_decision_checks": development_checks,
        "prospective_p167_evaluations": prospective, "prospective_decision_checks": prospective_checks,
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
