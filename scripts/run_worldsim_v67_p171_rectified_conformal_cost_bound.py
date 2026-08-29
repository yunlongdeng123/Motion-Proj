"""Rectify frozen P170 conformity residuals with a learned conditional quantile."""

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
import yaml

from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost
from scripts.run_worldsim_v67_p127_ensemble_continuous_selection import _ensemble_trajectory_score
from scripts.run_worldsim_v67_p158_crps_actor_ensemble import _subset
from scripts.run_worldsim_v67_p166_monotone_expected_cost_calibration import (
    HorizonOnlyCalibration, MonotoneScoreCalibration, _trajectory_horizon,
)
from scripts.run_worldsim_v67_p170_conformal_cost_upper_bound import (
    _conformal_offset, _pinball, _predict_log,
)


class ResidualQuantileRectifier(nn.Module):
    def __init__(self, hidden_dimensions: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = 2
        for hidden in hidden_dimensions:
            layers.extend((nn.Linear(width, int(hidden)), nn.SiLU()))
            width = int(hidden)
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, score: torch.Tensor, horizon: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((score[:, None], horizon), dim=1)).reshape(-1)


@torch.no_grad()
def _rectification(
    model: ResidualQuantileRectifier, score: np.ndarray, horizon: np.ndarray,
    norms: tuple[float, float, float, float],
) -> np.ndarray:
    score_mean, score_scale, horizon_mean, horizon_scale = norms
    outputs = []
    for start in range(0, len(score), 65536):
        normalized_score = torch.from_numpy(
            ((score[start:start + 65536] - score_mean) / score_scale).astype(np.float32)
        ).cuda()
        normalized_horizon = torch.from_numpy(
            ((horizon[start:start + 65536] - horizon_mean) / horizon_scale).astype(np.float32)
        ).cuda()[:, None]
        outputs.append(model(normalized_score, normalized_horizon).cpu().numpy())
    return np.concatenate(outputs)


def _evaluate(
    arrays: dict[str, np.ndarray], models: list[DirectionalActorGaussian], frozen: dict,
    base_model: MonotoneScoreCalibration, horizon_control: HorizonOnlyCalibration,
    rectifier: ResidualQuantileRectifier, norms: tuple[float, float, float, float],
    offsets: tuple[float, float], config: dict,
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
        raise RuntimeError("P171 trajectory grouping is not aligned")
    horizon = _trajectory_horizon(arrays)
    base_log, _ = _predict_log(base_model, horizon_control, score, horizon, norms)
    correction = _rectification(rectifier, score, horizon, norms)
    rectified_upper = np.expm1(np.maximum(base_log + correction + offsets[0], 0.0))
    p170_upper = np.expm1(np.maximum(base_log + offsets[1], 0.0))
    rectified_mean = float(rectified_upper.mean())
    p170_mean = float(p170_upper.mean())
    return {
        "row_count": int(len(arrays["features"])), "trajectory_count": int(len(actual)),
        "rectified_empirical_upper_coverage": float(np.mean(actual <= rectified_upper)),
        "p170_empirical_upper_coverage": float(np.mean(actual <= p170_upper)),
        "rectified_mean_cost_upper_bound": rectified_mean, "p170_mean_cost_upper_bound": p170_mean,
        "sharpness_reduction_over_p170": (p170_mean - rectified_mean) / max(p170_mean, 1e-12),
        "actual_mean_cost": float(actual.mean()),
    }


def _checks(results: dict[str, dict[str, float | int]], config: dict) -> dict[str, bool]:
    decision = config["decision"]
    return {
        "minimum_empirical_coverage_every_slice": all(
            float(row["rectified_empirical_upper_coverage"]) >= float(decision["minimum_empirical_coverage"])
            for row in results.values()
        ),
        "minimum_mean_sharpness_reduction_over_p170": float(np.mean([
            float(row["sharpness_reduction_over_p170"]) for row in results.values()
        ])) >= float(decision["minimum_mean_sharpness_reduction_over_p170"]),
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
    models = []
    for state in frozen["member_state_dicts"]:
        model = DirectionalActorGaussian(20, frozen["hidden_dimensions"]).cuda()
        model.load_state_dict(state)
        models.append(model.eval())
    p170 = torch.load(
        args.runs_root / config["frozen_p170"]["run"] / config["frozen_p170"]["artifact"], map_location="cuda",
    )
    base_model = MonotoneScoreCalibration([float(value) for value in p170["score_knots"]]).cuda()
    base_model.load_state_dict(p170["model_state_dict"])
    base_model.eval()
    horizon_control = HorizonOnlyCalibration().cuda()
    horizon_control.load_state_dict(p170["baseline_state_dict"])
    horizon_control.eval()
    norms = tuple(float(value) for value in p170["norms"])
    p170_offset = float(p170["conformal_offsets"][0])
    source = dict(np.load(
        args.runs_root / config["source_rows"]["run"] / config["source_rows"]["artifact"], allow_pickle=False,
    ))
    score, scenes = _ensemble_trajectory_score(
        source, models,
        np.asarray(frozen["feature_mean"], dtype=np.float32),
        np.asarray(frozen["feature_scale"], dtype=np.float32),
        np.asarray(frozen["target_mean"], dtype=np.float32),
        np.asarray(frozen["target_scale"], dtype=np.float32),
    )
    actual, cost_scenes = _continuous_cost(source, float(config["boundary_state_cost"]["clearance_floor_m"]))
    if not np.array_equal(scenes, cost_scenes):
        raise RuntimeError("P171 source grouping is not aligned")
    horizon = _trajectory_horizon(source)
    base_log, _ = _predict_log(base_model, horizon_control, score, horizon, norms)
    ordered_scenes = np.unique(scenes)
    modulus = int(config["split"]["ordered_unique_scene_position_modulus"])
    calibration_scenes = ordered_scenes[np.mod(np.arange(len(ordered_scenes)), modulus) == 0]
    calibration_mask = np.isin(scenes, calibration_scenes)
    train_mask = ~calibration_mask
    score_mean, score_scale, horizon_mean, horizon_scale = norms
    score_tensor = torch.from_numpy(((score - score_mean) / score_scale).astype(np.float32)).cuda()
    horizon_tensor = torch.from_numpy(((horizon - horizon_mean) / horizon_scale).astype(np.float32)).cuda()[:, None]
    residual_target = torch.from_numpy((np.log1p(actual) - base_log).astype(np.float32)).cuda()
    train_indices = torch.from_numpy(np.flatnonzero(train_mask).astype(np.int64)).cuda()
    model_config = config["model"]
    torch.manual_seed(int(config["seed"]))
    rectifier = ResidualQuantileRectifier(model_config["hidden_dimensions"]).cuda()
    optimizer = torch.optim.AdamW(
        rectifier.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    quantile = float(config["upper_bound"]["quantile"])
    final_loss = 0.0
    torch.cuda.reset_peak_memory_stats()
    for step in range(int(model_config["training_steps"])):
        draw = torch.randint(len(train_indices), (int(model_config["batch_size"]),), device="cuda")
        index = train_indices[draw]
        prediction = rectifier(score_tensor[index], horizon_tensor[index])
        loss = _pinball(prediction, residual_target[index], quantile)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step % 1000 == 0 or step + 1 == int(model_config["training_steps"]):
            print(f"P171 rectifier step={step + 1} qloss={final_loss:.6f}", flush=True)
    correction = _rectification(rectifier.eval(), score, horizon, norms)
    offset = _conformal_offset(
        (np.log1p(actual) - base_log - correction)[calibration_mask],
        float(config["upper_bound"]["coverage"]),
    )
    torch.save({
        "model_state_dict": rectifier.state_dict(), "hidden_dimensions": model_config["hidden_dimensions"],
        "rectified_conformal_offset": offset, "frozen_p170_offset": p170_offset,
    }, run_dir / config["model_artifact"])
    development = {}
    offsets = (offset, p170_offset)
    for cohort in config["decision_cohorts"]:
        arrays = dict(np.load(args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False))
        development[cohort["name"]] = _evaluate(
            arrays, models, frozen, base_model, horizon_control, rectifier.eval(), norms, offsets, config,
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
                raise TimeoutError(f"P171 prospective rows not ready: {rows_path}")
            time.sleep(5.0)
        arrays = dict(np.load(rows_path, allow_pickle=False))
        for horizon_value in spec["horizons_seconds"]:
            key = str(float(horizon_value))
            prospective[key] = _evaluate(
                _subset(arrays, float(horizon_value)), models, frozen, base_model, horizon_control,
                rectifier.eval(), norms, offsets, config,
            )
            print(json.dumps({f"P167_H{key}": prospective[key]}, indent=2), flush=True)
        prospective_checks = _checks(prospective, config)
    passed = all(development_checks.values()) and bool(prospective) and all(prospective_checks.values())
    verdict = config["verdict_on_pass"] if passed else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict, "role": config["role"],
        "training": {
            "final_residual_quantile_loss": final_loss, "rectified_conformal_offset_log_cost": offset,
            "train_trajectory_count": int(train_mask.sum()), "calibration_trajectory_count": int(calibration_mask.sum()),
        },
        "consumed_development_evaluations": development, "development_decision_checks": development_checks,
        "prospective_p167_evaluations": prospective, "prospective_decision_checks": prospective_checks,
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
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict}, indent=2), flush=True)


if __name__ == "__main__":
    main()
