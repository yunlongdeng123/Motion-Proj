"""Train a two-sided conformalized trajectory cost interval."""

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
from scripts.run_worldsim_v67_p166_monotone_expected_cost_calibration import (
    HorizonOnlyCalibration, MonotoneScoreCalibration, _trajectory_horizon,
)
from scripts.run_worldsim_v67_p170_conformal_cost_upper_bound import (
    _conformal_offset, _pinball, _predict_log,
)


def _evaluate(
    arrays: dict[str, np.ndarray], models: list[DirectionalActorGaussian], frozen: dict,
    lower: MonotoneScoreCalibration, upper: MonotoneScoreCalibration,
    lower_control: HorizonOnlyCalibration, upper_control: HorizonOnlyCalibration,
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
        raise RuntimeError("P172 trajectory grouping is not aligned")
    horizon = _trajectory_horizon(arrays)
    lower_log, lower_control_log = _predict_log(lower, lower_control, score, horizon, norms)
    upper_log, upper_control_log = _predict_log(upper, upper_control, score, horizon, norms)
    low_log = np.minimum(lower_log, upper_log) - offsets[0]
    high_log = np.maximum(lower_log, upper_log) + offsets[0]
    control_low_log = np.minimum(lower_control_log, upper_control_log) - offsets[1]
    control_high_log = np.maximum(lower_control_log, upper_control_log) + offsets[1]
    low = np.expm1(np.maximum(low_log, 0.0))
    high = np.expm1(np.maximum(high_log, 0.0))
    control_low = np.expm1(np.maximum(control_low_log, 0.0))
    control_high = np.expm1(np.maximum(control_high_log, 0.0))
    width = float(np.mean(high - low))
    control_width = float(np.mean(control_high - control_low))
    return {
        "row_count": int(len(arrays["features"])), "trajectory_count": int(len(actual)),
        "empirical_interval_coverage": float(np.mean((actual >= low) & (actual <= high))),
        "horizon_only_empirical_interval_coverage": float(
            np.mean((actual >= control_low) & (actual <= control_high))
        ),
        "mean_interval_width": width, "horizon_only_mean_interval_width": control_width,
        "interval_width_reduction": (control_width - width) / max(control_width, 1e-12),
        "actual_mean_cost": float(actual.mean()),
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
    actual, cost_scenes = _continuous_cost(
        source, float(config["boundary_state_cost"]["clearance_floor_m"]),
    )
    if not np.array_equal(scenes, cost_scenes):
        raise RuntimeError("P172 source grouping is not aligned")
    horizon = _trajectory_horizon(source)
    ordered_scenes = np.unique(scenes)
    modulus = int(config["split"]["ordered_unique_scene_position_modulus"])
    calibration_scenes = ordered_scenes[np.mod(np.arange(len(ordered_scenes)), modulus) == 0]
    calibration_mask = np.isin(scenes, calibration_scenes)
    train_mask = ~calibration_mask
    score_mean = float(score[train_mask].mean())
    score_scale = float(max(score[train_mask].std(), 1e-4))
    horizon_mean = float(horizon[train_mask].mean())
    horizon_scale = float(max(horizon[train_mask].std(), 1e-4))
    norms = (score_mean, score_scale, horizon_mean, horizon_scale)
    score_tensor = torch.from_numpy(((score - score_mean) / score_scale).astype(np.float32)).cuda()
    horizon_tensor = torch.from_numpy(((horizon - horizon_mean) / horizon_scale).astype(np.float32)).cuda()[:, None]
    target = torch.from_numpy(np.log1p(actual).astype(np.float32)).cuda()
    train_indices = torch.from_numpy(np.flatnonzero(train_mask).astype(np.int64)).cuda()
    model_config = config["model"]
    knots = [float(value) for value in model_config["score_knots"]]
    torch.manual_seed(int(config["seed"]))
    lower, upper = MonotoneScoreCalibration(knots).cuda(), MonotoneScoreCalibration(knots).cuda()
    lower_control, upper_control = HorizonOnlyCalibration().cuda(), HorizonOnlyCalibration().cuda()
    parameters = list(lower.parameters()) + list(upper.parameters())
    parameters += list(lower_control.parameters()) + list(upper_control.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"]),
    )
    lower_q = float(config["interval"]["lower_quantile"])
    upper_q = float(config["interval"]["upper_quantile"])
    final_loss = 0.0
    torch.cuda.reset_peak_memory_stats()
    for step in range(int(model_config["training_steps"])):
        draw = torch.randint(len(train_indices), (int(model_config["batch_size"]),), device="cuda")
        index = train_indices[draw]
        loss = _pinball(lower(score_tensor[index], horizon_tensor[index]), target[index], lower_q)
        loss = loss + _pinball(upper(score_tensor[index], horizon_tensor[index]), target[index], upper_q)
        loss = loss + _pinball(lower_control(horizon_tensor[index]), target[index], lower_q)
        loss = loss + _pinball(upper_control(horizon_tensor[index]), target[index], upper_q)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step % 1000 == 0 or step + 1 == int(model_config["training_steps"]):
            print(f"P172 interval step={step + 1} total_pinball={final_loss:.6f}", flush=True)
    lower_log, lower_control_log = _predict_log(lower.eval(), lower_control.eval(), score, horizon, norms)
    upper_log, upper_control_log = _predict_log(upper.eval(), upper_control.eval(), score, horizon, norms)
    actual_log = np.log1p(actual)
    conformity = np.maximum(np.minimum(lower_log, upper_log) - actual_log, actual_log - np.maximum(lower_log, upper_log))
    control_conformity = np.maximum(
        np.minimum(lower_control_log, upper_control_log) - actual_log,
        actual_log - np.maximum(lower_control_log, upper_control_log),
    )
    coverage = float(config["interval"]["coverage"])
    offsets = (
        _conformal_offset(conformity[calibration_mask], coverage),
        _conformal_offset(control_conformity[calibration_mask], coverage),
    )
    torch.save({
        "lower_state_dict": lower.state_dict(), "upper_state_dict": upper.state_dict(),
        "lower_control_state_dict": lower_control.state_dict(), "upper_control_state_dict": upper_control.state_dict(),
        "norms": norms, "score_knots": knots, "conformal_offsets": offsets,
    }, run_dir / config["model_artifact"])
    results = {}
    for cohort in config["decision_cohorts"]:
        arrays = dict(np.load(args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False))
        results[cohort["name"]] = _evaluate(
            arrays, models, frozen, lower.eval(), upper.eval(), lower_control.eval(), upper_control.eval(),
            norms, offsets, config,
        )
        print(json.dumps({cohort["name"]: results[cohort["name"]]}, indent=2), flush=True)
    decision = config["decision"]
    checks = {
        "minimum_empirical_coverage_every_cohort": all(
            float(row["empirical_interval_coverage"]) >= float(decision["minimum_empirical_coverage"])
            for row in results.values()
        ),
        "minimum_mean_interval_width_reduction": float(np.mean([
            float(row["interval_width_reduction"]) for row in results.values()
        ])) >= float(decision["minimum_mean_interval_width_reduction"]),
    }
    verdict = config["verdict_on_pass"] if all(checks.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict, "role": config["role"],
        "training": {
            "final_total_pinball_loss": final_loss, "train_trajectory_count": int(train_mask.sum()),
            "calibration_trajectory_count": int(calibration_mask.sum()),
        },
        "consumed_development_evaluations": results, "decision_checks": checks,
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
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict, "decision_checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
