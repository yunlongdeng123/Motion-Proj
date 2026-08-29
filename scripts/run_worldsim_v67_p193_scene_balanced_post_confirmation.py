"""Compare frozen P192 and P182 densities on the already-consumed P183 rows."""

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
from scripts.run_worldsim_v67_p178_clearance_conditioned_reliability_cdf import _trajectory_clearance
from scripts.run_worldsim_v67_p182_log_cost_mixture_density import LogCostMixtureDensity, _predict_cdf


def _load_density(path: Path) -> tuple[LogCostMixtureDensity, dict]:
    frozen = torch.load(path, map_location="cuda")
    model = LogCostMixtureDensity(
        int(frozen["component_count"]), [int(value) for value in frozen["hidden_dimensions"]],
    ).cuda()
    model.load_state_dict(frozen["model_state_dict"])
    return model.eval(), frozen


def _metrics(predicted: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    return (
        float(np.mean(np.square(predicted - target))),
        float(np.mean(np.abs(predicted.mean(axis=0) - target.mean(axis=0)))),
    )


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
    torch.cuda.reset_peak_memory_stats()

    rows_path = args.runs_root / config["frozen_rows"]["run"] / config["frozen_rows"]["artifact"]
    with np.load(rows_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    actor_ensemble = torch.load(
        args.runs_root / config["frozen_p126"]["run"] / config["frozen_p126"]["artifact"],
        map_location="cuda",
    )
    actor_models = []
    for state in actor_ensemble["member_state_dicts"]:
        member = DirectionalActorGaussian(20, actor_ensemble["hidden_dimensions"]).cuda()
        member.load_state_dict(state)
        actor_models.append(member.eval())
    p182, frozen_p182 = _load_density(
        args.runs_root / config["frozen_p182"]["run"] / config["frozen_p182"]["artifact"]
    )
    p192, frozen_p192 = _load_density(
        args.runs_root / config["frozen_p192"]["run"] / config["frozen_p192"]["artifact"]
    )

    score, scenes = _ensemble_trajectory_score(
        arrays, actor_models,
        np.asarray(actor_ensemble["feature_mean"], dtype=np.float32),
        np.asarray(actor_ensemble["feature_scale"], dtype=np.float32),
        np.asarray(actor_ensemble["target_mean"], dtype=np.float32),
        np.asarray(actor_ensemble["target_scale"], dtype=np.float32),
    )
    cost, cost_scenes = _continuous_cost(
        arrays, float(config["boundary_state_cost"]["clearance_floor_m"]),
    )
    if not np.array_equal(scenes, cost_scenes):
        raise RuntimeError("P193 trajectory grouping is not aligned")
    horizon = _trajectory_horizon(arrays)
    clearance = _trajectory_clearance(arrays)
    budgets = np.asarray(config["reliability_budgets"], dtype=np.float32)
    target = cost[:, None] <= budgets[None]
    p182_probability = _predict_cdf(
        p182, score, horizon, clearance, budgets, tuple(frozen_p182["norms"]),
    )
    p192_probability = _predict_cdf(
        p192, score, horizon, clearance, budgets, tuple(frozen_p192["norms"]),
    )

    per_horizon = {}
    for value in config["horizons_seconds"]:
        mask = np.isclose(horizon, float(value))
        p182_brier, p182_calibration = _metrics(p182_probability[mask], target[mask])
        p192_brier, p192_calibration = _metrics(p192_probability[mask], target[mask])
        per_horizon[str(value)] = {
            "trajectory_count": int(np.count_nonzero(mask)),
            "p182_integrated_brier": p182_brier,
            "p192_integrated_brier": p192_brier,
            "p192_brier_reduction_vs_p182": float((p182_brier - p192_brier) / p182_brier),
            "p182_mean_absolute_reliability_error": p182_calibration,
            "p192_mean_absolute_reliability_error": p192_calibration,
            "p192_calibration_error_reduction_vs_p182": float(
                (p182_calibration - p192_calibration) / max(p182_calibration, 1e-12)
            ),
        }

    brier_reductions = [row["p192_brier_reduction_vs_p182"] for row in per_horizon.values()]
    calibration_reductions = [
        row["p192_calibration_error_reduction_vs_p182"] for row in per_horizon.values()
    ]
    macro = {
        "mean_brier_reduction_vs_p182": float(np.mean(brier_reductions)),
        "mean_calibration_error_reduction_vs_p182": float(np.mean(calibration_reductions)),
    }
    decisions = {
        "every_horizon_brier_noninferior_vs_p182": bool(np.min(brier_reductions) >= 0.0),
        "minimum_mean_calibration_error_reduction_vs_p182": macro[
            "mean_calibration_error_reduction_vs_p182"
        ] >= float(config["decision"]["minimum_mean_calibration_error_reduction_vs_p182"]),
    }
    verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"], "per_horizon": per_horizon, "macro_metrics": macro,
        "decision_checks": decisions,
        "resources": {
            "gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "wall_seconds": time.monotonic() - started,
        },
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({
        "status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
