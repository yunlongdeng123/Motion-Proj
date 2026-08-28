"""Train and read one held-out trajectory-conditioned Actor-state reliability model."""

from __future__ import annotations

import argparse
import json
import resource
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from motion_proj.worldsim_v67.actor_state_reliability import (
    FEATURE_NAMES, evaluate_reliability, materialize_actor_query_rows,
    predict_reliability, train_reliability_models,
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
    processed_root = Path(config["data"]["processed_root"])
    all_scenes = sorted(
        path for path in processed_root.iterdir()
        if path.is_dir() and (path / "instances" / "instances_info.json").exists()
        and (path / "lidar_pose").exists()
    )
    divisor = int(config["data"]["confirmation_scene_modulus"])
    remainder = int(config["data"]["confirmation_scene_remainder"])
    train_scenes = [path for path in all_scenes if int(path.name) % divisor != remainder]
    confirmation_scenes = [path for path in all_scenes if int(path.name) % divisor == remainder]
    train_arrays = materialize_actor_query_rows(
        train_scenes, config["data"]["training_horizons_seconds"], config["data"]
    )
    np.savez_compressed(run_dir / "TRAIN_ACTOR_QUERY_ROWS.npz", **train_arrays)
    with ThreadPoolExecutor(max_workers=1) as pool:
        confirmation_future = pool.submit(
            materialize_actor_query_rows, confirmation_scenes,
            [float(config["data"]["confirmation_horizon_seconds"])], config["data"],
        )
        torch.cuda.reset_peak_memory_stats()
        query_model, actor_model, mean, scale, training = train_reliability_models(
            train_arrays, config["model"], int(config["seed"])
        )
        confirmation_arrays = confirmation_future.result()
    np.savez_compressed(run_dir / "CONFIRMATION_ACTOR_QUERY_ROWS.npz", **confirmation_arrays)
    query_output = predict_reliability(
        query_model, confirmation_arrays["features"], mean, scale, actor_only=False
    )
    uncertainty = None
    if query_output.ndim == 2:
        query_prediction = query_output[:, 1]
        target_cost = np.asarray(confirmation_arrays["target_cost"], dtype=np.float64)
        interval_coverage = float(np.mean(
            (target_cost >= query_output[:, 0]) & (target_cost <= query_output[:, 2])
        ))
        uncertainty = {
            "quantile_levels": [float(value) for value in config["model"]["quantile_levels"]],
            "central_interval_coverage": interval_coverage,
            "mean_interval_width": float(np.mean(query_output[:, 2] - query_output[:, 0])),
            "lower_miss_fraction": float(np.mean(target_cost < query_output[:, 0])),
            "upper_miss_fraction": float(np.mean(target_cost > query_output[:, 2])),
        }
    else:
        query_prediction = query_output
    actor_prediction = predict_reliability(
        actor_model, confirmation_arrays["features"], mean, scale, actor_only=True
    )
    raw_evaluation = evaluate_reliability(
        confirmation_arrays, query_prediction, actor_prediction, config["evaluation"]
    )
    calibrator = None
    if bool(config["model"].get("train_only_monotone_affine_calibration", False)):
        train_prediction = predict_reliability(
            query_model, train_arrays["features"], mean, scale, actor_only=False
        ).astype(np.float64)
        train_target = np.asarray(train_arrays["target_cost"], dtype=np.float64)
        centered_prediction = train_prediction - train_prediction.mean()
        slope = max(
            float(np.dot(centered_prediction, train_target - train_target.mean())
                  / max(np.dot(centered_prediction, centered_prediction), 1e-12)),
            float(config["model"]["minimum_calibration_slope"]),
        )
        bias = float(train_target.mean() - slope * train_prediction.mean())
        query_prediction = np.maximum(slope * query_prediction + bias, 0.0)
        calibrator = {"slope": slope, "bias": bias, "fit_row_count": int(len(train_target))}
    evaluation = evaluate_reliability(
        confirmation_arrays, query_prediction, actor_prediction, config["evaluation"]
    )
    gates = {
        "minimum_query_conditioned_spearman": evaluation["query_conditioned_spearman"]
        >= float(config["gates"]["minimum_query_conditioned_spearman"]),
        "minimum_mae_reduction_over_actor_only": evaluation["mae_reduction_over_actor_only"]
        >= float(config["gates"]["minimum_mae_reduction_over_actor_only"]),
        "minimum_unreliable_auroc": evaluation["query_conditioned_unreliable_auroc"]
        >= float(config["gates"]["minimum_unreliable_auroc"]),
    }
    if "minimum_spearman_delta_over_actor_only" in config["gates"]:
        gates["minimum_spearman_delta_over_actor_only"] = (
            evaluation["spearman_delta_over_actor_only"]
            >= float(config["gates"]["minimum_spearman_delta_over_actor_only"])
        )
    if "minimum_interval_coverage" in config["gates"]:
        gates["central_interval_coverage"] = (
            uncertainty is not None
            and float(config["gates"]["minimum_interval_coverage"])
            <= uncertainty["central_interval_coverage"]
            <= float(config["gates"]["maximum_interval_coverage"])
        )
    verdict = config["verdict_on_pass"] if all(gates.values()) else config["verdict_on_failure"]
    artifact = {
        "feature_names": FEATURE_NAMES, "feature_mean": mean, "feature_scale": scale,
        "hidden_dimensions": list(config["model"]["hidden_dimensions"]),
        "query_model_state_dict": query_model.state_dict(),
        "actor_only_model_state_dict": actor_model.state_dict(),
        "quantile_levels": config["model"].get("quantile_levels"),
        "train_only_monotone_affine_calibrator": calibrator,
    }
    torch.save(artifact, run_dir / "TRAJECTORY_CONDITIONED_ACTOR_RELIABILITY.pt")
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"], "training": training,
        "data": {
            "processed_scene_count": len(all_scenes), "training_scene_count": len(train_scenes),
            "confirmation_scene_count": len(confirmation_scenes),
            "training_horizons_seconds": config["data"]["training_horizons_seconds"],
            "confirmation_horizon_seconds": config["data"]["confirmation_horizon_seconds"],
        },
        "confirmation": evaluation, "raw_confirmation": raw_evaluation,
        "uncertainty": uncertainty,
        "train_only_monotone_affine_calibrator": calibrator, "gate_results": gates,
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
        "status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict, "gate_results": gates}, indent=2))


if __name__ == "__main__":
    main()
