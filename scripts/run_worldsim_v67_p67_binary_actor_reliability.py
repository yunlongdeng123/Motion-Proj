"""Train a direct given-trajectory Actor-state unreliability classifier."""

from __future__ import annotations

import argparse, json, resource, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from motion_proj.worldsim_v67.actor_state_reliability import (
    FEATURE_NAMES, binary_auroc, materialize_actor_query_rows, predict_binary_reliability,
    predict_reliability, train_binary_reliability_models,
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
    root = Path(config["data"]["processed_root"])
    scenes = sorted(path for path in root.iterdir() if (path / "instances" / "instances_info.json").exists())
    divisor = int(config["data"]["confirmation_scene_modulus"])
    remainder = int(config["data"]["confirmation_scene_remainder"])
    train_scenes = [path for path in scenes if int(path.name) % divisor != remainder]
    confirmation_scenes = [path for path in scenes if int(path.name) % divisor == remainder]
    train_arrays = materialize_actor_query_rows(
        train_scenes, config["data"]["training_horizons_seconds"], config["data"]
    )
    np.savez_compressed(run_dir / "TRAIN_BINARY_ACTOR_QUERY_ROWS.npz", **train_arrays)
    with ThreadPoolExecutor(max_workers=1) as pool:
        confirmation_future = pool.submit(
            materialize_actor_query_rows, confirmation_scenes,
            [float(config["data"]["confirmation_horizon_seconds"])], config["data"],
        )
        torch.cuda.reset_peak_memory_stats()
        continuous_model, query_model, actor_model, mean, scale, training = train_binary_reliability_models(
            train_arrays, config["model"], config["evaluation"], int(config["seed"])
        )
        confirmation = confirmation_future.result()
    np.savez_compressed(run_dir / "CONFIRMATION_BINARY_ACTOR_QUERY_ROWS.npz", **confirmation)
    query_probability = predict_binary_reliability(query_model, confirmation["features"], mean, scale)
    actor_probability = predict_binary_reliability(
        actor_model, confirmation["features"], mean, scale, actor_only=True
    )
    continuous_score = predict_reliability(
        continuous_model, confirmation["features"], mean, scale
    )
    labels = (
        (np.asarray(confirmation["raw_actor_state_error_m"]) > float(config["evaluation"]["unreliable_actor_state_error_m"]))
        & (np.asarray(confirmation["predicted_minimum_separation_m"]) <= float(config["evaluation"]["unreliable_exposure_radius_m"]))
    )
    metrics = {
        "row_count": int(len(labels)), "unreliable_row_count": int(np.count_nonzero(labels)),
        "binary_query_auroc": binary_auroc(labels, query_probability),
        "continuous_query_auroc": binary_auroc(labels, continuous_score),
        "binary_actor_only_auroc": binary_auroc(labels, actor_probability),
        "delta_over_continuous_query": binary_auroc(labels, query_probability) - binary_auroc(labels, continuous_score),
        "delta_over_binary_actor_only": binary_auroc(labels, query_probability) - binary_auroc(labels, actor_probability),
        "binary_query_brier": float(np.mean((query_probability - labels.astype(np.float32)) ** 2)),
        "binary_actor_only_brier": float(np.mean((actor_probability - labels.astype(np.float32)) ** 2)),
    }
    gates = {
        "minimum_binary_query_auroc": metrics["binary_query_auroc"] >= float(config["gates"]["minimum_binary_query_auroc"]),
        "minimum_delta_over_continuous_query": metrics["delta_over_continuous_query"] >= float(config["gates"]["minimum_delta_over_continuous_query"]),
        "minimum_delta_over_binary_actor_only": metrics["delta_over_binary_actor_only"] >= float(config["gates"]["minimum_delta_over_binary_actor_only"]),
    }
    verdict = config["verdict_on_pass"] if all(gates.values()) else config["verdict_on_failure"]
    torch.save({
        "feature_names": FEATURE_NAMES, "feature_mean": mean, "feature_scale": scale,
        "hidden_dimensions": config["model"]["hidden_dimensions"],
        "continuous_model_state_dict": continuous_model.state_dict(),
        "binary_query_model_state_dict": query_model.state_dict(),
        "binary_actor_only_model_state_dict": actor_model.state_dict(),
    }, run_dir / "BINARY_ACTOR_STATE_RELIABILITY.pt")
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"], "training": training,
        "data": {"training_scene_count": len(train_scenes), "confirmation_scene_count": len(confirmation_scenes)},
        "confirmation": metrics, "gate_results": gates,
        "resources": {"gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "wall_seconds": time.monotonic() - started},
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict, "gate_results": gates}, indent=2))


if __name__ == "__main__":
    main()
