"""Ablate the conditional correlation term of the frozen P117 model."""

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

import scripts.run_worldsim_v67_p117_full_covariance_actor_uncertainty as p117


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config["runs_root"] = str(args.runs_root)
    run_dir = args.runs_root / "worldsim_v67" / config["task_id"] / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()

    frozen_path = args.runs_root / config["frozen_p117"]["run"] / config["frozen_p117"]["artifact"]
    frozen = torch.load(frozen_path, map_location="cuda")
    feature_mean = np.asarray(frozen["feature_mean"], dtype=np.float32)
    feature_scale = np.asarray(frozen["feature_scale"], dtype=np.float32)
    target_mean = np.asarray(frozen["target_mean"], dtype=np.float32)
    target_scale = np.asarray(frozen["target_scale"], dtype=np.float32)
    model = p117.CorrelatedActorGaussian(len(feature_mean), frozen["hidden_dimensions"]).cuda()
    model.load_state_dict(frozen["model_state_dict"])
    model.eval()
    prep_run = args.runs_root / config["source_rows"]["run"]

    conditional, zero_correlation = {}, {}
    original_predict = p117._predict
    for cohort in config["development_cohorts"]:
        arrays = dict(np.load(prep_run / cohort["artifact"], allow_pickle=False))
        conditional[cohort["name"]] = p117._evaluate(
            arrays, model, feature_mean, feature_scale, target_mean, target_scale, config,
        )

    def predict_zero_correlation(*predict_args):
        mean, scale, correlation = original_predict(*predict_args)
        return mean, scale, np.zeros_like(correlation)

    p117._predict = predict_zero_correlation
    try:
        for cohort in config["development_cohorts"]:
            arrays = dict(np.load(prep_run / cohort["artifact"], allow_pickle=False))
            zero_correlation[cohort["name"]] = p117._evaluate(
                arrays, model, feature_mean, feature_scale, target_mean, target_scale, config,
            )
    finally:
        p117._predict = original_predict

    gains = {
        name: float(conditional[name]["query_event_auroc"])
        - float(zero_correlation[name]["query_event_auroc"])
        for name in conditional
    }
    event_noninferior = all(
        conditional[name]["query_selected_occupancy_flip_events"]
        <= zero_correlation[name]["query_selected_occupancy_flip_events"]
        for name in conditional
    )
    mean_gain = float(np.mean(list(gains.values())))
    supported = event_noninferior and all(x > 0.0 for x in gains.values()) and mean_gain >= float(
        config["decision"]["minimum_mean_auroc_gain_from_conditional_correlation"]
    )
    verdict = "supported_conditional_correlation_mechanism" if supported else \
        "rejected_conditional_correlation_mechanism"
    summary = {
        "schema_version": config["output_schema_version"],
        "task_id": config["task_id"], "hypothesis_id": config["hypothesis_id"],
        "status": "done", "verdict": verdict, "role": config["role"],
        "conditional_correlation": conditional,
        "zero_correlation_same_checkpoint": zero_correlation,
        "decision_metrics": {"event_noninferior": event_noninferior,
                             "per_cohort_auroc_gain": gains, "mean_auroc_gain": mean_gain},
        "resources": {"gpu": torch.cuda.get_device_name(0),
                      "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                      "wall_seconds": time.monotonic() - started},
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({
        "status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict,
                      "decision_metrics": summary["decision_metrics"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
