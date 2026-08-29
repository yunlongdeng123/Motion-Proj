"""Train a correlated heavy-tailed Student-t Actor residual field."""

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

from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import _actor_entries
from scripts.run_worldsim_v67_p117_full_covariance_actor_uncertainty import (
    CorrelatedActorGaussian, _evaluate,
)


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
    torch.manual_seed(int(config["seed"]))
    source_root = args.runs_root / config["source_rows"]["run"]
    source = dict(np.load(source_root / config["source_rows"]["artifact"], allow_pickle=False))
    raw_features, raw_target, _ = _actor_entries(source)
    feature_mean, feature_scale = raw_features.mean(0), raw_features.std(0).clip(min=1e-4)
    target_mean, target_scale = raw_target.mean(0), raw_target.std(0).clip(min=0.05)
    features = torch.from_numpy((raw_features - feature_mean) / feature_scale).cuda()
    target = torch.from_numpy((raw_target - target_mean) / target_scale).cuda()
    model_config = config["model"]
    degrees_of_freedom = float(model_config["degrees_of_freedom"])
    model = CorrelatedActorGaussian(features.shape[1], model_config["hidden_dimensions"]).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    steps = int(model_config["steps"])
    batch_size = int(model_config["batch_size"])
    final_loss = 0.0
    torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        indices = torch.randint(len(features), (batch_size,), device="cuda")
        mean, scale, correlation = model(features[indices])
        residual = (target[indices] - mean) / scale
        one_minus_rho2 = 1.0 - correlation.square()
        mahalanobis = (
            residual[:, 0].square() + residual[:, 1].square()
            - 2.0 * correlation * residual[:, 0] * residual[:, 1]
        ) / one_minus_rho2
        loss = (
            torch.log(scale).sum(dim=1) + 0.5 * torch.log(one_minus_rho2)
            + 0.5 * (degrees_of_freedom + 2.0)
            * torch.log1p(mahalanobis / degrees_of_freedom)
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step % 250 == 0 or step + 1 == steps:
            print(f"P124 correlated-student-t step={step + 1} nll={final_loss:.6f}", flush=True)
    torch.save({
        "feature_mean": feature_mean, "feature_scale": feature_scale,
        "target_mean": target_mean, "target_scale": target_scale,
        "hidden_dimensions": model_config["hidden_dimensions"],
        "degrees_of_freedom": degrees_of_freedom,
        "model_state_dict": model.state_dict(),
    }, run_dir / config["model_artifact"])
    results = {}
    for cohort in config["development_cohorts"]:
        cohort_root = args.runs_root / cohort.get("run", config["source_rows"]["run"])
        arrays = dict(np.load(cohort_root / cohort["artifact"], allow_pickle=False))
        results[cohort["name"]] = _evaluate(
            arrays, model, feature_mean, feature_scale, target_mean, target_scale, config,
        )
        print(json.dumps({cohort["name"]: results[cohort["name"]]}, indent=2), flush=True)
    comparisons = config["frozen_p109_comparisons"]
    event_noninferior = all(
        results[name]["query_selected_occupancy_flip_events"] <= int(reference["events"])
        for name, reference in comparisons.items()
    )
    gains = [
        float(results[name]["query_event_auroc"]) - float(reference["auroc"])
        for name, reference in comparisons.items()
    ]
    mean_gain = float(np.mean(gains))
    decisions = {
        "event_noninferior_to_p109": event_noninferior,
        "minimum_mean_auroc_gain": mean_gain
        >= float(config["decision"]["minimum_mean_auroc_gain_over_p109"]),
    }
    verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "training": {"actor_time_tokens": int(len(features)),
                     "degrees_of_freedom": degrees_of_freedom,
                     "final_correlated_student_t_nll": final_loss},
        "development_evaluations": results,
        "decision_metrics": {"decision_checks": decisions,
                             "per_cohort_auroc_gain_over_p109": dict(zip(comparisons, gains)),
                             "mean_auroc_gain_over_p109": mean_gain},
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
