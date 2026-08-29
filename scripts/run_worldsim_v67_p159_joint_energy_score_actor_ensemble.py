"""Jointly train the three-member Actor ensemble with a multivariate Energy Score."""

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

from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian, _actor_entries
from scripts.run_worldsim_v67_p158_crps_actor_ensemble import _evaluate as _evaluate_candidate, _subset


def _energy_score(
    target: torch.Tensor, means: list[torch.Tensor], scales: list[torch.Tensor],
) -> torch.Tensor:
    sample_a = [mean + scale * torch.randn_like(mean) for mean, scale in zip(means, scales)]
    sample_b = [mean + scale * torch.randn_like(mean) for mean, scale in zip(means, scales)]
    observation_distance = torch.stack([
        torch.linalg.vector_norm(sample - target, dim=1) for sample in sample_a
    ]).mean()
    forecast_distance = torch.stack([
        torch.linalg.vector_norm(left - right, dim=1)
        for left in sample_a for right in sample_b
    ]).mean()
    return observation_distance - 0.5 * forecast_distance


def _evaluate(*args, **kwargs) -> dict[str, float | int]:
    result = _evaluate_candidate(*args, **kwargs)
    return {name.replace("crps_", "energy_"): value for name, value in result.items()}


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
    source = dict(np.load(
        args.runs_root / config["source_rows"]["run"] / config["source_rows"]["artifact"], allow_pickle=False,
    ))
    raw_features, raw_target, _ = _actor_entries(source)
    frozen = torch.load(
        args.runs_root / config["frozen_p126"]["run"] / config["frozen_p126"]["artifact"], map_location="cuda",
    )
    feature_mean = np.asarray(frozen["feature_mean"], dtype=np.float32)
    feature_scale = np.asarray(frozen["feature_scale"], dtype=np.float32)
    target_mean = np.asarray(frozen["target_mean"], dtype=np.float32)
    target_scale = np.asarray(frozen["target_scale"], dtype=np.float32)
    features = torch.from_numpy((raw_features - feature_mean) / feature_scale).cuda()
    target = torch.from_numpy((raw_target - target_mean) / target_scale).cuda()
    model_config = config["model"]
    baseline = []
    for state in frozen["member_state_dicts"]:
        model = DirectionalActorGaussian(features.shape[1], frozen["hidden_dimensions"]).cuda()
        model.load_state_dict(state)
        baseline.append(model.eval())
    candidate = []
    parameters = []
    for seed_value in config["member_seeds"]:
        torch.manual_seed(int(seed_value))
        model = DirectionalActorGaussian(features.shape[1], model_config["hidden_dimensions"]).cuda()
        candidate.append(model)
        parameters.extend(model.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"]),
    )
    final_loss = 0.0
    torch.cuda.reset_peak_memory_stats()
    for step in range(int(model_config["joint_steps"])):
        index = torch.randint(len(features), (int(model_config["batch_size"]),), device="cuda")
        predictions = [model(features[index]) for model in candidate]
        loss = _energy_score(target[index], [item[0] for item in predictions], [item[1] for item in predictions])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step % 500 == 0 or step + 1 == int(model_config["joint_steps"]):
            print(f"P159 joint step={step + 1} energy_score={final_loss:.6f}", flush=True)
    candidate = [model.eval() for model in candidate]
    torch.save({
        "feature_mean": feature_mean, "feature_scale": feature_scale,
        "target_mean": target_mean, "target_scale": target_scale,
        "hidden_dimensions": model_config["hidden_dimensions"], "member_seeds": config["member_seeds"],
        "member_state_dicts": [model.state_dict() for model in candidate],
    }, run_dir / config["model_artifact"])
    coverage = float(config["selection"]["coverage_fraction"])
    floor = float(config["boundary_state_cost"]["clearance_floor_m"])
    decision_results = {}
    for cohort in config["decision_cohorts"]:
        arrays = dict(np.load(args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False))
        decision_results[cohort["name"]] = _evaluate(
            arrays, candidate, baseline, feature_mean, feature_scale, target_mean, target_scale, coverage, floor,
        )
        print(json.dumps({cohort["name"]: decision_results[cohort["name"]]}, indent=2), flush=True)
    diagnostic_spec = config["post_confirmation_diagnostic"]
    diagnostic_arrays = dict(np.load(
        args.runs_root / diagnostic_spec["run"] / diagnostic_spec["artifact"], allow_pickle=False,
    ))
    diagnostic_results = {}
    for horizon in diagnostic_spec["horizons_seconds"]:
        key = str(float(horizon))
        diagnostic_results[key] = _evaluate(
            _subset(diagnostic_arrays, float(horizon)), candidate, baseline,
            feature_mean, feature_scale, target_mean, target_scale, coverage, floor,
        )
        print(json.dumps({f"P147_H{key}": diagnostic_results[key]}, indent=2), flush=True)
    gains = [row["spearman_gain_over_p126"] for row in decision_results.values()]
    decisions = {
        "no_selected_cost_regression": all(row["energy_selected_mean_cost"] <= row["p126_selected_mean_cost"] for row in decision_results.values()),
        "minimum_mean_spearman_gain": float(np.mean(gains)) >= float(config["decision"]["minimum_mean_spearman_gain_over_p126"]),
    }
    verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict, "role": config["role"],
        "training": {"actor_time_tokens": int(len(features)), "member_count": len(candidate), "final_joint_energy_score": final_loss},
        "consumed_development_evaluations": decision_results,
        "post_confirmation_consumed_p147_diagnostic": diagnostic_results,
        "decision_checks": decisions, "mean_spearman_gain": float(np.mean(gains)),
        "resources": {"gpu": torch.cuda.get_device_name(0), "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                      "wall_seconds": time.monotonic() - started},
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({
        "status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict, "decision_checks": decisions}, indent=2), flush=True)


if __name__ == "__main__":
    main()
