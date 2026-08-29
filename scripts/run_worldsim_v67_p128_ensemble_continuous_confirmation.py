"""Prospectively evaluate the frozen P126 ensemble on the P121 same read."""

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

from motion_proj.worldsim_v67.actor_state_reliability import spearman_correlation
from scripts.run_worldsim_v67_p86_direct_trajectory_reliability import _select_by_scene
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p114_monotone_tail_risk import _crossing_probability, _trajectory_tail_features
from scripts.run_worldsim_v67_p119_ranked_range_tail import _head_features
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost
from scripts.run_worldsim_v67_p127_ensemble_continuous_selection import _ensemble_trajectory_score


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
    rows_path = args.runs_root / config["p121_same_read"]["run"] / config["p121_same_read"]["artifact"]
    deadline = time.monotonic() + float(config["p121_same_read"]["readiness_timeout_seconds"])
    while not rows_path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"P128 P121 same-read rows not ready: {rows_path}")
        time.sleep(10.0)
    arrays = dict(np.load(rows_path, allow_pickle=False))
    ensemble = torch.load(
        args.runs_root / config["frozen_p126"]["run"] / config["frozen_p126"]["artifact"],
        map_location="cuda",
    )
    models = []
    for state in ensemble["member_state_dicts"]:
        model = DirectionalActorGaussian(20, ensemble["hidden_dimensions"]).cuda()
        model.load_state_dict(state)
        models.append(model.eval())
    ensemble_score, scenes = _ensemble_trajectory_score(
        arrays, models,
        np.asarray(ensemble["feature_mean"], dtype=np.float32),
        np.asarray(ensemble["feature_scale"], dtype=np.float32),
        np.asarray(ensemble["target_mean"], dtype=np.float32),
        np.asarray(ensemble["target_scale"], dtype=np.float32),
    )
    p109 = torch.load(
        args.runs_root / config["frozen_p109"]["run"] / config["frozen_p109"]["artifact"],
        map_location="cuda",
    )
    p109_model = DirectionalActorGaussian(20, p109["hidden_dimensions"]).cuda()
    p109_model.load_state_dict(p109["model_state_dict"])
    p109_model.eval()
    p109_probability, _ = _crossing_probability(
        arrays, p109_model,
        np.asarray(p109["feature_mean"], dtype=np.float32),
        np.asarray(p109["feature_scale"], dtype=np.float32),
        np.asarray(p109["target_mean"], dtype=np.float32),
        np.asarray(p109["target_scale"], dtype=np.float32),
    )
    grouped = _trajectory_tail_features(
        arrays, p109_probability, int(config["score"]["top_k_crossing_probabilities"]),
    )
    _, p109_score = _head_features(grouped)
    actual_cost, cost_scenes = _continuous_cost(
        arrays, float(config["boundary_state_cost"]["clearance_floor_m"]),
    )
    if not (np.array_equal(scenes, cost_scenes) and np.array_equal(scenes, grouped["scene_index"])):
        raise RuntimeError("P128 trajectory grouping is not aligned")
    clearance = grouped["clearance_score"]
    coverage = float(config["selection"]["coverage_fraction"])
    ensemble_selected = _select_by_scene(ensemble_score, scenes, coverage)
    p109_selected = _select_by_scene(p109_score, scenes, coverage)
    clearance_selected = _select_by_scene(clearance, scenes, coverage)
    ensemble_spearman = spearman_correlation(actual_cost, ensemble_score)
    p109_spearman = spearman_correlation(actual_cost, p109_score)
    mean_cost = float(actual_cost.mean())
    ensemble_cost = float(actual_cost[ensemble_selected].mean())
    p109_cost = float(actual_cost[p109_selected].mean())
    metrics = {
        "row_count": int(len(arrays["features"])),
        "trajectory_count": int(len(actual_cost)),
        "selected_trajectory_count": int(len(ensemble_selected)),
        "all_mean_boundary_state_cost": mean_cost,
        "ensemble_selected_mean_cost": ensemble_cost,
        "p109_selected_mean_cost": p109_cost,
        "clearance_selected_mean_cost": float(actual_cost[clearance_selected].mean()),
        "ensemble_selected_cost_reduction": float((mean_cost - ensemble_cost) / max(mean_cost, 1e-12)),
        "ensemble_cost_spearman": ensemble_spearman,
        "p109_cost_spearman": p109_spearman,
        "clearance_cost_spearman": spearman_correlation(actual_cost, clearance),
        "ensemble_spearman_gain_over_p109": ensemble_spearman - p109_spearman,
    }
    decisions = {
        "minimum_spearman_gain_over_p109": metrics["ensemble_spearman_gain_over_p109"]
        >= float(config["decision"]["minimum_spearman_gain_over_p109"]),
        "selected_cost_noninferior_to_p109": ensemble_cost <= p109_cost,
    }
    verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"], "same_read_secondary_evaluation": metrics,
        "decision_checks": decisions,
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
                      "same_read_secondary_evaluation": metrics,
                      "decision_checks": decisions}, indent=2), flush=True)


if __name__ == "__main__":
    main()
