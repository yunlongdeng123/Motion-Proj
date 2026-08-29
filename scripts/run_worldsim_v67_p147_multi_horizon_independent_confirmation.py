"""Confirm the frozen P126 ensemble across five horizons on new scenes."""

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

from motion_proj.worldsim_v67.actor_state_reliability import materialize_actor_query_rows, spearman_correlation
from scripts.run_worldsim_v67_p86_direct_trajectory_reliability import _select_by_scene
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p114_monotone_tail_risk import _crossing_probability, _trajectory_tail_features
from scripts.run_worldsim_v67_p119_ranked_range_tail import _head_features
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost
from scripts.run_worldsim_v67_p127_ensemble_continuous_selection import _ensemble_trajectory_score


def _slice_rows(arrays: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.asarray(value)[mask] for name, value in arrays.items()}


def _evaluate_horizon(
    arrays: dict[str, np.ndarray], models: list[DirectionalActorGaussian], ensemble: dict,
    p109_model: DirectionalActorGaussian, p109: dict, config: dict,
) -> dict[str, float | int]:
    ensemble_score, scenes = _ensemble_trajectory_score(
        arrays, models,
        np.asarray(ensemble["feature_mean"], dtype=np.float32),
        np.asarray(ensemble["feature_scale"], dtype=np.float32),
        np.asarray(ensemble["target_mean"], dtype=np.float32),
        np.asarray(ensemble["target_scale"], dtype=np.float32),
    )
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
        raise RuntimeError("P147 trajectory grouping is not aligned")
    coverage = float(config["selection"]["coverage_fraction"])
    ensemble_selected = _select_by_scene(ensemble_score, scenes, coverage)
    p109_selected = _select_by_scene(p109_score, scenes, coverage)
    ensemble_spearman = spearman_correlation(actual_cost, ensemble_score)
    p109_spearman = spearman_correlation(actual_cost, p109_score)
    ensemble_cost = float(actual_cost[ensemble_selected].mean())
    p109_cost = float(actual_cost[p109_selected].mean())
    return {
        "row_count": int(len(arrays["features"])),
        "trajectory_count": int(len(actual_cost)),
        "selected_trajectory_count": int(len(ensemble_selected)),
        "all_mean_boundary_state_cost": float(actual_cost.mean()),
        "ensemble_selected_mean_cost": ensemble_cost,
        "p109_selected_mean_cost": p109_cost,
        "ensemble_minus_p109_selected_cost": ensemble_cost - p109_cost,
        "ensemble_cost_spearman": ensemble_spearman,
        "p109_cost_spearman": p109_spearman,
        "ensemble_spearman_gain_over_p109": ensemble_spearman - p109_spearman,
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
    data = config["evaluation_data"]
    metadata = Path(data["metadata_root"]) / "v1.0-trainval"
    scene_table = json.loads((metadata / "scene.json").read_text(encoding="utf-8"))
    index_by_name = {str(row["name"]): index for index, row in enumerate(scene_table)}
    scene_dirs = [Path(data["processed_root"]) / f"{index_by_name[str(name)]:03d}" for name in data["scene_names"]]
    deadline = time.monotonic() + float(data["readiness_timeout_seconds"])
    while not all((scene / "instances" / "instances_info.json").is_file() and (scene / "lidar_pose").is_dir() for scene in scene_dirs):
        if time.monotonic() >= deadline:
            raise TimeoutError("P147 processed scenes not ready")
        time.sleep(10.0)
    arrays = materialize_actor_query_rows(scene_dirs, data["horizons_seconds"], data)
    partial = run_dir / "P147_MULTI_HORIZON_INDEPENDENT_ROWS.partial.npz"
    np.savez_compressed(partial, **arrays)
    partial.replace(run_dir / "P147_MULTI_HORIZON_INDEPENDENT_ROWS.npz")
    ensemble = torch.load(args.runs_root / config["frozen_p126"]["run"] / config["frozen_p126"]["artifact"], map_location="cuda")
    models = []
    for state in ensemble["member_state_dicts"]:
        model = DirectionalActorGaussian(20, ensemble["hidden_dimensions"]).cuda()
        model.load_state_dict(state)
        models.append(model.eval())
    p109 = torch.load(args.runs_root / config["frozen_p109"]["run"] / config["frozen_p109"]["artifact"], map_location="cuda")
    p109_model = DirectionalActorGaussian(20, p109["hidden_dimensions"]).cuda()
    p109_model.load_state_dict(p109["model_state_dict"])
    p109_model.eval()
    results = {}
    for horizon in data["horizons_seconds"]:
        mask = np.isclose(arrays["horizon_seconds"], float(horizon))
        results[str(horizon)] = _evaluate_horizon(
            _slice_rows(arrays, mask), models, ensemble, p109_model, p109, config,
        )
        print(json.dumps({str(horizon): results[str(horizon)]}, indent=2), flush=True)
    gains = [float(row["ensemble_spearman_gain_over_p109"]) for row in results.values()]
    cost_differences = [float(row["ensemble_minus_p109_selected_cost"]) for row in results.values()]
    mean_gain = float(np.mean(gains))
    mean_cost_difference = float(np.mean(cost_differences))
    decisions = {
        "minimum_mean_per_horizon_spearman_gain_over_p109": mean_gain >= float(config["decision"]["minimum_mean_per_horizon_spearman_gain_over_p109"]),
        "mean_per_horizon_selected_cost_noninferior_to_p109": mean_cost_difference <= 0.0,
    }
    verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"], "fresh_independent_per_horizon_evaluations": results,
        "macro_metrics": {"mean_per_horizon_spearman_gain_over_p109": mean_gain,
                          "mean_per_horizon_ensemble_minus_p109_selected_cost": mean_cost_difference},
        "decision_checks": decisions,
        "resources": {"gpu": torch.cuda.get_device_name(0),
                      "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                      "wall_seconds": time.monotonic() - started},
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict, "macro_metrics": summary["macro_metrics"], "decision_checks": decisions}, indent=2), flush=True)


if __name__ == "__main__":
    main()
