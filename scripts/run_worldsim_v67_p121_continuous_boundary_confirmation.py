"""Confirm frozen P109 ranking of continuous trajectory-conditioned boundary-state cost."""

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

from motion_proj.worldsim_v67.actor_state_reliability import (
    materialize_actor_query_rows, spearman_correlation,
)
from scripts.run_worldsim_v67_p86_direct_trajectory_reliability import _select_by_scene
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p114_monotone_tail_risk import (
    _crossing_probability, _trajectory_tail_features,
)
from scripts.run_worldsim_v67_p119_ranked_range_tail import _head_features
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost


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
    scene_dirs = [
        Path(data["processed_root"]) / f"{index_by_name[str(name)]:03d}"
        for name in data["scene_names"]
    ]
    deadline = time.monotonic() + float(data["readiness_timeout_seconds"])
    while not all(
        (scene / "instances" / "instances_info.json").is_file() and (scene / "lidar_pose").is_dir()
        for scene in scene_dirs
    ):
        if time.monotonic() >= deadline:
            raise TimeoutError("P121 processed scenes not ready")
        time.sleep(10.0)
    raw = materialize_actor_query_rows(scene_dirs, data["horizons_seconds"], data)
    partial = run_dir / "P121_CONTINUOUS_BOUNDARY_CONFIRMATION_ROWS.partial.npz"
    np.savez_compressed(partial, **raw)
    partial.replace(run_dir / "P121_CONTINUOUS_BOUNDARY_CONFIRMATION_ROWS.npz")
    checkpoint = torch.load(
        args.runs_root / config["frozen_p109"]["run"] / config["frozen_p109"]["artifact"],
        map_location="cuda",
    )
    actor_model = DirectionalActorGaussian(20, checkpoint["hidden_dimensions"]).cuda()
    actor_model.load_state_dict(checkpoint["model_state_dict"])
    actor_model.eval()
    torch.cuda.reset_peak_memory_stats()
    probability, _ = _crossing_probability(
        raw, actor_model,
        np.asarray(checkpoint["feature_mean"], dtype=np.float32),
        np.asarray(checkpoint["feature_scale"], dtype=np.float32),
        np.asarray(checkpoint["target_mean"], dtype=np.float32),
        np.asarray(checkpoint["target_scale"], dtype=np.float32),
    )
    top_k = int(config["evaluation"]["top_k_crossing_probabilities"])
    grouped = _trajectory_tail_features(raw, probability, top_k)
    _, p109_score = _head_features(grouped)
    actual_cost, scenes = _continuous_cost(
        raw, float(config["boundary_state_cost"]["clearance_floor_m"]),
    )
    if len(actual_cost) != len(p109_score) or not np.array_equal(scenes, grouped["scene_index"]):
        raise RuntimeError("P121 continuous cost is not aligned with P109 trajectory groups")
    clearance_score = grouped["clearance_score"]
    coverage = float(config["selection"]["coverage_fraction"])
    selected = _select_by_scene(p109_score, scenes, coverage)
    clearance_selected = _select_by_scene(clearance_score, scenes, coverage)
    mean_cost = float(actual_cost.mean())
    selected_cost = float(actual_cost[selected].mean())
    clearance_cost = float(actual_cost[clearance_selected].mean())
    p109_spearman = spearman_correlation(actual_cost, p109_score)
    clearance_spearman = spearman_correlation(actual_cost, clearance_score)
    metrics = {
        "row_count": int(len(raw["features"])),
        "trajectory_count": int(len(actual_cost)),
        "selected_trajectory_count": int(len(selected)),
        "all_mean_boundary_state_cost": mean_cost,
        "p109_selected_mean_boundary_state_cost": selected_cost,
        "clearance_selected_mean_boundary_state_cost": clearance_cost,
        "p109_selected_cost_reduction": float((mean_cost - selected_cost) / max(mean_cost, 1e-12)),
        "clearance_selected_cost_reduction": float((mean_cost - clearance_cost) / max(mean_cost, 1e-12)),
        "p109_cost_spearman": p109_spearman,
        "clearance_cost_spearman": clearance_spearman,
        "p109_spearman_gain_over_clearance": p109_spearman - clearance_spearman,
    }
    decision = config["decision"]
    decisions = {
        "ranking_and_geometry_gain": p109_spearman >= float(decision["minimum_p109_spearman"])
        and p109_spearman - clearance_spearman >= float(decision["minimum_spearman_gain_over_clearance"]),
        "fixed_coverage_cost_and_geometry_gain": metrics["p109_selected_cost_reduction"]
        >= float(decision["minimum_selected_cost_reduction"])
        and selected_cost <= clearance_cost,
    }
    verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"], "fresh_confirmation_evaluation": metrics,
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
                      "fresh_confirmation_evaluation": metrics,
                      "decision_checks": decisions}, indent=2), flush=True)


if __name__ == "__main__":
    main()
