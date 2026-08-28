"""Evaluate a frozen risk-coverage operating point for Actor-state reliability."""

from __future__ import annotations

import argparse, json, resource, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from motion_proj.worldsim_v67.actor_state_reliability import (
    ACTOR_FEATURE_NAMES, BinaryReliabilityMLP, ReliabilityMLP,
    predict_binary_reliability, predict_reliability,
)


def _select_by_scene(score: np.ndarray, scenes: np.ndarray, fraction: float) -> np.ndarray:
    selected = []
    for scene in np.unique(scenes):
        members = np.flatnonzero(scenes == scene)
        count = max(1, int(np.floor(len(members) * fraction)))
        selected.extend(members[np.argsort(score[members], kind="mergesort")[:count]].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


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
    source = args.runs_root / config["source"]["run"]
    artifact = torch.load(source / config["source"]["artifact"], map_location="cuda")
    arrays = dict(np.load(source / config["source"]["confirmation_rows"], allow_pickle=False))
    hidden = artifact["hidden_dimensions"]
    continuous = ReliabilityMLP(len(artifact["feature_names"]), hidden).cuda()
    continuous.load_state_dict(artifact["continuous_model_state_dict"])
    binary_query = BinaryReliabilityMLP(len(artifact["feature_names"]), hidden).cuda()
    binary_query.load_state_dict(artifact["binary_query_model_state_dict"])
    mean = np.asarray(artifact["feature_mean"], dtype=np.float32)
    scale = np.asarray(artifact["feature_scale"], dtype=np.float32)
    continuous_score = predict_reliability(continuous.eval(), arrays["features"], mean, scale)
    binary_score = predict_binary_reliability(binary_query.eval(), arrays["features"], mean, scale)
    scenes = np.asarray(arrays["scene_index"])
    fraction = float(config["selection"]["coverage_fraction"])
    continuous_selected = _select_by_scene(continuous_score, scenes, fraction)
    binary_selected = _select_by_scene(binary_score, scenes, fraction)
    target = np.asarray(arrays["target_cost"], dtype=np.float64)
    unreliable = (
        (np.asarray(arrays["raw_actor_state_error_m"]) > float(config["evaluation"]["unreliable_actor_state_error_m"]))
        & (np.asarray(arrays["predicted_minimum_separation_m"]) <= float(config["evaluation"]["unreliable_exposure_radius_m"]))
    )
    all_cost = float(target.mean())
    all_prevalence = float(unreliable.mean())
    continuous_cost = float(target[continuous_selected].mean())
    binary_cost = float(target[binary_selected].mean())
    continuous_prevalence = float(unreliable[continuous_selected].mean())
    binary_prevalence = float(unreliable[binary_selected].mean())
    scene_rows = []
    for scene in np.unique(scenes):
        members = np.flatnonzero(scenes == scene)
        chosen = continuous_selected[np.isin(continuous_selected, members)]
        scene_rows.append({"scene_index": int(scene), "row_count": int(len(members)),
            "selected_count": int(len(chosen)), "all_mean_cost": float(target[members].mean()),
            "selected_mean_cost": float(target[chosen].mean())})
    metrics = {
        "row_count": int(len(target)), "selected_row_count": int(len(continuous_selected)),
        "achieved_coverage": float(len(continuous_selected) / len(target)),
        "all_mean_cost": all_cost, "continuous_selected_mean_cost": continuous_cost,
        "binary_selected_mean_cost": binary_cost,
        "continuous_cost_reduction": (all_cost - continuous_cost) / max(all_cost, 1e-12),
        "all_unreliable_prevalence": all_prevalence,
        "continuous_selected_unreliable_prevalence": continuous_prevalence,
        "binary_selected_unreliable_prevalence": binary_prevalence,
        "continuous_unreliable_prevalence_reduction": (all_prevalence - continuous_prevalence) / max(all_prevalence, 1e-12),
        "continuous_cost_delta_below_binary": binary_cost - continuous_cost,
        "scene_nonincreasing_count": int(sum(row["selected_mean_cost"] <= row["all_mean_cost"] for row in scene_rows)),
        "scene_count": int(len(scene_rows)), "scene_rows": scene_rows,
    }
    gates = {
        "minimum_cost_reduction": metrics["continuous_cost_reduction"] >= float(config["gates"]["minimum_cost_reduction"]),
        "minimum_unreliable_prevalence_reduction": metrics["continuous_unreliable_prevalence_reduction"] >= float(config["gates"]["minimum_unreliable_prevalence_reduction"]),
        "not_worse_than_binary_selection_cost": continuous_cost <= binary_cost,
    }
    verdict = config["verdict_on_pass"] if all(gates.values()) else config["verdict_on_failure"]
    summary = {"schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"], "selection": metrics, "gate_results": gates,
        "resources": {"gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "wall_seconds": time.monotonic() - started}, "claim_boundary": config["claim_boundary"]}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict, "gate_results": gates}, indent=2))


if __name__ == "__main__":
    main()
