"""Compare fixed-sample nonlinear Gaussian occupancy crossing with P109 linear projection."""

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

from motion_proj.worldsim_v67.actor_state_reliability import binary_auroc
from scripts.run_worldsim_v67_p86_direct_trajectory_reliability import _select_by_scene
from scripts.run_worldsim_v67_p87_deepset_trajectory_reliability import _build_sets
from scripts.run_worldsim_v67_p104_temporal_flip_supervision import _aligned_group_max
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import (
    DirectionalActorGaussian, _actor_entries, _predict,
)


def _evaluate(
    arrays: dict[str, np.ndarray], model: DirectionalActorGaussian, checkpoint: dict,
    sample_count: int, batch_size: int,
) -> dict[str, float | int]:
    actor_features, _, inverse = _actor_entries(arrays)
    point_count = int(arrays["actor_position_error_vector_ego_profile_m"].shape[1])
    mean, scale = _predict(
        model, actor_features, checkpoint["feature_mean"], checkpoint["feature_scale"],
        checkpoint["target_mean"], checkpoint["target_scale"],
    )
    mean = mean.reshape(-1, point_count, 2)[inverse]
    scale = scale.reshape(-1, point_count, 2)[inverse]
    normal = np.asarray(arrays["query_boundary_normal_ego_profile"], dtype=np.float32)
    distance = np.asarray(arrays["predicted_separation_profile_m"], dtype=np.float32)
    relative = normal * distance[..., None]
    radius = np.asarray(arrays["occupancy_interaction_radius_m"], dtype=np.float32)
    row_scores = []
    for start in range(0, len(distance), batch_size):
        end = min(len(distance), start + batch_size)
        relative_batch = torch.from_numpy(relative[start:end]).cuda()[:, :, None, :]
        mean_batch = torch.from_numpy(mean[start:end]).cuda()[:, :, None, :]
        scale_batch = torch.from_numpy(scale[start:end]).cuda()[:, :, None, :]
        radius_batch = torch.from_numpy(radius[start:end]).cuda()[:, None, None]
        samples = relative_batch + mean_batch + torch.randn(
            (end - start, point_count, sample_count, 2), device="cuda",
        ) * scale_batch
        actual_occupied = torch.linalg.vector_norm(samples, dim=-1) <= radius_batch
        predicted_occupied = torch.from_numpy(distance[start:end]).cuda()[:, :, None] <= radius_batch
        crossing = (actual_occupied != predicted_occupied).float().mean(dim=2)
        row_scores.append(crossing.max(dim=1).values.cpu().numpy())
    row_score = np.concatenate(row_scores)
    target = dict(arrays)
    target["raw_actor_state_error_m"] = arrays["occupancy_decision_flip"].astype(np.float32)
    evaluation = _build_sets(target, 20.0, 0.5, 16)
    row_keys = np.stack((
        arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["query_id"],
    ), axis=1)
    score = _aligned_group_max(row_keys, row_score, evaluation["identity"])
    selected = _select_by_scene(score, evaluation["scene_index"], 0.50)
    events = evaluation["events"]
    return {
        "trajectory_count": int(len(events)), "all_occupancy_flip_events": int(np.count_nonzero(events)),
        "selected_trajectory_count": int(len(selected)),
        "nonlinear_selected_occupancy_flip_events": int(np.count_nonzero(events[selected])),
        "nonlinear_event_auroc": binary_auroc(events, score),
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
    torch.manual_seed(int(config["seed"]))
    checkpoint = torch.load(
        args.runs_root / config["frozen_p109"]["run"] / config["frozen_p109"]["artifact"],
        map_location="cuda",
    )
    model = DirectionalActorGaussian(20, checkpoint["hidden_dimensions"]).cuda()
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    torch.cuda.reset_peak_memory_stats()
    prep = args.runs_root / config["development_rows"]["run"]
    results = {
        cohort["name"]: _evaluate(
            dict(np.load(prep / cohort["artifact"], allow_pickle=False)), model, checkpoint,
            int(config["sampling"]["sample_count"]), int(config["sampling"]["row_batch_size"]),
        ) for cohort in config["development_rows"]["cohorts"]
    }
    linear_summary = json.loads((
        args.runs_root / config["frozen_p109"]["run"] / "summary.json"
    ).read_text(encoding="utf-8"))
    comparison = {}
    for name, metrics in results.items():
        linear = linear_summary["development_evaluations"][name]
        comparison[name] = {
            "linear_selected_events": linear["query_selected_occupancy_flip_events"],
            "linear_event_auroc": linear["query_event_auroc"],
            "nonlinear_selected_events": metrics["nonlinear_selected_occupancy_flip_events"],
            "nonlinear_event_auroc": metrics["nonlinear_event_auroc"],
        }
    supported = all(
        row["nonlinear_selected_events"] <= row["linear_selected_events"]
        for row in comparison.values()
    )
    verdict = "supported_development_nonlinear_gaussian_crossing" if supported else \
        "rejected_development_nonlinear_gaussian_crossing"
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "status": "done", "verdict": verdict, "role": config["role"],
        "development_evaluations": results, "comparison_to_linear_P109": comparison,
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
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
