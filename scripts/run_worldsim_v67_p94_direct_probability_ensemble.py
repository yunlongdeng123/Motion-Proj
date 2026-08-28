"""Evaluate the frozen three-member direct trajectory failure ensemble."""

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
    ACTOR_FEATURE_NAMES, FEATURE_NAMES, ReliabilityMLP, binary_auroc, predict_reliability,
)
from scripts.run_worldsim_v67_p86_direct_trajectory_reliability import (
    _group_max_visited_score, _select_by_scene,
)
from scripts.run_worldsim_v67_p87_deepset_trajectory_reliability import DeepSetRisk, _build_sets
from scripts.run_worldsim_v67_p90_plain_trajectory_max_error import _predict_error


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
    evaluation_path = args.runs_root / config["evaluation_rows"]["run"] / config["evaluation_rows"]["artifact"]
    member_paths = [
        args.runs_root / "worldsim_v67" / member["task_id"] / member["run_id"] / member["artifact"]
        for member in config["ensemble"]["members"]
    ]
    deadline = time.monotonic() + float(config["evaluation_rows"]["readiness_timeout_seconds"])
    while not evaluation_path.is_file() or not all(path.is_file() for path in member_paths):
        if time.monotonic() >= deadline:
            raise TimeoutError("trajectory rows or ensemble checkpoints not ready")
        time.sleep(10.0)
    evaluation_raw = dict(np.load(evaluation_path, allow_pickle=False))
    model_config = config["model"]
    evaluation = _build_sets(
        evaluation_raw, float(config["evaluation"]["visited_region_radius_m"]),
        float(config["evaluation"]["unreliable_actor_state_error_m"]),
        int(model_config["maximum_visited_actors"]),
    )
    mask = torch.from_numpy(evaluation["mask"]).cuda()
    failure_threshold_log = float(np.log1p(config["evaluation"]["unreliable_actor_state_error_m"]))
    query_member_scores, actor_member_scores = [], []
    torch.cuda.reset_peak_memory_stats()
    for checkpoint_path in member_paths:
        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        query_np = (evaluation["query_sets"] - np.asarray(checkpoint["query_feature_mean"], dtype=np.float32)) / np.asarray(
            checkpoint["query_feature_scale"], dtype=np.float32,
        )
        actor_np = (evaluation["actor_sets"] - np.asarray(checkpoint["actor_feature_mean"], dtype=np.float32)) / np.asarray(
            checkpoint["actor_feature_scale"], dtype=np.float32,
        )
        query_np[~evaluation["mask"]] = 0.0
        actor_np[~evaluation["mask"]] = 0.0
        query_sets = torch.from_numpy(query_np).cuda()
        actor_sets = torch.from_numpy(actor_np).cuda()
        query_model = DeepSetRisk(
            len(FEATURE_NAMES), checkpoint["element_dimensions"], checkpoint["decoder_dimensions"],
        ).cuda()
        actor_model = DeepSetRisk(
            len(ACTOR_FEATURE_NAMES), checkpoint["element_dimensions"], checkpoint["decoder_dimensions"],
        ).cuda()
        query_model.load_state_dict(checkpoint["query_model_state_dict"])
        actor_model.load_state_dict(checkpoint["actor_model_state_dict"])
        query_member_scores.append(_predict_error(
            query_model.eval(), query_sets, mask, model_config, failure_threshold_log,
        ))
        actor_member_scores.append(_predict_error(
            actor_model.eval(), actor_sets, mask, model_config, failure_threshold_log,
        ))
        del checkpoint, query_sets, actor_sets, query_model, actor_model
        torch.cuda.empty_cache()
    query_score = np.mean(np.stack(query_member_scores), axis=0)
    actor_score = np.mean(np.stack(actor_member_scores), axis=0)

    frozen = torch.load(
        args.runs_root / config["frozen_p75"]["run"] / config["frozen_p75"]["artifact"], map_location="cuda",
    )
    frozen_model = ReliabilityMLP(len(FEATURE_NAMES), frozen["hidden_dimensions"]).cuda()
    frozen_model.load_state_dict(frozen["query_model_state_dict"])
    frozen_row_score = predict_reliability(
        frozen_model.eval(), evaluation_raw["features"],
        np.asarray(frozen["feature_mean"], dtype=np.float32),
        np.asarray(frozen["feature_scale"], dtype=np.float32),
    )
    row_keys = np.stack((
        evaluation_raw["scene_index"], np.rint(evaluation_raw["horizon_seconds"] * 10).astype(np.int32),
        evaluation_raw["anchor_frame"], evaluation_raw["query_id"],
    ), axis=1)
    frozen_score = _group_max_visited_score(
        row_keys, frozen_row_score,
        np.asarray(evaluation_raw["predicted_minimum_separation_m"])
        <= float(config["evaluation"]["visited_region_radius_m"]),
    )
    scenes = evaluation["scene_index"]
    events = evaluation["max_error"] > float(config["evaluation"]["unreliable_actor_state_error_m"])
    max_error = evaluation["max_error"]
    fraction = float(config["selection"]["coverage_fraction"])
    selected = _select_by_scene(query_score, scenes, fraction)
    actor_selected = _select_by_scene(actor_score, scenes, fraction)
    frozen_selected = _select_by_scene(frozen_score, scenes, fraction)
    query_events, actor_events, frozen_events = (
        int(np.count_nonzero(events[index])) for index in (selected, actor_selected, frozen_selected)
    )
    all_prevalence = float(events.mean())
    selected_prevalence = float(events[selected].mean())
    query_max_error = float(max_error[selected].mean())
    frozen_max_error = float(max_error[frozen_selected].mean())
    metrics = {
        "ensemble_member_count": len(member_paths),
        "evaluation_trajectory_count": int(len(events)), "selected_trajectory_count": int(len(selected)),
        "achieved_coverage": float(len(selected) / len(events)),
        "all_unreliable_events": int(np.count_nonzero(events)),
        "query_selected_unreliable_events": query_events,
        "actor_selected_unreliable_events": actor_events,
        "frozen_p75_selected_unreliable_events": frozen_events,
        "all_unreliable_prevalence": all_prevalence,
        "query_selected_unreliable_prevalence": selected_prevalence,
        "actor_selected_unreliable_prevalence": float(events[actor_selected].mean()),
        "frozen_p75_selected_unreliable_prevalence": float(events[frozen_selected].mean()),
        "query_event_reduction": float((all_prevalence - selected_prevalence) / max(all_prevalence, 1e-12)),
        "query_event_reduction_over_actor_only": float((actor_events - query_events) / max(actor_events, 1)),
        "query_event_auroc": binary_auroc(events, query_score),
        "actor_event_auroc": binary_auroc(events, actor_score),
        "query_selected_mean_max_error_m": query_max_error,
        "frozen_p75_selected_mean_max_error_m": frozen_max_error,
        "query_max_error_ratio_to_frozen_p75": query_max_error / max(frozen_max_error, 1e-12),
    }
    gates = {
        "minimum_event_reduction_over_actor_only": metrics["query_event_reduction_over_actor_only"]
        >= float(config["gates"]["minimum_event_reduction_over_actor_only"]),
        "minimum_absolute_trajectory_event_reduction": metrics["query_event_reduction"]
        >= float(config["gates"]["minimum_absolute_trajectory_event_reduction"]),
        "no_more_events_than_frozen_p75": query_events <= frozen_events,
        "maximum_max_error_ratio_to_frozen_p75": metrics["query_max_error_ratio_to_frozen_p75"]
        <= float(config["gates"]["maximum_max_error_ratio_to_frozen_p75"]),
    }
    verdict = "supported_direct_probability_ensemble" if all(gates.values()) else "rejected_direct_probability_ensemble"
    np.savez_compressed(
        run_dir / "ENSEMBLE_TRAJECTORY_SCORES.npz", scene_index=scenes, query_score=query_score,
        actor_score=actor_score, unreliable_event=events, maximum_actor_error_m=max_error,
    )
    summary = {
        "schema_version": "worldsim_v67.p94_direct_probability_ensemble_summary.v1",
        "task_id": config["task_id"], "hypothesis_id": config["hypothesis_id"], "status": "done",
        "verdict": verdict, "role": "prospective_direct_probability_deep_ensemble",
        "ensemble": {"aggregation": config["ensemble"]["aggregation"], "members": config["ensemble"]["members"]},
        "fresh_test_evaluation": metrics, "gate_results": gates,
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
                      "fresh_test_evaluation": metrics, "gate_results": gates}, indent=2), flush=True)


if __name__ == "__main__":
    main()
