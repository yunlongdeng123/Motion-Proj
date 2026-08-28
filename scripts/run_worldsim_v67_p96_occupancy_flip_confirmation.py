"""Evaluate frozen P95 occupancy-flip reliability on the remaining test-role scenes."""

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
    ACTOR_FEATURE_NAMES, FEATURE_NAMES, ReliabilityMLP, binary_auroc,
    materialize_actor_query_rows, predict_reliability,
)
from scripts.run_worldsim_v67_p86_direct_trajectory_reliability import (
    _group_max_visited_score, _select_by_scene,
)
from scripts.run_worldsim_v67_p87_deepset_trajectory_reliability import DeepSetRisk, _build_sets
from scripts.run_worldsim_v67_p90_plain_trajectory_max_error import _predict_error


def _group_any(arrays: dict[str, np.ndarray], field: str, identities: np.ndarray) -> np.ndarray:
    keys = np.stack((
        arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["query_id"],
    ), axis=1)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    values = np.zeros(len(unique), dtype=bool)
    np.logical_or.at(values, inverse, np.asarray(arrays[field], dtype=bool))
    table = {tuple(key.tolist()): bool(value) for key, value in zip(unique, values)}
    return np.asarray([table[tuple(key.tolist())] for key in identities], dtype=bool)


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
    scene_dirs = [Path(data["processed_root"]) / f"{index:03d}" for index in (
        599, 37, 489, 81, 83, 806, 485, 696, 440, 808,
    )]
    deadline = time.monotonic() + float(data["readiness_timeout_seconds"])
    while not all((scene / "instances" / "instances_info.json").is_file() and (scene / "lidar_pose").is_dir()
                  for scene in scene_dirs):
        if time.monotonic() >= deadline:
            raise TimeoutError("P96 processed scenes not ready")
        time.sleep(10.0)
    raw = materialize_actor_query_rows(scene_dirs, data["horizons_seconds"], data)
    raw["actor_position_error_m"] = raw["raw_actor_state_error_m"].copy()
    raw["raw_actor_state_error_m"] = raw["occupancy_decision_flip"].astype(np.float32)
    raw["target_cost"] = raw["raw_actor_state_error_m"].copy()
    np.savez_compressed(run_dir / "P96_CONFIRMATION_OCCUPANCY_FLIP_ROWS.npz", **raw)

    model_config = config["model"]
    evaluation = _build_sets(
        raw, float(config["evaluation"]["visited_region_radius_m"]),
        float(config["evaluation"]["unreliable_actor_state_error_m"]),
        int(model_config["maximum_visited_actors"]),
    )
    checkpoint = torch.load(
        args.runs_root / config["frozen_p95"]["run"] / config["frozen_p95"]["artifact"], map_location="cuda",
    )
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
    mask = torch.from_numpy(evaluation["mask"]).cuda()
    query_model = DeepSetRisk(len(FEATURE_NAMES), checkpoint["element_dimensions"], checkpoint["decoder_dimensions"]).cuda()
    actor_model = DeepSetRisk(len(ACTOR_FEATURE_NAMES), checkpoint["element_dimensions"], checkpoint["decoder_dimensions"]).cuda()
    query_model.load_state_dict(checkpoint["query_model_state_dict"])
    actor_model.load_state_dict(checkpoint["actor_model_state_dict"])
    torch.cuda.reset_peak_memory_stats()
    threshold_log = float(np.log1p(model_config["failure_threshold_m"]))
    query_score = _predict_error(query_model.eval(), query_sets, mask, model_config, threshold_log)
    actor_score = _predict_error(actor_model.eval(), actor_sets, mask, model_config, threshold_log)

    frozen = torch.load(
        args.runs_root / config["frozen_p75"]["run"] / config["frozen_p75"]["artifact"], map_location="cuda",
    )
    frozen_model = ReliabilityMLP(len(FEATURE_NAMES), frozen["hidden_dimensions"]).cuda()
    frozen_model.load_state_dict(frozen["query_model_state_dict"])
    frozen_row_score = predict_reliability(
        frozen_model.eval(), raw["features"], np.asarray(frozen["feature_mean"], dtype=np.float32),
        np.asarray(frozen["feature_scale"], dtype=np.float32),
    )
    row_keys = np.stack((
        raw["scene_index"], np.rint(raw["horizon_seconds"] * 10).astype(np.int32),
        raw["anchor_frame"], raw["query_id"],
    ), axis=1)
    frozen_score = _group_max_visited_score(
        row_keys, frozen_row_score,
        np.asarray(raw["predicted_minimum_separation_m"]) <= float(config["evaluation"]["visited_region_radius_m"]),
    )
    scenes, events = evaluation["scene_index"], evaluation["events"]
    false_safe = _group_any(raw, "occupancy_false_safe", evaluation["identity"])
    false_alarm = _group_any(raw, "occupancy_false_alarm", evaluation["identity"])
    fraction = float(config["selection"]["coverage_fraction"])
    selected = _select_by_scene(query_score, scenes, fraction)
    actor_selected = _select_by_scene(actor_score, scenes, fraction)
    frozen_selected = _select_by_scene(frozen_score, scenes, fraction)
    query_events, actor_events, frozen_events = (
        int(np.count_nonzero(events[index])) for index in (selected, actor_selected, frozen_selected)
    )
    all_prevalence = float(events.mean())
    selected_prevalence = float(events[selected].mean())
    metrics = {
        "row_count": int(len(raw["features"])), "evaluation_trajectory_count": int(len(events)),
        "selected_trajectory_count": int(len(selected)), "achieved_coverage": float(len(selected) / len(events)),
        "all_occupancy_flip_events": int(np.count_nonzero(events)),
        "query_selected_occupancy_flip_events": query_events,
        "actor_selected_occupancy_flip_events": actor_events,
        "frozen_p75_selected_occupancy_flip_events": frozen_events,
        "all_occupancy_flip_prevalence": all_prevalence,
        "query_selected_occupancy_flip_prevalence": selected_prevalence,
        "actor_selected_occupancy_flip_prevalence": float(events[actor_selected].mean()),
        "frozen_p75_selected_occupancy_flip_prevalence": float(events[frozen_selected].mean()),
        "query_event_reduction": float((all_prevalence - selected_prevalence) / max(all_prevalence, 1e-12)),
        "query_event_reduction_over_actor_only": float((actor_events - query_events) / max(actor_events, 1)),
        "query_event_auroc": binary_auroc(events, query_score), "actor_event_auroc": binary_auroc(events, actor_score),
        "all_false_safe_events": int(np.count_nonzero(false_safe)),
        "query_selected_false_safe_events": int(np.count_nonzero(false_safe[selected])),
        "actor_selected_false_safe_events": int(np.count_nonzero(false_safe[actor_selected])),
        "frozen_p75_selected_false_safe_events": int(np.count_nonzero(false_safe[frozen_selected])),
        "all_false_alarm_events": int(np.count_nonzero(false_alarm)),
        "query_selected_false_alarm_events": int(np.count_nonzero(false_alarm[selected])),
    }
    gates = {
        "minimum_event_reduction_over_actor_only": metrics["query_event_reduction_over_actor_only"]
        >= float(config["gates"]["minimum_event_reduction_over_actor_only"]),
        "minimum_absolute_trajectory_event_reduction": metrics["query_event_reduction"]
        >= float(config["gates"]["minimum_absolute_trajectory_event_reduction"]),
        "no_more_events_than_frozen_p75": query_events <= frozen_events,
        "maximum_max_error_ratio_to_frozen_p75": selected_prevalence
        <= float(events[frozen_selected].mean()) * float(config["gates"]["maximum_max_error_ratio_to_frozen_p75"]),
    }
    verdict = config["verdict_on_pass"] if all(gates.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"], "fresh_confirmation_evaluation": metrics, "gate_results": gates,
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
                      "fresh_confirmation_evaluation": metrics, "gate_results": gates}, indent=2), flush=True)


if __name__ == "__main__":
    main()
