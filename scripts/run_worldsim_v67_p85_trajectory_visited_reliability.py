"""Aggregate visited Actor failures into one reliability decision per candidate Ego trajectory."""

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
from scripts.run_worldsim_v67_p84_visited_actor_failure import ActorFailureMLP


def _select_by_scene(score: np.ndarray, scenes: np.ndarray, fraction: float) -> np.ndarray:
    selected: list[int] = []
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
    data = config["evaluation_data"]
    metadata = Path(data["metadata_root"]) / "v1.0-trainval"
    scene_rows = json.loads((metadata / "scene.json").read_text(encoding="utf-8"))
    index_by_name = {str(row["name"]): index for index, row in enumerate(scene_rows)}
    names = [str(name) for name in data["scene_names"]]
    identities = [(name, int(index_by_name[name])) for name in names]
    processed_root = Path(data["processed_root"])
    deadline = time.monotonic() + float(data["readiness_timeout_seconds"])
    model_path = args.runs_root / config["source_model"]["run"] / config["source_model"]["artifact"]
    while True:
        ready = [
            (processed_root / f"{index:03d}" / "instances" / "instances_info.json").is_file()
            and (processed_root / f"{index:03d}" / "lidar_pose").is_dir()
            for _, index in identities
        ]
        if all(ready) and model_path.is_file():
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"trajectory Actor inputs/model not ready: scenes={sum(ready)}/{len(ready)} model={model_path.is_file()}")
        print(f"waiting trajectory inputs ready={sum(ready)}/{len(ready)} model={model_path.is_file()}", flush=True)
        time.sleep(10.0)
    arrays = materialize_actor_query_rows(
        [processed_root / f"{index:03d}" for _, index in identities],
        [float(data["horizon_seconds"])], data,
    )
    np.savez_compressed(run_dir / "FRESH_TEST_H3P5_TRAJECTORY_ACTOR_ROWS.npz", **arrays)
    artifact = torch.load(model_path, map_location="cuda")
    actor_model = ActorFailureMLP(artifact["hidden_dimensions"]).cuda()
    actor_model.load_state_dict(artifact["model_state_dict"])
    actor_features = np.asarray(arrays["features"], dtype=np.float32)[:, :len(ACTOR_FEATURE_NAMES)]
    with torch.no_grad():
        actor_score, _ = actor_model.eval()(torch.from_numpy(
            (actor_features - np.asarray(artifact["feature_mean"], dtype=np.float32))
            / np.asarray(artifact["feature_scale"], dtype=np.float32)
        ).cuda())
    actor_score = actor_score.cpu().numpy()
    frozen = torch.load(
        args.runs_root / config["frozen_p75"]["run"] / config["frozen_p75"]["artifact"], map_location="cuda"
    )
    frozen_model = ReliabilityMLP(len(FEATURE_NAMES), frozen["hidden_dimensions"]).cuda()
    frozen_model.load_state_dict(frozen["query_model_state_dict"])
    frozen_score = predict_reliability(
        frozen_model.eval(), arrays["features"], np.asarray(frozen["feature_mean"], dtype=np.float32),
        np.asarray(frozen["feature_scale"], dtype=np.float32),
    )
    keys = np.stack((arrays["scene_index"], arrays["anchor_frame"], arrays["query_id"]), axis=1)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    radius = float(config["evaluation"]["visited_region_radius_m"])
    threshold = float(config["evaluation"]["unreliable_actor_state_error_m"])
    separation = np.asarray(arrays["predicted_minimum_separation_m"])
    error = np.asarray(arrays["raw_actor_state_error_m"], dtype=np.float64)
    group_scene = []
    group_actor_score = []
    group_frozen_score = []
    group_event = []
    group_max_error = []
    group_visited_actor_count = []
    group_identity = []
    for group in range(int(inverse.max()) + 1):
        members = np.flatnonzero(inverse == group)
        visited = members[separation[members] <= radius]
        if not len(visited):
            continue
        identity = keys[members[0]]
        group_scene.append(int(identity[0]))
        group_identity.append([int(value) for value in identity])
        group_actor_score.append(float(np.max(actor_score[visited])))
        group_frozen_score.append(float(np.max(frozen_score[visited])))
        group_event.append(bool(np.any(error[visited] > threshold)))
        group_max_error.append(float(np.max(error[visited])))
        group_visited_actor_count.append(int(len(visited)))
    scenes = np.asarray(group_scene, dtype=np.int32)
    candidate_score = np.asarray(group_actor_score, dtype=np.float64)
    baseline_score = np.asarray(group_frozen_score, dtype=np.float64)
    events = np.asarray(group_event, dtype=bool)
    max_error = np.asarray(group_max_error, dtype=np.float64)
    fraction = float(config["selection"]["coverage_fraction"])
    selected = _select_by_scene(candidate_score, scenes, fraction)
    baseline_selected = _select_by_scene(baseline_score, scenes, fraction)
    selected_events = int(np.count_nonzero(events[selected]))
    baseline_events = int(np.count_nonzero(events[baseline_selected]))
    all_prevalence = float(events.mean())
    selected_prevalence = float(events[selected].mean())
    scene_metrics = []
    for scene in np.unique(scenes):
        members = np.flatnonzero(scenes == scene)
        chosen = selected[np.isin(selected, members)]
        scene_metrics.append({"scene_index": int(scene), "trajectory_count": int(len(members)),
            "selected_count": int(len(chosen)), "all_event_prevalence": float(events[members].mean()),
            "selected_event_prevalence": float(events[chosen].mean())})
    candidate_max_error = float(max_error[selected].mean())
    baseline_max_error = float(max_error[baseline_selected].mean())
    metrics = {"actor_query_row_count": int(len(error)), "visited_trajectory_count": int(len(events)),
        "selected_trajectory_count": int(len(selected)), "achieved_coverage": float(len(selected) / len(events)),
        "all_trajectory_unreliable_events": int(np.count_nonzero(events)),
        "candidate_selected_unreliable_events": selected_events,
        "frozen_p75_selected_unreliable_events": baseline_events,
        "all_trajectory_unreliable_prevalence": all_prevalence,
        "candidate_selected_unreliable_prevalence": selected_prevalence,
        "frozen_p75_selected_unreliable_prevalence": float(events[baseline_selected].mean()),
        "candidate_trajectory_event_reduction": float((all_prevalence - selected_prevalence) / max(all_prevalence, 1e-12)),
        "candidate_trajectory_event_auroc": binary_auroc(events, candidate_score),
        "candidate_selected_mean_max_actor_error_m": candidate_max_error,
        "frozen_p75_selected_mean_max_actor_error_m": baseline_max_error,
        "candidate_max_error_ratio_to_frozen_p75": candidate_max_error / max(baseline_max_error, 1e-12),
        "mean_visited_actor_count": float(np.mean(group_visited_actor_count)),
        "scene_nonincreasing_count": int(sum(row["selected_event_prevalence"] <= row["all_event_prevalence"] for row in scene_metrics)),
        "scene_count": int(len(scene_metrics)), "scene_rows": scene_metrics}
    gates = {"minimum_trajectory_event_reduction": metrics["candidate_trajectory_event_reduction"] >= float(config["gates"]["minimum_trajectory_event_reduction"]),
        "no_more_events_than_frozen_p75": selected_events <= baseline_events,
        "maximum_max_error_ratio_to_frozen_p75": metrics["candidate_max_error_ratio_to_frozen_p75"] <= float(config["gates"]["maximum_max_error_ratio_to_frozen_p75"]),
        "minimum_scene_nonincreasing_fraction": metrics["scene_nonincreasing_count"] / max(metrics["scene_count"], 1) >= float(config["gates"]["minimum_scene_nonincreasing_fraction"])}
    verdict = config["verdict_on_pass"] if all(gates.values()) else config["verdict_on_failure"]
    summary = {"schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict, "role": config["role"],
        "fresh_scene_names": names, "fresh_scene_indices": [index for _, index in identities],
        "trajectory_evaluation": metrics, "gate_results": gates,
        "resources": {"gpu": torch.cuda.get_device_name(0), "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "wall_seconds": time.monotonic() - started}, "claim_boundary": config["claim_boundary"]}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict, "trajectory_evaluation": metrics, "gate_results": gates}, indent=2), flush=True)


if __name__ == "__main__":
    main()
