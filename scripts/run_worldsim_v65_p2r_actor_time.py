"""Run the preregistered legacy train-only Actor×time outcome probe."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from motion_proj.worldsim_v64.gaussian_route_consumer import _future_route_in_target_lidar
from motion_proj.worldsim_v65.actor_time_outcome import (
    fit_actor_outcome,
    ranking_metrics,
    score_actor_outcome,
    selected_outcome_metrics,
)
from motion_proj.worldsim_v65.task_contract import continuous_trajectory_features


STATIC_FEATURE_NAMES = (
    "log_current_voxels", "current_extent_x", "current_extent_y", "current_extent_z",
    "current_centroid_x", "current_centroid_y", "current_centroid_z",
    "current_route_distance_min", "current_route_distance_mean", "current_abs_lateral_min",
    "current_along_at_nearest", "current_horizon_at_nearest", "route_length", "actor_count",
)
TIME_FEATURE_NAMES = (
    "log_swept_voxels", "swept_extent_x", "swept_extent_y", "swept_extent_z",
    "history_centroid_delta_x", "history_centroid_delta_y", "history_centroid_delta_z",
    "swept_route_distance_min", "swept_route_distance_mean", "route_distance_approach",
    "motion_route_alignment", "observed_hit_fraction", "log_swept_current_ratio",
    "swept_lateral_at_nearest", "swept_along_at_nearest",
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_unit(unit: Path, processed_scene: Path, frame: int, future_frames: int) -> tuple[dict, dict, np.ndarray]:
    with np.load(unit / "METHOD_EVIDENCE.npz", allow_pickle=False) as source:
        method = {name: np.asarray(source[name]) for name in source.files}
    with np.load(unit / "TARGET_EVIDENCE.npz", allow_pickle=False) as source:
        target = {name: np.asarray(source[name]) for name in source.files}
    route = _future_route_in_target_lidar(processed_scene, frame, future_frames)
    return method, target, route


def _coords(indices: np.ndarray, origin: np.ndarray, voxel_size: float) -> np.ndarray:
    return origin[None] + (indices.astype(np.float32) + 0.5) * float(voxel_size)


def _extent(points: np.ndarray) -> np.ndarray:
    return points.max(axis=0) - points.min(axis=0)


def _materialize(config: dict, cache_path: Path) -> None:
    evidence_root = Path(config["inputs"]["evidence_root"])
    processed_root = Path(config["inputs"]["processed_root"])
    future_frames = int(config["evidence_contract"]["future_frame_count"])
    # The continuous-cost task shares this materializer but does not consume the
    # legacy binary label; keep that task-specific field optional.
    radius = float(config["evidence_contract"].get("route_corridor_radius_m", 0.0))
    scenes = []
    for role in ("train", "evaluation"):
        for scene in config["scenes"][role]:
            scenes.append((role, scene))
    descriptors = []
    for scene_index, (role, scene) in enumerate(scenes):
        for unit_index, frame in enumerate(config["targets"]["frame_indices"]):
            descriptors.append((scene_index, role, scene, unit_index, int(frame)))
    parts: dict[str, list[np.ndarray]] = {name: [] for name in ("static", "time", "label", "target_distance", "target_cost", "scene_index", "unit_index", "actor_id", "is_train")}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def submit(row: tuple):
        _, _, scene, _, frame = row
        return executor.submit(
            _load_unit,
            evidence_root / "units" / str(scene["name"]) / f"f{frame:03d}",
            processed_root / f"{int(scene['processed_index']):03d}",
            frame,
            future_frames,
        )

    future = submit(descriptors[0])
    for position, descriptor in enumerate(descriptors):
        scene_index, role, scene, unit_index, frame = descriptor
        method, target, route = future.result()
        if position + 1 < len(descriptors):
            future = submit(descriptors[position + 1])
        origin = np.asarray(method["grid_origin_m"], dtype=np.float32)
        voxel_size = float(method["voxel_size_m"])
        current_indices = np.asarray(method["actor_current_envelope_indices"], dtype=np.int32)
        current_ids = np.asarray(method["actor_current_envelope_ids"], dtype=np.int32)
        swept_indices = np.asarray(method["actor_swept_envelope_indices"], dtype=np.int32)
        swept_ids = np.asarray(method["actor_swept_envelope_ids"], dtype=np.int32)
        target_indices = np.asarray(target["actor_swept_envelope_indices"], dtype=np.int32)
        target_ids = np.asarray(target["actor_swept_envelope_ids"], dtype=np.int32)
        hit_ids = np.asarray(method["actor_hit_ids"], dtype=np.int32)
        current_points = _coords(current_indices, origin, voxel_size)
        swept_points = _coords(swept_indices, origin, voxel_size)
        target_points = _coords(target_indices, origin, voxel_size)
        merged = np.concatenate((current_points, swept_points, target_points), axis=0)
        trajectory = continuous_trajectory_features(merged, route, torch.device("cuda")).cpu().numpy()
        a = current_points.shape[0]
        b = a + swept_points.shape[0]
        current_trajectory, swept_trajectory, target_trajectory = trajectory[:a], trajectory[a:b], trajectory[b:]
        actors = np.unique(current_ids[current_ids >= 0])
        target_minimum = {
            int(actor): float(target_trajectory[target_ids == actor, 0].min())
            for actor in np.unique(target_ids[target_ids >= 0])
        }
        static_rows, time_rows, labels, distances, costs, actor_rows = [], [], [], [], [], []
        absent_distance = float(config["evidence_contract"].get("absent_actor_distance_m", 60.0))
        cost_scale = float(config["evidence_contract"].get("proximity_cost_scale_m", 6.0))
        for actor in actors:
            current_mask = current_ids == actor
            swept_mask = swept_ids == actor
            current_xyz, current_task = current_points[current_mask], current_trajectory[current_mask]
            swept_xyz, swept_task = swept_points[swept_mask], swept_trajectory[swept_mask]
            if swept_xyz.shape[0] == 0:
                swept_xyz, swept_task = current_xyz, current_task
            current_nearest = int(np.argmin(current_task[:, 0]))
            swept_nearest = int(np.argmin(swept_task[:, 0]))
            current_center = current_xyz.mean(axis=0)
            swept_center = swept_xyz.mean(axis=0)
            delta = swept_center - current_center
            tangent = current_task[current_nearest, 4:6]
            delta_norm = float(np.linalg.norm(delta[:2]))
            alignment = float(np.dot(delta[:2], tangent) / max(delta_norm, 1e-6))
            static_rows.append(np.asarray([
                math.log1p(current_xyz.shape[0]), *_extent(current_xyz), *current_center,
                current_task[:, 0].min(), current_task[:, 0].mean(), np.abs(current_task[:, 1]).min(),
                current_task[current_nearest, 2], current_task[current_nearest, 3],
                current_task[current_nearest, 6], actors.shape[0],
            ], dtype=np.float32))
            time_rows.append(np.asarray([
                math.log1p(swept_xyz.shape[0]), *_extent(swept_xyz), *delta,
                swept_task[:, 0].min(), swept_task[:, 0].mean(),
                current_task[:, 0].min() - swept_task[:, 0].min(), alignment,
                np.count_nonzero(hit_ids == actor) / max(1, current_xyz.shape[0]),
                math.log((swept_xyz.shape[0] + 1) / (current_xyz.shape[0] + 1)),
                swept_task[swept_nearest, 1], swept_task[swept_nearest, 2],
            ], dtype=np.float32))
            target_distance = min(target_minimum.get(int(actor), absent_distance), absent_distance)
            labels.append(target_distance <= radius)
            distances.append(target_distance)
            costs.append(math.exp(-target_distance / cost_scale))
            actor_rows.append(int(actor))
        count = len(labels)
        if count:
            parts["static"].append(np.stack(static_rows))
            parts["time"].append(np.stack(time_rows))
            parts["label"].append(np.asarray(labels, dtype=bool))
            parts["target_distance"].append(np.asarray(distances, dtype=np.float32))
            parts["target_cost"].append(np.asarray(costs, dtype=np.float32))
            parts["scene_index"].append(np.full(count, scene_index, dtype=np.uint8))
            parts["unit_index"].append(np.full(count, unit_index, dtype=np.uint8))
            parts["actor_id"].append(np.asarray(actor_rows, dtype=np.int32))
            parts["is_train"].append(np.full(count, role == "train", dtype=bool))
    executor.shutdown(wait=True)
    payload = {name: np.concatenate(values) for name, values in parts.items()}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp.npz")
    np.savez(temporary, **payload)
    os.replace(temporary, cache_path)


def _risk_comparison(snapshot: dict, actor_time: dict) -> dict[str, object]:
    base = float(snapshot["selected_outcome_rate"])
    candidate = float(actor_time["selected_outcome_rate"])
    base_scenes = {row["scene_index"]: row for row in snapshot["scene_rows"]}
    time_scenes = {row["scene_index"]: row for row in actor_time["scene_rows"]}
    deltas = {scene: time_scenes[scene]["selected_outcome_rate"] - row["selected_outcome_rate"] for scene, row in base_scenes.items()}
    return {
        "relative_selected_outcome_risk_reduction": float((base - candidate) / base if base > 0 else 0.0),
        "scene_lower_count": sum(value < 0 for value in deltas.values()),
        "scene_equal_count": sum(value == 0 for value in deltas.values()),
        "scene_higher_count": sum(value > 0 for value in deltas.values()),
        "scene_deltas": deltas,
    }


def run(config_path: Path, runs_root: Path, run_id: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v65" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    _write_json(run_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    cache_path = Path(config["inputs"]["compact_cache"])
    cache_reused = cache_path.is_file()
    if not cache_reused:
        _materialize(config, cache_path)
    with np.load(cache_path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    train, evaluate = arrays["is_train"], ~arrays["is_train"]
    snapshot_train = arrays["static"][train]
    full_train = np.concatenate((arrays["static"][train], arrays["time"][train]), axis=1)
    hidden = tuple(int(value) for value in config["model"]["hidden_dimensions"])
    model_kwargs = {
        "hidden_dimensions": hidden,
        "epochs": int(config["model"]["epochs"]),
        "batch_size": int(config["model"]["batch_size"]),
        "learning_rate": float(config["model"]["learning_rate"]),
        "weight_decay": float(config["model"]["weight_decay"]),
        "seed": int(config["seed"]),
    }
    snapshot_fit = fit_actor_outcome(snapshot_train, arrays["label"][train], **model_kwargs)
    actor_time_fit = fit_actor_outcome(full_train, arrays["label"][train], **model_kwargs)
    snapshot_eval = arrays["static"][evaluate]
    full_eval = np.concatenate((arrays["static"][evaluate], arrays["time"][evaluate]), axis=1)
    snapshot_scores = score_actor_outcome(snapshot_fit, snapshot_eval)
    actor_time_scores = score_actor_outcome(actor_time_fit, full_eval)
    shuffled = arrays["time"][evaluate].copy()
    rng = np.random.default_rng(int(config["seed"]) + 650)
    eval_scenes = arrays["scene_index"][evaluate]
    for scene in np.unique(eval_scenes):
        mask = eval_scenes == scene
        shuffled[mask] = shuffled[mask][rng.permutation(np.count_nonzero(mask))]
    shuffled_scores = score_actor_outcome(actor_time_fit, np.concatenate((snapshot_eval, shuffled), axis=1))
    labels = arrays["label"][evaluate]
    ranking = {
        "snapshot": ranking_metrics(labels, snapshot_scores),
        "actor_time": ranking_metrics(labels, actor_time_scores),
        "actor_time_shuffled": ranking_metrics(labels, shuffled_scores),
    }
    ranking["actor_time_minus_snapshot_auprc"] = ranking["actor_time"]["auprc"] - ranking["snapshot"]["auprc"]
    ranking["actor_time_minus_shuffled_auprc"] = ranking["actor_time"]["auprc"] - ranking["actor_time_shuffled"]["auprc"]
    coverage = float(config["evaluation"]["matched_safe_coverage"])
    snapshot_risk = selected_outcome_metrics(labels, snapshot_scores, eval_scenes, coverage=coverage)
    actor_time_risk = selected_outcome_metrics(labels, actor_time_scores, eval_scenes, coverage=coverage)
    comparison = _risk_comparison(snapshot_risk, actor_time_risk)
    gates = {
        "minimum_actor_auprc_gain": ranking["actor_time_minus_snapshot_auprc"] >= float(config["gates"]["minimum_actor_auprc_gain"]),
        "minimum_selected_outcome_risk_reduction": comparison["relative_selected_outcome_risk_reduction"] >= float(config["gates"]["minimum_selected_outcome_risk_reduction"]),
        "minimum_eval_scene_support": comparison["scene_lower_count"] >= int(config["gates"]["minimum_eval_scene_support"]),
        "temporal_shuffle_response": ranking["actor_time_minus_shuffled_auprc"] > 0.0,
    }
    verdict = "positive_train_only_actor_time_signal" if all(gates.values()) else "no_clear_train_only_actor_time_signal"
    torch.save({
        "snapshot_state_dict": {name: value.detach().cpu() for name, value in snapshot_fit.model.state_dict().items()},
        "actor_time_state_dict": {name: value.detach().cpu() for name, value in actor_time_fit.model.state_dict().items()},
        "snapshot_mean": snapshot_fit.mean, "snapshot_scale": snapshot_fit.scale,
        "actor_time_mean": actor_time_fit.mean, "actor_time_scale": actor_time_fit.scale,
        "static_feature_names": STATIC_FEATURE_NAMES, "time_feature_names": TIME_FEATURE_NAMES,
    }, run_dir / "actor_time_models.pt")
    summary = {
        "schema_version": "worldsim_v65.p2r_actor_time_summary.v1",
        "task_id": config["task_id"], "hypothesis_id": config["hypothesis_id"], "status": "done",
        "verdict": verdict, "claim_boundary": config["claim_boundary"], "cache_reused": cache_reused,
        "train_token_count": int(np.count_nonzero(train)), "evaluation_token_count": int(np.count_nonzero(evaluate)),
        "train_positive_count": int(np.count_nonzero(arrays["label"][train])),
        "evaluation_positive_count": int(np.count_nonzero(labels)),
        "ranking": ranking, "selected_outcome": {"snapshot": snapshot_risk, "actor_time": actor_time_risk},
        "comparison": comparison, "gate_results": gates, "formal_v65_selection_read": False,
        "resources": {"gpu": torch.cuda.get_device_name(0), "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024**3), "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2), "wall_seconds": time.monotonic() - started},
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "status.json", {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    return {"run_dir": str(run_dir), "verdict": verdict, "gate_results": gates, "ranking": ranking, "comparison": comparison, "resources": summary["resources"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.runs_root.resolve(), args.run_id), indent=2))


if __name__ == "__main__":
    main()
