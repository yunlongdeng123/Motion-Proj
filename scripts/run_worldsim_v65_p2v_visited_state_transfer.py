"""Run the single fresh Qagg trajectory-visited-state transfer read."""

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

import joblib
import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from motion_proj.worldsim_v61.occupancy import FREE
from motion_proj.worldsim_v64.conditional_state_bake import _target_free_boundary
from motion_proj.worldsim_v64.gaussian_route_consumer import _future_route_in_target_lidar
from motion_proj.worldsim_v64.native_voxel_uq import _evidence_on_native_grid, _native_unit_dir, _unit_dirs
from motion_proj.worldsim_v65.task_contract import continuous_trajectory_features


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_unit(
    evidence_unit: Path,
    native_unit: Path,
    processed_scene: Path,
    *,
    origin: np.ndarray,
    voxel_size: float,
    future_frame_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices, centers, features = _target_free_boundary(
        evidence_unit,
        native_unit,
        native_origin_m=origin,
        native_voxel_size_m=voxel_size,
    )
    with np.load(evidence_unit / "TARGET_EVIDENCE.npz", allow_pickle=False) as source:
        target = {name: np.asarray(source[name]) for name in source.files}
    native_shape = tuple(int(value) for value in np.load(native_unit / "ARGMAX.npy", mmap_mode="r").shape)
    target_state, target_valid = _evidence_on_native_grid(
        target,
        native_shape=native_shape,
        native_origin_m=origin,
        native_voxel_size_m=voxel_size,
    )
    x, y, z = indices.T
    valid = target_valid[x, y, z]
    frame = int(evidence_unit.name.removeprefix("f"))
    route = _future_route_in_target_lidar(processed_scene, frame, future_frame_count)
    return features[valid], centers[valid], target_state[x, y, z][valid] == FREE, route


def _frozen_q0_scores(model: object, features: np.ndarray) -> np.ndarray:
    values = (features.astype(np.float32) - model.mean) / model.scale
    network = model.model.cuda().eval()
    outputs = []
    with torch.inference_mode():
        for offset in range(0, values.shape[0], 131072):
            batch = torch.from_numpy(values[offset : offset + 131072]).cuda()
            with torch.cuda.amp.autocast():
                logits = network(batch).squeeze(1)
            outputs.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def _materialize(config: dict, runs_root: Path, cache_path: Path) -> dict[str, int]:
    inputs = config["inputs"]
    evidence_root = runs_root / inputs["evidence_run"]
    native_root = runs_root / inputs["native_run"]
    processed_root = Path(inputs["processed_root"])
    q0 = joblib.load(runs_root / inputs["risk_run"] / inputs["risk_model_relative_path"])
    origin = np.asarray(config["native_grid"]["origin_m"], dtype=np.float64)
    voxel_size = float(config["native_grid"]["voxel_size_m"])
    future_frames = int(config["trajectory"]["future_frame_count"])
    route_radius = float(config["trajectory"]["visited_corridor_radius_m"])
    minimum_visited = int(config["trajectory"]["minimum_visited_points_per_unit"])
    point_limit = int(config["sampling"]["evaluation_points_per_unit"])
    rng = np.random.default_rng(int(config["seed"]))
    device = torch.device("cuda")

    descriptors = []
    for scene_index, scene in enumerate(config["scenes"]):
        name = str(scene["name"])
        for unit_index, evidence_unit in enumerate(_unit_dirs(evidence_root, name)):
            descriptors.append((
                scene_index,
                unit_index,
                evidence_unit,
                _native_unit_dir(native_root, name, evidence_unit.name, {name: str(inputs["native_partition"])}),
                processed_root / f"{int(scene['processed_index']):03d}",
            ))
    if not descriptors:
        raise RuntimeError("no fresh P2V units found")

    parts = {name: [] for name in ("qagg", "target_cost", "unsafe", "visited_count", "hidden_free_count", "scene_index", "unit_index")}
    excluded = 0
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def submit(row: tuple):
        return executor.submit(
            _load_unit,
            row[2],
            row[3],
            row[4],
            origin=origin,
            voxel_size=voxel_size,
            future_frame_count=future_frames,
        )

    future = submit(descriptors[0])
    try:
        for position, descriptor in enumerate(descriptors):
            features, centers, labels, route_xy = future.result()
            if position + 1 < len(descriptors):
                future = submit(descriptors[position + 1])
            if features.shape[0] > point_limit:
                chosen = rng.choice(features.shape[0], size=point_limit, replace=False)
                features, centers, labels = features[chosen], centers[chosen], labels[chosen]
            scores = _frozen_q0_scores(q0, features)
            trajectory = continuous_trajectory_features(centers, route_xy, device).cpu().numpy()
            visited = trajectory[:, 0] <= route_radius
            visited_count = int(np.count_nonzero(visited))
            if visited_count < minimum_visited:
                excluded += 1
                continue
            hidden_free_count = int(np.count_nonzero(labels[visited]))
            parts["qagg"].append(float(scores[visited].mean()))
            parts["target_cost"].append(hidden_free_count / visited_count)
            parts["unsafe"].append(hidden_free_count > 0)
            parts["visited_count"].append(visited_count)
            parts["hidden_free_count"].append(hidden_free_count)
            parts["scene_index"].append(descriptor[0])
            parts["unit_index"].append(descriptor[1])
            print(f"fresh Qagg {position + 1}/{len(descriptors)} scene={descriptor[0]} unit={descriptor[1]} visited={visited_count}", flush=True)
    finally:
        executor.shutdown(wait=True)

    payload = {
        "qagg": np.asarray(parts["qagg"], dtype=np.float32),
        "target_cost": np.asarray(parts["target_cost"], dtype=np.float32),
        "unsafe": np.asarray(parts["unsafe"], dtype=bool),
        "visited_count": np.asarray(parts["visited_count"], dtype=np.int32),
        "hidden_free_count": np.asarray(parts["hidden_free_count"], dtype=np.int32),
        "scene_index": np.asarray(parts["scene_index"], dtype=np.uint8),
        "unit_index": np.asarray(parts["unit_index"], dtype=np.uint8),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp.npz")
    np.savez(temporary, **payload)
    os.replace(temporary, cache_path)
    return {"source_unit_count": len(descriptors), "excluded_unit_count": excluded}


def _selected(target: np.ndarray, scores: np.ndarray, scenes: np.ndarray, coverage: float) -> dict[str, object]:
    count = max(1, int(math.floor(float(coverage) * scores.shape[0])))
    selected = np.argsort(scores, kind="stable")[:count]
    all_mean = float(target.mean())
    selected_mean = float(target[selected].mean())
    scene_rows = []
    for scene in np.unique(scenes):
        members = np.flatnonzero(scenes == scene)
        local_count = max(1, int(math.floor(float(coverage) * members.shape[0])))
        local = members[np.argsort(scores[members], kind="stable")[:local_count]]
        local_all = float(target[members].mean())
        local_selected = float(target[local].mean())
        scene_rows.append({
            "scene_index": int(scene),
            "eligible_count": int(members.shape[0]),
            "selected_count": int(local_count),
            "all_mean_cost": local_all,
            "selected_mean_cost": local_selected,
            "delta": local_selected - local_all,
        })
    return {
        "eligible_count": int(scores.shape[0]),
        "selected_count": int(count),
        "realized_coverage": float(count / scores.shape[0]),
        "all_mean_cost": all_mean,
        "selected_mean_cost": selected_mean,
        "relative_cost_reduction": float((all_mean - selected_mean) / all_mean if all_mean > 0 else 0.0),
        "scene_lower_count": sum(row["delta"] < 0 for row in scene_rows),
        "scene_equal_count": sum(row["delta"] == 0 for row in scene_rows),
        "scene_higher_count": sum(row["delta"] > 0 for row in scene_rows),
        "scene_rows": scene_rows,
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
    materialization = {"source_unit_count": 72, "excluded_unit_count": 0}
    if not cache_reused:
        materialization = _materialize(config, runs_root, cache_path)
    with np.load(cache_path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}

    target = arrays["target_cost"]
    scores = arrays["qagg"]
    unsafe = arrays["unsafe"]
    spearman = float(spearmanr(target, scores).statistic)
    unsafe_metrics = {
        "auroc": float(roc_auc_score(unsafe, scores)) if np.unique(unsafe).size == 2 else float("nan"),
        "auprc": float(average_precision_score(unsafe, scores)),
    }
    selected = _selected(target, scores, arrays["scene_index"], float(config["evaluation"]["matched_safe_coverage"]))
    gates = {
        "minimum_q0_aggregate_spearman": spearman >= float(config["gates"]["minimum_q0_aggregate_spearman"]),
        "minimum_q0_aggregate_unsafe_auroc": unsafe_metrics["auroc"] >= float(config["gates"]["minimum_q0_aggregate_unsafe_auroc"]),
        "minimum_selected_cost_reduction": selected["relative_cost_reduction"] >= float(config["gates"]["minimum_selected_cost_reduction"]),
        "minimum_scene_support": selected["scene_lower_count"] >= int(config["gates"]["minimum_scene_support"]),
    }
    verdict = "supported_fresh_trajectory_visited_state_qagg" if all(gates.values()) else "rejected_fresh_trajectory_visited_state_qagg"
    summary = {
        "schema_version": "worldsim_v65.p2v_visited_state_transfer_summary.v1",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "claim_boundary": config["claim_boundary"],
        "cache_reused": cache_reused,
        **materialization,
        "eligible_unit_count": int(target.shape[0]),
        "unsafe_unit_count": int(np.count_nonzero(unsafe)),
        "visited_point_count": int(arrays["visited_count"].sum()),
        "hidden_free_count": int(arrays["hidden_free_count"].sum()),
        "qagg_target_spearman": spearman,
        "unsafe_ranking": unsafe_metrics,
        "selected_cost": selected,
        "gate_results": gates,
        "formal_v65_selection_read": True,
        "resources": {
            "gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
            "wall_seconds": time.monotonic() - started,
        },
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "status.json", {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    return {"run_dir": str(run_dir), "verdict": verdict, "gate_results": gates, "resources": summary["resources"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.runs_root.resolve(), args.run_id), indent=2))


if __name__ == "__main__":
    main()

