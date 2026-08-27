"""Run the V6.5 train-only map/context residual probe on one RTX 3090."""

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

from motion_proj.worldsim_v61.occupancy import FREE
from motion_proj.worldsim_v64.conditional_state_bake import _target_free_boundary
from motion_proj.worldsim_v64.gaussian_route_consumer import _future_route_in_target_lidar
from motion_proj.worldsim_v64.native_voxel_uq import (
    _evidence_on_native_grid,
    _native_unit_dir,
    _unit_dirs,
)
from motion_proj.worldsim_v65.conditional_validity import (
    fit_trajectory_residual,
    fixed_opportunity_metrics,
    ranking_metrics,
    score_trajectory_residual,
)
from motion_proj.worldsim_v65.map_context import (
    MAP_CONTEXT_FEATURE_NAMES,
    ego_aligned_map_context,
)
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return (
        features[valid],
        centers[valid],
        indices[valid],
        (target_state[x, y, z][valid] == FREE),
        route,
    )


def _frozen_q0_embedding(model: object, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda")
    values = (features.astype(np.float32) - model.mean) / model.scale
    network = model.model.to(device).eval()
    hidden_parts = []
    logit_parts = []
    with torch.inference_mode():
        for offset in range(0, values.shape[0], 131072):
            batch = torch.from_numpy(values[offset : offset + 131072]).to(device)
            with torch.cuda.amp.autocast():
                hidden = network.layers[:5](batch)
                logits = network.layers[5](hidden).squeeze(1)
            hidden_parts.append(hidden.half().cpu().numpy())
            logit_parts.append(logits.float().cpu().numpy())
    return np.concatenate(hidden_parts), np.concatenate(logit_parts).astype(np.float32)


def _materialize_cache(config: dict, runs_root: Path, cache_path: Path) -> None:
    inputs = config["inputs"]
    evidence_root = runs_root / inputs["evidence_run"]
    native_root = runs_root / inputs["native_run"]
    processed_root = Path(inputs["processed_root"])
    map_root = Path(inputs["map_root"])
    q0 = joblib.load(runs_root / inputs["risk_run"] / inputs["risk_model_relative_path"])
    origin = np.asarray(config["native_grid"]["origin_m"], dtype=np.float64)
    voxel_size = float(config["native_grid"]["voxel_size_m"])
    grid_shape = tuple(int(value) for value in config["native_grid"]["shape"])
    future_frame_count = int(config["trajectory"]["future_frame_count"])
    train_unit_count = int(config["sampling"]["train_unit_count_per_scene"])
    train_limit = int(config["sampling"]["train_points_per_unit"])
    eval_limit = int(config["sampling"]["evaluation_points_per_unit"])
    route_radius = float(config["evaluation"]["route_corridor_radius_m"])
    layers = tuple(str(value) for value in config["map"]["layers"])
    distance_clip = float(config["map"]["signed_distance_clip_m"])
    rng = np.random.default_rng(int(config["seed"]))
    device = torch.device("cuda")

    descriptors = []
    for scene_number, scene in enumerate(config["scenes"]):
        name = str(scene["name"])
        evidence_units = _unit_dirs(evidence_root, name)
        for unit_number, evidence_unit in enumerate(evidence_units):
            descriptors.append(
                {
                    "scene_index": scene_number,
                    "unit_index": unit_number,
                    "evidence_unit": evidence_unit,
                    "native_unit": _native_unit_dir(
                        native_root,
                        name,
                        evidence_unit.name,
                        {name: str(inputs["native_partition"])},
                    ),
                    "processed_scene": processed_root / f"{int(scene['processed_index']):03d}",
                    "map_location": str(scene["map_location"]),
                    "is_train": unit_number < train_unit_count,
                }
            )
    if not descriptors:
        raise RuntimeError("no R3 materialization units found")

    names = (
        "native_hidden",
        "base_logit",
        "map_context",
        "hidden_free",
        "route",
        "scene_index",
        "unit_index",
        "is_train",
    )
    parts = {name: [] for name in names}
    io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    map_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def submit_io(descriptor: dict):
        return io_executor.submit(
            _load_unit,
            descriptor["evidence_unit"],
            descriptor["native_unit"],
            descriptor["processed_scene"],
            origin=origin,
            voxel_size=voxel_size,
            future_frame_count=future_frame_count,
        )

    io_future = submit_io(descriptors[0])
    try:
        for position, descriptor in enumerate(descriptors):
            features, centers, indices, labels, route_xy = io_future.result()
            if position + 1 < len(descriptors):
                io_future = submit_io(descriptors[position + 1])
            limit = train_limit if descriptor["is_train"] else eval_limit
            if features.shape[0] > limit:
                chosen = rng.choice(features.shape[0], size=limit, replace=False)
                features = features[chosen]
                centers = centers[chosen]
                indices = indices[chosen]
                labels = labels[chosen]

            frame = int(descriptor["evidence_unit"].name.removeprefix("f"))
            map_future = map_executor.submit(
                ego_aligned_map_context,
                map_root,
                descriptor["map_location"],
                descriptor["processed_scene"],
                frame,
                indices,
                route_xy,
                origin_m=origin,
                voxel_size_m=voxel_size,
                grid_shape=grid_shape,
                layers=layers,
                signed_distance_clip_m=distance_clip,
            )
            hidden, logits = _frozen_q0_embedding(q0, features)
            trajectory = continuous_trajectory_features(centers, route_xy, device).cpu().numpy()
            map_context = map_future.result()

            parts["native_hidden"].append(hidden)
            parts["base_logit"].append(logits)
            parts["map_context"].append(map_context)
            parts["hidden_free"].append(labels.astype(bool))
            parts["route"].append(trajectory[:, 0] <= route_radius)
            parts["scene_index"].append(np.full(labels.shape, descriptor["scene_index"], dtype=np.uint8))
            parts["unit_index"].append(np.full(labels.shape, descriptor["unit_index"], dtype=np.uint8))
            parts["is_train"].append(np.full(labels.shape, descriptor["is_train"], dtype=bool))
            print(
                f"materialized {position + 1}/{len(descriptors)} "
                f"scene={descriptor['scene_index']} unit={descriptor['unit_index']} points={labels.size}",
                flush=True,
            )
    finally:
        io_executor.shutdown(wait=True)
        map_executor.shutdown(wait=True)

    payload = {name: np.concatenate(values) for name, values in parts.items()}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp.npz")
    np.savez(temporary, **payload)
    os.replace(temporary, cache_path)


def _comparison(q0: dict, candidate: dict) -> dict[str, object]:
    q0_by_scene = {row["scene_index"]: row for row in q0["scene_rows"]}
    candidate_by_scene = {row["scene_index"]: row for row in candidate["scene_rows"]}
    deltas = {
        scene: candidate_by_scene[scene]["fixed_route_conflict_density"]
        - q0_by_scene[scene]["fixed_route_conflict_density"]
        for scene in q0_by_scene
    }
    q0_density = float(q0["pooled_fixed_route_conflict_density"])
    candidate_density = float(candidate["pooled_fixed_route_conflict_density"])
    return {
        "r3_minus_q0_pooled_fixed_route_density": candidate_density - q0_density,
        "relative_fixed_route_risk_reduction": float(
            (q0_density - candidate_density) / q0_density if q0_density > 0 else 0.0
        ),
        "scene_lower_count": sum(value < 0 for value in deltas.values()),
        "scene_equal_count": sum(value == 0 for value in deltas.values()),
        "scene_higher_count": sum(value > 0 for value in deltas.values()),
        "maximum_scene_regression": max(deltas.values()),
        "scene_deltas": deltas,
    }


def _nonroute_emission_risk(
    hidden_free: np.ndarray,
    route: np.ndarray,
    scores: np.ndarray,
    scene_index: np.ndarray,
    unit_index: np.ndarray,
    coverage: float,
) -> dict[str, float | int]:
    selected_nonroute = 0
    conflicts = 0
    for scene in np.unique(scene_index):
        for unit in np.unique(unit_index[scene_index == scene]):
            mask = (scene_index == scene) & (unit_index == unit)
            count = max(1, int(math.floor(float(coverage) * np.count_nonzero(mask))))
            selected = np.argsort(scores[mask], kind="stable")[:count]
            nonroute = ~route[mask][selected]
            selected_nonroute += int(nonroute.sum())
            conflicts += int(np.count_nonzero(hidden_free[mask][selected] & nonroute))
    return {
        "selected_nonroute_count": selected_nonroute,
        "hidden_free_conflict_count": conflicts,
        "emitted_conflict_rate": float(conflicts / selected_nonroute if selected_nonroute else 0.0),
    }


def run(config_path: Path, runs_root: Path, run_id: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v65" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(run_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    cache_path = Path(config["inputs"]["compact_cache"])
    cache_reused = cache_path.is_file()
    if not cache_reused:
        _materialize_cache(config, runs_root, cache_path)
    with np.load(cache_path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}

    train = arrays["is_train"]
    evaluate = ~train
    fit = fit_trajectory_residual(
        arrays["native_hidden"][train],
        arrays["base_logit"][train],
        arrays["map_context"][train],
        arrays["hidden_free"][train],
        **config["model"],
        seed=int(config["seed"]),
    )
    q0_scores = torch.sigmoid(torch.from_numpy(arrays["base_logit"][evaluate])).numpy()
    r3_scores = score_trajectory_residual(
        fit,
        arrays["native_hidden"][evaluate],
        arrays["base_logit"][evaluate],
        arrays["map_context"][evaluate],
    )
    shuffled = arrays["map_context"][evaluate].copy()
    eval_scenes = arrays["scene_index"][evaluate]
    eval_units = arrays["unit_index"][evaluate]
    rng = np.random.default_rng(int(config["seed"]) + 653)
    for scene in np.unique(eval_scenes):
        for unit in np.unique(eval_units[eval_scenes == scene]):
            mask = (eval_scenes == scene) & (eval_units == unit)
            shuffled[mask] = shuffled[mask][rng.permutation(np.count_nonzero(mask))]
    shuffled_scores = score_trajectory_residual(
        fit,
        arrays["native_hidden"][evaluate],
        arrays["base_logit"][evaluate],
        shuffled,
    )

    common = {
        "hidden_free": arrays["hidden_free"][evaluate],
        "route": arrays["route"][evaluate],
        "scene_index": eval_scenes,
        "unit_index": eval_units,
        "coverage": float(config["evaluation"]["matched_total_coverage"]),
        "tail_fraction": float(config["evaluation"]["tail_fraction"]),
    }
    q0_fixed = fixed_opportunity_metrics(scores=q0_scores, **common)
    r3_fixed = fixed_opportunity_metrics(scores=r3_scores, **common)
    shuffled_fixed = fixed_opportunity_metrics(scores=shuffled_scores, **common)
    comparison = _comparison(q0_fixed, r3_fixed)
    ranking = {
        "q0": ranking_metrics(arrays["hidden_free"][evaluate], q0_scores),
        "r3": ranking_metrics(arrays["hidden_free"][evaluate], r3_scores),
        "r3_shuffled_map": ranking_metrics(arrays["hidden_free"][evaluate], shuffled_scores),
    }
    ranking["r3_minus_q0_auroc"] = ranking["r3"]["auroc"] - ranking["q0"]["auroc"]
    ranking["r3_minus_q0_auprc"] = ranking["r3"]["auprc"] - ranking["q0"]["auprc"]
    ranking["r3_minus_shuffled_auroc"] = ranking["r3"]["auroc"] - ranking["r3_shuffled_map"]["auroc"]

    coverage = float(config["evaluation"]["matched_total_coverage"])
    q0_nonroute = _nonroute_emission_risk(
        arrays["hidden_free"][evaluate], arrays["route"][evaluate], q0_scores, eval_scenes, eval_units, coverage
    )
    r3_nonroute = _nonroute_emission_risk(
        arrays["hidden_free"][evaluate], arrays["route"][evaluate], r3_scores, eval_scenes, eval_units, coverage
    )
    q0_nonroute_rate = float(q0_nonroute["emitted_conflict_rate"])
    r3_nonroute_rate = float(r3_nonroute["emitted_conflict_rate"])
    nonroute_relative_change = float(
        (r3_nonroute_rate - q0_nonroute_rate) / q0_nonroute_rate
        if q0_nonroute_rate > 0 else 0.0
    )
    gates = {
        "minimum_auroc_gain": ranking["r3_minus_q0_auroc"] >= float(config["gates"]["minimum_auroc_gain"]),
        "minimum_fixed_route_risk_reduction": comparison["relative_fixed_route_risk_reduction"] >= float(config["gates"]["minimum_fixed_route_risk_reduction"]),
        "scene_direction_support": comparison["scene_lower_count"] > comparison["scene_higher_count"],
        "nonroute_risk_not_worse": nonroute_relative_change <= float(config["gates"]["maximum_nonroute_relative_risk_increase"]),
        "map_shuffle_response": ranking["r3_minus_shuffled_auroc"] > 0.0,
    }
    verdict = "positive_train_only_map_context_signal" if all(gates.values()) else "no_clear_train_only_map_context_signal"
    torch.save(
        {
            "state_dict": {name: value.detach().cpu() for name, value in fit.model.state_dict().items()},
            "map_context_mean": fit.trajectory_mean,
            "map_context_scale": fit.trajectory_scale,
            "map_context_feature_names": MAP_CONTEXT_FEATURE_NAMES,
            "native_hidden_dimension": int(arrays["native_hidden"].shape[1]),
        },
        run_dir / "map_context_residual.pt",
    )
    map_context = arrays["map_context"]
    summary = {
        "schema_version": "worldsim_v65.p1r3_map_context_summary.v1",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "claim_boundary": config["claim_boundary"],
        "cache_reused": cache_reused,
        "train_point_count": int(train.sum()),
        "evaluation_point_count": int(evaluate.sum()),
        "evaluation_positive_count": int(arrays["hidden_free"][evaluate].sum()),
        "map_context_feature_names": MAP_CONTEXT_FEATURE_NAMES,
        "map_context_feature_mean": map_context.mean(axis=0).tolist(),
        "map_context_feature_std": map_context.std(axis=0).tolist(),
        "ranking": ranking,
        "fixed_opportunity": {
            "q0": q0_fixed,
            "r3": r3_fixed,
            "r3_shuffled_map": shuffled_fixed,
            "comparison": comparison,
        },
        "nonroute_emission": {
            "q0": q0_nonroute,
            "r3": r3_nonroute,
            "relative_risk_change": nonroute_relative_change,
        },
        "epoch_losses": fit.epoch_losses,
        "gate_results": gates,
        "quality_read": "legacy_train_only_mechanism",
        "formal_v65_selection_read": False,
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

