"""Run the V6.5 train-only continuous-trajectory signal atlas on one RTX 3090."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
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
from motion_proj.worldsim_v65.task_contract import (
    TRAJECTORY_FEATURE_NAMES,
    continuous_trajectory_features,
)


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
    return features[valid], centers[valid], (target_state[x, y, z][valid] == FREE), route


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


def _materialize_cache(config: dict, runs_root: Path, processed_root: Path, cache_path: Path) -> None:
    inputs = config["inputs"]
    evidence_root = runs_root / inputs["evidence_run"]
    native_root = runs_root / inputs["native_run"]
    q0 = joblib.load(runs_root / inputs["risk_run"] / inputs["risk_model_relative_path"])
    origin = np.asarray(config["native_grid"]["origin_m"], dtype=np.float64)
    voxel_size = float(config["native_grid"]["voxel_size_m"])
    future_frame_count = int(config["trajectory"]["future_frame_count"])
    train_unit_count = int(config["sampling"]["train_unit_count_per_scene"])
    train_limit = int(config["sampling"]["train_points_per_unit"])
    eval_limit = int(config["sampling"]["evaluation_points_per_unit"])
    rng = np.random.default_rng(int(config["seed"]))
    device = torch.device("cuda")

    descriptors = []
    for scene_number, scene in enumerate(config["scenes"]):
        name = str(scene["name"])
        evidence_units = _unit_dirs(evidence_root, name)
        for unit_number, evidence_unit in enumerate(evidence_units):
            native_unit = _native_unit_dir(
                native_root,
                name,
                evidence_unit.name,
                {name: str(inputs["native_partition"])},
            )
            descriptors.append(
                (
                    scene_number,
                    unit_number,
                    evidence_unit,
                    native_unit,
                    processed_root / f"{int(scene['processed_index']):03d}",
                    unit_number < train_unit_count,
                )
            )

    parts = {name: [] for name in ("native_hidden", "base_logit", "trajectory", "hidden_free", "route", "scene_index", "unit_index", "is_train")}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def submit(descriptor: tuple):
        return executor.submit(
            _load_unit,
            descriptor[2],
            descriptor[3],
            descriptor[4],
            origin=origin,
            voxel_size=voxel_size,
            future_frame_count=future_frame_count,
        )

    future = submit(descriptors[0])
    for position, descriptor in enumerate(descriptors):
        features, centers, labels, route_xy = future.result()
        if position + 1 < len(descriptors):
            future = submit(descriptors[position + 1])
        limit = train_limit if descriptor[5] else eval_limit
        if features.shape[0] > limit:
            chosen = rng.choice(features.shape[0], size=limit, replace=False)
            features, centers, labels = features[chosen], centers[chosen], labels[chosen]
        hidden, logits = _frozen_q0_embedding(q0, features)
        trajectory = continuous_trajectory_features(centers, route_xy, device).cpu().numpy()
        parts["native_hidden"].append(hidden)
        parts["base_logit"].append(logits)
        parts["trajectory"].append(trajectory.astype(np.float32))
        parts["hidden_free"].append(labels.astype(bool))
        parts["route"].append((trajectory[:, 0] <= float(config["evaluation"]["route_corridor_radius_m"])))
        parts["scene_index"].append(np.full(labels.shape, descriptor[0], dtype=np.uint8))
        parts["unit_index"].append(np.full(labels.shape, descriptor[1], dtype=np.uint8))
        parts["is_train"].append(np.full(labels.shape, descriptor[5], dtype=bool))
    executor.shutdown(wait=True)
    payload = {name: np.concatenate(values) for name, values in parts.items()}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp.npz")
    np.savez(temporary, **payload)
    os.replace(temporary, cache_path)


def _comparison(q0: dict, t0: dict) -> dict[str, object]:
    q0_by_scene = {row["scene_index"]: row for row in q0["scene_rows"]}
    t0_by_scene = {row["scene_index"]: row for row in t0["scene_rows"]}
    deltas = {
        scene: t0_by_scene[scene]["fixed_route_conflict_density"]
        - q0_by_scene[scene]["fixed_route_conflict_density"]
        for scene in q0_by_scene
    }
    q0_density = float(q0["pooled_fixed_route_conflict_density"])
    t0_density = float(t0["pooled_fixed_route_conflict_density"])
    return {
        "t0_minus_q0_pooled_fixed_route_density": t0_density - q0_density,
        "relative_fixed_route_risk_reduction": float(
            (q0_density - t0_density) / q0_density if q0_density > 0 else 0.0
        ),
        "scene_lower_count": sum(value < 0 for value in deltas.values()),
        "scene_equal_count": sum(value == 0 for value in deltas.values()),
        "scene_higher_count": sum(value > 0 for value in deltas.values()),
        "maximum_scene_regression": max(deltas.values()),
        "scene_deltas": deltas,
    }


def run(config_path: Path, runs_root: Path, processed_root: Path, run_id: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v65" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(run_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    cache_path = Path(config["inputs"]["compact_cache_path"])
    cache_reused = cache_path.is_file()
    if not cache_reused:
        _materialize_cache(config, runs_root, processed_root, cache_path)
    with np.load(cache_path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    train = arrays["is_train"]
    evaluate = ~train
    fit = fit_trajectory_residual(
        arrays["native_hidden"][train],
        arrays["base_logit"][train],
        arrays["trajectory"][train],
        arrays["hidden_free"][train],
        **config["model"],
        seed=int(config["seed"]),
    )
    q0_scores = torch.sigmoid(torch.from_numpy(arrays["base_logit"][evaluate])).numpy()
    t0_scores = score_trajectory_residual(
        fit,
        arrays["native_hidden"][evaluate],
        arrays["base_logit"][evaluate],
        arrays["trajectory"][evaluate],
    )
    shuffled = arrays["trajectory"][evaluate].copy()
    eval_scenes = arrays["scene_index"][evaluate]
    eval_units = arrays["unit_index"][evaluate]
    rng = np.random.default_rng(int(config["seed"]) + 65)
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
    t0_fixed = fixed_opportunity_metrics(scores=t0_scores, **common)
    shuffled_fixed = fixed_opportunity_metrics(scores=shuffled_scores, **common)
    comparison = _comparison(q0_fixed, t0_fixed)
    ranking = {
        "q0": ranking_metrics(arrays["hidden_free"][evaluate], q0_scores),
        "t0": ranking_metrics(arrays["hidden_free"][evaluate], t0_scores),
        "t0_shuffled_trajectory": ranking_metrics(arrays["hidden_free"][evaluate], shuffled_scores),
    }
    ranking["t0_minus_q0_auroc"] = ranking["t0"]["auroc"] - ranking["q0"]["auroc"]
    ranking["t0_minus_q0_auprc"] = ranking["t0"]["auprc"] - ranking["q0"]["auprc"]
    ranking["t0_minus_shuffled_auroc"] = ranking["t0"]["auroc"] - ranking["t0_shuffled_trajectory"]["auroc"]
    gates = {
        "minimum_auroc_gain": ranking["t0_minus_q0_auroc"] >= float(config["gates"]["minimum_auroc_gain"]),
        "minimum_fixed_route_risk_reduction": comparison["relative_fixed_route_risk_reduction"] >= float(config["gates"]["minimum_fixed_route_risk_reduction"]),
        "scene_direction_support": comparison["scene_lower_count"] > comparison["scene_higher_count"],
        "trajectory_perturbation_response": ranking["t0_minus_shuffled_auroc"] > 0.0,
    }
    verdict = "positive_train_only_trajectory_signal" if all(gates.values()) else "no_clear_train_only_trajectory_signal"
    artifact = {
        "state_dict": {name: value.detach().cpu() for name, value in fit.model.state_dict().items()},
        "trajectory_mean": fit.trajectory_mean,
        "trajectory_scale": fit.trajectory_scale,
        "trajectory_feature_names": TRAJECTORY_FEATURE_NAMES,
        "native_hidden_dimension": int(arrays["native_hidden"].shape[1]),
    }
    torch.save(artifact, run_dir / "trajectory_residual.pt")
    summary = {
        "schema_version": "worldsim_v65.p1_signal_atlas_summary.v1",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "claim_boundary": config["claim_boundary"],
        "cache_reused": cache_reused,
        "train_point_count": int(train.sum()),
        "evaluation_point_count": int(evaluate.sum()),
        "evaluation_positive_count": int(arrays["hidden_free"][evaluate].sum()),
        "trajectory_feature_names": TRAJECTORY_FEATURE_NAMES,
        "ranking": ranking,
        "fixed_opportunity": {"q0": q0_fixed, "t0": t0_fixed, "t0_shuffled_trajectory": shuffled_fixed, "comparison": comparison},
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
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.runs_root.resolve(), args.processed_root.resolve(), args.run_id), indent=2))


if __name__ == "__main__":
    main()
