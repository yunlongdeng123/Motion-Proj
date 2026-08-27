"""Run the preregistered Tier-L learned-admission mechanism probe."""

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
from motion_proj.worldsim_v64.route_aware_compiler import _constrained_order, _empirical_cvar
from motion_proj.worldsim_v65.learned_admission import fit_coverage_model, predict_coverage


CONTEXT_FEATURE_NAMES = [
    "log_eligible_count",
    "route_fraction",
    "normalized_target_frame",
    "risk_mean",
    "risk_std",
    "risk_q10",
    "risk_q25",
    "risk_q40",
    "risk_q50",
    "risk_q60",
    "risk_q75",
    "risk_q90",
    "route_risk_q25",
    "route_risk_q50",
    "route_risk_q75",
]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_case(
    evidence_unit: Path,
    native_unit: Path,
    origin: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    indices, _, features = _target_free_boundary(
        evidence_unit,
        native_unit,
        native_origin_m=origin,
        native_voxel_size_m=voxel_size,
    )
    with np.load(evidence_unit / "TARGET_EVIDENCE.npz", allow_pickle=False) as source:
        target = {name: np.asarray(source[name]) for name in source.files}
    return indices, features, target


def _context(scores: np.ndarray, in_route: np.ndarray, frame: int) -> np.ndarray:
    quantiles = np.quantile(scores, [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90])
    route_scores = scores[in_route]
    route_quantiles = (
        np.quantile(route_scores, [0.25, 0.50, 0.75])
        if route_scores.size
        else np.quantile(scores, [0.25, 0.50, 0.75])
    )
    return np.asarray(
        [
            math.log1p(scores.size),
            float(in_route.mean()),
            float(frame / 200.0),
            float(scores.mean()),
            float(scores.std()),
            *quantiles.tolist(),
            *route_quantiles.tolist(),
        ],
        dtype=np.float32,
    )


def _materialize(config: dict, cache_path: Path) -> None:
    origin = np.asarray(config["native_grid"]["origin_m"], dtype=np.float64)
    voxel_size = float(config["native_grid"]["voxel_size_m"])
    shape = tuple(int(value) for value in config["native_grid"]["shape"])
    processed_root = Path(config["inputs"]["processed_root"])
    risk_root = Path(config["inputs"]["risk_run"])
    risk_model = joblib.load(risk_root / config["inputs"]["risk_model_relative_path"])
    route_cap_coverage = float(config["route"]["nominal_coverage_cap"])
    future_frames = int(config["route"]["future_frame_count"])
    corridor_radius = float(config["route"]["corridor_radius_m"])
    minimum_coverage = float(config["admission"]["minimum_coverage"])
    maximum_coverage = float(config["admission"]["maximum_coverage"])
    conflict_threshold = float(config["admission"]["hidden_free_conflict_threshold"])

    x = origin[0] + (np.arange(shape[0], dtype=np.float32) + 0.5) * voxel_size
    y = origin[1] + (np.arange(shape[1], dtype=np.float32) + 0.5) * voxel_size
    xx, yy = np.meshgrid(x, y, indexing="ij")
    grid_xy = torch.from_numpy(np.stack((xx, yy), axis=-1)).cuda()
    strata = sorted(config["admission"]["m1_conditional_coverages"])
    stratum_to_index = {name: index for index, name in enumerate(strata)}

    descriptors = []
    scene_names = []
    for role_index, role in enumerate(("train", "evaluation")):
        role_config = config["inputs"][role]
        evidence_root = Path(role_config["evidence_root"])
        native_root = Path(role_config["native_root"])
        partition = str(role_config["native_partition"])
        for scene_row in role_config["scenes"]:
            scene = str(scene_row["name"])
            scene_index = len(scene_names)
            scene_names.append(scene)
            for unit in _unit_dirs(evidence_root, scene):
                descriptors.append(
                    (
                        role_index,
                        scene_index,
                        scene,
                        int(scene_row["processed_index"]),
                        str(scene_row["stratum"]),
                        unit,
                        _native_unit_dir(native_root, scene, unit.name, {scene: partition}),
                    )
                )

    contexts, oracle_coverages = [], []
    roles, scene_indices, stratum_indices, eligible_counts, route_eligible_counts = [], [], [], [], []
    offsets = [0]
    cumulative_hidden_parts, cumulative_route_hidden_parts, cumulative_route_selected_parts = [], [], []
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def submit(row: tuple):
        return executor.submit(_load_case, row[5], row[6], origin, voxel_size)

    pending = submit(descriptors[0])
    with torch.inference_mode():
        for row_index, row in enumerate(descriptors):
            indices, features, target = pending.result()
            if row_index + 1 < len(descriptors):
                pending = submit(descriptors[row_index + 1])
            role_index, scene_index, _, processed_index, stratum, unit, _ = row
            scores = np.asarray(risk_model.score(features, features[:, :17]), dtype=np.float32)
            frame = int(unit.name.removeprefix("f"))
            route_xy = _future_route_in_target_lidar(
                processed_root / f"{processed_index:03d}", frame, future_frames
            )
            route_tensor = torch.from_numpy(route_xy).cuda()
            corridor = (
                (grid_xy[None] - route_tensor[:, None, None]).square().sum(dim=-1).amin(dim=0)
                <= corridor_radius**2
            ).cpu().numpy()
            ix, iy, iz = indices.T
            in_route = corridor[ix, iy]
            target_state, target_valid = _evidence_on_native_grid(
                target,
                native_shape=shape,
                native_origin_m=origin,
                native_voxel_size_m=voxel_size,
            )
            hidden_free = (target_state[ix, iy, iz] == FREE) & target_valid[ix, iy, iz]
            route_cap = int(np.floor(route_cap_coverage * int(np.count_nonzero(in_route))))
            order = _constrained_order(scores, in_route, route_cap)
            maximum_count = min(order.size, max(1, int(np.floor(maximum_coverage * scores.size))))
            minimum_count = min(maximum_count, max(1, int(np.floor(minimum_coverage * scores.size))))
            selected_order = order[:maximum_count]
            selected_hidden = hidden_free[selected_order].astype(np.uint32)
            selected_route = in_route[selected_order]
            cumulative_hidden = np.cumsum(selected_hidden, dtype=np.uint32)
            cumulative_route_hidden = np.cumsum(selected_hidden * selected_route, dtype=np.uint32)
            cumulative_route_selected = np.cumsum(selected_route, dtype=np.uint32)
            rates = cumulative_hidden / np.arange(1, maximum_count + 1)
            eligible_safe = np.flatnonzero(
                (np.arange(1, maximum_count + 1) >= minimum_count) & (rates <= conflict_threshold)
            )
            oracle_count = int(eligible_safe[-1] + 1) if eligible_safe.size else minimum_count

            contexts.append(_context(scores, in_route, frame))
            oracle_coverages.append(oracle_count / scores.size)
            roles.append(role_index)
            scene_indices.append(scene_index)
            stratum_indices.append(stratum_to_index[stratum])
            eligible_counts.append(scores.size)
            route_eligible_counts.append(int(np.count_nonzero(in_route)))
            cumulative_hidden_parts.append(cumulative_hidden)
            cumulative_route_hidden_parts.append(cumulative_route_hidden)
            cumulative_route_selected_parts.append(cumulative_route_selected)
            offsets.append(offsets[-1] + maximum_count)
    executor.shutdown(wait=True)

    payload = {
        "context": np.stack(contexts),
        "oracle_coverage": np.asarray(oracle_coverages, dtype=np.float32),
        "role": np.asarray(roles, dtype=np.uint8),
        "scene_index": np.asarray(scene_indices, dtype=np.int16),
        "stratum_index": np.asarray(stratum_indices, dtype=np.int8),
        "eligible_count": np.asarray(eligible_counts, dtype=np.int32),
        "route_eligible_count": np.asarray(route_eligible_counts, dtype=np.int32),
        "prefix_offset": np.asarray(offsets, dtype=np.int64),
        "cumulative_hidden": np.concatenate(cumulative_hidden_parts),
        "cumulative_route_hidden": np.concatenate(cumulative_route_hidden_parts),
        "cumulative_route_selected": np.concatenate(cumulative_route_selected_parts),
        "scene_names": np.asarray(scene_names),
        "strata": np.asarray(strata),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp.npz")
    np.savez(temporary, **payload)
    os.replace(temporary, cache_path)


def _case_arm(arrays: dict[str, np.ndarray], index: int, coverage: float, threshold: float) -> dict[str, object]:
    eligible = int(arrays["eligible_count"][index])
    begin, end = int(arrays["prefix_offset"][index]), int(arrays["prefix_offset"][index + 1])
    count = min(end - begin, max(1, int(np.floor(float(coverage) * eligible))))
    position = begin + count - 1
    conflicts = int(arrays["cumulative_hidden"][position])
    route_conflicts = int(arrays["cumulative_route_hidden"][position])
    route_selected = int(arrays["cumulative_route_selected"][position])
    route_eligible = int(arrays["route_eligible_count"][index])
    return {
        "selected_count": count,
        "realized_coverage": count / eligible,
        "hidden_free_conflict": conflicts / count,
        "case_loss": conflicts / count > threshold,
        "route_eligible_count": route_eligible,
        "route_selected_count": route_selected,
        "route_conflict_count": route_conflicts,
        "fixed_route_density": route_conflicts / route_eligible if route_eligible else 0.0,
    }


def _summarize(cases: list[dict[str, object]], arm: str, tail_fraction: float) -> dict[str, object]:
    rows = [row[arm] for row in cases]
    tail, tail_count = _empirical_cvar([float(row["fixed_route_density"]) for row in rows], tail_fraction)
    route_eligible = sum(int(row["route_eligible_count"]) for row in rows)
    route_conflicts = sum(int(row["route_conflict_count"]) for row in rows)
    return {
        "case_count": len(rows),
        "failure_count": sum(bool(row["case_loss"]) for row in rows),
        "mean_realized_coverage": float(np.mean([row["realized_coverage"] for row in rows])),
        "pooled_fixed_route_density": route_conflicts / route_eligible if route_eligible else 0.0,
        "fixed_route_tail_cvar": tail,
        "tail_count": tail_count,
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

    train = arrays["role"] == 0
    evaluate = arrays["role"] == 1
    minimum_coverage = float(config["admission"]["minimum_coverage"])
    maximum_coverage = float(config["admission"]["maximum_coverage"])
    model_config = config["model"]
    fit = fit_coverage_model(
        arrays["context"][train],
        arrays["oracle_coverage"][train],
        hidden_dimensions=tuple(int(value) for value in model_config["hidden_dimensions"]),
        minimum_coverage=minimum_coverage,
        maximum_coverage=maximum_coverage,
        epochs=int(model_config["epochs"]),
        learning_rate=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
        seed=int(config["seed"]),
    )
    learned_coverage = predict_coverage(fit, arrays["context"][evaluate])
    evaluation_indices = np.flatnonzero(evaluate)
    threshold = float(config["admission"]["hidden_free_conflict_threshold"])
    coverage_by_stratum = config["admission"]["m1_conditional_coverages"]
    strata = arrays["strata"].tolist()
    cases = []
    for local_index, index in enumerate(evaluation_indices):
        baseline_coverage = float(coverage_by_stratum[strata[int(arrays["stratum_index"][index])]])
        cases.append({
            "scene_index": int(arrays["scene_index"][index]),
            "stratum": strata[int(arrays["stratum_index"][index])],
            "oracle_coverage": float(arrays["oracle_coverage"][index]),
            "predicted_coverage": float(learned_coverage[local_index]),
            "m1": _case_arm(arrays, int(index), baseline_coverage, threshold),
            "g0": _case_arm(arrays, int(index), float(learned_coverage[local_index]), threshold),
        })
    with (run_dir / "CASE_METRICS.jsonl").open("w", encoding="utf-8") as handle:
        for row in cases:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    tail_fraction = float(config["evaluation"]["tail_fraction"])
    arms = {arm: _summarize(cases, arm, tail_fraction) for arm in ("m1", "g0")}
    coverage_uplift = arms["g0"]["mean_realized_coverage"] - arms["m1"]["mean_realized_coverage"]
    baseline_tail = float(arms["m1"]["fixed_route_tail_cvar"])
    tail_reduction = (
        (baseline_tail - float(arms["g0"]["fixed_route_tail_cvar"])) / baseline_tail
        if baseline_tail > 0 else 0.0
    )
    baseline_density = float(arms["m1"]["pooled_fixed_route_density"])
    density_regression = (
        (float(arms["g0"]["pooled_fixed_route_density"]) - baseline_density) / baseline_density
        if baseline_density > 0
        else (0.0 if float(arms["g0"]["pooled_fixed_route_density"]) == 0 else float("inf"))
    )
    scene_support = 0
    scene_rows = []
    for scene in sorted({int(row["scene_index"]) for row in cases}):
        members = [row for row in cases if int(row["scene_index"]) == scene]
        m1_failures = sum(bool(row["m1"]["case_loss"]) for row in members)
        g0_failures = sum(bool(row["g0"]["case_loss"]) for row in members)
        m1_coverage = float(np.mean([row["m1"]["realized_coverage"] for row in members]))
        g0_coverage = float(np.mean([row["g0"]["realized_coverage"] for row in members]))
        m1_density = float(np.mean([row["m1"]["fixed_route_density"] for row in members]))
        g0_density = float(np.mean([row["g0"]["fixed_route_density"] for row in members]))
        supported = g0_failures <= m1_failures and (g0_coverage > m1_coverage or g0_density < m1_density)
        scene_support += int(supported)
        scene_rows.append({
            "scene_index": scene,
            "m1_failure_count": m1_failures,
            "g0_failure_count": g0_failures,
            "coverage_delta": g0_coverage - m1_coverage,
            "fixed_route_density_delta": g0_density - m1_density,
            "supported": supported,
        })

    primary = config["gates"]["minimum_coverage_uplift_or_tail_reduction"]
    gates = {
        "coverage_or_tail_increment": coverage_uplift >= float(primary["coverage_uplift"])
        or tail_reduction >= float(primary["fixed_route_tail_reduction"]),
        "no_more_case_failures": int(arms["g0"]["failure_count"]) <= int(arms["m1"]["failure_count"]),
        "pooled_route_density_regression": density_regression
        <= float(config["gates"]["maximum_relative_pooled_route_density_regression"]),
        "minimum_scene_support": scene_support >= int(config["gates"]["minimum_scene_support"]),
    }
    verdict = "positive_train_only_learned_admission" if all(gates.values()) else "no_clear_train_only_learned_admission"
    oracle_eval = arrays["oracle_coverage"][evaluate]
    actual_mse = float(np.mean(np.square(learned_coverage - oracle_eval)))
    rng = np.random.default_rng(int(config["seed"]) + 654)
    shuffled_mse = float(np.mean(np.square(learned_coverage[rng.permutation(learned_coverage.size)] - oracle_eval)))
    torch.save({
        "state_dict": {name: value.detach().cpu() for name, value in fit.model.state_dict().items()},
        "mean": fit.mean,
        "scale": fit.scale,
        "feature_names": CONTEXT_FEATURE_NAMES,
        "minimum_coverage": minimum_coverage,
        "maximum_coverage": maximum_coverage,
    }, run_dir / "LEARNED_ADMISSION.pt")
    summary = {
        "schema_version": "worldsim_v65.p4t_learned_admission_summary.v1",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "claim_boundary": config["claim_boundary"],
        "cache_reused": cache_reused,
        "train_case_count": int(np.count_nonzero(train)),
        "evaluation_case_count": int(np.count_nonzero(evaluate)),
        "train_oracle_coverage_min_mean_max": [float(arrays["oracle_coverage"][train].min()), float(arrays["oracle_coverage"][train].mean()), float(arrays["oracle_coverage"][train].max())],
        "evaluation_oracle_coverage_min_mean_max": [float(oracle_eval.min()), float(oracle_eval.mean()), float(oracle_eval.max())],
        "coverage_prediction_mse": actual_mse,
        "shuffled_prediction_mse": shuffled_mse,
        "arms": arms,
        "comparison": {
            "coverage_uplift": coverage_uplift,
            "fixed_route_tail_reduction": tail_reduction,
            "relative_pooled_route_density_regression": density_regression,
            "scene_support_count": scene_support,
            "scene_rows": scene_rows,
        },
        "gate_results": gates,
        "formal_v65_admission_selection_read": False,
        "stratum_inference_feature": False,
        "resources": {
            "gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
            "wall_seconds": time.monotonic() - started,
        },
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "status.json", {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    return {"run_dir": str(run_dir), "verdict": verdict, "gate_results": gates, "comparison": summary["comparison"], "resources": summary["resources"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.runs_root.resolve(), args.run_id), indent=2))


if __name__ == "__main__":
    main()
