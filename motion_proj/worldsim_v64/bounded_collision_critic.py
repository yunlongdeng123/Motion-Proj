"""Bounded action-lattice collision critic without a large neural world model."""

from __future__ import annotations

import argparse
import json
import math
import resource
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import torch
import yaml

from motion_proj.worldsim_v64.conditional_state_bake import _target_free_boundary
from motion_proj.worldsim_v64.gaussian_route_consumer import _future_route_in_target_lidar
from motion_proj.worldsim_v64.native_voxel_uq import _native_unit_dir, _unit_dirs
from motion_proj.worldsim_v64.route_aware_compiler import _constrained_order


FEATURE_NAMES = (
    "progress_ratio",
    "signed_lateral_offset_normalized",
    "absolute_lateral_offset_normalized",
    "route_length_ratio",
    "acceleration_ratio",
    "route_eligible_fraction",
    "m0_route_selected_fraction",
    "m1_route_selected_fraction",
    "route_mean_risk",
    "m1_selected_route_mean_risk",
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _resampled_action(
    logged_route: np.ndarray,
    *,
    progress_ratio: float,
    lateral_offset_m: float,
) -> np.ndarray:
    base = np.vstack((np.zeros((1, 2), dtype=np.float64), np.asarray(logged_route, dtype=np.float64)))
    source = np.linspace(0.0, 1.0, len(base))
    query = np.linspace(0.0, float(progress_ratio), len(base))
    path = np.stack(
        (np.interp(query, source, base[:, 0]), np.interp(query, source, base[:, 1])), axis=1
    )
    if lateral_offset_m != 0.0:
        tangent = np.gradient(path, axis=0)
        norm = np.linalg.norm(tangent, axis=1, keepdims=True)
        tangent = tangent / np.maximum(norm, 1e-6)
        normal = np.stack((-tangent[:, 1], tangent[:, 0]), axis=1)
        profile = np.sin(np.linspace(0.0, math.pi / 2.0, len(path)))[:, None]
        path = path + normal * profile * float(lateral_offset_m)
        path[0] = 0.0
    return path.astype(np.float32)


def _path_length(path: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def _maximum_planar_acceleration(path: np.ndarray, horizon_seconds: float) -> float:
    if len(path) < 3:
        return 0.0
    dt = float(horizon_seconds) / float(len(path) - 1)
    velocity = np.diff(path.astype(np.float64), axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    return float(np.linalg.norm(acceleration, axis=1).max(initial=0.0))


def _action_rows(logged_route: np.ndarray, action_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    horizon = float(action_config["horizon_seconds"])
    lateral_scale = max(abs(float(value)) for value in action_config["lateral_offsets_m"])
    logged = np.vstack((np.zeros((1, 2), dtype=np.float64), np.asarray(logged_route)))
    logged_length = max(_path_length(logged), 1e-6)
    rows: list[dict[str, Any]] = []
    for lateral in action_config["lateral_offsets_m"]:
        for progress in action_config["progress_ratios"]:
            path = _resampled_action(
                logged_route,
                progress_ratio=float(progress),
                lateral_offset_m=float(lateral),
            )
            acceleration = _maximum_planar_acceleration(path, horizon)
            rows.append(
                {
                    "action_id": f"p{float(progress):.2f}_l{float(lateral):+.2f}",
                    "source_role": "real" if float(lateral) == 0.0 else "generated",
                    "progress_ratio": float(progress),
                    "lateral_offset_m": float(lateral),
                    "lateral_offset_scale_m": lateral_scale,
                    "path": path,
                    "route_length_m": _path_length(path),
                    "route_length_ratio": float(_path_length(path) / logged_length),
                    "maximum_planar_acceleration_mps2": acceleration,
                }
            )
    stop_path = np.zeros((len(logged), 2), dtype=np.float32)
    rows.append(
        {
            "action_id": "stop",
            "source_role": "stop",
            "progress_ratio": 0.0,
            "lateral_offset_m": 0.0,
            "lateral_offset_scale_m": lateral_scale,
            "path": stop_path,
            "route_length_m": 0.0,
            "route_length_ratio": 0.0,
            "maximum_planar_acceleration_mps2": 0.0,
        }
    )
    return rows


def _corridor_masks(
    paths: np.ndarray,
    *,
    origin_xy: np.ndarray,
    voxel_size_m: float,
    shape_xy: tuple[int, int],
    radius_m: float,
    device: torch.device,
    action_chunk: int = 2,
) -> np.ndarray:
    x = float(origin_xy[0]) + (torch.arange(shape_xy[0], device=device) + 0.5) * float(
        voxel_size_m
    )
    y = float(origin_xy[1]) + (torch.arange(shape_xy[1], device=device) + 0.5) * float(
        voxel_size_m
    )
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1)[None, :, :, None, :]
    path_tensor = torch.from_numpy(np.asarray(paths, dtype=np.float32)).to(device)
    masks = []
    for start in range(0, len(path_tensor), action_chunk):
        chunk = path_tensor[start : start + action_chunk, None, None, :, :]
        distance2 = (grid - chunk).square().sum(dim=-1).amin(dim=-1)
        masks.append((distance2 <= float(radius_m) ** 2).cpu().numpy())
    return np.concatenate(masks, axis=0)


def _mean(values: np.ndarray, default: float = 0.0) -> float:
    return float(np.mean(values)) if values.size else float(default)


def _cohort_rows(
    cohort: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    runs_root: Path,
    processed_root: Path,
    model: object,
    device: torch.device,
) -> list[dict[str, Any]]:
    evidence_root = runs_root / str(cohort["evidence_run"])
    native_root = runs_root / str(cohort["native_run"])
    scenes = [str(row["name"]) for row in cohort["scenes"]]
    strata = {str(row["name"]): str(row["stratum"]) for row in cohort["scenes"]}
    processed_indices = {str(row["name"]): int(row["processed_index"]) for row in cohort["scenes"]}
    partition_by_scene = {scene: str(cohort["native_partition"]) for scene in scenes}
    native_origin = np.asarray(config["native_grid"]["origin_m"], dtype=np.float64)
    native_voxel = float(config["native_grid"]["voxel_size_m"])
    native_shape = tuple(int(value) for value in config["native_grid"]["shape"])
    conditional_coverages = {
        str(key): float(value) for key, value in config["policy"]["m0_conditional_nominal_coverages"].items()
    }
    route_cap_coverage = float(config["policy"]["m1_route_nominal_coverage_cap"])
    future_frame_count = int(config["route"]["future_frame_count"])
    corridor_radius = float(config["route"]["corridor_radius_m"])
    evidence_radius = float(config["collision_proxy"]["ego_corridor_radius_m"])
    comfort_limit = float(config["metrics"]["maximum_planar_acceleration_mps2"])
    rows: list[dict[str, Any]] = []

    for scene in scenes:
        for evidence_unit in _unit_dirs(evidence_root, scene):
            native_unit = _native_unit_dir(native_root, scene, evidence_unit.name, partition_by_scene)
            indices, _, features = _target_free_boundary(
                evidence_unit,
                native_unit,
                native_origin_m=native_origin,
                native_voxel_size_m=native_voxel,
            )
            logits = features[:, :17]
            scores = np.asarray(model.score(features, logits), dtype=np.float32)
            selected_count = max(
                1, int(np.floor(conditional_coverages[strata[scene]] * scores.size))
            )
            m0_order = np.argsort(scores, kind="stable")[:selected_count]
            target_frame = int(evidence_unit.name.removeprefix("f"))
            logged_route = _future_route_in_target_lidar(
                processed_root / f"{processed_indices[scene]:03d}",
                target_frame,
                future_frame_count,
            )
            actions = _action_rows(logged_route, config["action_lattice"])
            paths = np.stack([row["path"] for row in actions])
            native_corridors = _corridor_masks(
                paths,
                origin_xy=native_origin[:2],
                voxel_size_m=native_voxel,
                shape_xy=native_shape[:2],
                radius_m=corridor_radius,
                device=device,
            )
            with np.load(evidence_unit / "TARGET_EVIDENCE.npz", allow_pickle=False) as source:
                evidence_origin = np.asarray(source["grid_origin_m"], dtype=np.float64)
                evidence_voxel = float(source["voxel_size_m"])
                evidence_shape = tuple(int(value) for value in source["grid_shape"])
                actor_indices = np.asarray(source["actor_swept_envelope_indices"], dtype=np.int64)
            actor_mask = np.zeros(evidence_shape[:2], dtype=bool)
            if actor_indices.size:
                actor_mask[actor_indices[:, 0], actor_indices[:, 1]] = True
            evidence_corridors = _corridor_masks(
                paths,
                origin_xy=evidence_origin[:2],
                voxel_size_m=evidence_voxel,
                shape_xy=evidence_shape[:2],
                radius_m=evidence_radius,
                device=device,
            )
            actual_unsafe = np.any(evidence_corridors & actor_mask[None], axis=(1, 2))
            actor_overlap_cells = np.count_nonzero(
                evidence_corridors & actor_mask[None], axis=(1, 2)
            )
            ix, iy, _ = indices.T
            case_id = f"{scene}/{evidence_unit.name}"
            for action_index, action in enumerate(actions):
                in_route = native_corridors[action_index, ix, iy]
                route_count = int(np.count_nonzero(in_route))
                route_cap = int(np.floor(route_cap_coverage * route_count))
                m1_order = _constrained_order(scores, in_route, route_cap)[:selected_count]
                if m1_order.size != selected_count:
                    raise RuntimeError(f"insufficient non-route capacity: {case_id}/{action['action_id']}")
                m0_route = m0_order[in_route[m0_order]]
                m1_route = m1_order[in_route[m1_order]]
                route_scores = scores[in_route]
                feature_values = [
                    float(action["progress_ratio"]),
                    float(action["lateral_offset_m"] / action["lateral_offset_scale_m"]),
                    float(abs(action["lateral_offset_m"]) / action["lateral_offset_scale_m"]),
                    float(action["route_length_ratio"]),
                    float(action["maximum_planar_acceleration_mps2"] / comfort_limit),
                    float(route_count / scores.size),
                    float(m0_route.size / route_count) if route_count else 0.0,
                    float(m1_route.size / route_count) if route_count else 0.0,
                    _mean(route_scores, default=1.0),
                    _mean(scores[m1_route], default=1.0),
                ]
                rows.append(
                    {
                        "cohort": str(cohort["role"]),
                        "case_id": case_id,
                        "scene": scene,
                        "stratum": strata[scene],
                        "unit": evidence_unit.name,
                        "action_id": action["action_id"],
                        "source_role": action["source_role"],
                        "progress_ratio": action["progress_ratio"],
                        "lateral_offset_m": action["lateral_offset_m"],
                        "route_length_m": action["route_length_m"],
                        "maximum_planar_acceleration_mps2": action[
                            "maximum_planar_acceleration_mps2"
                        ],
                        "comfort_pass": bool(
                            action["maximum_planar_acceleration_mps2"] <= comfort_limit
                        ),
                        "route_eligible_count": route_count,
                        "m0_route_selected_count": int(m0_route.size),
                        "m1_route_selected_count": int(m1_route.size),
                        "verification_score": feature_values[-1],
                        "feature_values": feature_values,
                        "actual_unsafe": bool(actual_unsafe[action_index]),
                        "actor_overlap_cell_count": int(actor_overlap_cells[action_index]),
                        "verified_generated": False,
                    }
                )
    return rows


def _mark_verified_generated(rows: list[dict[str, Any]], fraction: float) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["source_role"] == "generated":
            groups[str(row["case_id"])].append(row)
    for candidates in groups.values():
        count = max(1, int(math.ceil(len(candidates) * float(fraction))))
        ordered = sorted(candidates, key=lambda row: (row["verification_score"], row["action_id"]))
        for row in ordered[:count]:
            row["verified_generated"] = True


def _fit_linear_critic(
    rows: list[dict[str, Any]], *, config: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    x = np.asarray([row["feature_values"] for row in rows], dtype=np.float32)
    y = np.asarray([row["actual_unsafe"] for row in rows], dtype=np.float32)
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-6)
    standardized = (x - mean) / scale
    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    linear = torch.nn.Linear(standardized.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(
        linear.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    x_tensor = torch.from_numpy(standardized).to(device)
    y_tensor = torch.from_numpy(y).to(device)
    positives = max(int(y.sum()), 1)
    negatives = max(int(len(y) - y.sum()), 1)
    positive_weight = float(len(y) / (2.0 * positives))
    negative_weight = float(len(y) / (2.0 * negatives))
    final_loss = 0.0
    for _ in range(int(config["steps"])):
        logits = linear(x_tensor).squeeze(1)
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y_tensor, reduction="none"
        )
        weights = torch.where(y_tensor > 0.5, positive_weight, negative_weight)
        loss = (losses * weights).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    weight = linear.weight.detach().cpu().numpy()[0].astype(np.float64)
    bias = float(linear.bias.detach().cpu().numpy()[0])
    return {
        "feature_mean": mean.astype(np.float64).tolist(),
        "feature_scale": scale.astype(np.float64).tolist(),
        "weight": weight.tolist(),
        "bias": bias,
        "training_row_count": len(rows),
        "training_positive_count": int(y.sum()),
        "final_loss": final_loss,
    }


def _predict(model: Mapping[str, Any], rows: list[dict[str, Any]]) -> np.ndarray:
    x = np.asarray([row["feature_values"] for row in rows], dtype=np.float64)
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    weight = np.asarray(model["weight"], dtype=np.float64)
    logits = ((x - mean) / scale) @ weight + float(model["bias"])
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def _ece(probabilities: np.ndarray, labels: np.ndarray, bins: int) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        if index + 1 == bins:
            selected = (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
        else:
            selected = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        if np.any(selected):
            total += float(selected.mean()) * abs(
                float(probabilities[selected].mean()) - float(labels[selected].mean())
            )
    return total


def _evaluate(
    rows: list[dict[str, Any]],
    probabilities: np.ndarray,
    *,
    threshold: float,
    stuck_threshold: float,
    calibration_bins: int,
) -> dict[str, Any]:
    labels = np.asarray([row["actual_unsafe"] for row in rows], dtype=bool)
    predicted_unsafe = probabilities >= float(threshold)
    tp = int(np.count_nonzero(labels & predicted_unsafe))
    fn = int(np.count_nonzero(labels & ~predicted_unsafe))
    tn = int(np.count_nonzero(~labels & ~predicted_unsafe))
    fp = int(np.count_nonzero(~labels & predicted_unsafe))
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["case_id"])].append(index)
    selected_rows = []
    for case_id, indices in groups.items():
        safe = [index for index in indices if probabilities[index] < float(threshold)]
        if safe:
            chosen = min(
                safe,
                key=lambda index: (
                    -float(rows[index]["progress_ratio"]),
                    not bool(rows[index]["comfort_pass"]),
                    abs(float(rows[index]["lateral_offset_m"])),
                    float(probabilities[index]),
                    str(rows[index]["action_id"]),
                ),
            )
            fallback = False
        else:
            chosen = next(index for index in indices if rows[index]["source_role"] == "stop")
            fallback = True
        row = rows[chosen]
        unsafe = bool(row["actual_unsafe"])
        predicted_safe = bool(probabilities[chosen] < float(threshold))
        progress = float(row["progress_ratio"])
        comfortable = bool(row["comfort_pass"])
        selected_rows.append(
            {
                "case_id": case_id,
                "action_id": row["action_id"],
                "unsafe_probability": float(probabilities[chosen]),
                "actual_unsafe": unsafe,
                "predicted_safe": predicted_safe,
                "false_safe": unsafe and predicted_safe,
                "progress_ratio": progress,
                "stuck": progress <= float(stuck_threshold),
                "comfort_pass": comfortable,
                "fallback_stop": fallback,
                "reward": progress - float(unsafe) - 0.1 * float(not comfortable),
            }
        )
    return {
        "action_count": len(rows),
        "unsafe_action_count": int(labels.sum()),
        "predicted_safe_action_count": int(np.count_nonzero(~predicted_unsafe)),
        "action_false_safe_count": fn,
        "unsafe_action_recall": float(tp / (tp + fn)) if tp + fn else 1.0,
        "safe_action_precision": float(tn / (tn + fn)) if tn + fn else 1.0,
        "brier_score": float(np.mean((probabilities - labels.astype(np.float64)) ** 2)),
        "expected_calibration_error": _ece(
            probabilities, labels.astype(np.float64), int(calibration_bins)
        ),
        "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp},
        "policy": {
            "case_count": len(selected_rows),
            "collision_count": sum(bool(row["actual_unsafe"]) for row in selected_rows),
            "collision_false_safe_count": sum(bool(row["false_safe"]) for row in selected_rows),
            "mean_progress_ratio": float(
                np.mean([float(row["progress_ratio"]) for row in selected_rows])
            ),
            "stuck_rate": float(np.mean([bool(row["stuck"]) for row in selected_rows])),
            "comfort_pass_rate": float(
                np.mean([bool(row["comfort_pass"]) for row in selected_rows])
            ),
            "fallback_stop_count": sum(bool(row["fallback_stop"]) for row in selected_rows),
            "mean_total_reward": float(np.mean([float(row["reward"]) for row in selected_rows])),
        },
        "selected_actions": selected_rows,
    }


def run(config_path: Path, runs_root: Path, processed_root: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v64" / str(config["task_id"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(
        run_dir / "status.json",
        {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("bounded collision critic requires the configured single CUDA GPU")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model = joblib.load(
        runs_root
        / str(config["inputs"]["risk_run"])
        / str(config["inputs"]["risk_model_relative_path"])
    )

    training_rows = _cohort_rows(
        config["inputs"]["training"],
        config=config,
        runs_root=runs_root,
        processed_root=processed_root,
        model=model,
        device=device,
    )
    _mark_verified_generated(
        training_rows, float(config["augmentation"]["verified_generated_keep_fraction"])
    )
    _write_jsonl(run_dir / "TRAINING_ACTION_ROWS.jsonl", training_rows)
    arm_rows = {
        "real_only": [row for row in training_rows if row["source_role"] == "real"],
        "real_plus_naive_generated": [
            row for row in training_rows if row["source_role"] in {"real", "generated"}
        ],
        "real_plus_unc_verified": [
            row
            for row in training_rows
            if row["source_role"] == "real" or bool(row["verified_generated"])
        ],
    }
    critic_models = {
        name: _fit_linear_critic(rows, config=config["critic"], device=device)
        for name, rows in arm_rows.items()
    }
    _write_json(
        run_dir / "CRITIC_MODELS.json",
        {"feature_names": FEATURE_NAMES, "arms": critic_models},
    )

    test_rows = _cohort_rows(
        config["inputs"]["evaluation"],
        config=config,
        runs_root=runs_root,
        processed_root=processed_root,
        model=model,
        device=device,
    )
    threshold = float(config["critic"]["decision_threshold"])
    metrics = {}
    eval_rows = []
    for name, critic in critic_models.items():
        probabilities = _predict(critic, test_rows)
        evaluation = _evaluate(
            test_rows,
            probabilities,
            threshold=threshold,
            stuck_threshold=float(config["metrics"]["stuck_progress_ratio"]),
            calibration_bins=int(config["metrics"]["calibration_bins"]),
        )
        metrics[name] = {key: value for key, value in evaluation.items() if key != "selected_actions"}
        for row, probability in zip(test_rows, probabilities):
            eval_rows.append(
                {
                    **{key: value for key, value in row.items() if key != "feature_values"},
                    "arm": name,
                    "unsafe_probability": float(probability),
                    "predicted_unsafe": bool(probability >= threshold),
                }
            )
        _write_jsonl(run_dir / f"SELECTED_ACTIONS_{name.upper()}.jsonl", evaluation["selected_actions"])
    _write_jsonl(run_dir / "EVALUATION_ACTION_ROWS.jsonl", eval_rows)

    real = metrics["real_only"]["policy"]
    naive = metrics["real_plus_naive_generated"]["policy"]
    verified = metrics["real_plus_unc_verified"]["policy"]
    gates = {
        "verified_collision_false_safe_not_worse": int(verified["collision_false_safe_count"])
        <= min(
            int(real["collision_false_safe_count"]),
            int(naive["collision_false_safe_count"]),
        ),
        "verified_mean_progress_nontrivial": float(verified["mean_progress_ratio"])
        >= float(config["gates"]["minimum_verified_mean_progress_ratio"]),
        "verified_stuck_rate_bounded": float(verified["stuck_rate"])
        <= float(config["gates"]["maximum_verified_stuck_rate"]),
    }
    if not all(gates.values()):
        verdict = str(config["verdict_on_failure"])
    elif int(verified["collision_false_safe_count"]) < int(real["collision_false_safe_count"]):
        verdict = str(config["verdict_on_pass"])
    else:
        verdict = str(config["verdict_on_no_increment"])
    summary = {
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "claim_boundary": config["claim_boundary"],
        "training_case_count": len({row["case_id"] for row in training_rows}),
        "training_action_counts": {name: len(rows) for name, rows in arm_rows.items()},
        "training_positive_counts": {
            name: sum(bool(row["actual_unsafe"]) for row in rows) for name, rows in arm_rows.items()
        },
        "evaluation_case_count": len({row["case_id"] for row in test_rows}),
        "evaluation_action_count": len(test_rows),
        "arms": metrics,
        "gate_results": gates,
        "large_nwm_trained": False,
        "test_action_labels_read_after_model_fit": True,
        "model_or_action_selection_during_test": False,
        "resources": {
            "gpu_used": True,
            "wall_seconds": time.monotonic() - started,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
            "peak_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        },
        "references": config["references"],
        "failure_ledger_refs": config["failure_ledger_refs"],
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "resource.json", summary["resources"])
    _write_json(
        run_dir / "status.json",
        {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    return {
        "run_dir": str(run_dir),
        "verdict": verdict,
        "gate_results": gates,
        "policy_metrics": {name: values["policy"] for name, values in metrics.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.config.resolve(),
                args.runs_root.resolve(),
                args.processed_root.resolve(),
                args.run_id,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
