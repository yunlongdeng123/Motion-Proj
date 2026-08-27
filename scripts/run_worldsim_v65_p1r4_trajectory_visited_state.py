"""Run the V6.5 train-only trajectory-level visited-state reliability probe."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from motion_proj.worldsim_v65.actor_time_outcome import fit_actor_cost, score_actor_outcome
from motion_proj.worldsim_v65.map_context import MAP_CONTEXT_FEATURE_NAMES


FEATURE_NAMES = (
    "q0_route_mean",
    "q0_route_std",
    "q0_route_max",
    "q0_route_q50",
    "q0_route_q90",
    "q0_route_q95",
    "q0_route_entropy_mean",
    "log_visited_point_count",
    "visited_point_fraction",
    "q0_global_mean",
    "q0_global_q90",
    *(f"route_mean_{name}" for name in MAP_CONTEXT_FEATURE_NAMES),
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_remaining(cache_path: Path) -> dict[str, np.ndarray]:
    names = ("route", "hidden_free", "scene_index", "unit_index", "is_train", "map_context")
    with np.load(cache_path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in names}


def _q0_probabilities(cache_path: Path) -> np.ndarray:
    with np.load(cache_path, allow_pickle=False) as source:
        logits = np.asarray(source["base_logit"])
    device = torch.device("cuda")
    outputs = []
    with torch.inference_mode():
        for offset in range(0, logits.shape[0], 262144):
            values = torch.from_numpy(logits[offset : offset + 262144]).to(device)
            outputs.append(torch.sigmoid(values).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def _aggregate_units(
    arrays: dict[str, np.ndarray],
    q0: np.ndarray,
    *,
    minimum_visited_points: int,
) -> dict[str, np.ndarray]:
    rows: dict[str, list] = {
        name: []
        for name in (
            "features",
            "target_cost",
            "unsafe",
            "visited_count",
            "hidden_free_count",
            "scene_index",
            "unit_index",
            "is_train",
        )
    }
    scene_index = arrays["scene_index"]
    unit_index = arrays["unit_index"]
    route = arrays["route"]
    labels = arrays["hidden_free"]
    context = arrays["map_context"]
    for scene in np.unique(scene_index):
        for unit in np.unique(unit_index[scene_index == scene]):
            members = (scene_index == scene) & (unit_index == unit)
            visited = members & route
            visited_count = int(np.count_nonzero(visited))
            if visited_count < minimum_visited_points:
                continue
            route_q0 = q0[visited]
            global_q0 = q0[members]
            clipped = np.clip(route_q0, 1e-6, 1.0 - 1e-6)
            entropy = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))
            features = np.concatenate(
                (
                    np.asarray(
                        [
                            route_q0.mean(),
                            route_q0.std(),
                            route_q0.max(),
                            *np.quantile(route_q0, (0.50, 0.90, 0.95)),
                            entropy.mean(),
                            math.log1p(visited_count),
                            visited_count / int(np.count_nonzero(members)),
                            global_q0.mean(),
                            np.quantile(global_q0, 0.90),
                        ],
                        dtype=np.float32,
                    ),
                    context[visited].mean(axis=0, dtype=np.float64).astype(np.float32),
                )
            )
            hidden_free_count = int(np.count_nonzero(labels[visited]))
            roles = np.unique(arrays["is_train"][members])
            if roles.shape[0] != 1:
                raise RuntimeError("mixed train/evaluation role inside one unit")
            rows["features"].append(features)
            rows["target_cost"].append(hidden_free_count / visited_count)
            rows["unsafe"].append(hidden_free_count > 0)
            rows["visited_count"].append(visited_count)
            rows["hidden_free_count"].append(hidden_free_count)
            rows["scene_index"].append(int(scene))
            rows["unit_index"].append(int(unit))
            rows["is_train"].append(bool(roles[0]))
    if not rows["features"]:
        raise RuntimeError("no eligible trajectory-level units")
    payload = {
        "features": np.stack(rows["features"]).astype(np.float32),
        "target_cost": np.asarray(rows["target_cost"], dtype=np.float32),
        "unsafe": np.asarray(rows["unsafe"], dtype=bool),
        "visited_count": np.asarray(rows["visited_count"], dtype=np.int32),
        "hidden_free_count": np.asarray(rows["hidden_free_count"], dtype=np.int32),
        "scene_index": np.asarray(rows["scene_index"], dtype=np.uint8),
        "unit_index": np.asarray(rows["unit_index"], dtype=np.uint8),
        "is_train": np.asarray(rows["is_train"], dtype=bool),
    }
    if payload["features"].shape[1] != len(FEATURE_NAMES):
        raise RuntimeError("trajectory-level feature dimension mismatch")
    return payload


def _continuous_metrics(costs: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    correlation = float(spearmanr(costs, predictions).statistic)
    residual = predictions - costs
    return {
        "spearman": correlation,
        "mse": float(np.mean(np.square(residual))),
        "mae": float(np.mean(np.abs(residual))),
    }


def _unsafe_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, predictions)) if np.unique(labels).size == 2 else float("nan"),
        "auprc": float(average_precision_score(labels, predictions)),
    }


def _selected_cost(
    costs: np.ndarray,
    predictions: np.ndarray,
    scene_index: np.ndarray,
    *,
    coverage: float,
) -> dict[str, object]:
    count = max(1, int(math.floor(float(coverage) * predictions.shape[0])))
    selected = np.argsort(predictions, kind="stable")[:count]
    scene_rows = []
    for scene in np.unique(scene_index):
        members = np.flatnonzero(scene_index == scene)
        local_count = max(1, int(math.floor(float(coverage) * members.shape[0])))
        local = members[np.argsort(predictions[members], kind="stable")[:local_count]]
        scene_rows.append(
            {
                "scene_index": int(scene),
                "eligible_count": int(members.shape[0]),
                "selected_count": int(local_count),
                "selected_mean_cost": float(costs[local].mean()),
                "selected_unsafe_count": int(np.count_nonzero(costs[local] > 0.0)),
            }
        )
    all_mean = float(costs.mean())
    selected_mean = float(costs[selected].mean())
    return {
        "eligible_count": int(costs.shape[0]),
        "selected_count": int(count),
        "realized_coverage": float(count / costs.shape[0]),
        "all_mean_cost": all_mean,
        "selected_mean_cost": selected_mean,
        "relative_cost_reduction_vs_all": float(
            (all_mean - selected_mean) / all_mean if all_mean > 0 else 0.0
        ),
        "selected_unsafe_count": int(np.count_nonzero(costs[selected] > 0.0)),
        "scene_rows": scene_rows,
    }


def _compare_selected(baseline: dict, candidate: dict) -> dict[str, object]:
    baseline_rows = {row["scene_index"]: row for row in baseline["scene_rows"]}
    candidate_rows = {row["scene_index"]: row for row in candidate["scene_rows"]}
    deltas = {
        scene: candidate_rows[scene]["selected_mean_cost"] - row["selected_mean_cost"]
        for scene, row in baseline_rows.items()
    }
    baseline_cost = float(baseline["selected_mean_cost"])
    candidate_cost = float(candidate["selected_mean_cost"])
    return {
        "relative_selected_cost_reduction_vs_q0": float(
            (baseline_cost - candidate_cost) / baseline_cost if baseline_cost > 0 else 0.0
        ),
        "scene_lower_count": sum(value < 0 for value in deltas.values()),
        "scene_equal_count": sum(value == 0 for value in deltas.values()),
        "scene_higher_count": sum(value > 0 for value in deltas.values()),
        "scene_deltas": deltas,
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        remaining_future = executor.submit(_load_remaining, cache_path)
        q0 = _q0_probabilities(cache_path)
        arrays = remaining_future.result()
    print("loaded q0 on GPU while remaining compact-cache arrays were prefetched", flush=True)
    units = _aggregate_units(
        arrays,
        q0,
        minimum_visited_points=int(config["trajectory_contract"]["minimum_visited_points_per_unit"]),
    )
    train = units["is_train"]
    evaluate = ~train
    hidden = tuple(int(value) for value in config["model"]["hidden_dimensions"])
    fit = fit_actor_cost(
        units["features"][train],
        units["target_cost"][train],
        hidden_dimensions=hidden,
        epochs=int(config["model"]["epochs"]),
        batch_size=int(config["model"]["batch_size"]),
        learning_rate=float(config["model"]["learning_rate"]),
        weight_decay=float(config["model"]["weight_decay"]),
        seed=int(config["seed"]),
    )

    target = units["target_cost"][evaluate]
    unsafe = units["unsafe"][evaluate]
    scenes = units["scene_index"][evaluate]
    baseline_scores = units["features"][evaluate, 0]
    candidate_scores = score_actor_outcome(fit, units["features"][evaluate])
    shuffled_features = units["features"][evaluate].copy()
    rng = np.random.default_rng(int(config["seed"]) + 654)
    for scene in np.unique(scenes):
        mask = scenes == scene
        shuffled_features[mask] = shuffled_features[mask][rng.permutation(np.count_nonzero(mask))]
    shuffled_scores = score_actor_outcome(fit, shuffled_features)

    continuous = {
        "q0_aggregate": _continuous_metrics(target, baseline_scores),
        "visited_state_head": _continuous_metrics(target, candidate_scores),
        "shuffled_trajectory": _continuous_metrics(target, shuffled_scores),
    }
    continuous["head_minus_q0_spearman"] = (
        continuous["visited_state_head"]["spearman"] - continuous["q0_aggregate"]["spearman"]
    )
    continuous["head_minus_shuffled_spearman"] = (
        continuous["visited_state_head"]["spearman"] - continuous["shuffled_trajectory"]["spearman"]
    )
    q0_mse = continuous["q0_aggregate"]["mse"]
    head_mse = continuous["visited_state_head"]["mse"]
    continuous["relative_mse_reduction"] = float((q0_mse - head_mse) / q0_mse if q0_mse > 0 else 0.0)
    unsafe_ranking = {
        "q0_aggregate": _unsafe_metrics(unsafe, baseline_scores),
        "visited_state_head": _unsafe_metrics(unsafe, candidate_scores),
        "shuffled_trajectory": _unsafe_metrics(unsafe, shuffled_scores),
    }
    coverage = float(config["evaluation"]["matched_safe_coverage"])
    q0_selected = _selected_cost(target, baseline_scores, scenes, coverage=coverage)
    head_selected = _selected_cost(target, candidate_scores, scenes, coverage=coverage)
    shuffled_selected = _selected_cost(target, shuffled_scores, scenes, coverage=coverage)
    selected_comparison = _compare_selected(q0_selected, head_selected)

    viability_gates = {
        "minimum_q0_aggregate_spearman": continuous["q0_aggregate"]["spearman"] >= float(config["viability_gates"]["minimum_q0_aggregate_spearman"]),
        "minimum_q0_aggregate_unsafe_auroc": unsafe_ranking["q0_aggregate"]["auroc"] >= float(config["viability_gates"]["minimum_q0_aggregate_unsafe_auroc"]),
        "minimum_q0_selected_cost_reduction": q0_selected["relative_cost_reduction_vs_all"] >= float(config["viability_gates"]["minimum_q0_selected_cost_reduction"]),
    }
    incremental_gates = {
        "minimum_spearman_gain": continuous["head_minus_q0_spearman"] >= float(config["incremental_gates"]["minimum_spearman_gain"]),
        "minimum_mse_reduction": continuous["relative_mse_reduction"] >= float(config["incremental_gates"]["minimum_mse_reduction"]),
        "minimum_selected_cost_reduction_vs_q0": selected_comparison["relative_selected_cost_reduction_vs_q0"] >= float(config["incremental_gates"]["minimum_selected_cost_reduction_vs_q0"]),
        "scene_direction_support": selected_comparison["scene_lower_count"] > selected_comparison["scene_higher_count"],
        "trajectory_shuffle_response": continuous["head_minus_shuffled_spearman"] > 0.0,
    }
    if all(incremental_gates.values()):
        verdict = "positive_train_only_visited_state_context_increment"
    elif all(viability_gates.values()):
        verdict = "positive_train_only_visited_state_object_q0_aggregation_only"
    else:
        verdict = "no_clear_train_only_visited_state_reliability"

    torch.save(
        {
            "state_dict": {name: value.detach().cpu() for name, value in fit.model.state_dict().items()},
            "feature_mean": fit.mean,
            "feature_scale": fit.scale,
            "feature_names": FEATURE_NAMES,
            "target": config["target"],
        },
        run_dir / "trajectory_visited_state_head.pt",
    )
    summary = {
        "schema_version": "worldsim_v65.p1r4_trajectory_visited_state_summary.v1",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "claim_boundary": config["claim_boundary"],
        "source_cache": str(cache_path),
        "native_hidden_loaded": False,
        "feature_names": FEATURE_NAMES,
        "train_unit_count": int(np.count_nonzero(train)),
        "evaluation_unit_count": int(np.count_nonzero(evaluate)),
        "excluded_unit_count": int(192 - units["features"].shape[0]),
        "train_evaluation_unsafe_counts": [int(np.count_nonzero(units["unsafe"][train])), int(np.count_nonzero(unsafe))],
        "evaluation_visited_point_count": int(units["visited_count"][evaluate].sum()),
        "evaluation_hidden_free_count": int(units["hidden_free_count"][evaluate].sum()),
        "continuous_metrics": continuous,
        "unsafe_ranking": unsafe_ranking,
        "selected_cost": {
            "q0_aggregate": q0_selected,
            "visited_state_head": head_selected,
            "shuffled_trajectory": shuffled_selected,
            "comparison": selected_comparison,
        },
        "viability_gate_results": viability_gates,
        "incremental_gate_results": incremental_gates,
        "epoch_losses": fit.epoch_losses,
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
    return {
        "run_dir": str(run_dir),
        "verdict": verdict,
        "viability_gate_results": viability_gates,
        "incremental_gate_results": incremental_gates,
        "resources": summary["resources"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.runs_root.resolve(), args.run_id), indent=2))


if __name__ == "__main__":
    main()
