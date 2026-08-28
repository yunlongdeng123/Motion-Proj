"""Train a fixed-coverage boundary-pair Actor selector during validation IO."""

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
    spearman_correlation,
)
from scripts.run_worldsim_v67_p76_group_rank_actor_selector import (
    _combine, _load_npz, _predict, _select,
)
from scripts.run_worldsim_v67_p77_listnet_actor_selector import _load_query_score


def _boundary_pairs(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    costs = np.asarray(arrays["target_cost"], dtype=np.float64)
    scenes = np.asarray(arrays["scene_index"])
    horizons = np.asarray(arrays["horizon_seconds"])
    low_rows, high_rows, weights = [], [], []
    group_keys = sorted(set(zip(scenes.tolist(), horizons.tolist())))
    for scene, horizon in group_keys:
        members = np.flatnonzero((scenes == scene) & (horizons == horizon))
        order = members[np.argsort(costs[members], kind="mergesort")]
        count = len(order) // 2
        low, high = order[:count], order[-count:]
        gap = costs[high] - costs[low]
        low_rows.append(low)
        high_rows.append(high)
        weights.append(gap / max(float(gap.mean()), 1e-8) / (count * len(group_keys)))
    return (
        np.concatenate(low_rows).astype(np.int64),
        np.concatenate(high_rows).astype(np.int64),
        np.concatenate(weights).astype(np.float32),
        len(group_keys),
    )


def _train(query_model, actor_model, arrays, mean, scale, config):
    features = torch.from_numpy(
        (np.asarray(arrays["features"], dtype=np.float32) - mean) / scale
    ).cuda()
    actor_features = features[:, :len(ACTOR_FEATURE_NAMES)]
    low_array, high_array, weight_array, group_count = _boundary_pairs(arrays)
    low, high = torch.from_numpy(low_array).cuda(), torch.from_numpy(high_array).cuda()
    weights = torch.from_numpy(weight_array).cuda()
    temperature = float(config["pair_temperature"])
    optimizer = torch.optim.AdamW(
        list(query_model.parameters()) + list(actor_model.parameters()),
        lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]),
    )
    query_loss = actor_loss = torch.tensor(float("nan"), device="cuda")
    for epoch in range(int(config["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        query_score, actor_score = query_model(features), actor_model(actor_features)
        query_loss = (
            torch.nn.functional.softplus((query_score[low] - query_score[high]) / temperature)
            * weights
        ).sum()
        actor_loss = (
            torch.nn.functional.softplus((actor_score[low] - actor_score[high]) / temperature)
            * weights
        ).sum()
        (query_loss + actor_loss).backward()
        optimizer.step()
        if epoch % 250 == 0 or epoch + 1 == int(config["epochs"]):
            print(
                f"boundary-pair epoch={epoch + 1} query={float(query_loss):.6f} "
                f"actor={float(actor_loss):.6f}", flush=True,
            )
    result = float(query_loss.detach().cpu()), float(actor_loss.detach().cpu())
    pair_count = int(len(low_array))
    del features, actor_features, low, high, weights
    torch.cuda.empty_cache()
    return result[0], result[1], pair_count, group_count


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
    (run_dir / "status.json").write_text(json.dumps({"status": "running"}, indent=2) + "\n", encoding="utf-8")
    started = time.monotonic()
    torch.manual_seed(int(config["seed"]))

    source = _combine([
        _load_npz(args.runs_root / row["run"] / row["artifact"])
        for row in config["source_rows"]
    ])
    raw_features = np.asarray(source["features"], dtype=np.float32)
    mean, scale = raw_features.mean(axis=0), raw_features.std(axis=0).clip(min=1e-4)
    query_model = ReliabilityMLP(len(FEATURE_NAMES), config["model"]["hidden_dimensions"]).cuda()
    actor_model = ReliabilityMLP(len(ACTOR_FEATURE_NAMES), config["model"]["hidden_dimensions"]).cuda()
    final_query, final_actor, pair_count, group_count = _train(
        query_model, actor_model, source, mean, scale, config["model"]
    )
    torch.save({
        "feature_names": FEATURE_NAMES, "feature_mean": mean, "feature_scale": scale,
        "hidden_dimensions": config["model"]["hidden_dimensions"],
        "query_model_state_dict": query_model.state_dict(),
        "actor_only_model_state_dict": actor_model.state_dict(),
    }, run_dir / "BOUNDARY_PAIR_ACTOR_SELECTOR.pt")

    p75_dir = args.runs_root / config["evaluation_source"]["run"]
    evaluation_path = p75_dir / config["evaluation_source"]["rows"]
    baseline_paths = {
        name: args.runs_root / row["run"] / row["artifact"]
        for name, row in config["baselines"].items()
    }
    deadline = time.monotonic() + float(config["evaluation_source"]["readiness_timeout_seconds"])
    while not (evaluation_path.is_file() and all(path.is_file() for path in baseline_paths.values())):
        if time.monotonic() >= deadline:
            raise TimeoutError("P75 rows or frozen baseline model not ready")
        print("boundary-pair training done; waiting for P75 fresh validation read", flush=True)
        time.sleep(10.0)

    evaluation = _load_npz(evaluation_path)
    query_score = _predict(query_model.eval(), evaluation["features"], mean, scale)
    actor_score = _predict(actor_model.eval(), evaluation["features"], mean, scale, actor_only=True)
    baseline_scores = {
        name: _load_query_score(path, evaluation["features"], cost_model=name in {"p75", "p73"})
        for name, path in baseline_paths.items()
    }
    costs = np.asarray(evaluation["target_cost"], dtype=np.float64)
    scenes = np.asarray(evaluation["scene_index"])
    fraction = float(config["selection"]["coverage_fraction"])
    selected = {
        "query": _select(query_score, scenes, fraction),
        "actor_only": _select(actor_score, scenes, fraction),
        **{name: _select(score, scenes, fraction) for name, score in baseline_scores.items()},
    }
    selected_costs = {name: float(costs[index].mean()) for name, index in selected.items()}
    best_blind_cost = min(selected_costs["p76"], selected_costs["p77"])
    all_cost = float(costs.mean())
    unreliable = (
        np.asarray(evaluation["raw_actor_state_error_m"]) > float(config["evaluation"]["unreliable_actor_state_error_m"])
    ) & (
        np.asarray(evaluation["predicted_minimum_separation_m"]) <= float(config["evaluation"]["unreliable_exposure_radius_m"])
    )
    metrics = {
        "row_count": int(len(costs)), "scene_count": int(len(np.unique(scenes))),
        "query_spearman": spearman_correlation(query_score, costs),
        "actor_only_spearman": spearman_correlation(actor_score, costs),
        "query_unreliable_auroc": binary_auroc(unreliable, query_score),
        "actor_only_unreliable_auroc": binary_auroc(unreliable, actor_score),
        "all_mean_cost": all_cost,
        **{f"{name}_selected_mean_cost": value for name, value in selected_costs.items()},
        "best_blind_rank_selected_mean_cost": best_blind_cost,
        "query_cost_reduction": (all_cost - selected_costs["query"]) / max(all_cost, 1e-12),
        "query_cost_reduction_over_actor_only": (selected_costs["actor_only"] - selected_costs["query"]) / max(selected_costs["actor_only"], 1e-12),
        "query_cost_reduction_over_p75": (selected_costs["p75"] - selected_costs["query"]) / max(selected_costs["p75"], 1e-12),
        "query_cost_reduction_over_best_blind_rank": (best_blind_cost - selected_costs["query"]) / max(best_blind_cost, 1e-12),
        "query_selected_unreliable_prevalence": float(unreliable[selected["query"]].mean()),
    }
    gates = {
        "minimum_cost_reduction_over_actor_only": metrics["query_cost_reduction_over_actor_only"] >= float(config["gates"]["minimum_cost_reduction_over_actor_only"]),
        "minimum_cost_reduction_over_p75": metrics["query_cost_reduction_over_p75"] >= float(config["gates"]["minimum_cost_reduction_over_p75"]),
        "minimum_cost_reduction_over_best_blind_rank": metrics["query_cost_reduction_over_best_blind_rank"] >= float(config["gates"]["minimum_cost_reduction_over_best_blind_rank"]),
        "minimum_absolute_cost_reduction": metrics["query_cost_reduction"] >= float(config["gates"]["minimum_absolute_cost_reduction"]),
    }
    verdict = config["verdict_on_pass"] if all(gates.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "training": {
            "row_count": int(len(source["target_cost"])), "pair_count": pair_count,
            "group_count": group_count, "final_query_boundary_loss": final_query,
            "final_actor_only_boundary_loss": final_actor,
            "frozen_before_p75_validation_rows_became_available": True,
        },
        "fresh_validation_metrics": metrics, "gate_results": gates,
        "resources": {
            "gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "wall_seconds": time.monotonic() - started,
        },
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({
        "status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict, "gate_results": gates}, indent=2), flush=True)


if __name__ == "__main__":
    main()
