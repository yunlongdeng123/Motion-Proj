"""Train a horizon-FiLM rank selector while fresh validation data streams."""

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
    _combine, _group_percentile_target, _load_npz, _select,
)
from scripts.run_worldsim_v67_p77_listnet_actor_selector import _load_query_score


def _augment(features: np.ndarray, horizons: np.ndarray, center: float, scale: float) -> np.ndarray:
    raw = np.asarray(features, dtype=np.float32)
    modulation = ((np.asarray(horizons, dtype=np.float32) - center) / scale)[:, None]
    return np.concatenate((raw, raw * modulation), axis=1)


def _predict(model, features, mean, scale):
    normalized = (np.asarray(features, dtype=np.float32) - mean) / scale
    with torch.no_grad():
        return model(torch.from_numpy(normalized).cuda()).cpu().numpy()


def _train(query_model, actor_model, query_features, actor_features, target, config):
    query_tensor = torch.from_numpy(query_features).cuda()
    actor_tensor = torch.from_numpy(actor_features).cuda()
    target_tensor = torch.from_numpy(target).cuda()
    optimizer = torch.optim.AdamW(
        list(query_model.parameters()) + list(actor_model.parameters()),
        lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]),
    )
    query_loss = actor_loss = torch.tensor(float("nan"), device="cuda")
    for epoch in range(int(config["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        query_loss = torch.nn.functional.smooth_l1_loss(
            query_model(query_tensor), target_tensor, beta=float(config["huber_beta"])
        )
        actor_loss = torch.nn.functional.smooth_l1_loss(
            actor_model(actor_tensor), target_tensor, beta=float(config["huber_beta"])
        )
        (query_loss + actor_loss).backward()
        optimizer.step()
        if epoch % 250 == 0 or epoch + 1 == int(config["epochs"]):
            print(
                f"horizon-film epoch={epoch + 1} query={float(query_loss):.6f} "
                f"actor={float(actor_loss):.6f}", flush=True,
            )
    result = float(query_loss.detach().cpu()), float(actor_loss.detach().cpu())
    del query_tensor, actor_tensor, target_tensor
    torch.cuda.empty_cache()
    return result


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
    center, horizon_scale = float(config["model"]["horizon_center_seconds"]), float(config["model"]["horizon_scale_seconds"])
    query_raw = _augment(source["features"], source["horizon_seconds"], center, horizon_scale)
    actor_base = np.asarray(source["features"], dtype=np.float32)[:, :len(ACTOR_FEATURE_NAMES)]
    actor_raw = _augment(actor_base, source["horizon_seconds"], center, horizon_scale)
    query_mean, query_scale = query_raw.mean(axis=0), query_raw.std(axis=0).clip(min=1e-4)
    actor_mean, actor_scale = actor_raw.mean(axis=0), actor_raw.std(axis=0).clip(min=1e-4)
    query_features = (query_raw - query_mean) / query_scale
    actor_features = (actor_raw - actor_mean) / actor_scale
    target = _group_percentile_target(source)
    query_model = ReliabilityMLP(query_features.shape[1], config["model"]["hidden_dimensions"]).cuda()
    actor_model = ReliabilityMLP(actor_features.shape[1], config["model"]["hidden_dimensions"]).cuda()
    final_query, final_actor = _train(
        query_model, actor_model, query_features, actor_features, target, config["model"]
    )
    torch.save({
        "base_feature_names": FEATURE_NAMES, "query_feature_mean": query_mean,
        "query_feature_scale": query_scale, "actor_feature_mean": actor_mean,
        "actor_feature_scale": actor_scale, "hidden_dimensions": config["model"]["hidden_dimensions"],
        "horizon_center_seconds": center, "horizon_scale_seconds": horizon_scale,
        "query_model_state_dict": query_model.state_dict(),
        "actor_only_model_state_dict": actor_model.state_dict(),
    }, run_dir / "HORIZON_FILM_ACTOR_SELECTOR.pt")

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
        print("horizon-film training done; waiting for P75 fresh validation read", flush=True)
        time.sleep(10.0)

    evaluation = _load_npz(evaluation_path)
    eval_query = _augment(evaluation["features"], evaluation["horizon_seconds"], center, horizon_scale)
    eval_actor_base = np.asarray(evaluation["features"], dtype=np.float32)[:, :len(ACTOR_FEATURE_NAMES)]
    eval_actor = _augment(eval_actor_base, evaluation["horizon_seconds"], center, horizon_scale)
    query_score = _predict(query_model.eval(), eval_query, query_mean, query_scale)
    actor_score = _predict(actor_model.eval(), eval_actor, actor_mean, actor_scale)
    baseline_scores = {
        name: _load_query_score(path, evaluation["features"], cost_model=name in {"p75", "p73"})
        for name, path in baseline_paths.items()
    }
    costs, scenes = np.asarray(evaluation["target_cost"], dtype=np.float64), np.asarray(evaluation["scene_index"])
    fraction = float(config["selection"]["coverage_fraction"])
    selected = {
        "query": _select(query_score, scenes, fraction),
        "actor_only": _select(actor_score, scenes, fraction),
        **{name: _select(score, scenes, fraction) for name, score in baseline_scores.items()},
    }
    selected_costs = {name: float(costs[index].mean()) for name, index in selected.items()}
    best_blind_cost = min(selected_costs[name] for name in ("p76", "p77", "p78", "p79"))
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
        "best_blind_selector_mean_cost": best_blind_cost,
        "query_cost_reduction": (all_cost - selected_costs["query"]) / max(all_cost, 1e-12),
        "query_cost_reduction_over_actor_only": (selected_costs["actor_only"] - selected_costs["query"]) / max(selected_costs["actor_only"], 1e-12),
        "query_cost_reduction_over_p75": (selected_costs["p75"] - selected_costs["query"]) / max(selected_costs["p75"], 1e-12),
        "query_cost_reduction_over_best_blind": (best_blind_cost - selected_costs["query"]) / max(best_blind_cost, 1e-12),
        "query_selected_unreliable_prevalence": float(unreliable[selected["query"]].mean()),
    }
    gates = {
        "minimum_cost_reduction_over_actor_only": metrics["query_cost_reduction_over_actor_only"] >= float(config["gates"]["minimum_cost_reduction_over_actor_only"]),
        "minimum_cost_reduction_over_p75": metrics["query_cost_reduction_over_p75"] >= float(config["gates"]["minimum_cost_reduction_over_p75"]),
        "minimum_cost_reduction_over_best_blind": metrics["query_cost_reduction_over_best_blind"] >= float(config["gates"]["minimum_cost_reduction_over_best_blind"]),
        "minimum_absolute_cost_reduction": metrics["query_cost_reduction"] >= float(config["gates"]["minimum_absolute_cost_reduction"]),
    }
    verdict = config["verdict_on_pass"] if all(gates.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "training": {
            "row_count": int(len(source["target_cost"])),
            "query_augmented_feature_count": int(query_features.shape[1]),
            "actor_augmented_feature_count": int(actor_features.shape[1]),
            "final_query_rank_loss": final_query, "final_actor_only_rank_loss": final_actor,
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
