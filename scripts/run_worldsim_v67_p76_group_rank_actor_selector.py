"""Train dense group-rank Actor reliability while P75 validation inputs stream in."""

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
    ACTOR_FEATURE_NAMES,
    FEATURE_NAMES,
    ReliabilityMLP,
    binary_auroc,
    predict_reliability,
    spearman_correlation,
)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    return dict(np.load(path, allow_pickle=False))


def _combine(arrays: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([part[key] for part in arrays], axis=0) for key in arrays[0]}


def _group_percentile_target(arrays: dict[str, np.ndarray]) -> np.ndarray:
    costs = np.asarray(arrays["target_cost"], dtype=np.float64)
    scenes = np.asarray(arrays["scene_index"])
    horizons = np.asarray(arrays["horizon_seconds"])
    target = np.empty(len(costs), dtype=np.float32)
    for scene, horizon in sorted(set(zip(scenes.tolist(), horizons.tolist()))):
        members = np.flatnonzero((scenes == scene) & (horizons == horizon))
        order = members[np.argsort(costs[members], kind="mergesort")]
        target[order] = np.linspace(0.0, 1.0, len(order), dtype=np.float32)
    return target


def _train(
    query_model: ReliabilityMLP,
    actor_model: ReliabilityMLP,
    arrays: dict[str, np.ndarray],
    mean: np.ndarray,
    scale: np.ndarray,
    config: dict,
) -> tuple[float, float]:
    features = torch.from_numpy(
        (np.asarray(arrays["features"], dtype=np.float32) - mean) / scale
    ).cuda()
    actor_features = features[:, :len(ACTOR_FEATURE_NAMES)]
    target = torch.from_numpy(_group_percentile_target(arrays)).cuda()
    optimizer = torch.optim.AdamW(
        list(query_model.parameters()) + list(actor_model.parameters()),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    query_loss = actor_loss = torch.tensor(float("nan"), device="cuda")
    for epoch in range(int(config["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        query_loss = torch.nn.functional.smooth_l1_loss(
            query_model(features), target, beta=float(config["huber_beta"])
        )
        actor_loss = torch.nn.functional.smooth_l1_loss(
            actor_model(actor_features), target, beta=float(config["huber_beta"])
        )
        (query_loss + actor_loss).backward()
        optimizer.step()
        if epoch % 250 == 0 or epoch + 1 == int(config["epochs"]):
            print(
                f"group-rank epoch={epoch + 1} query={float(query_loss):.6f} "
                f"actor={float(actor_loss):.6f}",
                flush=True,
            )
    result = float(query_loss.detach().cpu()), float(actor_loss.detach().cpu())
    del features, actor_features, target
    torch.cuda.empty_cache()
    return result


def _predict(model: ReliabilityMLP, features: np.ndarray, mean: np.ndarray, scale: np.ndarray, actor_only: bool = False) -> np.ndarray:
    normalized = (np.asarray(features, dtype=np.float32) - mean) / scale
    if actor_only:
        normalized = normalized[:, :len(ACTOR_FEATURE_NAMES)]
    with torch.no_grad():
        return model(torch.from_numpy(normalized).cuda()).cpu().numpy()


def _select(score: np.ndarray, scenes: np.ndarray, fraction: float) -> np.ndarray:
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
    (run_dir / "status.json").write_text(json.dumps({"status": "running"}, indent=2) + "\n", encoding="utf-8")
    started = time.monotonic()
    torch.manual_seed(int(config["seed"]))

    source = _combine([
        _load_npz(args.runs_root / row["run"] / row["artifact"])
        for row in config["source_rows"]
    ])
    raw_features = np.asarray(source["features"], dtype=np.float32)
    mean = raw_features.mean(axis=0)
    scale = raw_features.std(axis=0).clip(min=1e-4)
    query_model = ReliabilityMLP(len(FEATURE_NAMES), config["model"]["hidden_dimensions"]).cuda()
    actor_model = ReliabilityMLP(len(ACTOR_FEATURE_NAMES), config["model"]["hidden_dimensions"]).cuda()
    final_query, final_actor = _train(
        query_model, actor_model, source, mean, scale, config["model"]
    )
    torch.save({
        "feature_names": FEATURE_NAMES,
        "feature_mean": mean,
        "feature_scale": scale,
        "hidden_dimensions": config["model"]["hidden_dimensions"],
        "query_model_state_dict": query_model.state_dict(),
        "actor_only_model_state_dict": actor_model.state_dict(),
    }, run_dir / "GROUP_RANK_ACTOR_SELECTOR.pt")

    p75_dir = args.runs_root / config["evaluation_source"]["run"]
    evaluation_path = p75_dir / config["evaluation_source"]["rows"]
    p75_artifact_path = p75_dir / config["evaluation_source"]["model"]
    deadline = time.monotonic() + float(config["evaluation_source"]["readiness_timeout_seconds"])
    while not (evaluation_path.is_file() and p75_artifact_path.is_file()):
        if time.monotonic() >= deadline:
            raise TimeoutError("P75 fresh validation rows/model not ready")
        print("group-rank training done; waiting for P75 fresh validation read", flush=True)
        time.sleep(10.0)

    evaluation = _load_npz(evaluation_path)
    query_score = _predict(query_model.eval(), evaluation["features"], mean, scale)
    actor_score = _predict(actor_model.eval(), evaluation["features"], mean, scale, actor_only=True)
    p75_artifact = torch.load(p75_artifact_path, map_location="cuda")
    p75_model = ReliabilityMLP(len(FEATURE_NAMES), p75_artifact["hidden_dimensions"]).cuda()
    p75_model.load_state_dict(p75_artifact["query_model_state_dict"])
    p75_score = predict_reliability(
        p75_model.eval(), evaluation["features"],
        np.asarray(p75_artifact["feature_mean"], dtype=np.float32),
        np.asarray(p75_artifact["feature_scale"], dtype=np.float32),
    )
    p73_artifact = torch.load(
        args.runs_root / config["p73_baseline"]["run"] / config["p73_baseline"]["artifact"],
        map_location="cuda",
    )
    p73_model = ReliabilityMLP(len(FEATURE_NAMES), p73_artifact["hidden_dimensions"]).cuda()
    p73_model.load_state_dict(p73_artifact["query_model_state_dict"])
    p73_score = predict_reliability(
        p73_model.eval(), evaluation["features"],
        np.asarray(p73_artifact["feature_mean"], dtype=np.float32),
        np.asarray(p73_artifact["feature_scale"], dtype=np.float32),
    )

    costs = np.asarray(evaluation["target_cost"], dtype=np.float64)
    scenes = np.asarray(evaluation["scene_index"])
    fraction = float(config["selection"]["coverage_fraction"])
    selected = {
        "query": _select(query_score, scenes, fraction),
        "actor_only": _select(actor_score, scenes, fraction),
        "p75": _select(p75_score, scenes, fraction),
        "p73": _select(p73_score, scenes, fraction),
    }
    selected_costs = {name: float(costs[index].mean()) for name, index in selected.items()}
    all_cost = float(costs.mean())
    unreliable = (
        np.asarray(evaluation["raw_actor_state_error_m"]) > float(config["evaluation"]["unreliable_actor_state_error_m"])
    ) & (
        np.asarray(evaluation["predicted_minimum_separation_m"]) <= float(config["evaluation"]["unreliable_exposure_radius_m"])
    )
    metrics = {
        "row_count": int(len(costs)),
        "scene_count": int(len(np.unique(scenes))),
        "query_spearman": spearman_correlation(query_score, costs),
        "actor_only_spearman": spearman_correlation(actor_score, costs),
        "query_unreliable_auroc": binary_auroc(unreliable, query_score),
        "actor_only_unreliable_auroc": binary_auroc(unreliable, actor_score),
        "all_mean_cost": all_cost,
        "query_selected_mean_cost": selected_costs["query"],
        "actor_only_selected_mean_cost": selected_costs["actor_only"],
        "p75_selected_mean_cost": selected_costs["p75"],
        "p73_selected_mean_cost": selected_costs["p73"],
        "query_cost_reduction": (all_cost - selected_costs["query"]) / max(all_cost, 1e-12),
        "query_cost_reduction_over_actor_only": (selected_costs["actor_only"] - selected_costs["query"]) / max(selected_costs["actor_only"], 1e-12),
        "query_cost_reduction_over_p75": (selected_costs["p75"] - selected_costs["query"]) / max(selected_costs["p75"], 1e-12),
        "query_selected_unreliable_prevalence": float(unreliable[selected["query"]].mean()),
    }
    gates = {
        "minimum_cost_reduction_over_actor_only": metrics["query_cost_reduction_over_actor_only"] >= float(config["gates"]["minimum_cost_reduction_over_actor_only"]),
        "minimum_cost_reduction_over_p75": metrics["query_cost_reduction_over_p75"] >= float(config["gates"]["minimum_cost_reduction_over_p75"]),
        "minimum_absolute_cost_reduction": metrics["query_cost_reduction"] >= float(config["gates"]["minimum_absolute_cost_reduction"]),
    }
    verdict = config["verdict_on_pass"] if all(gates.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"],
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "training": {
            "row_count": int(len(source["target_cost"])),
            "group_count": int(len(set(zip(source["scene_index"].tolist(), source["horizon_seconds"].tolist())))),
            "final_query_rank_loss": final_query,
            "final_actor_only_rank_loss": final_actor,
            "frozen_before_p75_validation_rows_became_available": True,
        },
        "fresh_validation_metrics": metrics,
        "gate_results": gates,
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
