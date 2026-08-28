"""Train a group-balanced ListNet Actor selector while fresh inputs stream in."""

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
from scripts.run_worldsim_v67_p76_group_rank_actor_selector import (
    _combine,
    _group_percentile_target,
    _load_npz,
    _predict,
    _select,
)


def _group_ids(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, int]:
    keys = list(zip(arrays["scene_index"].tolist(), arrays["horizon_seconds"].tolist()))
    index = {key: value for value, key in enumerate(sorted(set(keys)))}
    return np.asarray([index[key] for key in keys], dtype=np.int64), len(index)


def _group_log_softmax(logits: torch.Tensor, groups: torch.Tensor, count: int) -> torch.Tensor:
    maxima = torch.full((count,), -torch.inf, dtype=logits.dtype, device=logits.device)
    maxima.scatter_reduce_(0, groups, logits.detach(), reduce="amax", include_self=True)
    shifted = logits - maxima[groups]
    normalizer = torch.zeros(count, dtype=logits.dtype, device=logits.device)
    normalizer.scatter_add_(0, groups, shifted.exp())
    return shifted - normalizer[groups].log()


def _train(
    query_model: ReliabilityMLP,
    actor_model: ReliabilityMLP,
    arrays: dict[str, np.ndarray],
    mean: np.ndarray,
    scale: np.ndarray,
    config: dict,
) -> tuple[float, float, int]:
    features = torch.from_numpy(
        (np.asarray(arrays["features"], dtype=np.float32) - mean) / scale
    ).cuda()
    actor_features = features[:, :len(ACTOR_FEATURE_NAMES)]
    group_array, group_count = _group_ids(arrays)
    groups = torch.from_numpy(group_array).cuda()
    temperature = float(config["listnet_temperature"])
    target_logits = -torch.from_numpy(_group_percentile_target(arrays)).cuda() / temperature
    target_probability = _group_log_softmax(target_logits, groups, group_count).exp()
    optimizer = torch.optim.AdamW(
        list(query_model.parameters()) + list(actor_model.parameters()),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    query_loss = actor_loss = torch.tensor(float("nan"), device="cuda")
    for epoch in range(int(config["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        query_log_probability = _group_log_softmax(
            -query_model(features) / temperature, groups, group_count
        )
        actor_log_probability = _group_log_softmax(
            -actor_model(actor_features) / temperature, groups, group_count
        )
        query_loss = -(target_probability * query_log_probability).sum() / group_count
        actor_loss = -(target_probability * actor_log_probability).sum() / group_count
        (query_loss + actor_loss).backward()
        optimizer.step()
        if epoch % 250 == 0 or epoch + 1 == int(config["epochs"]):
            print(
                f"listnet epoch={epoch + 1} query={float(query_loss):.6f} "
                f"actor={float(actor_loss):.6f}",
                flush=True,
            )
    result = float(query_loss.detach().cpu()), float(actor_loss.detach().cpu()), group_count
    del features, actor_features, groups, target_logits, target_probability
    torch.cuda.empty_cache()
    return result


def _load_query_score(path: Path, features: np.ndarray, cost_model: bool) -> np.ndarray:
    artifact = torch.load(path, map_location="cuda")
    model = ReliabilityMLP(len(FEATURE_NAMES), artifact["hidden_dimensions"]).cuda()
    model.load_state_dict(artifact["query_model_state_dict"])
    mean = np.asarray(artifact["feature_mean"], dtype=np.float32)
    scale = np.asarray(artifact["feature_scale"], dtype=np.float32)
    return predict_reliability(model.eval(), features, mean, scale) if cost_model else _predict(
        model.eval(), features, mean, scale
    )


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
    final_query, final_actor, group_count = _train(
        query_model, actor_model, source, mean, scale, config["model"]
    )
    torch.save({
        "feature_names": FEATURE_NAMES,
        "feature_mean": mean,
        "feature_scale": scale,
        "hidden_dimensions": config["model"]["hidden_dimensions"],
        "query_model_state_dict": query_model.state_dict(),
        "actor_only_model_state_dict": actor_model.state_dict(),
    }, run_dir / "LISTNET_ACTOR_SELECTOR.pt")

    p75_dir = args.runs_root / config["evaluation_source"]["run"]
    evaluation_path = p75_dir / config["evaluation_source"]["rows"]
    p75_artifact_path = p75_dir / config["evaluation_source"]["model"]
    p76_artifact_path = args.runs_root / config["p76_baseline"]["run"] / config["p76_baseline"]["artifact"]
    deadline = time.monotonic() + float(config["evaluation_source"]["readiness_timeout_seconds"])
    while not (evaluation_path.is_file() and p75_artifact_path.is_file() and p76_artifact_path.is_file()):
        if time.monotonic() >= deadline:
            raise TimeoutError("P75 rows/model or P76 model not ready")
        print("listnet training done; waiting for P75 fresh validation read", flush=True)
        time.sleep(10.0)

    evaluation = _load_npz(evaluation_path)
    query_score = _predict(query_model.eval(), evaluation["features"], mean, scale)
    actor_score = _predict(actor_model.eval(), evaluation["features"], mean, scale, actor_only=True)
    p75_score = _load_query_score(p75_artifact_path, evaluation["features"], cost_model=True)
    p76_score = _load_query_score(p76_artifact_path, evaluation["features"], cost_model=False)
    p73_score = _load_query_score(
        args.runs_root / config["p73_baseline"]["run"] / config["p73_baseline"]["artifact"],
        evaluation["features"], cost_model=True,
    )

    costs = np.asarray(evaluation["target_cost"], dtype=np.float64)
    scenes = np.asarray(evaluation["scene_index"])
    fraction = float(config["selection"]["coverage_fraction"])
    selected = {
        "query": _select(query_score, scenes, fraction),
        "actor_only": _select(actor_score, scenes, fraction),
        "p75": _select(p75_score, scenes, fraction),
        "p76": _select(p76_score, scenes, fraction),
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
        **{f"{name}_selected_mean_cost": value for name, value in selected_costs.items()},
        "query_cost_reduction": (all_cost - selected_costs["query"]) / max(all_cost, 1e-12),
        "query_cost_reduction_over_actor_only": (selected_costs["actor_only"] - selected_costs["query"]) / max(selected_costs["actor_only"], 1e-12),
        "query_cost_reduction_over_p75": (selected_costs["p75"] - selected_costs["query"]) / max(selected_costs["p75"], 1e-12),
        "query_cost_reduction_over_p76": (selected_costs["p76"] - selected_costs["query"]) / max(selected_costs["p76"], 1e-12),
        "query_selected_unreliable_prevalence": float(unreliable[selected["query"]].mean()),
    }
    gates = {
        "minimum_cost_reduction_over_actor_only": metrics["query_cost_reduction_over_actor_only"] >= float(config["gates"]["minimum_cost_reduction_over_actor_only"]),
        "minimum_cost_reduction_over_p75": metrics["query_cost_reduction_over_p75"] >= float(config["gates"]["minimum_cost_reduction_over_p75"]),
        "minimum_cost_reduction_over_p76": metrics["query_cost_reduction_over_p76"] >= float(config["gates"]["minimum_cost_reduction_over_p76"]),
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
            "group_count": group_count,
            "final_query_listnet_loss": final_query,
            "final_actor_only_listnet_loss": final_actor,
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
