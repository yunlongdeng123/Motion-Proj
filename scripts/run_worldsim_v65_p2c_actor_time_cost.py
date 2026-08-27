"""Run the preregistered continuous Actor×time proximity-cost probe."""

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

from motion_proj.worldsim_v65.actor_time_outcome import (
    continuous_cost_metrics,
    fit_actor_cost,
    score_actor_outcome,
    selected_cost_metrics,
)
from scripts.run_worldsim_v65_p2r_actor_time import STATIC_FEATURE_NAMES, TIME_FEATURE_NAMES, _materialize


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _comparison(snapshot_metrics: dict, actor_time_metrics: dict, snapshot_selected: dict, actor_time_selected: dict) -> dict[str, object]:
    snapshot_cost = float(snapshot_selected["selected_mean_cost"])
    actor_time_cost = float(actor_time_selected["selected_mean_cost"])
    snapshot_scenes = {row["scene_index"]: row for row in snapshot_selected["scene_rows"]}
    actor_time_scenes = {row["scene_index"]: row for row in actor_time_selected["scene_rows"]}
    deltas = {scene: actor_time_scenes[scene]["selected_mean_cost"] - row["selected_mean_cost"] for scene, row in snapshot_scenes.items()}
    return {
        "spearman_gain": float(actor_time_metrics["spearman"] - snapshot_metrics["spearman"]),
        "relative_mse_reduction": float((snapshot_metrics["mse"] - actor_time_metrics["mse"]) / snapshot_metrics["mse"] if snapshot_metrics["mse"] > 0 else 0.0),
        "relative_selected_cost_reduction": float((snapshot_cost - actor_time_cost) / snapshot_cost if snapshot_cost > 0 else 0.0),
        "scene_lower_count": sum(value < 0 for value in deltas.values()),
        "scene_equal_count": sum(value == 0 for value in deltas.values()),
        "scene_higher_count": sum(value > 0 for value in deltas.values()),
        "scene_deltas": deltas,
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
    train, evaluate = arrays["is_train"], ~arrays["is_train"]
    snapshot_train = arrays["static"][train]
    actor_time_train = np.concatenate((arrays["static"][train], arrays["time"][train]), axis=1)
    hidden = tuple(int(value) for value in config["model"]["hidden_dimensions"])
    kwargs = {
        "hidden_dimensions": hidden,
        "epochs": int(config["model"]["epochs"]),
        "batch_size": int(config["model"]["batch_size"]),
        "learning_rate": float(config["model"]["learning_rate"]),
        "weight_decay": float(config["model"]["weight_decay"]),
        "seed": int(config["seed"]),
    }
    snapshot_fit = fit_actor_cost(snapshot_train, arrays["target_cost"][train], **kwargs)
    actor_time_fit = fit_actor_cost(actor_time_train, arrays["target_cost"][train], **kwargs)
    snapshot_eval = arrays["static"][evaluate]
    actor_time_eval = np.concatenate((arrays["static"][evaluate], arrays["time"][evaluate]), axis=1)
    snapshot_prediction = score_actor_outcome(snapshot_fit, snapshot_eval)
    actor_time_prediction = score_actor_outcome(actor_time_fit, actor_time_eval)
    shuffled_time = arrays["time"][evaluate].copy()
    rng = np.random.default_rng(int(config["seed"]) + 651)
    eval_scenes = arrays["scene_index"][evaluate]
    for scene in np.unique(eval_scenes):
        mask = eval_scenes == scene
        shuffled_time[mask] = shuffled_time[mask][rng.permutation(np.count_nonzero(mask))]
    shuffled_prediction = score_actor_outcome(actor_time_fit, np.concatenate((snapshot_eval, shuffled_time), axis=1))
    costs = arrays["target_cost"][evaluate]
    metrics = {
        "snapshot": continuous_cost_metrics(costs, snapshot_prediction),
        "actor_time": continuous_cost_metrics(costs, actor_time_prediction),
        "actor_time_shuffled": continuous_cost_metrics(costs, shuffled_prediction),
    }
    coverage = float(config["evaluation"]["matched_safe_coverage"])
    selected = {
        "snapshot": selected_cost_metrics(costs, snapshot_prediction, eval_scenes, coverage=coverage),
        "actor_time": selected_cost_metrics(costs, actor_time_prediction, eval_scenes, coverage=coverage),
    }
    comparison = _comparison(metrics["snapshot"], metrics["actor_time"], selected["snapshot"], selected["actor_time"])
    comparison["actor_time_minus_shuffled_spearman"] = float(metrics["actor_time"]["spearman"] - metrics["actor_time_shuffled"]["spearman"])
    gates = {
        "minimum_spearman_gain": comparison["spearman_gain"] >= float(config["gates"]["minimum_spearman_gain"]),
        "minimum_mse_reduction": comparison["relative_mse_reduction"] >= float(config["gates"]["minimum_mse_reduction"]),
        "minimum_selected_cost_reduction": comparison["relative_selected_cost_reduction"] >= float(config["gates"]["minimum_selected_cost_reduction"]),
        "minimum_eval_scene_support": comparison["scene_lower_count"] >= int(config["gates"]["minimum_eval_scene_support"]),
        "temporal_shuffle_response": comparison["actor_time_minus_shuffled_spearman"] > 0.0,
    }
    verdict = "positive_train_only_continuous_actor_time_cost" if all(gates.values()) else "no_clear_train_only_continuous_actor_time_cost"
    torch.save({
        "snapshot_state_dict": {name: value.detach().cpu() for name, value in snapshot_fit.model.state_dict().items()},
        "actor_time_state_dict": {name: value.detach().cpu() for name, value in actor_time_fit.model.state_dict().items()},
        "snapshot_mean": snapshot_fit.mean, "snapshot_scale": snapshot_fit.scale,
        "actor_time_mean": actor_time_fit.mean, "actor_time_scale": actor_time_fit.scale,
        "static_feature_names": STATIC_FEATURE_NAMES, "time_feature_names": TIME_FEATURE_NAMES,
    }, run_dir / "actor_time_cost_models.pt")
    summary = {
        "schema_version": "worldsim_v65.p2c_actor_time_cost_summary.v1", "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "claim_boundary": config["claim_boundary"], "cache_reused": cache_reused,
        "train_token_count": int(np.count_nonzero(train)), "evaluation_token_count": int(np.count_nonzero(evaluate)),
        "target_cost_train_min_mean_max": [float(arrays["target_cost"][train].min()), float(arrays["target_cost"][train].mean()), float(arrays["target_cost"][train].max())],
        "target_cost_eval_min_mean_max": [float(costs.min()), float(costs.mean()), float(costs.max())],
        "metrics": metrics, "selected_cost": selected, "comparison": comparison, "gate_results": gates,
        "formal_v65_selection_read": False,
        "resources": {"gpu": torch.cuda.get_device_name(0), "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024**3), "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2), "wall_seconds": time.monotonic() - started},
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "status.json", {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    return {"run_dir": str(run_dir), "verdict": verdict, "gate_results": gates, "metrics": metrics, "comparison": comparison, "resources": summary["resources"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.runs_root.resolve(), args.run_id), indent=2))


if __name__ == "__main__":
    main()
