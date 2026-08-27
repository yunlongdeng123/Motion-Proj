"""Audit frozen Actor-route forecasts and a deterministic false-safe monitor."""

from __future__ import annotations

import argparse
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

from motion_proj.worldsim_v65.actor_time_outcome import (
    ActorOutcomeFit,
    ActorOutcomeMLP,
    score_actor_outcome,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _restore_fit(
    state_dict: dict,
    mean: np.ndarray,
    scale: np.ndarray,
    input_dimension: int,
) -> ActorOutcomeFit:
    model = ActorOutcomeMLP(input_dimension, (32, 16))
    model.load_state_dict(state_dict)
    return ActorOutcomeFit(model=model.cuda(), mean=mean, scale=scale, epoch_losses=[])


def _aggregate(
    arrays: dict[str, np.ndarray],
    snapshot_scores: np.ndarray,
    actor_time_scores: np.ndarray,
) -> dict[str, np.ndarray]:
    rows = {name: [] for name in ("snapshot", "actor_time", "target", "gap", "monitor", "scene_index", "unit_index", "is_train", "actor_count")}
    for scene in np.unique(arrays["scene_index"]):
        for unit in np.unique(arrays["unit_index"][arrays["scene_index"] == scene]):
            mask = (arrays["scene_index"] == scene) & (arrays["unit_index"] == unit)
            snapshot = float(snapshot_scores[mask].max())
            actor_time = float(actor_time_scores[mask].max())
            target = float(arrays["target_cost"][mask].max())
            roles = np.unique(arrays["is_train"][mask])
            if roles.shape[0] != 1:
                raise RuntimeError("mixed roles in Actor trajectory unit")
            rows["snapshot"].append(snapshot)
            rows["actor_time"].append(actor_time)
            rows["target"].append(target)
            rows["gap"].append(max(target - snapshot, 0.0))
            rows["monitor"].append(max(actor_time - snapshot, 0.0))
            rows["scene_index"].append(int(scene))
            rows["unit_index"].append(int(unit))
            rows["is_train"].append(bool(roles[0]))
            rows["actor_count"].append(int(np.count_nonzero(mask)))
    return {
        name: np.asarray(values, dtype=(bool if name == "is_train" else np.uint8 if name in ("scene_index", "unit_index") else np.int32 if name == "actor_count" else np.float32))
        for name, values in rows.items()
    }


def _continuous(target: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    residual = scores - target
    return {
        "spearman": float(spearmanr(target, scores).statistic),
        "mse": float(np.mean(np.square(residual))),
        "mae": float(np.mean(np.abs(residual))),
    }


def _ranking(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else float("nan"),
        "auprc": float(average_precision_score(labels, scores)),
    }


def _selected(
    target: np.ndarray,
    scores: np.ndarray,
    scene_index: np.ndarray,
    *,
    coverage: float,
) -> dict[str, object]:
    count = max(1, int(math.floor(float(coverage) * scores.shape[0])))
    selected = np.argsort(scores, kind="stable")[:count]
    all_mean = float(target.mean())
    selected_mean = float(target[selected].mean())
    scene_rows = []
    for scene in np.unique(scene_index):
        members = np.flatnonzero(scene_index == scene)
        local_count = max(1, int(math.floor(float(coverage) * members.shape[0])))
        local = members[np.argsort(scores[members], kind="stable")[:local_count]]
        scene_rows.append({
            "scene_index": int(scene),
            "eligible_count": int(members.shape[0]),
            "selected_count": int(local_count),
            "all_mean": float(target[members].mean()),
            "selected_mean": float(target[local].mean()),
        })
    return {
        "eligible_count": int(scores.shape[0]),
        "selected_count": int(count),
        "all_mean": all_mean,
        "selected_mean": selected_mean,
        "relative_reduction_vs_all": float((all_mean - selected_mean) / all_mean if all_mean > 0 else 0.0),
        "scene_rows": scene_rows,
    }


def run(config_path: Path, runs_root: Path, run_id: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v65" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    _write_json(run_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()

    with np.load(config["inputs"]["compact_cache"], allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    artifact = torch.load(config["inputs"]["frozen_models"], map_location="cpu", weights_only=False)
    snapshot_fit = _restore_fit(
        artifact["snapshot_state_dict"], artifact["snapshot_mean"], artifact["snapshot_scale"], arrays["static"].shape[1]
    )
    full = np.concatenate((arrays["static"], arrays["time"]), axis=1)
    actor_time_fit = _restore_fit(
        artifact["actor_time_state_dict"], artifact["actor_time_mean"], artifact["actor_time_scale"], full.shape[1]
    )
    snapshot_scores = score_actor_outcome(snapshot_fit, arrays["static"])
    actor_time_scores = score_actor_outcome(actor_time_fit, full)
    units = _aggregate(arrays, snapshot_scores, actor_time_scores)
    evaluate = ~units["is_train"]
    target = units["target"][evaluate]
    snapshot = units["snapshot"][evaluate]
    actor_time = units["actor_time"][evaluate]
    gap = units["gap"][evaluate]
    monitor = units["monitor"][evaluate]
    scenes = units["scene_index"][evaluate]
    coverage = float(config["evaluation"]["matched_safe_coverage"])

    forecast = {
        "snapshot": _continuous(target, snapshot),
        "actor_time_descriptive": _continuous(target, actor_time),
        "snapshot_selected_target_cost": _selected(target, snapshot, scenes, coverage=coverage),
    }
    false_safe = {
        "gap_metrics": _continuous(gap, monitor),
        "positive_gap_ranking": _ranking(gap > 0.0, monitor),
        "monitor_selected_gap": _selected(gap, monitor, scenes, coverage=coverage),
        "positive_gap_count": int(np.count_nonzero(gap > 0.0)),
        "zero_monitor_count": int(np.count_nonzero(monitor == 0.0)),
    }
    forecast_gates = {
        "minimum_snapshot_target_spearman": forecast["snapshot"]["spearman"] >= float(config["forecast_viability_gates"]["minimum_snapshot_target_spearman"]),
        "minimum_snapshot_selected_target_cost_reduction": forecast["snapshot_selected_target_cost"]["relative_reduction_vs_all"] >= float(config["forecast_viability_gates"]["minimum_snapshot_selected_target_cost_reduction"]),
    }
    monitor_gates = {
        "minimum_monitor_gap_spearman": false_safe["gap_metrics"]["spearman"] >= float(config["monitor_gates"]["minimum_monitor_gap_spearman"]),
        "minimum_monitor_false_safe_auroc": false_safe["positive_gap_ranking"]["auroc"] >= float(config["monitor_gates"]["minimum_monitor_false_safe_auroc"]),
        "minimum_monitor_selected_gap_reduction": false_safe["monitor_selected_gap"]["relative_reduction_vs_all"] >= float(config["monitor_gates"]["minimum_monitor_selected_gap_reduction"]),
    }
    if all(forecast_gates.values()) and all(monitor_gates.values()):
        verdict = "positive_train_only_actor_forecast_and_false_safe_monitor"
    elif all(forecast_gates.values()):
        verdict = "positive_train_only_actor_forecast_false_safe_monitor_rejected"
    else:
        verdict = "no_clear_train_only_actor_trajectory_forecast"

    summary = {
        "schema_version": "worldsim_v65.p1r5_actor_false_safe_summary.v1",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "claim_boundary": config["claim_boundary"],
        "train_evaluation_trajectory_counts": [int(np.count_nonzero(units["is_train"])), int(np.count_nonzero(evaluate))],
        "evaluation_actor_token_count": int(units["actor_count"][evaluate].sum()),
        "forecast": forecast,
        "false_safe": false_safe,
        "forecast_gate_results": forecast_gates,
        "monitor_gate_results": monitor_gates,
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
    return {"run_dir": str(run_dir), "verdict": verdict, "forecast_gate_results": forecast_gates, "monitor_gate_results": monitor_gates, "resources": summary["resources"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.runs_root.resolve(), args.run_id), indent=2))


if __name__ == "__main__":
    main()

