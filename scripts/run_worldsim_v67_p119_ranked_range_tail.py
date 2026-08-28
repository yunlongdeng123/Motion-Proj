"""Train a bounded ranked-range residual around the fixed-coverage selection boundary."""

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

from motion_proj.worldsim_v67.actor_state_reliability import binary_auroc
from scripts.run_worldsim_v67_p86_direct_trajectory_reliability import _select_by_scene
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p114_monotone_tail_risk import (
    _crossing_probability, _trajectory_tail_features,
)


class RankedRangeResidual(torch.nn.Module):
    def __init__(self, feature_count: int, hidden_dimension: int, bound: float) -> None:
        super().__init__()
        self.bound = float(bound)
        self.network = torch.nn.Sequential(
            torch.nn.Linear(feature_count, hidden_dimension),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dimension, 1),
        )

    def forward(self, features: torch.Tensor, base_logit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.bound * torch.tanh(self.network(features).reshape(-1))
        return base_logit + residual, residual


def _head_features(grouped: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    probability = np.clip(np.asarray(grouped["features"], dtype=np.float32), 1e-5, 1.0 - 1e-5)
    logits = np.log(probability) - np.log1p(-probability)
    clearance = np.log1p(np.asarray(grouped["clearance_score"], dtype=np.float32))[:, None]
    return np.concatenate((logits, clearance), axis=1), logits[:, 0]


def _within_scene_percentile(base_score: np.ndarray, scenes: np.ndarray) -> np.ndarray:
    percentile = np.zeros(len(base_score), dtype=np.float32)
    for scene in np.unique(scenes):
        members = np.flatnonzero(scenes == scene)
        order = members[np.argsort(base_score[members], kind="stable")]
        percentile[order] = np.arange(len(order), dtype=np.float32) / max(len(order) - 1, 1)
    return percentile


@torch.no_grad()
def _score(
    model: RankedRangeResidual, raw_features: np.ndarray, base_logit: np.ndarray,
    feature_mean: np.ndarray, feature_scale: np.ndarray,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(raw_features), 65536):
        features = torch.from_numpy(
            (raw_features[start:start + 65536] - feature_mean) / feature_scale,
        ).cuda()
        base = torch.from_numpy(base_logit[start:start + 65536]).cuda()
        outputs.append(model(features, base)[0].cpu().numpy())
    return np.concatenate(outputs)


def _evaluate(
    model: RankedRangeResidual, grouped: dict[str, np.ndarray], feature_mean: np.ndarray,
    feature_scale: np.ndarray, coverage: float,
) -> dict[str, float | int]:
    raw_features, base_logit = _head_features(grouped)
    learned = _score(model.eval(), raw_features, base_logit, feature_mean, feature_scale)
    clearance = grouped["clearance_score"]
    scenes, events = grouped["scene_index"], grouped["events"]
    selected = _select_by_scene(learned, scenes, coverage)
    base_selected = _select_by_scene(base_logit, scenes, coverage)
    clearance_selected = _select_by_scene(clearance, scenes, coverage)
    learned_auroc = binary_auroc(events, learned)
    base_auroc = binary_auroc(events, base_logit)
    return {
        "trajectory_count": int(len(events)),
        "all_occupancy_flip_events": int(np.count_nonzero(events)),
        "selected_trajectory_count": int(len(selected)),
        "ranked_range_selected_occupancy_flip_events": int(np.count_nonzero(events[selected])),
        "p109_selected_occupancy_flip_events": int(np.count_nonzero(events[base_selected])),
        "clearance_selected_occupancy_flip_events": int(np.count_nonzero(events[clearance_selected])),
        "ranked_range_event_auroc": learned_auroc,
        "p109_event_auroc": base_auroc,
        "clearance_event_auroc": binary_auroc(events, clearance),
        "ranked_range_auroc_gain_over_p109": learned_auroc - base_auroc,
    }


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
    started = time.monotonic()
    torch.manual_seed(int(config["seed"]))
    checkpoint = torch.load(
        args.runs_root / config["frozen_p109"]["run"] / config["frozen_p109"]["artifact"],
        map_location="cuda",
    )
    actor_model = DirectionalActorGaussian(20, checkpoint["hidden_dimensions"]).cuda()
    actor_model.load_state_dict(checkpoint["model_state_dict"])
    actor_model.eval()
    feature_mean_actor = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_scale_actor = np.asarray(checkpoint["feature_scale"], dtype=np.float32)
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
    target_scale = np.asarray(checkpoint["target_scale"], dtype=np.float32)
    rows_root = args.runs_root / config["rows"]["run"]
    source_raw = dict(np.load(rows_root / config["rows"]["source_artifact"], allow_pickle=False))
    source_probability, _ = _crossing_probability(
        source_raw, actor_model, feature_mean_actor, feature_scale_actor, target_mean, target_scale,
    )
    top_k = int(config["model"]["top_k_crossing_probabilities"])
    source = _trajectory_tail_features(source_raw, source_probability, top_k)
    raw_features, base_logit = _head_features(source)
    feature_mean = raw_features.mean(0)
    feature_scale = raw_features.std(0).clip(min=1e-4)
    features = torch.from_numpy((raw_features - feature_mean) / feature_scale).cuda()
    base = torch.from_numpy(base_logit).cuda()
    events = np.asarray(source["events"], dtype=bool)
    percentile = _within_scene_percentile(base_logit, source["scene_index"])
    band = config["ranked_range"]
    hard_positive = np.flatnonzero(events & (percentile <= float(band["positive_max_percentile"])))
    boundary_negative = np.flatnonzero(
        (~events) & (percentile >= float(band["negative_min_percentile"]))
        & (percentile <= float(band["negative_max_percentile"]))
    )
    positive_gpu = torch.from_numpy(hard_positive).long().cuda()
    negative_gpu = torch.from_numpy(boundary_negative).long().cuda()
    model = RankedRangeResidual(
        raw_features.shape[1], int(config["model"]["hidden_dimension"]),
        float(config["model"]["residual_bound"]),
    ).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["model"]["learning_rate"]),
        weight_decay=float(config["model"]["weight_decay"]),
    )
    batch_size = int(config["model"]["pair_batch_size"])
    regularization = float(config["model"]["residual_regularization"])
    final_loss = 0.0
    torch.cuda.reset_peak_memory_stats()
    for step in range(int(config["model"]["steps"])):
        positive = positive_gpu[torch.randint(len(positive_gpu), (batch_size,), device="cuda")]
        negative = negative_gpu[torch.randint(len(negative_gpu), (batch_size,), device="cuda")]
        positive_score, positive_residual = model(features[positive], base[positive])
        negative_score, negative_residual = model(features[negative], base[negative])
        rank_loss = torch.nn.functional.softplus(-(positive_score - negative_score)).mean()
        penalty = 0.5 * (positive_residual.square().mean() + negative_residual.square().mean())
        loss = rank_loss + regularization * penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step % 250 == 0 or step + 1 == int(config["model"]["steps"]):
            print(f"P119 ranked-range step={step + 1} loss={final_loss:.6f}", flush=True)
    torch.save({
        "top_k_crossing_probabilities": top_k,
        "feature_mean": feature_mean, "feature_scale": feature_scale,
        "hidden_dimension": int(config["model"]["hidden_dimension"]),
        "residual_bound": float(config["model"]["residual_bound"]),
        "model_state_dict": model.state_dict(),
    }, run_dir / config["model_artifact"])
    results = {}
    for cohort in config["development_cohorts"]:
        cohort_root = args.runs_root / cohort.get("run", config["rows"]["run"])
        raw = dict(np.load(cohort_root / cohort["artifact"], allow_pickle=False))
        probability, _ = _crossing_probability(
            raw, actor_model, feature_mean_actor, feature_scale_actor, target_mean, target_scale,
        )
        grouped = _trajectory_tail_features(raw, probability, top_k)
        results[cohort["name"]] = _evaluate(
            model, grouped, feature_mean, feature_scale,
            float(config["selection"]["coverage_fraction"]),
        )
        print(json.dumps({cohort["name"]: results[cohort["name"]]}, indent=2), flush=True)
    decisions = {
        name: int(results[name]["ranked_range_selected_occupancy_flip_events"])
        <= int(limit)
        for name, limit in config["decision"]["maximum_selected_events"].items()
    }
    verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "training": {"source_trajectory_count": int(len(events)),
                     "source_event_count": int(np.count_nonzero(events)),
                     "ranked_range_positive_count": int(len(hard_positive)),
                     "ranked_range_negative_count": int(len(boundary_negative)),
                     "final_ranked_range_loss": final_loss},
        "development_evaluations": results, "decision_checks": decisions,
        "resources": {"gpu": torch.cuda.get_device_name(0),
                      "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                      "wall_seconds": time.monotonic() - started},
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({
        "status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict,
                      "decision_checks": decisions}, indent=2), flush=True)


if __name__ == "__main__":
    main()
