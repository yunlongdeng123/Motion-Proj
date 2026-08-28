"""Learn a tiny monotone trajectory tail pool over frozen Actor crossing probabilities."""

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

from motion_proj.worldsim_v67.actor_state_reliability import binary_auroc
from scripts.run_worldsim_v67_p86_direct_trajectory_reliability import _select_by_scene
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import (
    DirectionalActorGaussian, _actor_entries, _predict,
)


class MonotoneTailPool(torch.nn.Module):
    """Positive linear pool over ordered crossing probabilities and their union proxy."""

    def __init__(self, feature_count: int, initial_bias: float) -> None:
        super().__init__()
        self.raw_weight = torch.nn.Parameter(torch.full((feature_count,), -2.0))
        self.bias = torch.nn.Parameter(torch.tensor(initial_bias, dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features @ torch.nn.functional.softplus(self.raw_weight) + self.bias


@torch.no_grad()
def _crossing_probability(
    arrays: dict[str, np.ndarray], model: DirectionalActorGaussian,
    feature_mean: np.ndarray, feature_scale: np.ndarray,
    target_mean: np.ndarray, target_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    actor_features, _, inverse = _actor_entries(arrays)
    point_count = int(arrays["actor_position_error_vector_ego_profile_m"].shape[1])
    mean, scale = _predict(
        model.eval(), actor_features, feature_mean, feature_scale, target_mean, target_scale,
    )
    row_mean = mean.reshape(-1, point_count, 2)[inverse]
    row_scale = scale.reshape(-1, point_count, 2)[inverse]
    normal = np.asarray(arrays["query_boundary_normal_ego_profile"], dtype=np.float32)
    projected_mean = np.sum(normal * row_mean, axis=2)
    projected_scale = np.sqrt(np.sum(np.square(normal * row_scale), axis=2)).clip(min=1e-4)
    predicted = np.asarray(arrays["predicted_separation_profile_m"], dtype=np.float32)
    radius = np.asarray(arrays["occupancy_interaction_radius_m"], dtype=np.float32)[:, None]
    signed = predicted - radius
    margin = (np.abs(signed) + np.sign(signed) * projected_mean) / projected_scale
    probability = torch.special.ndtr(torch.from_numpy(-margin).cuda()).cpu().numpy()
    return probability, signed


def _trajectory_tail_features(
    arrays: dict[str, np.ndarray], probability: np.ndarray, top_k: int,
) -> dict[str, np.ndarray]:
    keys = np.stack((
        arrays["scene_index"],
        np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["query_id"],
    ), axis=1)
    identities, inverse = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_inverse)) + 1]
    ends = np.r_[starts[1:], len(order)]
    features = np.zeros((len(starts), top_k + 1), dtype=np.float32)
    events = np.zeros(len(starts), dtype=bool)
    clearance = np.zeros(len(starts), dtype=np.float32)
    flip = np.asarray(arrays["occupancy_decision_flip"], dtype=bool)
    predicted = np.asarray(arrays["predicted_separation_profile_m"], dtype=np.float32)
    radius = np.asarray(arrays["occupancy_interaction_radius_m"], dtype=np.float32)[:, None]
    for group, (start, end) in enumerate(zip(starts, ends)):
        members = order[start:end]
        values = np.sort(probability[members].reshape(-1))[::-1]
        count = min(top_k, len(values))
        features[group, :count] = values[:count]
        clipped = np.clip(values, 0.0, 1.0 - 1e-7)
        features[group, -1] = float(-np.expm1(np.log1p(-clipped).sum()))
        events[group] = bool(np.any(flip[members]))
        signed = predicted[members] - radius[members]
        clearance[group] = float(np.max(1.0 / np.maximum(np.abs(signed), 0.05)))
    return {
        "features": features, "events": events,
        "scene_index": identities[:, 0].astype(np.int32),
        "identity": identities.astype(np.int32), "clearance_score": clearance,
    }


@torch.no_grad()
def _score(model: MonotoneTailPool, features: np.ndarray) -> np.ndarray:
    outputs = []
    for start in range(0, len(features), 65536):
        outputs.append(model(torch.from_numpy(features[start:start + 65536]).cuda()).cpu().numpy())
    return np.concatenate(outputs)


def _evaluate(model: MonotoneTailPool, cohort: dict[str, np.ndarray], coverage: float) -> dict[str, float | int]:
    learned = _score(model.eval(), cohort["features"])
    directional = cohort["features"][:, 0]
    clearance = cohort["clearance_score"]
    scenes, events = cohort["scene_index"], cohort["events"]
    selected = _select_by_scene(learned, scenes, coverage)
    directional_selected = _select_by_scene(directional, scenes, coverage)
    clearance_selected = _select_by_scene(clearance, scenes, coverage)
    learned_events = int(np.count_nonzero(events[selected]))
    directional_events = int(np.count_nonzero(events[directional_selected]))
    clearance_events = int(np.count_nonzero(events[clearance_selected]))
    learned_auroc = binary_auroc(events, learned)
    directional_auroc = binary_auroc(events, directional)
    return {
        "trajectory_count": int(len(events)),
        "all_occupancy_flip_events": int(np.count_nonzero(events)),
        "selected_trajectory_count": int(len(selected)),
        "learned_tail_selected_occupancy_flip_events": learned_events,
        "directional_max_selected_occupancy_flip_events": directional_events,
        "clearance_only_selected_occupancy_flip_events": clearance_events,
        "learned_tail_event_auroc": learned_auroc,
        "directional_max_event_auroc": directional_auroc,
        "clearance_only_event_auroc": binary_auroc(events, clearance),
        "learned_tail_auroc_gain_over_directional_max": learned_auroc - directional_auroc,
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
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_scale = np.asarray(checkpoint["feature_scale"], dtype=np.float32)
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
    target_scale = np.asarray(checkpoint["target_scale"], dtype=np.float32)
    source_root = args.runs_root / config["rows"]["run"]
    source_raw = dict(np.load(source_root / config["rows"]["source_artifact"], allow_pickle=False))
    source_probability, _ = _crossing_probability(
        source_raw, actor_model, feature_mean, feature_scale, target_mean, target_scale,
    )
    top_k = int(config["model"]["top_k_crossing_probabilities"])
    source = _trajectory_tail_features(source_raw, source_probability, top_k)
    labels = source["events"]
    positive = np.flatnonzero(labels)
    negative = np.flatnonzero(~labels)
    prevalence = float(labels.mean())
    initial_bias = math.log(max(prevalence, 1e-6) / max(1.0 - prevalence, 1e-6))
    model = MonotoneTailPool(source["features"].shape[1], initial_bias).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["model"]["learning_rate"]),
        weight_decay=float(config["model"]["weight_decay"]),
    )
    features = torch.from_numpy(source["features"]).cuda()
    positive_gpu = torch.from_numpy(positive).long().cuda()
    negative_gpu = torch.from_numpy(negative).long().cuda()
    half = int(config["model"]["balanced_batch_size"]) // 2
    final_loss = 0.0
    torch.cuda.reset_peak_memory_stats()
    for step in range(int(config["model"]["steps"])):
        pos = positive_gpu[torch.randint(len(positive_gpu), (half,), device="cuda")]
        neg = negative_gpu[torch.randint(len(negative_gpu), (half,), device="cuda")]
        indices = torch.cat((pos, neg))
        target = torch.cat((torch.ones(half, device="cuda"), torch.zeros(half, device="cuda")))
        logits = model(features[indices])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step % 250 == 0 or step + 1 == int(config["model"]["steps"]):
            print(f"P114 monotone-tail step={step + 1} balanced_bce={final_loss:.6f}", flush=True)
    torch.save({
        "top_k_crossing_probabilities": top_k,
        "raw_weight": model.raw_weight.detach().cpu(),
        "bias": model.bias.detach().cpu(),
        "model_state_dict": model.state_dict(),
    }, run_dir / config["model_artifact"])
    results = {}
    for cohort in config["rows"]["development_cohorts"]:
        raw = dict(np.load(source_root / cohort["artifact"], allow_pickle=False))
        probability, _ = _crossing_probability(
            raw, actor_model, feature_mean, feature_scale, target_mean, target_scale,
        )
        grouped = _trajectory_tail_features(raw, probability, top_k)
        results[cohort["name"]] = _evaluate(
            model, grouped, float(config["selection"]["coverage_fraction"]),
        )
        print(json.dumps({cohort["name"]: results[cohort["name"]]}, indent=2), flush=True)
    gains = [float(value["learned_tail_auroc_gain_over_directional_max"]) for value in results.values()]
    decisions = {
        "no_event_regression_on_either_consumed_cohort": all(
            value["learned_tail_selected_occupancy_flip_events"]
            <= value["directional_max_selected_occupancy_flip_events"]
            for value in results.values()
        ),
        "nonnegative_auroc_gain_on_either_consumed_cohort": all(gain >= 0.0 for gain in gains),
        "minimum_mean_auroc_gain": float(np.mean(gains))
        >= float(config["decision"]["minimum_mean_auroc_gain"]),
    }
    verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "training": {"source_trajectory_count": int(len(labels)),
                     "source_event_count": int(np.count_nonzero(labels)),
                     "final_balanced_bce": final_loss},
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
