"""Train time-local occupancy-flip risk and aggregate it to candidate trajectories."""

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
    ACTOR_FEATURE_NAMES, FEATURE_NAMES, ReliabilityMLP, binary_auroc, predict_reliability,
)
from scripts.run_worldsim_v67_p86_direct_trajectory_reliability import (
    _group_max_visited_score, _select_by_scene,
)
from scripts.run_worldsim_v67_p87_deepset_trajectory_reliability import _build_sets


class TemporalTokenRisk(torch.nn.Module):
    def __init__(self, feature_count: int, hidden_dimensions: list[int]) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        width = feature_count
        for hidden in hidden_dimensions:
            layers.extend((torch.nn.Linear(width, int(hidden)), torch.nn.SiLU()))
            width = int(hidden)
        layers.append(torch.nn.Linear(width, 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def _token_features(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(arrays["features"], dtype=np.float32)
    predicted = np.asarray(arrays["predicted_separation_profile_m"], dtype=np.float32)
    radius = np.asarray(arrays["occupancy_interaction_radius_m"], dtype=np.float32)[:, None]
    signed = predicted - radius
    point_count = predicted.shape[1]
    fractions = np.linspace(0.0, 1.0, point_count, dtype=np.float32)
    query_base = np.broadcast_to(features[:, None, :21], (len(features), point_count, 21))
    actor_base = np.broadcast_to(features[:, None, :len(ACTOR_FEATURE_NAMES)],
                                 (len(features), point_count, len(ACTOR_FEATURE_NAMES)))
    time = np.broadcast_to(fractions[None, :, None], (len(features), point_count, 1))
    query = np.concatenate((query_base, time, signed[..., None], np.abs(signed)[..., None]), axis=2)
    actor = np.concatenate((actor_base, time), axis=2)
    return query.reshape(-1, query.shape[-1]), actor.reshape(-1, actor.shape[-1])


@torch.no_grad()
def _predict(model: TemporalTokenRisk, features: torch.Tensor, batch_size: int = 65536) -> np.ndarray:
    outputs = []
    for start in range(0, len(features), batch_size):
        outputs.append(torch.sigmoid(model(features[start:start + batch_size])).cpu().numpy())
    return np.concatenate(outputs)


def _aligned_group_max(keys: np.ndarray, values: np.ndarray, identities: np.ndarray) -> np.ndarray:
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    maxima = np.full(len(unique), -np.inf, dtype=np.float32)
    np.maximum.at(maxima, inverse, values.astype(np.float32))
    table = {tuple(key.tolist()): float(value) for key, value in zip(unique, maxima)}
    return np.asarray([table[tuple(key.tolist())] for key in identities], dtype=np.float32)


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
    source = dict(np.load(args.runs_root / config["source_rows"]["run"] /
                          config["source_rows"]["artifact"], allow_pickle=False))
    query_np, actor_np = _token_features(source)
    query_mean, query_scale = query_np.mean(0), query_np.std(0).clip(min=1e-4)
    actor_mean, actor_scale = actor_np.mean(0), actor_np.std(0).clip(min=1e-4)
    query = torch.from_numpy((query_np - query_mean) / query_scale).cuda()
    actor = torch.from_numpy((actor_np - actor_mean) / actor_scale).cuda()
    labels_np = np.asarray(source["occupancy_decision_flip_profile"], dtype=bool).reshape(-1)
    labels = torch.from_numpy(labels_np.astype(np.float32)).cuda()
    horizons = np.asarray(source["horizon_seconds"], dtype=np.float32)
    point_count = int(source["occupancy_decision_flip_profile"].shape[1])
    groups = []
    for horizon in sorted(np.unique(horizons).tolist()):
        rows = np.flatnonzero(horizons == horizon)
        for temporal_index in range(point_count):
            indices = rows * point_count + temporal_index
            positive = indices[labels_np[indices]]
            negative = indices[~labels_np[indices]]
            if len(positive) and len(negative):
                groups.append((torch.from_numpy(positive).long().cuda(), torch.from_numpy(negative).long().cuda()))
    model_config = config["model"]
    query_model = TemporalTokenRisk(query.shape[1], model_config["hidden_dimensions"]).cuda()
    actor_model = TemporalTokenRisk(actor.shape[1], model_config["hidden_dimensions"]).cuda()
    optimizer = torch.optim.AdamW(
        list(query_model.parameters()) + list(actor_model.parameters()),
        lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"]),
    )
    pairs = max(1, int(model_config["pair_batch_size"]) // len(groups))
    final_query = final_actor = 0.0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(int(model_config["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        positive = torch.cat([x[torch.randint(len(x), (pairs,), device="cuda")] for x, _ in groups])
        negative = torch.cat([x[torch.randint(len(x), (pairs,), device="cuda")] for _, x in groups])
        batch = torch.cat((positive, negative))
        target = labels[batch]
        query_loss = torch.nn.functional.binary_cross_entropy_with_logits(query_model(query[batch]), target)
        actor_loss = torch.nn.functional.binary_cross_entropy_with_logits(actor_model(actor[batch]), target)
        (query_loss + actor_loss).backward(); optimizer.step()
        final_query, final_actor = float(query_loss.detach().cpu()), float(actor_loss.detach().cpu())
        if epoch % 250 == 0 or epoch + 1 == int(model_config["epochs"]):
            print(f"temporal-flip epoch={epoch + 1} query={final_query:.6f} actor={final_actor:.6f}", flush=True)
    torch.save({
        "query_feature_mean": query_mean, "query_feature_scale": query_scale,
        "actor_feature_mean": actor_mean, "actor_feature_scale": actor_scale,
        "hidden_dimensions": model_config["hidden_dimensions"],
        "query_model_state_dict": query_model.state_dict(), "actor_model_state_dict": actor_model.state_dict(),
    }, run_dir / config["model_artifact"])

    evaluation_raw = dict(np.load(args.runs_root / config["evaluation_rows"]["run"] /
                                  config["evaluation_rows"]["artifact"], allow_pickle=False))
    evaluation_query_np, evaluation_actor_np = _token_features(evaluation_raw)
    evaluation_query = torch.from_numpy((evaluation_query_np - query_mean) / query_scale).cuda()
    evaluation_actor = torch.from_numpy((evaluation_actor_np - actor_mean) / actor_scale).cuda()
    query_token_score = _predict(query_model.eval(), evaluation_query)
    actor_token_score = _predict(actor_model.eval(), evaluation_actor)
    temporal_count = int(evaluation_raw["occupancy_decision_flip_profile"].shape[1])
    query_row_score = query_token_score.reshape(-1, temporal_count).max(axis=1)
    actor_row_score = actor_token_score.reshape(-1, temporal_count).max(axis=1)
    target_raw = dict(evaluation_raw)
    target_raw["raw_actor_state_error_m"] = target_raw["occupancy_decision_flip"].astype(np.float32)
    evaluation = _build_sets(
        target_raw, float(config["evaluation"]["visited_region_radius_m"]),
        float(config["evaluation"]["unreliable_actor_state_error_m"]),
        int(config["evaluation"]["maximum_visited_actors"]),
    )
    row_keys = np.stack((
        evaluation_raw["scene_index"], np.rint(evaluation_raw["horizon_seconds"] * 10).astype(np.int32),
        evaluation_raw["anchor_frame"], evaluation_raw["query_id"],
    ), axis=1)
    query_score = _aligned_group_max(row_keys, query_row_score, evaluation["identity"])
    actor_score = _aligned_group_max(row_keys, actor_row_score, evaluation["identity"])
    frozen = torch.load(args.runs_root / config["frozen_p75"]["run"] /
                        config["frozen_p75"]["artifact"], map_location="cuda")
    frozen_model = ReliabilityMLP(len(FEATURE_NAMES), frozen["hidden_dimensions"]).cuda()
    frozen_model.load_state_dict(frozen["query_model_state_dict"])
    frozen_row_score = predict_reliability(
        frozen_model.eval(), evaluation_raw["features"][:, :len(FEATURE_NAMES)],
        np.asarray(frozen["feature_mean"], dtype=np.float32),
        np.asarray(frozen["feature_scale"], dtype=np.float32),
    )
    frozen_score = _group_max_visited_score(
        row_keys, frozen_row_score,
        np.asarray(evaluation_raw["predicted_minimum_separation_m"])
        <= float(config["evaluation"]["visited_region_radius_m"]),
    )
    scenes, events = evaluation["scene_index"], evaluation["events"]
    fraction = float(config["selection"]["coverage_fraction"])
    selected = _select_by_scene(query_score, scenes, fraction)
    actor_selected = _select_by_scene(actor_score, scenes, fraction)
    frozen_selected = _select_by_scene(frozen_score, scenes, fraction)
    query_events, actor_events, frozen_events = (
        int(np.count_nonzero(events[index])) for index in (selected, actor_selected, frozen_selected)
    )
    all_prevalence, selected_prevalence = float(events.mean()), float(events[selected].mean())
    metrics = {
        "source_temporal_token_count": int(len(labels_np)),
        "source_temporal_flip_count": int(np.count_nonzero(labels_np)),
        "evaluation_trajectory_count": int(len(events)), "all_occupancy_flip_events": int(np.count_nonzero(events)),
        "selected_trajectory_count": int(len(selected)), "achieved_coverage": float(len(selected) / len(events)),
        "query_selected_occupancy_flip_events": query_events,
        "actor_selected_occupancy_flip_events": actor_events,
        "frozen_p75_selected_occupancy_flip_events": frozen_events,
        "all_occupancy_flip_prevalence": all_prevalence,
        "query_selected_occupancy_flip_prevalence": selected_prevalence,
        "actor_selected_occupancy_flip_prevalence": float(events[actor_selected].mean()),
        "frozen_p75_selected_occupancy_flip_prevalence": float(events[frozen_selected].mean()),
        "query_event_reduction": float((all_prevalence - selected_prevalence) / max(all_prevalence, 1e-12)),
        "query_event_reduction_over_actor_only": float((actor_events - query_events) / max(actor_events, 1)),
        "query_event_auroc": binary_auroc(events, query_score), "actor_event_auroc": binary_auroc(events, actor_score),
    }
    gates = {
        "minimum_event_reduction_over_actor_only": metrics["query_event_reduction_over_actor_only"]
        >= float(config["gates"]["minimum_event_reduction_over_actor_only"]),
        "minimum_absolute_trajectory_event_reduction": metrics["query_event_reduction"]
        >= float(config["gates"]["minimum_absolute_trajectory_event_reduction"]),
        "no_more_events_than_frozen_p75": query_events <= frozen_events,
        "maximum_max_error_ratio_to_frozen_p75": selected_prevalence
        <= float(events[frozen_selected].mean()) * float(config["gates"]["maximum_max_error_ratio_to_frozen_p75"]),
    }
    verdict = config["verdict_on_pass"] if all(gates.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "training": {"temporal_pair_groups": len(groups), "final_query_loss": final_query,
                     "final_actor_loss": final_actor},
        "development_evaluation": metrics, "gate_results": gates,
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
                      "development_evaluation": metrics, "gate_results": gates}, indent=2), flush=True)


if __name__ == "__main__":
    main()
