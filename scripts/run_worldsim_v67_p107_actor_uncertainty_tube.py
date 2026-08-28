"""Train Actor-only q90 error tubes and intersect them with candidate-Ego clearances."""

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
from scripts.run_worldsim_v67_p104_temporal_flip_supervision import _aligned_group_max


class ActorQuantileTube(torch.nn.Module):
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
        return torch.nn.functional.softplus(self.network(features).squeeze(-1))


def _actor_entries(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = np.stack((
        arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["actor_id"],
    ), axis=1)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    actor = np.asarray(arrays["features"], dtype=np.float32)[first, :len(ACTOR_FEATURE_NAMES)]
    error = np.asarray(arrays["actor_position_error_profile_m"], dtype=np.float32)[first]
    fractions = np.linspace(0.0, 1.0, error.shape[1], dtype=np.float32)
    features = np.concatenate((
        np.broadcast_to(actor[:, None, :], (len(actor), error.shape[1], actor.shape[1])),
        np.broadcast_to(fractions[None, :, None], (len(actor), error.shape[1], 1)),
    ), axis=2)
    return features.reshape(-1, features.shape[-1]), error.reshape(-1), inverse


@torch.no_grad()
def _predict(model: ActorQuantileTube, features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    outputs = []
    for start in range(0, len(features), 65536):
        batch = torch.from_numpy((features[start:start + 65536] - mean) / scale).cuda()
        outputs.append(torch.expm1(model(batch)).clamp(min=0).cpu().numpy())
    return np.concatenate(outputs)


def _evaluate(
    arrays: dict[str, np.ndarray], model: ActorQuantileTube, mean: np.ndarray, scale: np.ndarray,
    config: dict,
) -> dict[str, float | int]:
    actor_features, _, actor_inverse = _actor_entries(arrays)
    point_count = int(arrays["actor_position_error_profile_m"].shape[1])
    actor_tube = _predict(model.eval(), actor_features, mean, scale).reshape(-1, point_count)
    row_tube = actor_tube[actor_inverse]
    predicted = np.asarray(arrays["predicted_separation_profile_m"], dtype=np.float32)
    radius = np.asarray(arrays["occupancy_interaction_radius_m"], dtype=np.float32)[:, None]
    boundary_distance = np.abs(predicted - radius)
    query_row_score = np.max(
        row_tube / np.maximum(boundary_distance, float(config["model"]["clearance_floor_m"])), axis=1,
    )
    actor_row_score = np.max(row_tube, axis=1)

    target_raw = dict(arrays)
    target_raw["raw_actor_state_error_m"] = arrays["occupancy_decision_flip"].astype(np.float32)
    evaluation = _build_sets(
        target_raw, float(config["evaluation"]["visited_region_radius_m"]),
        float(config["evaluation"]["unreliable_actor_state_error_m"]),
        int(config["evaluation"]["maximum_visited_actors"]),
    )
    row_keys = np.stack((
        arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["query_id"],
    ), axis=1)
    query_score = _aligned_group_max(row_keys, query_row_score, evaluation["identity"])
    actor_score = _aligned_group_max(row_keys, actor_row_score, evaluation["identity"])

    frozen = torch.load(
        Path(config["runs_root"]) / config["frozen_p75"]["run"] / config["frozen_p75"]["artifact"],
        map_location="cuda",
    )
    frozen_model = ReliabilityMLP(len(FEATURE_NAMES), frozen["hidden_dimensions"]).cuda()
    frozen_model.load_state_dict(frozen["query_model_state_dict"])
    frozen_row_score = predict_reliability(
        frozen_model.eval(), arrays["features"][:, :len(FEATURE_NAMES)],
        np.asarray(frozen["feature_mean"], dtype=np.float32),
        np.asarray(frozen["feature_scale"], dtype=np.float32),
    )
    frozen_score = _group_max_visited_score(
        row_keys, frozen_row_score,
        np.asarray(arrays["predicted_minimum_separation_m"])
        <= float(config["evaluation"]["visited_region_radius_m"]),
    )
    scenes = evaluation["scene_index"]
    events = evaluation["events"]
    fraction = float(config["selection"]["coverage_fraction"])
    query_selected = _select_by_scene(query_score, scenes, fraction)
    actor_selected = _select_by_scene(actor_score, scenes, fraction)
    frozen_selected = _select_by_scene(frozen_score, scenes, fraction)
    query_events = int(np.count_nonzero(events[query_selected]))
    actor_events = int(np.count_nonzero(events[actor_selected]))
    frozen_events = int(np.count_nonzero(events[frozen_selected]))
    prevalence = float(events.mean())
    selected_prevalence = float(events[query_selected].mean())
    return {
        "row_count": int(len(arrays["features"])),
        "trajectory_count": int(len(events)),
        "all_occupancy_flip_events": int(np.count_nonzero(events)),
        "selected_trajectory_count": int(len(query_selected)),
        "query_selected_occupancy_flip_events": query_events,
        "actor_selected_occupancy_flip_events": actor_events,
        "frozen_p75_selected_occupancy_flip_events": frozen_events,
        "query_event_reduction": float((prevalence - selected_prevalence) / max(prevalence, 1e-12)),
        "query_event_reduction_over_actor_only": float((actor_events - query_events) / max(actor_events, 1)),
        "query_event_auroc": binary_auroc(events, query_score),
        "actor_event_auroc": binary_auroc(events, actor_score),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config["runs_root"] = str(args.runs_root)
    run_dir = args.runs_root / "worldsim_v67" / config["task_id"] / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    started = time.monotonic()
    torch.manual_seed(int(config["seed"]))
    source_path = args.runs_root / config["source_rows"]["run"] / config["source_rows"]["artifact"]
    deadline = time.monotonic() + float(config["readiness_timeout_seconds"])
    while not source_path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"P107 source rows not ready: {source_path}")
        print("P107 waiting for source Actor tubes", flush=True)
        time.sleep(5.0)
    source = dict(np.load(source_path, allow_pickle=False))
    raw_features, raw_error, _ = _actor_entries(source)
    mean = raw_features.mean(0)
    scale = raw_features.std(0).clip(min=1e-4)
    features = torch.from_numpy((raw_features - mean) / scale).cuda()
    target = torch.from_numpy(np.log1p(raw_error)).cuda()
    model_config = config["model"]
    model = ActorQuantileTube(features.shape[1], model_config["hidden_dimensions"]).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    quantile = float(model_config["quantile"])
    batch_size = int(model_config["batch_size"])
    final_loss = 0.0
    torch.cuda.reset_peak_memory_stats()
    for step in range(int(model_config["steps"])):
        indices = torch.randint(len(features), (batch_size,), device="cuda")
        prediction = model(features[indices])
        residual = target[indices] - prediction
        loss = torch.maximum(quantile * residual, (quantile - 1.0) * residual).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step % 250 == 0 or step + 1 == int(model_config["steps"]):
            print(f"P107 actor-q90 step={step + 1} pinball={final_loss:.6f}", flush=True)
    torch.save({
        "feature_mean": mean, "feature_scale": scale,
        "hidden_dimensions": model_config["hidden_dimensions"], "quantile": quantile,
        "model_state_dict": model.state_dict(),
    }, run_dir / config["model_artifact"])

    results = {}
    for cohort in config["development_cohorts"]:
        path = args.runs_root / config["source_rows"]["run"] / cohort["artifact"]
        while not path.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"P107 development rows not ready: {path}")
            print(f"P107 waiting for development cohort {cohort['name']}", flush=True)
            time.sleep(5.0)
        results[cohort["name"]] = _evaluate(
            dict(np.load(path, allow_pickle=False)), model, mean, scale, config,
        )
        print(json.dumps({cohort["name"]: results[cohort["name"]]}, indent=2), flush=True)

    supported = all(
        metrics["query_selected_occupancy_flip_events"] < metrics["actor_selected_occupancy_flip_events"]
        and metrics["query_selected_occupancy_flip_events"]
        <= metrics["frozen_p75_selected_occupancy_flip_events"]
        for metrics in results.values()
    )
    verdict = (
        "supported_development_actor_uncertainty_boundary_factorization"
        if supported else "rejected_development_actor_uncertainty_boundary_factorization"
    )
    summary = {
        "schema_version": config["output_schema_version"],
        "task_id": config["task_id"], "hypothesis_id": config["hypothesis_id"],
        "status": "done", "verdict": verdict, "role": config["role"],
        "training": {
            "deduplicated_actor_time_tokens": int(len(features)),
            "quantile": quantile, "final_pinball_loss": final_loss,
        },
        "development_evaluations": results,
        "resources": {
            "gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "wall_seconds": time.monotonic() - started,
        },
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    (run_dir / "status.json").write_text(json.dumps({
        "status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict,
                      "development_evaluations": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
