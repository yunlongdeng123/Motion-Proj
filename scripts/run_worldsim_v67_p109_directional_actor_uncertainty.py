"""Train a directional Gaussian Actor error field and query boundary-crossing margin."""

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


class DirectionalActorGaussian(torch.nn.Module):
    def __init__(self, feature_count: int, hidden_dimensions: list[int]) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        width = feature_count
        for hidden in hidden_dimensions:
            layers.extend((torch.nn.Linear(width, int(hidden)), torch.nn.SiLU()))
            width = int(hidden)
        layers.append(torch.nn.Linear(width, 4))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.network(features)
        return output[:, :2], torch.nn.functional.softplus(output[:, 2:]) + 0.02


def _actor_entries(
    arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = np.stack((
        arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["actor_id"],
    ), axis=1)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    actor = np.asarray(arrays["features"], dtype=np.float32)[first, :len(ACTOR_FEATURE_NAMES)]
    residual = np.asarray(
        arrays["actor_position_error_vector_ego_profile_m"], dtype=np.float32,
    )[first]
    fractions = np.linspace(0.0, 1.0, residual.shape[1], dtype=np.float32)
    features = np.concatenate((
        np.broadcast_to(actor[:, None, :], (len(actor), residual.shape[1], actor.shape[1])),
        np.broadcast_to(fractions[None, :, None], (len(actor), residual.shape[1], 1)),
    ), axis=2)
    return features.reshape(-1, features.shape[-1]), residual.reshape(-1, 2), inverse


@torch.no_grad()
def _predict(
    model: DirectionalActorGaussian, features: np.ndarray, feature_mean: np.ndarray,
    feature_scale: np.ndarray, target_mean: np.ndarray, target_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    means, scales = [], []
    for start in range(0, len(features), 65536):
        batch = torch.from_numpy(
            (features[start:start + 65536] - feature_mean) / feature_scale,
        ).cuda()
        mean, scale = model(batch)
        means.append(mean.cpu().numpy() * target_scale + target_mean)
        scales.append(scale.cpu().numpy() * target_scale)
    return np.concatenate(means), np.concatenate(scales)


def _evaluate(
    arrays: dict[str, np.ndarray], model: DirectionalActorGaussian,
    feature_mean: np.ndarray, feature_scale: np.ndarray,
    target_mean: np.ndarray, target_scale: np.ndarray, config: dict,
) -> dict[str, float | int]:
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
    standardized_crossing_margin = (
        np.abs(signed) + np.sign(signed) * projected_mean
    ) / projected_scale
    query_row_score = np.max(-standardized_crossing_margin, axis=1)
    actor_row_score = np.max(np.linalg.norm(row_scale, axis=2), axis=1)

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
    scenes, events = evaluation["scene_index"], evaluation["events"]
    fraction = float(config["selection"]["coverage_fraction"])
    selected = _select_by_scene(query_score, scenes, fraction)
    actor_selected = _select_by_scene(actor_score, scenes, fraction)
    frozen_selected = _select_by_scene(frozen_score, scenes, fraction)
    query_events = int(np.count_nonzero(events[selected]))
    actor_events = int(np.count_nonzero(events[actor_selected]))
    frozen_events = int(np.count_nonzero(events[frozen_selected]))
    prevalence, selected_prevalence = float(events.mean()), float(events[selected].mean())
    return {
        "row_count": int(len(arrays["features"])), "trajectory_count": int(len(events)),
        "all_occupancy_flip_events": int(np.count_nonzero(events)),
        "selected_trajectory_count": int(len(selected)),
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
    prep_run = args.runs_root / config["source_rows"]["run"]
    source_path = prep_run / config["source_rows"]["artifact"]
    deadline = time.monotonic() + float(config["readiness_timeout_seconds"])
    while not source_path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"P109 source rows not ready: {source_path}")
        print("P109 waiting for directional source rows", flush=True)
        time.sleep(5.0)
    source = dict(np.load(source_path, allow_pickle=False))
    raw_features, raw_target, _ = _actor_entries(source)
    feature_mean, feature_scale = raw_features.mean(0), raw_features.std(0).clip(min=1e-4)
    target_mean, target_scale = raw_target.mean(0), raw_target.std(0).clip(min=0.05)
    features = torch.from_numpy((raw_features - feature_mean) / feature_scale).cuda()
    target = torch.from_numpy((raw_target - target_mean) / target_scale).cuda()
    model_config = config["model"]
    model = DirectionalActorGaussian(features.shape[1], model_config["hidden_dimensions"]).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    final_loss = 0.0
    torch.cuda.reset_peak_memory_stats()
    for step in range(int(model_config["steps"])):
        indices = torch.randint(len(features), (int(model_config["batch_size"]),), device="cuda")
        mean, scale = model(features[indices])
        residual = (target[indices] - mean) / scale
        loss = (0.5 * residual.square() + torch.log(scale)).sum(dim=1).mean()
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step % 250 == 0 or step + 1 == int(model_config["steps"]):
            print(f"P109 directional-gaussian step={step + 1} nll={final_loss:.6f}", flush=True)
    torch.save({
        "feature_mean": feature_mean, "feature_scale": feature_scale,
        "target_mean": target_mean, "target_scale": target_scale,
        "hidden_dimensions": model_config["hidden_dimensions"],
        "model_state_dict": model.state_dict(),
    }, run_dir / config["model_artifact"])
    results = {}
    for cohort in config["development_cohorts"]:
        path = prep_run / cohort["artifact"]
        while not path.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"P109 development rows not ready: {path}")
            time.sleep(5.0)
        results[cohort["name"]] = _evaluate(
            dict(np.load(path, allow_pickle=False)), model, feature_mean, feature_scale,
            target_mean, target_scale, config,
        )
        print(json.dumps({cohort["name"]: results[cohort["name"]]}, indent=2), flush=True)
    supported = all(
        x["query_selected_occupancy_flip_events"] < x["actor_selected_occupancy_flip_events"]
        and x["query_selected_occupancy_flip_events"] <= x["frozen_p75_selected_occupancy_flip_events"]
        for x in results.values()
    )
    verdict = "supported_development_directional_actor_uncertainty" if supported else \
        "rejected_development_directional_actor_uncertainty"
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "training": {"actor_time_tokens": int(len(features)), "final_gaussian_nll": final_loss},
        "development_evaluations": results,
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
                      "development_evaluations": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
