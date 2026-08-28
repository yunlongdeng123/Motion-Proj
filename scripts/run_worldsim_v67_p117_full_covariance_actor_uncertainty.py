"""Train a correlated bivariate Actor residual field for boundary-crossing risk."""

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
    FEATURE_NAMES, ReliabilityMLP, binary_auroc, predict_reliability,
)
from scripts.run_worldsim_v67_p86_direct_trajectory_reliability import (
    _group_max_visited_score, _select_by_scene,
)
from scripts.run_worldsim_v67_p87_deepset_trajectory_reliability import _build_sets
from scripts.run_worldsim_v67_p104_temporal_flip_supervision import _aligned_group_max
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import _actor_entries


class CorrelatedActorGaussian(torch.nn.Module):
    def __init__(self, feature_count: int, hidden_dimensions: list[int]) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        width = feature_count
        for hidden in hidden_dimensions:
            layers.extend((torch.nn.Linear(width, int(hidden)), torch.nn.SiLU()))
            width = int(hidden)
        layers.append(torch.nn.Linear(width, 5))
        self.network = torch.nn.Sequential(*layers)

    def forward(
        self, features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.network(features)
        mean = output[:, :2]
        scale = torch.nn.functional.softplus(output[:, 2:4]) + 0.02
        correlation = 0.95 * torch.tanh(output[:, 4])
        return mean, scale, correlation


@torch.no_grad()
def _predict(
    model: CorrelatedActorGaussian, features: np.ndarray, feature_mean: np.ndarray,
    feature_scale: np.ndarray, target_mean: np.ndarray, target_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means, scales, correlations = [], [], []
    for start in range(0, len(features), 65536):
        batch = torch.from_numpy(
            (features[start:start + 65536] - feature_mean) / feature_scale,
        ).cuda()
        mean, scale, correlation = model(batch)
        means.append(mean.cpu().numpy() * target_scale + target_mean)
        scales.append(scale.cpu().numpy() * target_scale)
        correlations.append(correlation.cpu().numpy())
    return np.concatenate(means), np.concatenate(scales), np.concatenate(correlations)


def _evaluate(
    arrays: dict[str, np.ndarray], model: CorrelatedActorGaussian,
    feature_mean: np.ndarray, feature_scale: np.ndarray,
    target_mean: np.ndarray, target_scale: np.ndarray, config: dict,
) -> dict[str, float | int]:
    actor_features, _, inverse = _actor_entries(arrays)
    point_count = int(arrays["actor_position_error_vector_ego_profile_m"].shape[1])
    mean, scale, correlation = _predict(
        model.eval(), actor_features, feature_mean, feature_scale, target_mean, target_scale,
    )
    row_mean = mean.reshape(-1, point_count, 2)[inverse]
    row_scale = scale.reshape(-1, point_count, 2)[inverse]
    row_correlation = correlation.reshape(-1, point_count)[inverse]
    normal = np.asarray(arrays["query_boundary_normal_ego_profile"], dtype=np.float32)
    projected_mean = np.sum(normal * row_mean, axis=2)
    nx_sx = normal[:, :, 0] * row_scale[:, :, 0]
    ny_sy = normal[:, :, 1] * row_scale[:, :, 1]
    projected_variance = (
        np.square(nx_sx) + np.square(ny_sy)
        + 2.0 * row_correlation * nx_sx * ny_sy
    )
    projected_scale = np.sqrt(np.maximum(projected_variance, 1e-8))
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
    clearance_row_score = np.max(
        1.0 / np.maximum(np.abs(signed), float(config["clearance_baseline_floor_m"])), axis=1,
    )
    clearance_score = _aligned_group_max(row_keys, clearance_row_score, evaluation["identity"])
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
    clearance_selected = _select_by_scene(clearance_score, scenes, fraction)
    frozen_selected = _select_by_scene(frozen_score, scenes, fraction)
    query_events = int(np.count_nonzero(events[selected]))
    actor_events = int(np.count_nonzero(events[actor_selected]))
    frozen_events = int(np.count_nonzero(events[frozen_selected]))
    prevalence, selected_prevalence = float(events.mean()), float(events[selected].mean())
    return {
        "row_count": int(len(arrays["features"])),
        "trajectory_count": int(len(events)),
        "all_occupancy_flip_events": int(np.count_nonzero(events)),
        "selected_trajectory_count": int(len(selected)),
        "query_selected_occupancy_flip_events": query_events,
        "actor_selected_occupancy_flip_events": actor_events,
        "clearance_only_selected_occupancy_flip_events": int(
            np.count_nonzero(events[clearance_selected])
        ),
        "frozen_p75_selected_occupancy_flip_events": frozen_events,
        "query_event_reduction": float((prevalence - selected_prevalence) / max(prevalence, 1e-12)),
        "query_event_reduction_over_actor_only": float((actor_events - query_events) / max(actor_events, 1)),
        "query_event_auroc": binary_auroc(events, query_score),
        "actor_event_auroc": binary_auroc(events, actor_score),
        "clearance_only_event_auroc": binary_auroc(events, clearance_score),
        "mean_absolute_predicted_correlation": float(np.mean(np.abs(row_correlation))),
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
            raise TimeoutError(f"P117 source rows not ready: {source_path}")
        time.sleep(5.0)
    source = dict(np.load(source_path, allow_pickle=False))
    raw_features, raw_target, _ = _actor_entries(source)
    feature_mean, feature_scale = raw_features.mean(0), raw_features.std(0).clip(min=1e-4)
    target_mean, target_scale = raw_target.mean(0), raw_target.std(0).clip(min=0.05)
    features = torch.from_numpy((raw_features - feature_mean) / feature_scale).cuda()
    target = torch.from_numpy((raw_target - target_mean) / target_scale).cuda()
    model_config = config["model"]
    model = CorrelatedActorGaussian(features.shape[1], model_config["hidden_dimensions"]).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    final_loss = 0.0
    torch.cuda.reset_peak_memory_stats()
    for step in range(int(model_config["steps"])):
        indices = torch.randint(len(features), (int(model_config["batch_size"]),), device="cuda")
        mean, scale, correlation = model(features[indices])
        residual = (target[indices] - mean) / scale
        one_minus_rho2 = 1.0 - correlation.square()
        mahalanobis = (
            residual[:, 0].square() + residual[:, 1].square()
            - 2.0 * correlation * residual[:, 0] * residual[:, 1]
        ) / one_minus_rho2
        loss = (
            torch.log(scale).sum(dim=1) + 0.5 * torch.log(one_minus_rho2)
            + 0.5 * mahalanobis
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step % 250 == 0 or step + 1 == int(model_config["steps"]):
            print(f"P117 correlated-gaussian step={step + 1} nll={final_loss:.6f}", flush=True)
    torch.save({
        "feature_mean": feature_mean, "feature_scale": feature_scale,
        "target_mean": target_mean, "target_scale": target_scale,
        "hidden_dimensions": model_config["hidden_dimensions"],
        "model_state_dict": model.state_dict(),
    }, run_dir / config["model_artifact"])
    results = {}
    for cohort in config["development_cohorts"]:
        path = prep_run / cohort["artifact"]
        results[cohort["name"]] = _evaluate(
            dict(np.load(path, allow_pickle=False)), model, feature_mean, feature_scale,
            target_mean, target_scale, config,
        )
        print(json.dumps({cohort["name"]: results[cohort["name"]]}, indent=2), flush=True)
    comparisons = config["frozen_p109_comparisons"]
    event_noninferior = all(
        results[name]["query_selected_occupancy_flip_events"] <= int(reference["events"])
        for name, reference in comparisons.items()
    )
    auroc_gains = [
        float(results[name]["query_event_auroc"]) - float(reference["auroc"])
        for name, reference in comparisons.items()
    ]
    mean_auroc_gain = float(np.mean(auroc_gains))
    supported = event_noninferior and mean_auroc_gain >= float(
        config["decision"]["minimum_mean_auroc_gain_over_p109"]
    )
    verdict = "supported_development_correlated_actor_uncertainty" if supported else \
        "rejected_development_correlated_actor_uncertainty"
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "training": {"actor_time_tokens": int(len(features)), "final_correlated_gaussian_nll": final_loss},
        "development_evaluations": results,
        "decision_metrics": {"event_noninferior_to_p109": event_noninferior,
                             "per_cohort_auroc_gain_over_p109": dict(zip(comparisons, auroc_gains)),
                             "mean_auroc_gain_over_p109": mean_auroc_gain},
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
                      "decision_metrics": summary["decision_metrics"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
