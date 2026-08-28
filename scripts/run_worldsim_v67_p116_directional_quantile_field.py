"""Train an Actor-only directional q90 residual field for analytic boundary queries."""

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
from scripts.run_worldsim_v67_p87_deepset_trajectory_reliability import _build_sets
from scripts.run_worldsim_v67_p104_temporal_flip_supervision import _aligned_group_max
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import (
    DirectionalActorGaussian, _actor_entries, _predict,
)
from scripts.run_worldsim_v67_p115_spectral_actor_uncertainty import _row_score


class DirectionalQuantileField(torch.nn.Module):
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


@torch.no_grad()
def _predict_quantile(
    model: DirectionalQuantileField, actor_features: np.ndarray, direction: np.ndarray,
    feature_mean: np.ndarray, feature_scale: np.ndarray,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(actor_features), 65536):
        normalized = (actor_features[start:start + 65536] - feature_mean) / feature_scale
        inputs = np.concatenate((normalized, direction[start:start + 65536]), axis=1)
        outputs.append(model(torch.from_numpy(inputs).cuda()).cpu().numpy())
    return np.concatenate(outputs)


def _evaluate(
    arrays: dict[str, np.ndarray], model: DirectionalQuantileField,
    feature_mean: np.ndarray, feature_scale: np.ndarray,
    frozen_directional: DirectionalActorGaussian, directional_checkpoint: dict,
    config: dict,
) -> dict[str, float | int]:
    actor_features, _, inverse = _actor_entries(arrays)
    point_count = int(arrays["actor_position_error_vector_ego_profile_m"].shape[1])
    row_actor_features = actor_features.reshape(-1, point_count, actor_features.shape[1])[inverse]
    normal = np.asarray(arrays["query_boundary_normal_ego_profile"], dtype=np.float32)
    predicted = np.asarray(arrays["predicted_separation_profile_m"], dtype=np.float32)
    radius = np.asarray(arrays["occupancy_interaction_radius_m"], dtype=np.float32)[:, None]
    signed = predicted - radius
    adverse_direction = -np.sign(signed)[:, :, None] * normal
    zero = np.abs(signed) < 1e-8
    adverse_direction[zero] = -normal[zero]
    q90 = _predict_quantile(
        model.eval(), row_actor_features.reshape(-1, row_actor_features.shape[-1]),
        adverse_direction.reshape(-1, 2), feature_mean, feature_scale,
    ).reshape(-1, point_count)
    floor = float(config["boundary_query"]["clearance_floor_m"])
    quantile_row_score = np.max(q90 / np.maximum(np.abs(signed), floor), axis=1)

    directional_mean, directional_scale = _predict(
        frozen_directional.eval(), actor_features,
        np.asarray(directional_checkpoint["feature_mean"], dtype=np.float32),
        np.asarray(directional_checkpoint["feature_scale"], dtype=np.float32),
        np.asarray(directional_checkpoint["target_mean"], dtype=np.float32),
        np.asarray(directional_checkpoint["target_scale"], dtype=np.float32),
    )
    directional_row_score = _row_score(
        arrays,
        directional_mean.reshape(-1, point_count, 2)[inverse],
        directional_scale.reshape(-1, point_count, 2)[inverse],
    )
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
    quantile_score = _aligned_group_max(row_keys, quantile_row_score, evaluation["identity"])
    directional_score = _aligned_group_max(row_keys, directional_row_score, evaluation["identity"])
    clearance_row = np.max(1.0 / np.maximum(np.abs(signed), floor), axis=1)
    clearance_score = _aligned_group_max(row_keys, clearance_row, evaluation["identity"])
    scenes, events = evaluation["scene_index"], evaluation["events"]
    coverage = float(config["selection"]["coverage_fraction"])
    selected = _select_by_scene(quantile_score, scenes, coverage)
    directional_selected = _select_by_scene(directional_score, scenes, coverage)
    clearance_selected = _select_by_scene(clearance_score, scenes, coverage)
    quantile_events = int(np.count_nonzero(events[selected]))
    directional_events = int(np.count_nonzero(events[directional_selected]))
    quantile_auroc = binary_auroc(events, quantile_score)
    directional_auroc = binary_auroc(events, directional_score)
    return {
        "trajectory_count": int(len(events)),
        "all_occupancy_flip_events": int(np.count_nonzero(events)),
        "selected_trajectory_count": int(len(selected)),
        "directional_quantile_selected_occupancy_flip_events": quantile_events,
        "directional_gaussian_selected_occupancy_flip_events": directional_events,
        "clearance_only_selected_occupancy_flip_events": int(
            np.count_nonzero(events[clearance_selected])
        ),
        "directional_quantile_event_auroc": quantile_auroc,
        "directional_gaussian_event_auroc": directional_auroc,
        "clearance_only_event_auroc": binary_auroc(events, clearance_score),
        "quantile_auroc_gain_over_gaussian": quantile_auroc - directional_auroc,
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
    rows_root = args.runs_root / config["rows"]["run"]
    source = dict(np.load(rows_root / config["rows"]["source_artifact"], allow_pickle=False))
    raw_features, raw_residual, _ = _actor_entries(source)
    feature_mean, feature_scale = raw_features.mean(0), raw_features.std(0).clip(min=1e-4)
    features = torch.from_numpy((raw_features - feature_mean) / feature_scale).cuda()
    residual = torch.from_numpy(raw_residual).cuda()
    direction_count = int(config["model"]["training_direction_count"])
    angles = torch.arange(direction_count, device="cuda", dtype=torch.float32) * (
        2.0 * torch.pi / direction_count
    )
    directions = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    model_config = config["model"]
    model = DirectionalQuantileField(features.shape[1] + 2, model_config["hidden_dimensions"]).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    quantile = float(model_config["quantile"])
    final_loss = 0.0
    torch.cuda.reset_peak_memory_stats()
    for step in range(int(model_config["steps"])):
        indices = torch.randint(len(features), (int(model_config["batch_size"]),), device="cuda")
        direction_index = torch.randint(direction_count, (len(indices),), device="cuda")
        direction = directions[direction_index]
        target = torch.sum(residual[indices] * direction, dim=1)
        prediction = model(torch.cat((features[indices], direction), dim=1))
        error = target - prediction
        loss = torch.maximum(quantile * error, (quantile - 1.0) * error).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step % 250 == 0 or step + 1 == int(model_config["steps"]):
            print(f"P116 directional-q90 step={step + 1} pinball={final_loss:.6f}", flush=True)
    torch.save({
        "feature_mean": feature_mean, "feature_scale": feature_scale,
        "hidden_dimensions": model_config["hidden_dimensions"],
        "quantile": quantile, "training_direction_count": direction_count,
        "model_state_dict": model.state_dict(),
    }, run_dir / config["model_artifact"])
    directional_checkpoint = torch.load(
        args.runs_root / config["frozen_p109"]["run"] / config["frozen_p109"]["artifact"],
        map_location="cuda",
    )
    frozen_directional = DirectionalActorGaussian(20, directional_checkpoint["hidden_dimensions"]).cuda()
    frozen_directional.load_state_dict(directional_checkpoint["model_state_dict"])
    results = {}
    for cohort in config["rows"]["development_cohorts"]:
        raw = dict(np.load(rows_root / cohort["artifact"], allow_pickle=False))
        results[cohort["name"]] = _evaluate(
            raw, model, feature_mean, feature_scale,
            frozen_directional, directional_checkpoint, config,
        )
        print(json.dumps({cohort["name"]: results[cohort["name"]]}, indent=2), flush=True)
    gains = [float(value["quantile_auroc_gain_over_gaussian"]) for value in results.values()]
    decisions = {
        "no_event_regression_on_either_consumed_cohort": all(
            value["directional_quantile_selected_occupancy_flip_events"]
            <= value["directional_gaussian_selected_occupancy_flip_events"]
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
        "training": {"actor_time_token_count": int(len(features)),
                     "training_direction_count": direction_count,
                     "quantile": quantile, "final_pinball_loss": final_loss},
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

