"""Train a joint low-frequency Actor residual sequence distribution for boundary queries."""

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

from motion_proj.worldsim_v67.actor_state_reliability import ACTOR_FEATURE_NAMES, binary_auroc
from scripts.run_worldsim_v67_p86_direct_trajectory_reliability import _select_by_scene
from scripts.run_worldsim_v67_p87_deepset_trajectory_reliability import _build_sets
from scripts.run_worldsim_v67_p104_temporal_flip_supervision import _aligned_group_max
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import (
    DirectionalActorGaussian, _actor_entries, _predict,
)


class SpectralActorGaussian(torch.nn.Module):
    def __init__(self, feature_count: int, hidden_dimensions: list[int], coefficient_count: int) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        width = feature_count
        for hidden in hidden_dimensions:
            layers.extend((torch.nn.Linear(width, int(hidden)), torch.nn.SiLU()))
            width = int(hidden)
        layers.append(torch.nn.Linear(width, coefficient_count * 4))
        self.network = torch.nn.Sequential(*layers)
        self.coefficient_count = coefficient_count

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.network(features).reshape(-1, self.coefficient_count, 4)
        return output[:, :, :2], torch.nn.functional.softplus(output[:, :, 2:]) + 0.02


def _dct_basis(point_count: int, coefficient_count: int) -> np.ndarray:
    time_index = np.arange(point_count, dtype=np.float32)[:, None]
    frequency = np.arange(coefficient_count, dtype=np.float32)[None, :]
    basis = np.sqrt(2.0 / point_count) * np.cos(
        np.pi / point_count * (time_index + 0.5) * frequency,
    )
    basis[:, 0] = np.sqrt(1.0 / point_count)
    return basis.astype(np.float32)


def _actor_sequences(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = np.stack((
        arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["actor_id"],
    ), axis=1)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    actor = np.asarray(arrays["features"], dtype=np.float32)[first, :len(ACTOR_FEATURE_NAMES)]
    residual = np.asarray(arrays["actor_position_error_vector_ego_profile_m"], dtype=np.float32)[first]
    return actor, residual, inverse


@torch.no_grad()
def _predict_spectral(
    model: SpectralActorGaussian, features: np.ndarray,
    feature_mean: np.ndarray, feature_scale: np.ndarray,
    target_mean: np.ndarray, target_scale: np.ndarray,
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


def _row_score(
    arrays: dict[str, np.ndarray], row_mean: np.ndarray, row_scale: np.ndarray,
) -> np.ndarray:
    normal = np.asarray(arrays["query_boundary_normal_ego_profile"], dtype=np.float32)
    projected_mean = np.sum(normal * row_mean, axis=2)
    projected_scale = np.sqrt(np.sum(np.square(normal * row_scale), axis=2)).clip(min=1e-4)
    predicted = np.asarray(arrays["predicted_separation_profile_m"], dtype=np.float32)
    radius = np.asarray(arrays["occupancy_interaction_radius_m"], dtype=np.float32)[:, None]
    signed = predicted - radius
    margin = (np.abs(signed) + np.sign(signed) * projected_mean) / projected_scale
    return np.max(-margin, axis=1)


def _evaluate(
    arrays: dict[str, np.ndarray], model: SpectralActorGaussian,
    feature_mean: np.ndarray, feature_scale: np.ndarray,
    target_mean: np.ndarray, target_scale: np.ndarray, basis: np.ndarray,
    frozen_directional: DirectionalActorGaussian, directional_checkpoint: dict,
    config: dict,
) -> dict[str, float | int]:
    actor_features, _, inverse = _actor_sequences(arrays)
    coefficient_mean, coefficient_scale = _predict_spectral(
        model.eval(), actor_features, feature_mean, feature_scale, target_mean, target_scale,
    )
    time_mean = np.einsum("tk,bkd->btd", basis, coefficient_mean)
    time_variance = np.einsum("tk,bkd->btd", np.square(basis), np.square(coefficient_scale))
    spectral_score_row = _row_score(arrays, time_mean[inverse], np.sqrt(time_variance)[inverse])

    directional_features, _, directional_inverse = _actor_entries(arrays)
    point_count = int(arrays["actor_position_error_vector_ego_profile_m"].shape[1])
    directional_mean, directional_scale = _predict(
        frozen_directional.eval(), directional_features,
        np.asarray(directional_checkpoint["feature_mean"], dtype=np.float32),
        np.asarray(directional_checkpoint["feature_scale"], dtype=np.float32),
        np.asarray(directional_checkpoint["target_mean"], dtype=np.float32),
        np.asarray(directional_checkpoint["target_scale"], dtype=np.float32),
    )
    directional_score_row = _row_score(
        arrays,
        directional_mean.reshape(-1, point_count, 2)[directional_inverse],
        directional_scale.reshape(-1, point_count, 2)[directional_inverse],
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
    spectral_score = _aligned_group_max(row_keys, spectral_score_row, evaluation["identity"])
    directional_score = _aligned_group_max(row_keys, directional_score_row, evaluation["identity"])
    predicted = np.asarray(arrays["predicted_separation_profile_m"], dtype=np.float32)
    radius = np.asarray(arrays["occupancy_interaction_radius_m"], dtype=np.float32)[:, None]
    clearance_row = np.max(1.0 / np.maximum(np.abs(predicted - radius), 0.05), axis=1)
    clearance_score = _aligned_group_max(row_keys, clearance_row, evaluation["identity"])
    scenes, events = evaluation["scene_index"], evaluation["events"]
    coverage = float(config["selection"]["coverage_fraction"])
    selected = _select_by_scene(spectral_score, scenes, coverage)
    directional_selected = _select_by_scene(directional_score, scenes, coverage)
    clearance_selected = _select_by_scene(clearance_score, scenes, coverage)
    spectral_events = int(np.count_nonzero(events[selected]))
    directional_events = int(np.count_nonzero(events[directional_selected]))
    spectral_auroc = binary_auroc(events, spectral_score)
    directional_auroc = binary_auroc(events, directional_score)
    return {
        "trajectory_count": int(len(events)),
        "all_occupancy_flip_events": int(np.count_nonzero(events)),
        "selected_trajectory_count": int(len(selected)),
        "spectral_selected_occupancy_flip_events": spectral_events,
        "directional_selected_occupancy_flip_events": directional_events,
        "clearance_only_selected_occupancy_flip_events": int(
            np.count_nonzero(events[clearance_selected])
        ),
        "spectral_event_auroc": spectral_auroc,
        "directional_event_auroc": directional_auroc,
        "clearance_only_event_auroc": binary_auroc(events, clearance_score),
        "spectral_auroc_gain_over_directional": spectral_auroc - directional_auroc,
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
    raw_features, residual, _ = _actor_sequences(source)
    coefficient_count = int(config["model"]["dct_coefficient_count"])
    basis = _dct_basis(residual.shape[1], coefficient_count)
    coefficient_target = np.einsum("tk,btd->bkd", basis, residual)
    feature_mean, feature_scale = raw_features.mean(0), raw_features.std(0).clip(min=1e-4)
    target_mean = coefficient_target.mean(0)
    target_scale = coefficient_target.std(0).clip(min=0.05)
    features = torch.from_numpy((raw_features - feature_mean) / feature_scale).cuda()
    target = torch.from_numpy((coefficient_target - target_mean) / target_scale).cuda()
    model_config = config["model"]
    model = SpectralActorGaussian(
        features.shape[1], model_config["hidden_dimensions"], coefficient_count,
    ).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    final_loss = 0.0
    torch.cuda.reset_peak_memory_stats()
    for step in range(int(model_config["steps"])):
        indices = torch.randint(len(features), (int(model_config["batch_size"]),), device="cuda")
        mean, scale = model(features[indices])
        residual_normalized = (target[indices] - mean) / scale
        loss = (0.5 * residual_normalized.square() + torch.log(scale)).sum(dim=(1, 2)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step % 250 == 0 or step + 1 == int(model_config["steps"]):
            print(f"P115 spectral-actor step={step + 1} nll={final_loss:.6f}", flush=True)
    torch.save({
        "feature_mean": feature_mean, "feature_scale": feature_scale,
        "target_mean": target_mean, "target_scale": target_scale,
        "dct_basis": basis, "hidden_dimensions": model_config["hidden_dimensions"],
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
            raw, model, feature_mean, feature_scale, target_mean, target_scale, basis,
            frozen_directional, directional_checkpoint, config,
        )
        print(json.dumps({cohort["name"]: results[cohort["name"]]}, indent=2), flush=True)
    gains = [float(value["spectral_auroc_gain_over_directional"]) for value in results.values()]
    decisions = {
        "no_event_regression_on_either_consumed_cohort": all(
            value["spectral_selected_occupancy_flip_events"]
            <= value["directional_selected_occupancy_flip_events"]
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
        "training": {"actor_sequence_count": int(len(features)),
                     "dct_coefficient_count": coefficient_count,
                     "final_spectral_gaussian_nll": final_loss},
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

