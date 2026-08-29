"""Train two additional P109 members and decompose ensemble Actor uncertainty."""

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
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import (
    DirectionalActorGaussian, _actor_entries, _predict,
)


def _evaluate(
    arrays: dict[str, np.ndarray], models: list[DirectionalActorGaussian],
    feature_mean: np.ndarray, feature_scale: np.ndarray,
    target_mean: np.ndarray, target_scale: np.ndarray, config: dict,
) -> dict[str, float | int]:
    actor_features, _, inverse = _actor_entries(arrays)
    point_count = int(arrays["actor_position_error_vector_ego_profile_m"].shape[1])
    member_means, member_scales = [], []
    for model in models:
        mean, scale = _predict(
            model.eval(), actor_features, feature_mean, feature_scale, target_mean, target_scale,
        )
        member_means.append(mean.reshape(-1, point_count, 2)[inverse])
        member_scales.append(scale.reshape(-1, point_count, 2)[inverse])
    means = np.stack(member_means, axis=0)
    scales = np.stack(member_scales, axis=0)
    normal = np.asarray(arrays["query_boundary_normal_ego_profile"], dtype=np.float32)
    projected_member_mean = np.sum(normal[None] * means, axis=3)
    projected_aleatoric_variance = np.sum(np.square(normal[None] * scales), axis=3)
    projected_mean = projected_member_mean.mean(axis=0)
    epistemic_variance = projected_member_mean.var(axis=0)
    aleatoric_variance = projected_aleatoric_variance.mean(axis=0)
    total_variance = np.maximum(aleatoric_variance + epistemic_variance, 1e-8)
    predicted = np.asarray(arrays["predicted_separation_profile_m"], dtype=np.float32)
    radius = np.asarray(arrays["occupancy_interaction_radius_m"], dtype=np.float32)[:, None]
    signed = predicted - radius
    standardized_crossing_margin = (
        np.abs(signed) + np.sign(signed) * projected_mean
    ) / np.sqrt(total_variance)
    query_row_score = np.max(-standardized_crossing_margin, axis=1)
    coordinate_mean = means.mean(axis=0)
    coordinate_variance = np.mean(np.square(scales) + np.square(means), axis=0) - np.square(coordinate_mean)
    actor_row_score = np.max(np.sqrt(np.maximum(coordinate_variance, 1e-8)).sum(axis=2), axis=1)
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
    coverage = float(config["selection"]["coverage_fraction"])
    selected = _select_by_scene(query_score, scenes, coverage)
    actor_selected = _select_by_scene(actor_score, scenes, coverage)
    clearance_selected = _select_by_scene(clearance_score, scenes, coverage)
    frozen_selected = _select_by_scene(frozen_score, scenes, coverage)
    return {
        "row_count": int(len(arrays["features"])),
        "trajectory_count": int(len(events)),
        "all_occupancy_flip_events": int(np.count_nonzero(events)),
        "selected_trajectory_count": int(len(selected)),
        "query_selected_occupancy_flip_events": int(np.count_nonzero(events[selected])),
        "actor_selected_occupancy_flip_events": int(np.count_nonzero(events[actor_selected])),
        "clearance_only_selected_occupancy_flip_events": int(np.count_nonzero(events[clearance_selected])),
        "frozen_p75_selected_occupancy_flip_events": int(np.count_nonzero(events[frozen_selected])),
        "query_event_auroc": binary_auroc(events, query_score),
        "actor_event_auroc": binary_auroc(events, actor_score),
        "clearance_only_event_auroc": binary_auroc(events, clearance_score),
        "mean_projected_epistemic_fraction": float(np.mean(
            epistemic_variance / total_variance,
        )),
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
    source_root = args.runs_root / config["source_rows"]["run"]
    source = dict(np.load(source_root / config["source_rows"]["artifact"], allow_pickle=False))
    raw_features, raw_target, _ = _actor_entries(source)
    frozen = torch.load(
        args.runs_root / config["frozen_p109"]["run"] / config["frozen_p109"]["artifact"],
        map_location="cuda",
    )
    feature_mean = np.asarray(frozen["feature_mean"], dtype=np.float32)
    feature_scale = np.asarray(frozen["feature_scale"], dtype=np.float32)
    target_mean = np.asarray(frozen["target_mean"], dtype=np.float32)
    target_scale = np.asarray(frozen["target_scale"], dtype=np.float32)
    features = torch.from_numpy((raw_features - feature_mean) / feature_scale).cuda()
    target = torch.from_numpy((raw_target - target_mean) / target_scale).cuda()
    model_config = config["model"]
    models: list[DirectionalActorGaussian] = []
    seed0 = DirectionalActorGaussian(features.shape[1], model_config["hidden_dimensions"]).cuda()
    seed0.load_state_dict(frozen["model_state_dict"])
    models.append(seed0.eval())
    final_losses = {}
    torch.cuda.reset_peak_memory_stats()
    for seed in config["new_member_seeds"]:
        seed = int(seed)
        torch.manual_seed(seed)
        model = DirectionalActorGaussian(features.shape[1], model_config["hidden_dimensions"]).cuda()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(model_config["learning_rate"]),
            weight_decay=float(model_config["weight_decay"]),
        )
        final_loss = 0.0
        for step in range(int(model_config["steps"])):
            indices = torch.randint(len(features), (int(model_config["batch_size"]),), device="cuda")
            mean, scale = model(features[indices])
            residual = (target[indices] - mean) / scale
            loss = (0.5 * residual.square() + torch.log(scale)).sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
            if step % 500 == 0 or step + 1 == int(model_config["steps"]):
                print(f"P126 ensemble seed={seed} step={step + 1} nll={final_loss:.6f}", flush=True)
        final_losses[str(seed)] = final_loss
        models.append(model.eval())
    torch.save({
        "feature_mean": feature_mean, "feature_scale": feature_scale,
        "target_mean": target_mean, "target_scale": target_scale,
        "hidden_dimensions": model_config["hidden_dimensions"],
        "member_seeds": [0] + [int(x) for x in config["new_member_seeds"]],
        "member_state_dicts": [model.state_dict() for model in models],
    }, run_dir / config["model_artifact"])
    results = {}
    for cohort in config["development_cohorts"]:
        cohort_root = args.runs_root / cohort.get("run", config["source_rows"]["run"])
        arrays = dict(np.load(cohort_root / cohort["artifact"], allow_pickle=False))
        results[cohort["name"]] = _evaluate(
            arrays, models, feature_mean, feature_scale, target_mean, target_scale, config,
        )
        print(json.dumps({cohort["name"]: results[cohort["name"]]}, indent=2), flush=True)
    comparisons = config["frozen_p109_comparisons"]
    event_noninferior = all(
        results[name]["query_selected_occupancy_flip_events"] <= int(reference["events"])
        for name, reference in comparisons.items()
    )
    gains = [
        float(results[name]["query_event_auroc"]) - float(reference["auroc"])
        for name, reference in comparisons.items()
    ]
    mean_gain = float(np.mean(gains))
    decisions = {
        "event_noninferior_to_p109": event_noninferior,
        "minimum_mean_auroc_gain": mean_gain
        >= float(config["decision"]["minimum_mean_auroc_gain_over_p109"]),
    }
    verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "training": {"actor_time_tokens": int(len(features)), "ensemble_member_count": 3,
                     "reused_seed0_p109": True, "new_member_final_nll": final_losses},
        "development_evaluations": results,
        "decision_metrics": {"decision_checks": decisions,
                             "per_cohort_auroc_gain_over_p109": dict(zip(comparisons, gains)),
                             "mean_auroc_gain_over_p109": mean_gain},
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
