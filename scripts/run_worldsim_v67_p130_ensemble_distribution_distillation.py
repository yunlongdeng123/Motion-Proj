"""Distil P126 ensemble moments into one correlated Gaussian Actor model."""

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

from motion_proj.worldsim_v67.actor_state_reliability import spearman_correlation
from scripts.run_worldsim_v67_p86_direct_trajectory_reliability import _select_by_scene
from scripts.run_worldsim_v67_p104_temporal_flip_supervision import _aligned_group_max
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import (
    DirectionalActorGaussian, _actor_entries,
)
from scripts.run_worldsim_v67_p117_full_covariance_actor_uncertainty import CorrelatedActorGaussian
from scripts.run_worldsim_v67_p114_monotone_tail_risk import (
    _crossing_probability, _trajectory_tail_features,
)
from scripts.run_worldsim_v67_p119_ranked_range_tail import _head_features
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost


@torch.no_grad()
def _teacher_moments(
    models: list[DirectionalActorGaussian], features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    means, variances = [], []
    for model in models:
        mean, scale = model(features)
        means.append(mean)
        variances.append(scale.square())
    member_mean = torch.stack(means)
    mean = member_mean.mean(0)
    variance = torch.stack(variances).mean(0) + member_mean.var(0, unbiased=False)
    centered = member_mean - mean.unsqueeze(0)
    covariance = (centered[:, :, 0] * centered[:, :, 1]).mean(0)
    correlation = covariance / torch.sqrt(variance[:, 0] * variance[:, 1]).clamp_min(1e-8)
    return mean, variance.clamp_min(1e-6), correlation.clamp(-0.95, 0.95)


def _student_trajectory_score(
    arrays: dict[str, np.ndarray], model: CorrelatedActorGaussian,
    feature_mean: np.ndarray, feature_scale: np.ndarray,
    target_mean: np.ndarray, target_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    actor_features, _, inverse = _actor_entries(arrays)
    point_count = int(arrays["actor_position_error_vector_ego_profile_m"].shape[1])
    outputs = []
    with torch.no_grad():
        for start in range(0, len(actor_features), 65536):
            batch = torch.from_numpy(
                (actor_features[start:start + 65536] - feature_mean) / feature_scale,
            ).cuda()
            outputs.append(tuple(x.cpu().numpy() for x in model(batch)))
    mean = np.concatenate([x[0] for x in outputs]) * target_scale + target_mean
    scale = np.concatenate([x[1] for x in outputs]) * target_scale
    correlation = np.concatenate([x[2] for x in outputs])
    row_mean = mean.reshape(-1, point_count, 2)[inverse]
    row_scale = scale.reshape(-1, point_count, 2)[inverse]
    row_correlation = correlation.reshape(-1, point_count)[inverse]
    normal = np.asarray(arrays["query_boundary_normal_ego_profile"], dtype=np.float32)
    projected_mean = np.sum(normal * row_mean, axis=2)
    nx_sx = normal[:, :, 0] * row_scale[:, :, 0]
    ny_sy = normal[:, :, 1] * row_scale[:, :, 1]
    projected_variance = np.maximum(
        np.square(nx_sx) + np.square(ny_sy) + 2.0 * row_correlation * nx_sx * ny_sy,
        1e-8,
    )
    signed = (
        np.asarray(arrays["predicted_separation_profile_m"], dtype=np.float32)
        - np.asarray(arrays["occupancy_interaction_radius_m"], dtype=np.float32)[:, None]
    )
    row_score = np.max(-(
        np.abs(signed) + np.sign(signed) * projected_mean
    ) / np.sqrt(projected_variance), axis=1)
    keys = np.stack((
        arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["query_id"],
    ), axis=1)
    identities = np.unique(keys, axis=0)
    return _aligned_group_max(keys, row_score, identities), identities[:, 0].astype(np.int32)


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
    ensemble = torch.load(
        args.runs_root / config["frozen_p126"]["run"] / config["frozen_p126"]["artifact"],
        map_location="cuda",
    )
    feature_mean = np.asarray(ensemble["feature_mean"], dtype=np.float32)
    feature_scale = np.asarray(ensemble["feature_scale"], dtype=np.float32)
    target_mean = np.asarray(ensemble["target_mean"], dtype=np.float32)
    target_scale = np.asarray(ensemble["target_scale"], dtype=np.float32)
    models = []
    for state in ensemble["member_state_dicts"]:
        member = DirectionalActorGaussian(20, ensemble["hidden_dimensions"]).cuda()
        member.load_state_dict(state)
        models.append(member.eval())
    source = dict(np.load(
        args.runs_root / config["source_rows"]["run"] / config["source_rows"]["artifact"],
        allow_pickle=False,
    ))
    raw_features, _, _ = _actor_entries(source)
    features = torch.from_numpy((raw_features - feature_mean) / feature_scale).cuda()
    teacher_mean, teacher_variance, teacher_correlation = _teacher_moments(models, features)
    model_config = config["model"]
    student = CorrelatedActorGaussian(20, model_config["hidden_dimensions"]).cuda()
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    final_loss = 0.0
    torch.cuda.reset_peak_memory_stats()
    for step in range(int(model_config["steps"])):
        index = torch.randint(len(features), (int(model_config["batch_size"]),), device="cuda")
        mean, scale, correlation = student(features[index])
        student_variance = scale.square()
        student_covariance = correlation * scale[:, 0] * scale[:, 1]
        determinant = (student_variance[:, 0] * student_variance[:, 1]
                       - student_covariance.square()).clamp_min(1e-8)
        target_variance = teacher_variance[index]
        target_covariance = (
            teacher_correlation[index]
            * torch.sqrt(target_variance[:, 0] * target_variance[:, 1])
        )
        target_determinant = (
            target_variance[:, 0] * target_variance[:, 1] - target_covariance.square()
        ).clamp_min(1e-8)
        delta = mean - teacher_mean[index]
        trace = (
            student_variance[:, 1] * target_variance[:, 0]
            + student_variance[:, 0] * target_variance[:, 1]
            - 2.0 * student_covariance * target_covariance
        ) / determinant
        quadratic = (
            student_variance[:, 1] * delta[:, 0].square()
            + student_variance[:, 0] * delta[:, 1].square()
            - 2.0 * student_covariance * delta[:, 0] * delta[:, 1]
        ) / determinant
        loss = 0.5 * (trace + quadratic + torch.log(determinant / target_determinant) - 2.0).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step % 250 == 0 or step + 1 == int(model_config["steps"]):
            print(f"P130 distillation step={step + 1} gaussian_kl={final_loss:.6f}", flush=True)
    torch.save({
        "feature_mean": feature_mean, "feature_scale": feature_scale,
        "target_mean": target_mean, "target_scale": target_scale,
        "hidden_dimensions": model_config["hidden_dimensions"],
        "model_state_dict": student.state_dict(),
    }, run_dir / config["model_artifact"])
    p109 = torch.load(
        args.runs_root / config["frozen_p109"]["run"] / config["frozen_p109"]["artifact"],
        map_location="cuda",
    )
    p109_model = DirectionalActorGaussian(20, p109["hidden_dimensions"]).cuda()
    p109_model.load_state_dict(p109["model_state_dict"])
    p109_model.eval()
    floor = float(config["boundary_state_cost"]["clearance_floor_m"])
    coverage = float(config["selection"]["coverage_fraction"])
    top_k = int(config["score"]["top_k_crossing_probabilities"])
    results = {}
    for cohort in config["development_cohorts"]:
        arrays = dict(np.load(
            args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False,
        ))
        score, scenes = _student_trajectory_score(
            arrays, student.eval(), feature_mean, feature_scale, target_mean, target_scale,
        )
        p109_probability, _ = _crossing_probability(
            arrays, p109_model, np.asarray(p109["feature_mean"], dtype=np.float32),
            np.asarray(p109["feature_scale"], dtype=np.float32),
            np.asarray(p109["target_mean"], dtype=np.float32),
            np.asarray(p109["target_scale"], dtype=np.float32),
        )
        grouped = _trajectory_tail_features(arrays, p109_probability, top_k)
        _, p109_score = _head_features(grouped)
        actual_cost, cost_scenes = _continuous_cost(arrays, floor)
        if not (np.array_equal(scenes, cost_scenes) and np.array_equal(scenes, grouped["scene_index"])):
            raise RuntimeError("P130 trajectory grouping is not aligned")
        selected = _select_by_scene(score, scenes, coverage)
        p109_selected = _select_by_scene(p109_score, scenes, coverage)
        reference = config["frozen_p126_comparisons"][cohort["name"]]
        student_spearman = spearman_correlation(actual_cost, score)
        results[cohort["name"]] = {
            "trajectory_count": int(len(actual_cost)),
            "selected_trajectory_count": int(len(selected)),
            "student_selected_mean_cost": float(actual_cost[selected].mean()),
            "ensemble_selected_mean_cost": float(reference["selected_cost"]),
            "p109_selected_mean_cost": float(actual_cost[p109_selected].mean()),
            "student_cost_spearman": student_spearman,
            "ensemble_cost_spearman": float(reference["spearman"]),
            "student_spearman_difference_from_ensemble": float(
                student_spearman - float(reference["spearman"])
            ),
        }
        print(json.dumps({cohort["name"]: results[cohort["name"]]}, indent=2), flush=True)
    differences = [x["student_spearman_difference_from_ensemble"] for x in results.values()]
    decisions = {
        "no_selected_cost_regression_from_ensemble": all(
            x["student_selected_mean_cost"] <= x["ensemble_selected_mean_cost"]
            for x in results.values()
        ),
        "mean_spearman_retention": float(np.mean(differences))
        >= float(config["decision"]["minimum_mean_spearman_difference_from_ensemble"]),
    }
    verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "training": {"actor_time_tokens": int(len(features)), "final_teacher_student_gaussian_kl": final_loss},
        "development_evaluations": results, "decision_checks": decisions,
        "mean_spearman_difference_from_ensemble": float(np.mean(differences)),
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
