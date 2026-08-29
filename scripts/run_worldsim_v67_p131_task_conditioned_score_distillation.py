"""Functionally distil the P126 ensemble boundary score into one query model."""

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
    DirectionalActorGaussian, _actor_entries, _predict,
)
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost


class BoundaryScoreStudent(torch.nn.Module):
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
        return self.network(features).squeeze(1)


def _student_features(arrays: dict[str, np.ndarray], floor: float) -> np.ndarray:
    signed = (
        np.asarray(arrays["predicted_separation_profile_m"], dtype=np.float32)
        - np.asarray(arrays["occupancy_interaction_radius_m"], dtype=np.float32)[:, None]
    )
    return np.concatenate((
        np.asarray(arrays["features"], dtype=np.float32),
        np.sign(signed) * np.log1p(np.abs(signed) / floor),
        np.asarray(arrays["query_boundary_normal_ego_profile"], dtype=np.float32).reshape(len(signed), -1),
    ), axis=1)


def _ensemble_row_score(
    arrays: dict[str, np.ndarray], models: list[DirectionalActorGaussian],
    feature_mean: np.ndarray, feature_scale: np.ndarray,
    target_mean: np.ndarray, target_scale: np.ndarray,
) -> np.ndarray:
    actor_features, _, inverse = _actor_entries(arrays)
    point_count = int(arrays["actor_position_error_vector_ego_profile_m"].shape[1])
    means, scales = [], []
    for model in models:
        mean, scale = _predict(
            model.eval(), actor_features, feature_mean, feature_scale, target_mean, target_scale,
        )
        means.append(mean.reshape(-1, point_count, 2)[inverse])
        scales.append(scale.reshape(-1, point_count, 2)[inverse])
    means = np.stack(means)
    scales = np.stack(scales)
    normal = np.asarray(arrays["query_boundary_normal_ego_profile"], dtype=np.float32)
    projected_member_mean = np.sum(normal[None] * means, axis=3)
    projected_mean = projected_member_mean.mean(axis=0)
    total_variance = np.maximum(
        np.mean(np.sum(np.square(normal[None] * scales), axis=3), axis=0)
        + projected_member_mean.var(axis=0),
        1e-8,
    )
    signed = (
        np.asarray(arrays["predicted_separation_profile_m"], dtype=np.float32)
        - np.asarray(arrays["occupancy_interaction_radius_m"], dtype=np.float32)[:, None]
    )
    return np.max(-(
        np.abs(signed) + np.sign(signed) * projected_mean
    ) / np.sqrt(total_variance), axis=1).astype(np.float32)


@torch.no_grad()
def _predict_student(
    model: BoundaryScoreStudent, features: np.ndarray,
    feature_mean: np.ndarray, feature_scale: np.ndarray,
    score_mean: float, score_scale: float,
) -> np.ndarray:
    result = []
    for start in range(0, len(features), 65536):
        batch = torch.from_numpy(
            (features[start:start + 65536] - feature_mean) / feature_scale,
        ).cuda()
        result.append(model(batch).cpu().numpy())
    return np.concatenate(result) * score_scale + score_mean


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
    actor_feature_mean = np.asarray(ensemble["feature_mean"], dtype=np.float32)
    actor_feature_scale = np.asarray(ensemble["feature_scale"], dtype=np.float32)
    target_mean = np.asarray(ensemble["target_mean"], dtype=np.float32)
    target_scale = np.asarray(ensemble["target_scale"], dtype=np.float32)
    members = []
    for state in ensemble["member_state_dicts"]:
        member = DirectionalActorGaussian(20, ensemble["hidden_dimensions"]).cuda()
        member.load_state_dict(state)
        members.append(member.eval())
    source = dict(np.load(
        args.runs_root / config["source_rows"]["run"] / config["source_rows"]["artifact"],
        allow_pickle=False,
    ))
    floor = float(config["boundary_state_cost"]["clearance_floor_m"])
    raw_features = _student_features(source, floor)
    teacher_score = _ensemble_row_score(
        source, members, actor_feature_mean, actor_feature_scale, target_mean, target_scale,
    )
    feature_mean = raw_features.mean(0)
    feature_scale = raw_features.std(0).clip(min=1e-4)
    score_mean = float(teacher_score.mean())
    score_scale = float(max(teacher_score.std(), 1e-4))
    features = torch.from_numpy((raw_features - feature_mean) / feature_scale).cuda()
    targets = torch.from_numpy((teacher_score - score_mean) / score_scale).cuda()
    model_config = config["model"]
    student = BoundaryScoreStudent(features.shape[1], model_config["hidden_dimensions"]).cuda()
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    final_loss = 0.0
    torch.cuda.reset_peak_memory_stats()
    for step in range(int(model_config["steps"])):
        index = torch.randint(len(features), (int(model_config["batch_size"]),), device="cuda")
        predicted = student(features[index])
        loss = torch.nn.functional.smooth_l1_loss(predicted, targets[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step % 250 == 0 or step + 1 == int(model_config["steps"]):
            print(f"P131 functional distillation step={step + 1} smooth_l1={final_loss:.6f}", flush=True)
    torch.save({
        "feature_mean": feature_mean, "feature_scale": feature_scale,
        "score_mean": score_mean, "score_scale": score_scale,
        "hidden_dimensions": model_config["hidden_dimensions"],
        "model_state_dict": student.state_dict(),
    }, run_dir / config["model_artifact"])
    coverage = float(config["selection"]["coverage_fraction"])
    results = {}
    for cohort in config["development_cohorts"]:
        arrays = dict(np.load(
            args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False,
        ))
        row_score = _predict_student(
            student.eval(), _student_features(arrays, floor), feature_mean, feature_scale,
            score_mean, score_scale,
        )
        keys = np.stack((
            arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
            arrays["anchor_frame"], arrays["query_id"],
        ), axis=1)
        identities = np.unique(keys, axis=0)
        score = _aligned_group_max(keys, row_score, identities)
        scenes = identities[:, 0].astype(np.int32)
        actual_cost, cost_scenes = _continuous_cost(arrays, floor)
        if not np.array_equal(scenes, cost_scenes):
            raise RuntimeError("P131 trajectory grouping is not aligned")
        selected = _select_by_scene(score, scenes, coverage)
        reference = config["frozen_p126_comparisons"][cohort["name"]]
        student_spearman = spearman_correlation(actual_cost, score)
        results[cohort["name"]] = {
            "trajectory_count": int(len(actual_cost)),
            "selected_trajectory_count": int(len(selected)),
            "student_selected_mean_cost": float(actual_cost[selected].mean()),
            "ensemble_selected_mean_cost": float(reference["selected_cost"]),
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
        "training": {"source_rows": int(len(features)), "final_teacher_score_smooth_l1": final_loss},
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
