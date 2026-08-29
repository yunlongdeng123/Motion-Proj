"""Distil P126 with the deployed trajectory-max order as the training object."""

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
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost
from scripts.run_worldsim_v67_p131_task_conditioned_score_distillation import (
    BoundaryScoreStudent, _ensemble_row_score, _student_features,
)


def _groups(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = np.stack((
        arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["query_id"],
    ), axis=1)
    identities, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    rows = np.zeros((len(identities), int(counts.max())), dtype=np.int64)
    mask = np.zeros_like(rows, dtype=bool)
    order = np.argsort(inverse, kind="stable")
    starts = np.concatenate(([0], np.cumsum(counts)))
    for group in range(len(identities)):
        selected = order[starts[group]:starts[group + 1]]
        rows[group, :len(selected)] = selected
        mask[group, :len(selected)] = True
    return identities, rows, mask


def _scene_group_table(identities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scenes, inverse = np.unique(identities[:, 0], return_inverse=True)
    counts = np.bincount(inverse)
    table = np.zeros((len(scenes), int(counts.max())), dtype=np.int64)
    order = np.argsort(inverse, kind="stable")
    starts = np.concatenate(([0], np.cumsum(counts)))
    for scene in range(len(scenes)):
        selected = order[starts[scene]:starts[scene + 1]]
        table[scene, :len(selected)] = selected
    return table, counts.astype(np.int64)


@torch.no_grad()
def _trajectory_scores(
    model: BoundaryScoreStudent, features: torch.Tensor,
    group_rows: torch.Tensor, group_mask: torch.Tensor,
) -> torch.Tensor:
    result = []
    for start in range(0, len(group_rows), 8192):
        rows = group_rows[start:start + 8192]
        mask = group_mask[start:start + 8192]
        score = model(features[rows.reshape(-1)]).reshape(rows.shape)
        result.append(score.masked_fill(~mask, -torch.inf).max(1).values.cpu())
    return torch.cat(result)


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
    teacher_row = _ensemble_row_score(
        source, members, actor_feature_mean, actor_feature_scale, target_mean, target_scale,
    )
    identities, group_rows_np, group_mask_np = _groups(source)
    teacher_trajectory = np.where(
        group_mask_np, teacher_row[group_rows_np], -np.inf,
    ).max(1).astype(np.float32)
    feature_mean = raw_features.mean(0)
    feature_scale = raw_features.std(0).clip(min=1e-4)
    features = torch.from_numpy((raw_features - feature_mean) / feature_scale).cuda()
    group_rows = torch.from_numpy(group_rows_np).cuda()
    group_mask = torch.from_numpy(group_mask_np).cuda()
    teacher = torch.from_numpy(teacher_trajectory).cuda()
    scene_table_np, scene_counts_np = _scene_group_table(identities)
    scene_table = torch.from_numpy(scene_table_np).cuda()
    scene_counts = torch.from_numpy(scene_counts_np).cuda()
    model_config = config["model"]
    student = BoundaryScoreStudent(features.shape[1], model_config["hidden_dimensions"]).cuda()
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    pair_batch = int(model_config["pair_batch_size"])
    final_loss = 0.0
    torch.cuda.reset_peak_memory_stats()
    for step in range(int(model_config["steps"])):
        scene = torch.randint(len(scene_table), (pair_batch,), device="cuda")
        count = scene_counts[scene]
        first_position = torch.floor(torch.rand(pair_batch, device="cuda") * count).long()
        offset = 1 + torch.floor(torch.rand(pair_batch, device="cuda") * (count - 1)).long()
        second_position = (first_position + offset) % count
        first_group = scene_table[scene, first_position]
        second_group = scene_table[scene, second_position]
        first_rows, second_rows = group_rows[first_group], group_rows[second_group]
        first_mask, second_mask = group_mask[first_group], group_mask[second_group]
        first_score = student(features[first_rows.reshape(-1)]).reshape(first_rows.shape)
        second_score = student(features[second_rows.reshape(-1)]).reshape(second_rows.shape)
        first_score = first_score.masked_fill(~first_mask, -torch.inf).max(1).values
        second_score = second_score.masked_fill(~second_mask, -torch.inf).max(1).values
        direction = torch.where(teacher[first_group] >= teacher[second_group], 1.0, -1.0)
        loss = torch.nn.functional.softplus(-direction * (first_score - second_score)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step % 250 == 0 or step + 1 == int(model_config["steps"]):
            print(f"P132 trajectory-rank step={step + 1} pairwise_logistic={final_loss:.6f}", flush=True)
    torch.save({
        "feature_mean": feature_mean, "feature_scale": feature_scale,
        "hidden_dimensions": model_config["hidden_dimensions"],
        "model_state_dict": student.state_dict(),
    }, run_dir / config["model_artifact"])
    coverage = float(config["selection"]["coverage_fraction"])
    results = {}
    for cohort in config["development_cohorts"]:
        arrays = dict(np.load(
            args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False,
        ))
        raw = _student_features(arrays, floor)
        cohort_features = torch.from_numpy((raw - feature_mean) / feature_scale).cuda()
        cohort_identities, cohort_rows_np, cohort_mask_np = _groups(arrays)
        score = _trajectory_scores(
            student.eval(), cohort_features, torch.from_numpy(cohort_rows_np).cuda(),
            torch.from_numpy(cohort_mask_np).cuda(),
        ).numpy()
        scenes = cohort_identities[:, 0].astype(np.int32)
        actual_cost, cost_scenes = _continuous_cost(arrays, floor)
        if not np.array_equal(scenes, cost_scenes):
            raise RuntimeError("P132 trajectory grouping is not aligned")
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
        "training": {"source_trajectories": int(len(group_rows)), "final_pairwise_logistic": final_loss},
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
