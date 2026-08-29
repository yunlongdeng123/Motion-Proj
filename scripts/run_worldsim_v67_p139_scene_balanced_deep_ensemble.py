"""Train a three-member Actor ensemble with uniform source-scene sampling."""

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

from motion_proj.worldsim_v67.actor_state_reliability import ACTOR_FEATURE_NAMES, spearman_correlation
from scripts.run_worldsim_v67_p86_direct_trajectory_reliability import _select_by_scene
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost
from scripts.run_worldsim_v67_p127_ensemble_continuous_selection import _ensemble_trajectory_score


def _actor_entries_with_scene(
    arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = np.stack((
        arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["actor_id"],
    ), axis=1)
    _, first = np.unique(keys, axis=0, return_index=True)
    actor = np.asarray(arrays["features"], dtype=np.float32)[first, :len(ACTOR_FEATURE_NAMES)]
    residual = np.asarray(
        arrays["actor_position_error_vector_ego_profile_m"], dtype=np.float32,
    )[first]
    fractions = np.linspace(0.0, 1.0, residual.shape[1], dtype=np.float32)
    features = np.concatenate((
        np.broadcast_to(actor[:, None, :], (len(actor), residual.shape[1], actor.shape[1])),
        np.broadcast_to(fractions[None, :, None], (len(actor), residual.shape[1], 1)),
    ), axis=2)
    scenes = np.broadcast_to(
        np.asarray(arrays["scene_index"], dtype=np.int32)[first, None],
        residual.shape[:2],
    )
    return features.reshape(-1, features.shape[-1]), residual.reshape(-1, 2), scenes.reshape(-1)


def _scene_table(scene_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, inverse = np.unique(scene_ids, return_inverse=True)
    counts = np.bincount(inverse)
    table = np.zeros((len(counts), int(counts.max())), dtype=np.int64)
    order = np.argsort(inverse, kind="stable")
    starts = np.concatenate(([0], np.cumsum(counts)))
    for scene in range(len(counts)):
        selected = order[starts[scene]:starts[scene + 1]]
        table[scene, :len(selected)] = selected
    return table, counts.astype(np.int64)


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
    source = dict(np.load(
        args.runs_root / config["source_rows"]["run"] / config["source_rows"]["artifact"],
        allow_pickle=False,
    ))
    raw_features, raw_target, scene_ids = _actor_entries_with_scene(source)
    feature_mean = raw_features.mean(0)
    feature_scale = raw_features.std(0).clip(min=1e-4)
    target_mean = raw_target.mean(0)
    target_scale = raw_target.std(0).clip(min=0.05)
    features = torch.from_numpy((raw_features - feature_mean) / feature_scale).cuda()
    targets = torch.from_numpy((raw_target - target_mean) / target_scale).cuda()
    scene_table_np, scene_counts_np = _scene_table(scene_ids)
    scene_table = torch.from_numpy(scene_table_np).cuda()
    scene_counts = torch.from_numpy(scene_counts_np).cuda()
    model_config = config["model"]
    models = []
    final_losses = {}
    torch.cuda.reset_peak_memory_stats()
    for seed_value in config["member_seeds"]:
        seed_value = int(seed_value)
        torch.manual_seed(seed_value)
        model = DirectionalActorGaussian(features.shape[1], model_config["hidden_dimensions"]).cuda()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(model_config["learning_rate"]),
            weight_decay=float(model_config["weight_decay"]),
        )
        final_loss = 0.0
        batch_size = int(model_config["batch_size"])
        for step in range(int(model_config["steps"])):
            scene = torch.randint(len(scene_table), (batch_size,), device="cuda")
            position = torch.floor(torch.rand(batch_size, device="cuda") * scene_counts[scene]).long()
            index = scene_table[scene, position]
            mean, scale = model(features[index])
            residual = (targets[index] - mean) / scale
            loss = (0.5 * residual.square() + torch.log(scale)).sum(1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
            if step % 500 == 0 or step + 1 == int(model_config["steps"]):
                print(
                    f"P139 scene-balanced seed={seed_value} step={step + 1} nll={final_loss:.6f}",
                    flush=True,
                )
        final_losses[str(seed_value)] = final_loss
        models.append(model.eval())
    torch.save({
        "feature_mean": feature_mean, "feature_scale": feature_scale,
        "target_mean": target_mean, "target_scale": target_scale,
        "hidden_dimensions": model_config["hidden_dimensions"],
        "member_seeds": [int(x) for x in config["member_seeds"]],
        "member_state_dicts": [model.state_dict() for model in models],
    }, run_dir / config["model_artifact"])
    floor = float(config["boundary_state_cost"]["clearance_floor_m"])
    coverage = float(config["selection"]["coverage_fraction"])
    results = {}
    for cohort in config["development_cohorts"]:
        arrays = dict(np.load(
            args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False,
        ))
        score, scenes = _ensemble_trajectory_score(
            arrays, models, feature_mean, feature_scale, target_mean, target_scale,
        )
        actual_cost, cost_scenes = _continuous_cost(arrays, floor)
        if not np.array_equal(scenes, cost_scenes):
            raise RuntimeError("P139 trajectory grouping is not aligned")
        selected = _select_by_scene(score, scenes, coverage)
        reference = config["frozen_p126_comparisons"][cohort["name"]]
        model_spearman = spearman_correlation(actual_cost, score)
        results[cohort["name"]] = {
            "trajectory_count": int(len(actual_cost)),
            "selected_trajectory_count": int(len(selected)),
            "scene_balanced_selected_mean_cost": float(actual_cost[selected].mean()),
            "uniform_token_selected_mean_cost": float(reference["selected_cost"]),
            "scene_balanced_cost_spearman": model_spearman,
            "uniform_token_cost_spearman": float(reference["spearman"]),
            "spearman_gain_over_uniform_token_ensemble": float(
                model_spearman - float(reference["spearman"])
            ),
        }
        print(json.dumps({cohort["name"]: results[cohort["name"]]}, indent=2), flush=True)
    gains = [x["spearman_gain_over_uniform_token_ensemble"] for x in results.values()]
    decisions = {
        "no_selected_cost_regression": all(
            x["scene_balanced_selected_mean_cost"] <= x["uniform_token_selected_mean_cost"]
            for x in results.values()
        ),
        "minimum_mean_spearman_gain": float(np.mean(gains))
        >= float(config["decision"]["minimum_mean_spearman_gain"]),
    }
    verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "training": {"actor_time_tokens": int(len(features)), "source_scenes": int(len(scene_table)),
                     "member_final_scene_balanced_nll": final_losses},
        "development_evaluations": results, "decision_checks": decisions,
        "mean_spearman_gain": float(np.mean(gains)),
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
