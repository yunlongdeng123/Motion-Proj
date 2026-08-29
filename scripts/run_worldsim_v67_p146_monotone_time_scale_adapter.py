"""Fit monotone absolute-time scale adapters on the frozen P126 ensemble."""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
import yaml

from motion_proj.worldsim_v67.actor_state_reliability import spearman_correlation
from scripts.run_worldsim_v67_p86_direct_trajectory_reliability import _select_by_scene
from scripts.run_worldsim_v67_p104_temporal_flip_supervision import _aligned_group_max
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import (
    DirectionalActorGaussian, _predict,
)
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost
from scripts.run_worldsim_v67_p145_absolute_time_actor_ensemble import (
    _absolute_time_actor_entries,
)


def _trajectory_score(
    arrays: dict[str, np.ndarray], models: list[DirectionalActorGaussian],
    adapter_biases: list[np.ndarray], adapter_raw_slopes: list[np.ndarray],
    feature_mean: np.ndarray, feature_scale: np.ndarray,
    target_mean: np.ndarray, target_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    absolute_features, _, inverse = _absolute_time_actor_entries(arrays)
    base_features = absolute_features[:, :-1]
    absolute_time = absolute_features[:, -1:]
    point_count = int(arrays["actor_position_error_vector_ego_profile_m"].shape[1])
    member_means, member_scales = [], []
    for model, bias, raw_slope in zip(models, adapter_biases, adapter_raw_slopes):
        mean, scale = _predict(
            model, base_features, feature_mean, feature_scale, target_mean, target_scale,
        )
        slope = np.log1p(np.exp(raw_slope)).astype(np.float32)
        multiplier = np.exp(bias[None] + absolute_time * slope[None])
        member_means.append(mean.reshape(-1, point_count, 2)[inverse])
        member_scales.append((scale * multiplier).reshape(-1, point_count, 2)[inverse])
    means = np.stack(member_means, axis=0)
    scales = np.stack(member_scales, axis=0)
    normal = np.asarray(arrays["query_boundary_normal_ego_profile"], dtype=np.float32)
    projected_member_mean = np.sum(normal[None] * means, axis=3)
    projected_mean = projected_member_mean.mean(axis=0)
    epistemic_variance = projected_member_mean.var(axis=0)
    aleatoric_variance = np.mean(np.sum(np.square(normal[None] * scales), axis=3), axis=0)
    total_variance = np.maximum(epistemic_variance + aleatoric_variance, 1e-8)
    predicted = np.asarray(arrays["predicted_separation_profile_m"], dtype=np.float32)
    radius = np.asarray(arrays["occupancy_interaction_radius_m"], dtype=np.float32)[:, None]
    signed = predicted - radius
    row_score = np.max(-(
        np.abs(signed) + np.sign(signed) * projected_mean
    ) / np.sqrt(total_variance), axis=1)
    keys = np.stack((
        arrays["scene_index"], np.rint(arrays["horizon_seconds"] * 10).astype(np.int32),
        arrays["anchor_frame"], arrays["query_id"],
    ), axis=1)
    identities = np.unique(keys, axis=0)
    return (
        _aligned_group_max(keys, row_score, identities),
        identities[:, 0].astype(np.int32),
        float(np.mean(epistemic_variance / total_variance)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_dir = args.runs_root / "worldsim_v67" / config["task_id"] / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8",
    )
    started = time.monotonic()
    frozen = torch.load(
        args.runs_root / config["frozen_p126"]["run"] / config["frozen_p126"]["artifact"],
        map_location="cuda",
    )
    feature_mean = np.asarray(frozen["feature_mean"], dtype=np.float32)
    feature_scale = np.asarray(frozen["feature_scale"], dtype=np.float32)
    target_mean = np.asarray(frozen["target_mean"], dtype=np.float32)
    target_scale = np.asarray(frozen["target_scale"], dtype=np.float32)
    source = dict(np.load(
        args.runs_root / config["source_rows"]["run"] / config["source_rows"]["artifact"],
        allow_pickle=False,
    ))
    absolute_features, raw_target, _ = _absolute_time_actor_entries(source)
    features = torch.from_numpy(
        (absolute_features[:, :-1] - feature_mean) / feature_scale,
    ).cuda()
    absolute_time = torch.from_numpy(absolute_features[:, -1:]).cuda()
    targets = torch.from_numpy((raw_target - target_mean) / target_scale).cuda()
    models = []
    biases = []
    raw_slopes = []
    final_losses = {}
    adapter_config = config["adapter"]
    torch.cuda.reset_peak_memory_stats()
    for seed_value, state_dict in zip(frozen["member_seeds"], frozen["member_state_dicts"]):
        model = DirectionalActorGaussian(features.shape[1], frozen["hidden_dimensions"]).cuda()
        model.load_state_dict(state_dict)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        bias = torch.nn.Parameter(torch.zeros(2, device="cuda"))
        raw_slope = torch.nn.Parameter(torch.full(
            (2,), float(adapter_config["initial_raw_slope"]), device="cuda",
        ))
        optimizer = torch.optim.AdamW(
            [bias, raw_slope], lr=float(adapter_config["learning_rate"]),
            weight_decay=0.0,
        )
        final_loss = 0.0
        for step in range(int(adapter_config["steps"])):
            index = torch.randint(
                len(features), (int(adapter_config["batch_size"]),), device="cuda",
            )
            with torch.no_grad():
                mean, base_scale = model(features[index])
            multiplier = torch.exp(bias[None] + absolute_time[index] * functional.softplus(raw_slope)[None])
            scale = base_scale * multiplier
            residual = (targets[index] - mean) / scale
            loss = (0.5 * residual.square() + torch.log(scale)).sum(1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
            if step % 250 == 0 or step + 1 == int(adapter_config["steps"]):
                print(
                    f"P146 time-scale seed={seed_value} step={step + 1} nll={final_loss:.6f} "
                    f"bias={bias.detach().cpu().tolist()} slope={functional.softplus(raw_slope).detach().cpu().tolist()}",
                    flush=True,
                )
        models.append(model)
        biases.append(bias.detach().cpu().numpy().astype(np.float32))
        raw_slopes.append(raw_slope.detach().cpu().numpy().astype(np.float32))
        final_losses[str(seed_value)] = final_loss
    torch.save({
        "frozen_base": config["frozen_p126"],
        "member_seeds": [int(x) for x in frozen["member_seeds"]],
        "adapter_biases": biases, "adapter_raw_slopes": raw_slopes,
    }, run_dir / config["model_artifact"])
    floor = float(config["boundary_state_cost"]["clearance_floor_m"])
    coverage = float(config["selection"]["coverage_fraction"])
    results = {}
    for cohort in config["development_cohorts"]:
        arrays = dict(np.load(
            args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False,
        ))
        score, scenes, epistemic_fraction = _trajectory_score(
            arrays, models, biases, raw_slopes,
            feature_mean, feature_scale, target_mean, target_scale,
        )
        actual_cost, cost_scenes = _continuous_cost(arrays, floor)
        if not np.array_equal(scenes, cost_scenes):
            raise RuntimeError("P146 trajectory grouping is not aligned")
        selected = _select_by_scene(score, scenes, coverage)
        reference = config["frozen_p126_comparisons"][cohort["name"]]
        model_spearman = spearman_correlation(actual_cost, score)
        results[cohort["name"]] = {
            "trajectory_count": int(len(actual_cost)),
            "selected_trajectory_count": int(len(selected)),
            "adapted_selected_mean_cost": float(actual_cost[selected].mean()),
            "p126_selected_mean_cost": float(reference["selected_cost"]),
            "adapted_cost_spearman": model_spearman,
            "p126_cost_spearman": float(reference["spearman"]),
            "spearman_gain_over_p126": float(model_spearman - float(reference["spearman"])),
            "mean_projected_epistemic_fraction": epistemic_fraction,
        }
        print(json.dumps({cohort["name"]: results[cohort["name"]]}, indent=2), flush=True)
    gains = [x["spearman_gain_over_p126"] for x in results.values()]
    decisions = {
        "no_selected_cost_regression": all(
            x["adapted_selected_mean_cost"] <= x["p126_selected_mean_cost"]
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
        "training": {"actor_time_tokens": int(len(features)),
                     "member_final_nll": final_losses,
                     "adapter_biases": [x.tolist() for x in biases],
                     "adapter_positive_slopes": [np.log1p(np.exp(x)).tolist() for x in raw_slopes]},
        "development_evaluations": results, "decision_checks": decisions,
        "mean_spearman_gain": float(np.mean(gains)),
        "resources": {"gpu": torch.cuda.get_device_name(0),
                      "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                      "wall_seconds": time.monotonic() - started},
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    (run_dir / "status.json").write_text(json.dumps({
        "status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict,
                      "decision_checks": decisions}, indent=2), flush=True)


if __name__ == "__main__":
    main()
