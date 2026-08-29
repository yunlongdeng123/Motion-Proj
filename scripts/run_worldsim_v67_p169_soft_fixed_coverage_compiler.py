"""Train a P126-anchored compiler on a soft scene-wise fixed-coverage objective."""

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
from scripts.run_worldsim_v67_p144_trajectory_set_rank_compiler import (
    TrajectorySetResidual, _build_sets, _p126_row_score, _predict, _row_features, _scene_table,
)
from scripts.run_worldsim_v67_p158_crps_actor_ensemble import _subset


def _materialize(
    arrays: dict[str, np.ndarray], base_models: list[DirectionalActorGaussian], frozen: dict,
    maximum_rows: int, floor: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    row_score = _p126_row_score(
        arrays, base_models,
        np.asarray(frozen["feature_mean"], dtype=np.float32),
        np.asarray(frozen["feature_scale"], dtype=np.float32),
        np.asarray(frozen["target_mean"], dtype=np.float32),
        np.asarray(frozen["target_scale"], dtype=np.float32),
    )
    grouped = _build_sets(arrays, row_score, maximum_rows)
    cost, scenes = _continuous_cost(arrays, floor)
    if not np.array_equal(grouped["scene_index"], scenes):
        raise RuntimeError("P169 trajectory grouping is not aligned")
    return grouped, cost, scenes, row_score


@torch.no_grad()
def _evaluate(
    arrays: dict[str, np.ndarray], base_models: list[DirectionalActorGaussian], frozen: dict,
    model: TrajectorySetResidual, token_mean: np.ndarray, token_scale: np.ndarray,
    config: dict, frozen_reference: dict | None = None,
) -> dict[str, float | int]:
    maximum_rows = int(config["model"]["maximum_actor_query_rows"])
    floor = float(config["boundary_state_cost"]["clearance_floor_m"])
    grouped, actual_cost, scenes, _ = _materialize(
        arrays, base_models, frozen, maximum_rows, floor,
    )
    normalized = (grouped["sets"] - token_mean) / token_scale
    normalized[~grouped["mask"]] = 0.0
    score = _predict(
        model, torch.from_numpy(normalized).cuda(), torch.from_numpy(grouped["mask"]).cuda(),
        torch.from_numpy(grouped["base_score"]).cuda(),
    )
    coverage = float(config["selection"]["coverage_fraction"])
    selected = _select_by_scene(score, scenes, coverage)
    base_selected = _select_by_scene(grouped["base_score"], scenes, coverage)
    rank = spearman_correlation(actual_cost, score)
    base_rank = spearman_correlation(actual_cost, grouped["base_score"])
    base_cost = float(actual_cost[base_selected].mean())
    if frozen_reference is not None:
        base_rank = float(frozen_reference["spearman"])
        base_cost = float(frozen_reference["selected_cost"])
    learned_cost = float(actual_cost[selected].mean())
    return {
        "trajectory_count": int(len(actual_cost)), "selected_trajectory_count": int(len(selected)),
        "soft_coverage_selected_mean_cost": learned_cost, "p126_selected_mean_cost": base_cost,
        "soft_coverage_minus_p126_selected_cost": learned_cost - base_cost,
        "soft_coverage_cost_spearman": rank, "p126_cost_spearman": base_rank,
        "spearman_gain_over_p126": rank - base_rank,
    }


def _checks(results: dict[str, dict[str, float | int]], minimum_gain: float) -> dict[str, bool]:
    return {
        "no_selected_cost_regression": all(
            float(row["soft_coverage_minus_p126_selected_cost"]) <= 0.0 for row in results.values()
        ),
        "minimum_mean_spearman_gain": float(np.mean([
            float(row["spearman_gain_over_p126"]) for row in results.values()
        ])) >= minimum_gain,
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
    frozen = torch.load(
        args.runs_root / config["frozen_p126"]["run"] / config["frozen_p126"]["artifact"],
        map_location="cuda",
    )
    base_models = []
    for state in frozen["member_state_dicts"]:
        base = DirectionalActorGaussian(20, frozen["hidden_dimensions"]).cuda()
        base.load_state_dict(state)
        base_models.append(base.eval())
    source = dict(np.load(
        args.runs_root / config["source_rows"]["run"] / config["source_rows"]["artifact"],
        allow_pickle=False,
    ))
    model_config = config["model"]
    maximum_rows = int(model_config["maximum_actor_query_rows"])
    floor = float(config["boundary_state_cost"]["clearance_floor_m"])
    source_grouped, source_cost, source_scenes, source_row_score = _materialize(
        source, base_models, frozen, maximum_rows, floor,
    )
    all_tokens = _row_features(source, source_row_score)
    token_mean = all_tokens.mean(axis=0)
    token_scale = all_tokens.std(axis=0).clip(min=1e-4)
    normalized = (source_grouped["sets"] - token_mean) / token_scale
    normalized[~source_grouped["mask"]] = 0.0
    features = torch.from_numpy(normalized).cuda()
    mask = torch.from_numpy(source_grouped["mask"]).cuda()
    base_score = torch.from_numpy(source_grouped["base_score"]).cuda()
    costs = torch.from_numpy(source_cost).cuda()
    scene_table_np, scene_counts_np = _scene_table(source_scenes)
    scene_table = torch.from_numpy(scene_table_np).cuda()
    scene_counts = torch.from_numpy(scene_counts_np).cuda()
    torch.manual_seed(int(config["seed"]))
    model = TrajectorySetResidual(
        features.shape[2], model_config["element_dimensions"], model_config["decoder_dimensions"],
        float(model_config["residual_bound"]),
    ).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    batch_scenes = int(model_config["batch_scenes"])
    list_size = int(model_config["list_size"])
    temperature = float(model_config["soft_cutoff_temperature"])
    penalty_weight = float(model_config["residual_regularization"])
    steps = int(model_config["steps"])
    final_loss = 0.0
    torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        scene = torch.randint(len(scene_table), (batch_scenes,), device="cuda")
        positions = torch.floor(
            torch.rand((batch_scenes, list_size), device="cuda") * scene_counts[scene, None]
        ).long()
        indices = scene_table[scene[:, None], positions]
        flat = indices.reshape(-1)
        score, residual = model(features[flat], mask[flat], base_score[flat])
        score = score.reshape(batch_scenes, list_size)
        residual = residual.reshape(batch_scenes, list_size)
        batch_cost = costs[indices]
        cutoff = score.detach().median(dim=1, keepdim=True).values
        scale = (score.detach() - cutoff).abs().median(dim=1, keepdim=True).values.clamp(min=1e-3)
        weight = torch.sigmoid(-(score - cutoff) / (temperature * scale))
        selected_cost = (weight * batch_cost).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)
        loss = selected_cost.mean() + penalty_weight * residual.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step % 500 == 0 or step + 1 == steps:
            print(f"P169 soft-fixed50 step={step + 1} loss={final_loss:.6f}", flush=True)
    torch.save({
        "token_mean": token_mean, "token_scale": token_scale,
        "maximum_actor_query_rows": maximum_rows, "model_state_dict": model.state_dict(),
        "element_dimensions": model_config["element_dimensions"],
        "decoder_dimensions": model_config["decoder_dimensions"],
        "residual_bound": float(model_config["residual_bound"]),
    }, run_dir / config["model_artifact"])
    development = {}
    for cohort in config["decision_cohorts"]:
        arrays = dict(np.load(args.runs_root / cohort["run"] / cohort["artifact"], allow_pickle=False))
        development[cohort["name"]] = _evaluate(
            arrays, base_models, frozen, model.eval(), token_mean, token_scale, config,
            config["frozen_p126_comparisons"][cohort["name"]],
        )
        print(json.dumps({cohort["name"]: development[cohort["name"]]}, indent=2), flush=True)
    minimum_gain = float(config["decision"]["minimum_mean_spearman_gain_over_p126"])
    development_checks = _checks(development, minimum_gain)
    (run_dir / "development.json").write_text(json.dumps({
        "evaluations": development, "decision_checks": development_checks,
    }, indent=2) + "\n", encoding="utf-8")
    prospective = {}
    prospective_checks: dict[str, bool] = {}
    if all(development_checks.values()):
        spec = config["prospective_p167"]
        rows_path = args.runs_root / spec["run"] / spec["artifact"]
        deadline = time.monotonic() + float(spec["readiness_timeout_seconds"])
        while not rows_path.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"P169 prospective rows not ready: {rows_path}")
            time.sleep(5.0)
        arrays = dict(np.load(rows_path, allow_pickle=False))
        for horizon in spec["horizons_seconds"]:
            key = str(float(horizon))
            prospective[key] = _evaluate(
                _subset(arrays, float(horizon)), base_models, frozen, model.eval(),
                token_mean, token_scale, config,
            )
            print(json.dumps({f"P167_H{key}": prospective[key]}, indent=2), flush=True)
        prospective_checks = _checks(prospective, minimum_gain)
    passed = all(development_checks.values()) and bool(prospective) and all(prospective_checks.values())
    verdict = config["verdict_on_pass"] if passed else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "training": {"source_trajectory_count": int(len(source_cost)), "final_soft_fixed50_loss": final_loss},
        "consumed_development_evaluations": development,
        "development_decision_checks": development_checks,
        "prospective_p167_evaluations": prospective, "prospective_decision_checks": prospective_checks,
        "resources": {
            "gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "wall_seconds": time.monotonic() - started,
        },
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({
        "status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict}, indent=2), flush=True)


if __name__ == "__main__":
    main()
