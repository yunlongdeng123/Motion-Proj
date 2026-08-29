"""Confirm the frozen P173 visited-state reliability CDF as scenes become ready."""

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

from motion_proj.worldsim_v67.actor_state_reliability import materialize_actor_query_rows
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost
from scripts.run_worldsim_v67_p127_ensemble_continuous_selection import _ensemble_trajectory_score
from scripts.run_worldsim_v67_p166_monotone_expected_cost_calibration import _trajectory_horizon
from scripts.run_worldsim_v67_p173_monotone_visit_reliability_cdf import (
    HorizonOnlyReliabilityCDF,
    MonotoneReliabilityCDF,
    _predict_surface,
)


def _slice(arrays: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.asarray(value)[mask] for name, value in arrays.items()}


def _score_payload(
    arrays: dict[str, np.ndarray],
    models: list[DirectionalActorGaussian],
    ensemble: dict,
    model: MonotoneReliabilityCDF,
    baseline: HorizonOnlyReliabilityCDF,
    frozen_cdf: dict,
    config: dict,
) -> dict[str, np.ndarray]:
    score, scenes = _ensemble_trajectory_score(
        arrays, models,
        np.asarray(ensemble["feature_mean"], dtype=np.float32),
        np.asarray(ensemble["feature_scale"], dtype=np.float32),
        np.asarray(ensemble["target_mean"], dtype=np.float32),
        np.asarray(ensemble["target_scale"], dtype=np.float32),
    )
    actual_cost, cost_scenes = _continuous_cost(
        arrays, float(config["boundary_state_cost"]["clearance_floor_m"]),
    )
    if not np.array_equal(scenes, cost_scenes):
        raise RuntimeError("P175 trajectory grouping is not aligned")
    horizon = _trajectory_horizon(arrays)
    budgets = np.asarray(config["reliability_budgets"], dtype=np.float32)
    predicted, control = _predict_surface(
        model, baseline, score, horizon, budgets, tuple(frozen_cdf["norms"]),
    )
    target = actual_cost[:, None] <= budgets[None]
    return {
        "predicted": predicted.astype(np.float32),
        "control": control.astype(np.float32),
        "target": target.astype(np.bool_),
        "actual_cost": actual_cost.astype(np.float32),
        "scene_index": scenes.astype(np.int32),
    }


def _summarize(payload: dict[str, np.ndarray], budgets: np.ndarray) -> dict:
    predicted = payload["predicted"]
    control = payload["control"]
    target = payload["target"]
    brier = float(np.mean(np.square(predicted - target)))
    control_brier = float(np.mean(np.square(control - target)))
    calibration_error = float(np.mean(np.abs(predicted.mean(axis=0) - target.mean(axis=0))))
    control_calibration_error = float(
        np.mean(np.abs(control.mean(axis=0) - target.mean(axis=0)))
    )
    return {
        "trajectory_count": int(len(target)),
        "integrated_brier": brier,
        "horizon_only_integrated_brier": control_brier,
        "integrated_brier_reduction": float(
            (control_brier - brier) / max(control_brier, 1e-12)
        ),
        "mean_absolute_reliability_error": calibration_error,
        "horizon_only_mean_absolute_reliability_error": control_calibration_error,
        "per_budget": {
            str(float(budget)): {
                "empirical_reliability": float(target[:, index].mean()),
                "predicted_reliability": float(predicted[:, index].mean()),
                "horizon_only_predicted_reliability": float(control[:, index].mean()),
                "brier": float(np.mean(np.square(predicted[:, index] - target[:, index]))),
                "horizon_only_brier": float(
                    np.mean(np.square(control[:, index] - target[:, index]))
                ),
            }
            for index, budget in enumerate(budgets)
        },
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
    data = config["evaluation_data"]
    budgets = np.asarray(config["reliability_budgets"], dtype=np.float32)

    ensemble = torch.load(
        args.runs_root / config["frozen_p126"]["run"] / config["frozen_p126"]["artifact"],
        map_location="cuda",
    )
    models = []
    for state in ensemble["member_state_dicts"]:
        member = DirectionalActorGaussian(20, ensemble["hidden_dimensions"]).cuda()
        member.load_state_dict(state)
        models.append(member.eval())
    frozen_cdf = torch.load(
        args.runs_root / config["frozen_p173"]["run"] / config["frozen_p173"]["artifact"],
        map_location="cuda",
    )
    model = MonotoneReliabilityCDF(
        [float(value) for value in frozen_cdf["score_knots"]],
        [float(value) for value in frozen_cdf["budget_knots"]],
    ).cuda()
    model.load_state_dict(frozen_cdf["model_state_dict"])
    model.eval()
    baseline = HorizonOnlyReliabilityCDF(
        [float(value) for value in frozen_cdf["budget_knots"]],
    ).cuda()
    baseline.load_state_dict(frozen_cdf["baseline_state_dict"])
    baseline.eval()

    metadata = Path(data["metadata_root"]) / "v1.0-trainval"
    scenes = json.loads((metadata / "scene.json").read_text(encoding="utf-8"))
    index_by_name = {str(row["name"]): index for index, row in enumerate(scenes)}
    pending = {
        str(name): Path(data["processed_root"]) / f"{index_by_name[str(name)]:03d}"
        for name in data["scene_names"]
    }
    deadline = time.monotonic() + float(data["readiness_timeout_seconds"])
    row_parts = []
    horizon_payloads = {str(value): [] for value in data["horizons_seconds"]}
    ready_sequence = []
    torch.cuda.reset_peak_memory_stats()
    while pending:
        ready = [
            name for name, scene in pending.items()
            if (scene / "instances" / "instances_info.json").is_file()
            and (scene / "lidar_pose").is_dir()
        ]
        if not ready:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"P175 processed scenes not ready: {sorted(pending)}")
            time.sleep(5.0)
            continue
        for name in ready:
            scene = pending.pop(name)
            scene_started = time.monotonic()
            arrays = materialize_actor_query_rows([scene], data["horizons_seconds"], data)
            row_parts.append(arrays)
            scene_metrics = {}
            for horizon in data["horizons_seconds"]:
                key = str(horizon)
                payload = _score_payload(
                    _slice(arrays, np.isclose(arrays["horizon_seconds"], float(horizon))),
                    models, ensemble, model, baseline, frozen_cdf, config,
                )
                horizon_payloads[key].append(payload)
                local = _summarize(payload, budgets)
                scene_metrics[key] = {
                    "trajectory_count": local["trajectory_count"],
                    "integrated_brier_reduction": local["integrated_brier_reduction"],
                    "mean_absolute_reliability_error": local["mean_absolute_reliability_error"],
                }
            ready_row = {
                "scene": name,
                "remaining_scenes": len(pending),
                "wall_seconds": time.monotonic() - scene_started,
                "per_horizon": scene_metrics,
            }
            ready_sequence.append(ready_row)
            print(json.dumps({"scene_ready_gpu_scored": ready_row}, indent=2), flush=True)

    arrays = {
        name: np.concatenate([part[name] for part in row_parts], axis=0) for name in row_parts[0]
    }
    partial = run_dir / "P175_VISIT_RELIABILITY_CDF_ROWS.partial.npz"
    np.savez_compressed(partial, **arrays)
    partial.replace(run_dir / "P175_VISIT_RELIABILITY_CDF_ROWS.npz")
    results = {}
    for horizon in data["horizons_seconds"]:
        key = str(horizon)
        payload = {
            field: np.concatenate([part[field] for part in horizon_payloads[key]], axis=0)
            for field in horizon_payloads[key][0]
        }
        results[key] = _summarize(payload, budgets)
        results[key]["row_count"] = int(
            np.count_nonzero(np.isclose(arrays["horizon_seconds"], float(horizon)))
        )
        print(json.dumps({key: results[key]}, indent=2), flush=True)

    mean_brier_reduction = float(np.mean([
        float(row["integrated_brier_reduction"]) for row in results.values()
    ]))
    mean_calibration_error = float(np.mean([
        float(row["mean_absolute_reliability_error"]) for row in results.values()
    ]))
    mean_control_calibration_error = float(np.mean([
        float(row["horizon_only_mean_absolute_reliability_error"]) for row in results.values()
    ]))
    decisions = {
        "minimum_mean_integrated_brier_reduction_over_horizon_only": mean_brier_reduction
        >= float(config["decision"]["minimum_mean_integrated_brier_reduction_over_horizon_only"]),
        "mean_calibration_error_noninferior_to_horizon_only": mean_calibration_error
        <= mean_control_calibration_error,
    }
    verdict = config["verdict_on_pass"] if all(decisions.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"],
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "role": config["role"],
        "fresh_independent_per_horizon_evaluations": results,
        "scene_ready_gpu_sequence": ready_sequence,
        "macro_metrics": {
            "mean_integrated_brier_reduction_over_horizon_only": mean_brier_reduction,
            "mean_absolute_reliability_error": mean_calibration_error,
            "horizon_only_mean_absolute_reliability_error": mean_control_calibration_error,
        },
        "decision_checks": decisions,
        "resources": {
            "gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "wall_seconds": time.monotonic() - started,
        },
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    (run_dir / "status.json").write_text(json.dumps({
        "status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "run_dir": str(run_dir), "verdict": verdict,
        "macro_metrics": summary["macro_metrics"], "decision_checks": decisions,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
