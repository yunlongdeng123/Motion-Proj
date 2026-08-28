"""Fit rank-preserving affine calibration maps for frozen Actor reliability scores."""

from __future__ import annotations

import argparse
import json
import math
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from motion_proj.worldsim_v67.actor_state_reliability import (
    ACTOR_FEATURE_NAMES, ReliabilityMLP, evaluate_reliability, predict_reliability,
)


def _select_by_scene(score: np.ndarray, scenes: np.ndarray, fraction: float) -> np.ndarray:
    selected: list[int] = []
    for scene in np.unique(scenes):
        members = np.flatnonzero(scenes == scene)
        count = max(1, int(np.floor(len(members) * fraction)))
        selected.extend(members[np.argsort(score[members], kind="mergesort")[:count]].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def _fit_affine(
    base_prediction: np.ndarray, target_cost: np.ndarray, model_config: dict,
) -> tuple[float, float, float]:
    base_log = torch.from_numpy(np.log1p(base_prediction).astype(np.float32)).cuda()
    target_log = torch.from_numpy(np.log1p(target_cost).astype(np.float32)).cuda()
    raw_scale = torch.nn.Parameter(torch.tensor(math.log(math.expm1(1.0)), device="cuda"))
    bias = torch.nn.Parameter(torch.zeros((), device="cuda"))
    optimizer = torch.optim.AdamW(
        [raw_scale, bias], lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    final_loss = 0.0
    for epoch in range(int(model_config["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        scale = torch.nn.functional.softplus(raw_scale) + 1e-6
        prediction = torch.clamp(scale * base_log + bias, min=0.0)
        loss = torch.nn.functional.smooth_l1_loss(
            prediction, target_log, beta=float(model_config["huber_beta"])
        )
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if epoch % 250 == 0 or epoch + 1 == int(model_config["epochs"]):
            print(
                f"monotone actor calibration epoch={epoch + 1} loss={final_loss:.6f}",
                flush=True,
            )
    return float(torch.nn.functional.softplus(raw_scale).detach().cpu() + 1e-6), float(bias.detach().cpu()), final_loss


def _apply_affine(prediction: np.ndarray, scale: float, bias: float) -> np.ndarray:
    calibrated_log = np.maximum(scale * np.log1p(prediction) + bias, 0.0)
    return np.expm1(calibrated_log).clip(min=0.0)


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

    source = args.runs_root / config["source"]["run"]
    artifact = torch.load(source / config["source"]["artifact"], map_location="cuda")
    hidden = artifact["hidden_dimensions"]
    query_model = ReliabilityMLP(len(artifact["feature_names"]), hidden).cuda()
    query_model.load_state_dict(artifact["query_model_state_dict"])
    actor_model = ReliabilityMLP(len(ACTOR_FEATURE_NAMES), hidden).cuda()
    actor_model.load_state_dict(artifact["actor_only_model_state_dict"])
    query_model.eval()
    actor_model.eval()
    mean = np.asarray(artifact["feature_mean"], dtype=np.float32)
    scale = np.asarray(artifact["feature_scale"], dtype=np.float32)

    calibration = dict(np.load(
        args.runs_root / config["calibration_data"]["run"] / config["calibration_data"]["rows"],
        allow_pickle=False,
    ))
    fresh = dict(np.load(
        args.runs_root / config["evaluation_data"]["run"] / config["evaluation_data"]["rows"],
        allow_pickle=False,
    ))
    calibration_query = predict_reliability(query_model, calibration["features"], mean, scale)
    calibration_actor = predict_reliability(actor_model, calibration["features"], mean, scale, actor_only=True)
    query_scale, query_bias, query_loss = _fit_affine(
        calibration_query, calibration["target_cost"], config["calibrator"]
    )
    actor_scale, actor_bias, actor_loss = _fit_affine(
        calibration_actor, calibration["target_cost"], config["calibrator"]
    )
    frozen_query = predict_reliability(query_model, fresh["features"], mean, scale)
    frozen_actor = predict_reliability(actor_model, fresh["features"], mean, scale, actor_only=True)
    calibrated_query = _apply_affine(frozen_query, query_scale, query_bias)
    calibrated_actor = _apply_affine(frozen_actor, actor_scale, actor_bias)
    frozen_evaluation = evaluate_reliability(fresh, frozen_query, frozen_actor, config["evaluation"])
    calibrated_evaluation = evaluate_reliability(
        fresh, calibrated_query, calibrated_actor, config["evaluation"]
    )
    query_mae_improvement = (
        frozen_evaluation["query_conditioned_mae"] - calibrated_evaluation["query_conditioned_mae"]
    ) / max(frozen_evaluation["query_conditioned_mae"], 1e-12)

    target = np.asarray(fresh["target_cost"], dtype=np.float64)
    scenes = np.asarray(fresh["scene_index"])
    unreliable = (
        np.asarray(fresh["raw_actor_state_error_m"]) > float(config["evaluation"]["unreliable_actor_state_error_m"])
    ) & (
        np.asarray(fresh["predicted_minimum_separation_m"]) <= float(config["evaluation"]["unreliable_exposure_radius_m"])
    )
    selected = _select_by_scene(calibrated_query, scenes, float(config["selection"]["coverage_fraction"]))
    all_cost = float(target.mean())
    selected_cost = float(target[selected].mean())
    all_prevalence = float(unreliable.mean())
    selected_prevalence = float(unreliable[selected].mean())
    selection = {
        "selected_row_count": int(len(selected)),
        "achieved_coverage": float(len(selected) / len(target)),
        "all_mean_cost": all_cost,
        "calibrated_query_selected_mean_cost": selected_cost,
        "calibrated_query_cost_reduction": (all_cost - selected_cost) / max(all_cost, 1e-12),
        "all_unreliable_prevalence": all_prevalence,
        "calibrated_query_selected_unreliable_prevalence": selected_prevalence,
        "calibrated_query_unreliable_prevalence_reduction": (
            all_prevalence - selected_prevalence
        ) / max(all_prevalence, 1e-12),
    }
    gates = {
        "minimum_mae_reduction_over_calibrated_actor_only": (
            calibrated_evaluation["mae_reduction_over_actor_only"]
            >= float(config["gates"]["minimum_mae_reduction_over_calibrated_actor_only"])
        ),
        "minimum_query_mae_improvement_over_frozen": (
            query_mae_improvement >= float(config["gates"]["minimum_query_mae_improvement_over_frozen"])
        ),
        "minimum_selective_cost_reduction": (
            selection["calibrated_query_cost_reduction"]
            >= float(config["gates"]["minimum_selective_cost_reduction"])
        ),
    }
    verdict = config["verdict_on_pass"] if all(gates.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"],
        "calibrators": {
            "query": {"scale": query_scale, "bias": query_bias, "final_loss": query_loss},
            "actor_only": {"scale": actor_scale, "bias": actor_bias, "final_loss": actor_loss},
        },
        "frozen_evaluation": frozen_evaluation,
        "calibrated_evaluation": calibrated_evaluation,
        "query_mae_improvement_over_frozen": query_mae_improvement,
        "selection": selection, "gate_results": gates,
        "resources": {"gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "wall_seconds": time.monotonic() - started},
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({
        "status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict, "gate_results": gates}, indent=2))


if __name__ == "__main__":
    main()
