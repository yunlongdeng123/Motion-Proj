"""Run the task-aligned monotone trajectory-risk recovery probe."""

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

from motion_proj.worldsim_v65.conditional_validity import (
    fit_monotone_task_risk,
    fixed_opportunity_metrics,
    ranking_metrics,
    score_monotone_task_risk,
)
from motion_proj.worldsim_v65.task_contract import TRAJECTORY_FEATURE_NAMES


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _compare(q0: dict, task: dict) -> dict[str, object]:
    q0_scene = {row["scene_index"]: row for row in q0["scene_rows"]}
    task_scene = {row["scene_index"]: row for row in task["scene_rows"]}
    deltas = {
        scene: task_scene[scene]["fixed_route_conflict_density"]
        - q0_scene[scene]["fixed_route_conflict_density"]
        for scene in q0_scene
    }
    q0_density = float(q0["pooled_fixed_route_conflict_density"])
    task_density = float(task["pooled_fixed_route_conflict_density"])
    return {
        "task_minus_q0_pooled_fixed_route_density": task_density - q0_density,
        "relative_fixed_route_risk_reduction": float(
            (q0_density - task_density) / q0_density if q0_density > 0 else 0.0
        ),
        "scene_lower_count": sum(value < 0 for value in deltas.values()),
        "scene_equal_count": sum(value == 0 for value in deltas.values()),
        "scene_higher_count": sum(value > 0 for value in deltas.values()),
        "maximum_scene_regression": max(deltas.values()),
        "scene_deltas": deltas,
    }


def _nonroute_emission_risk(
    hidden_free: np.ndarray,
    route: np.ndarray,
    scores: np.ndarray,
    scene_index: np.ndarray,
    unit_index: np.ndarray,
    coverage: float,
) -> dict[str, float | int]:
    selected_nonroute = 0
    conflicts = 0
    for scene in np.unique(scene_index):
        for unit in np.unique(unit_index[scene_index == scene]):
            mask = (scene_index == scene) & (unit_index == unit)
            local_scores = scores[mask]
            count = max(1, int(math.floor(float(coverage) * local_scores.size)))
            selected = np.argsort(local_scores, kind="stable")[:count]
            nonroute = ~route[mask][selected]
            selected_nonroute += int(nonroute.sum())
            conflicts += int(np.count_nonzero(hidden_free[mask][selected] & nonroute))
    return {
        "selected_nonroute_count": selected_nonroute,
        "hidden_free_conflict_count": conflicts,
        "emitted_conflict_rate": float(conflicts / selected_nonroute if selected_nonroute else 0.0),
    }


def run(config_path: Path, runs_root: Path, run_id: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v65" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(run_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with np.load(config["inputs"]["compact_cache_path"], allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    train = arrays["is_train"]
    evaluate = ~train
    fit = fit_monotone_task_risk(
        arrays["native_hidden"][train],
        arrays["base_logit"][train],
        arrays["trajectory"][train],
        arrays["hidden_free"][train],
        **config["model"],
        seed=int(config["seed"]),
    )
    q0_scores = torch.sigmoid(torch.from_numpy(arrays["base_logit"][evaluate])).numpy()
    task_scores = score_monotone_task_risk(
        fit,
        arrays["native_hidden"][evaluate],
        arrays["base_logit"][evaluate],
        arrays["trajectory"][evaluate],
    )
    shuffled = arrays["trajectory"][evaluate].copy()
    eval_scenes = arrays["scene_index"][evaluate]
    eval_units = arrays["unit_index"][evaluate]
    rng = np.random.default_rng(int(config["seed"]) + 651)
    for scene in np.unique(eval_scenes):
        for unit in np.unique(eval_units[eval_scenes == scene]):
            mask = (eval_scenes == scene) & (eval_units == unit)
            shuffled[mask] = shuffled[mask][rng.permutation(np.count_nonzero(mask))]
    shuffled_scores = score_monotone_task_risk(
        fit,
        arrays["native_hidden"][evaluate],
        arrays["base_logit"][evaluate],
        shuffled,
    )
    common = {
        "hidden_free": arrays["hidden_free"][evaluate],
        "route": arrays["route"][evaluate],
        "scene_index": eval_scenes,
        "unit_index": eval_units,
        "coverage": float(config["evaluation"]["matched_total_coverage"]),
        "tail_fraction": float(config["evaluation"]["tail_fraction"]),
    }
    q0_fixed = fixed_opportunity_metrics(scores=q0_scores, **common)
    task_fixed = fixed_opportunity_metrics(scores=task_scores, **common)
    shuffled_fixed = fixed_opportunity_metrics(scores=shuffled_scores, **common)
    comparison = _compare(q0_fixed, task_fixed)
    coverage = float(config["evaluation"]["matched_total_coverage"])
    q0_nonroute = _nonroute_emission_risk(
        arrays["hidden_free"][evaluate], arrays["route"][evaluate], q0_scores, eval_scenes, eval_units, coverage
    )
    task_nonroute = _nonroute_emission_risk(
        arrays["hidden_free"][evaluate], arrays["route"][evaluate], task_scores, eval_scenes, eval_units, coverage
    )
    q0_nonroute_rate = float(q0_nonroute["emitted_conflict_rate"])
    task_nonroute_rate = float(task_nonroute["emitted_conflict_rate"])
    nonroute_relative_change = float(
        (task_nonroute_rate - q0_nonroute_rate) / q0_nonroute_rate
        if q0_nonroute_rate > 0 else 0.0
    )
    perturbation_density = float(shuffled_fixed["pooled_fixed_route_conflict_density"])
    gates = {
        "minimum_fixed_route_risk_reduction": comparison["relative_fixed_route_risk_reduction"] >= float(config["gates"]["minimum_fixed_route_risk_reduction"]),
        "scene_direction_support": comparison["scene_lower_count"] >= comparison["scene_higher_count"],
        "nonroute_risk_not_worse": nonroute_relative_change <= float(config["gates"]["maximum_nonroute_relative_risk_increase"]),
        "trajectory_perturbation_response": float(task_fixed["pooled_fixed_route_conflict_density"]) < perturbation_density,
    }
    verdict = "positive_train_only_task_risk_signal" if all(gates.values()) else "no_clear_train_only_task_risk_signal"
    torch.save(
        {
            "state_dict": {name: value.detach().cpu() for name, value in fit.model.state_dict().items()},
            "trajectory_mean": fit.trajectory_mean,
            "trajectory_scale": fit.trajectory_scale,
            "trajectory_feature_names": TRAJECTORY_FEATURE_NAMES,
            "semantics": "frozen_physical_logit_plus_nonnegative_relevance_scaled_task_risk",
        },
        run_dir / "monotone_task_risk.pt",
    )
    summary = {
        "schema_version": "worldsim_v65.p1r_task_risk_summary.v1",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "claim_boundary": config["claim_boundary"],
        "train_point_count": int(train.sum()),
        "evaluation_point_count": int(evaluate.sum()),
        "ranking_descriptive": {
            "q0": ranking_metrics(arrays["hidden_free"][evaluate], q0_scores),
            "task": ranking_metrics(arrays["hidden_free"][evaluate], task_scores),
        },
        "fixed_opportunity": {"q0": q0_fixed, "task": task_fixed, "shuffled_trajectory": shuffled_fixed, "comparison": comparison},
        "nonroute_emission": {"q0": q0_nonroute, "task": task_nonroute, "relative_risk_change": nonroute_relative_change},
        "epoch_losses": fit.epoch_losses,
        "gate_results": gates,
        "quality_read": "legacy_train_only_mechanism",
        "formal_v65_selection_read": False,
        "resources": {
            "gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
            "wall_seconds": time.monotonic() - started,
        },
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "status.json", {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    return {"run_dir": str(run_dir), "verdict": verdict, "gate_results": gates, "resources": summary["resources"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.runs_root.resolve(), args.run_id), indent=2))


if __name__ == "__main__":
    main()
