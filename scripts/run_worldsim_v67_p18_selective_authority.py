"""Train a selective authority compiler without changing trajectory qmean order."""

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
from scipy.stats import spearmanr

from motion_proj.worldsim_v67.selective_authority import (
    FEATURE_NAMES,
    authority_metrics,
    case_dataset,
    score_benefit_head,
    train_benefit_head,
)
from scripts.run_worldsim_v65_p10v_action_visited_state_transfer import _within_case_selection
from scripts.run_worldsim_v67_p17_quantile_trajectory import _combine, _load


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(config_path: Path, runs_root: Path, run_id: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v67" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    _write_json(run_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    train_actions = _combine([Path(path) for path in config["inputs"]["train_action_caches"]])
    selection_actions = _load(Path(config["inputs"]["selection_action_cache"]))
    fraction = float(config["compiler"]["within_case_selected_fraction"])
    train_cases = case_dataset(train_actions, fraction)
    selection_cases = case_dataset(selection_actions, fraction)
    model, mean, scale, training = train_benefit_head(
        train_cases, config["model"], int(config["seed"])
    )
    train_scores = score_benefit_head(model, train_cases, mean, scale)
    selection_scores = score_benefit_head(model, selection_cases, mean, scale)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": list(FEATURE_NAMES),
            "mean": mean,
            "scale": scale,
            "qmean_action_order_frozen": True,
        },
        run_dir / "SELECTIVE_AUTHORITY_COMPILER.pt",
    )
    _write_json(
        run_dir / "model_frozen.json",
        {"model_frozen_before_selection_scoring": True, "qmean_action_order_frozen": True, "train_case_count": int(len(train_scores))},
    )
    train_metrics = authority_metrics(
        train_actions, train_cases, train_scores, float(config["compiler"]["authority_case_fraction"])
    )
    selection_metrics = authority_metrics(
        selection_actions, selection_cases, selection_scores, float(config["compiler"]["authority_case_fraction"])
    )
    qmean_all = _within_case_selection(
        np.asarray(selection_actions["target_cost"], dtype=np.float32),
        np.asarray(selection_actions["qmean"], dtype=np.float32),
        np.asarray(selection_actions["case_index"]),
        np.asarray(selection_actions["scene_index"]),
        fraction,
    )
    benefit_ranking = {
        "train_spearman": float(spearmanr(train_cases["benefit"], train_scores).statistic),
        "selection_spearman": float(spearmanr(selection_cases["benefit"], selection_scores).statistic),
    }
    improvement = {
        "authority_reduction_delta_over_ungated_qmean": float(
            selection_metrics["relative_cost_reduction"] - qmean_all["relative_cost_reduction"]
        )
    }
    gates_config = config["gates"]
    gates = {
        "minimum_authority_fraction": selection_metrics["authority_fraction"] >= float(gates_config["minimum_authority_fraction"]),
        "minimum_authority_cost_reduction": selection_metrics["relative_cost_reduction"] >= float(gates_config["minimum_authority_cost_reduction"]),
        "minimum_reduction_delta_over_ungated_qmean": improvement["authority_reduction_delta_over_ungated_qmean"] >= float(gates_config["minimum_reduction_delta_over_ungated_qmean"]),
        "minimum_nonincreasing_scene_support": selection_metrics["scene_nonincreasing_count"] >= int(gates_config["minimum_nonincreasing_scene_support"]),
    }
    verdict = "supported_selective_trajectory_authority_compiler" if all(gates.values()) else "rejected_selective_trajectory_authority_compiler"
    _write_json(
        run_dir / "CASE_AUTHORITY_SCORES.json",
        {
            "case_index": [int(v) for v in selection_cases["case_index"]],
            "scene_index": [int(v) for v in selection_cases["scene_index"]],
            "predicted_benefit": [float(v) for v in selection_scores],
            "actual_benefit": [float(v) for v in selection_cases["benefit"]],
            "authorized_case_indices": selection_metrics["authorized_case_indices"],
        },
    )
    summary = {
        "schema_version": "worldsim_v67.p18_selective_authority_summary.v1",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "role": config["role"],
        "claim_boundary": config["claim_boundary"],
        "training": training,
        "benefit_ranking": benefit_ranking,
        "train_authority_metrics": train_metrics,
        "selection_authority_metrics": selection_metrics,
        "selection_ungated_qmean": qmean_all,
        "selection_improvement": improvement,
        "gate_results": gates,
        "failure_ledger_delta": "pending_result",
        "resources": {
            "gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
            "wall_seconds": time.monotonic() - started,
        },
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "status.json", {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    return {"run_dir": str(run_dir), "verdict": verdict, "gate_results": gates}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.runs_root.resolve(), args.run_id), indent=2))


if __name__ == "__main__":
    main()
