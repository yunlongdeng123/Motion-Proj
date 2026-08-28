"""Train and confirm a bounded multi-domain listwise action compiler."""

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

from motion_proj.worldsim_v67.listwise_action_compiler import (
    FEATURE_NAMES,
    score_listwise_compiler,
    train_listwise_compiler,
)
from motion_proj.worldsim_v67.trajectory_quantile import materialize_quantiles
from scripts.run_worldsim_v67_p15_trajectory_reliability_train import _metrics
from scripts.run_worldsim_v67_p17_quantile_trajectory import _combine, _load


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(config_path: Path, runs_root: Path, run_id: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v67" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _write_json(run_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    train = _combine([Path(path) for path in config["inputs"]["train_action_caches"]])
    model, mean, scale, training = train_listwise_compiler(train, config["model"], int(config["seed"]))
    train_scores = score_listwise_compiler(model, train, mean, scale)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": list(FEATURE_NAMES),
            "hidden_dimensions": list(config["model"]["hidden_dimensions"]),
            "maximum_residual_cost": float(config["model"]["maximum_residual_cost"]),
            "mean": mean,
            "scale": scale,
        },
        run_dir / "LISTWISE_ACTION_COMPILER.pt",
    )
    _write_json(
        run_dir / "model_frozen.json",
        {"model_frozen_before_confirmation_materialization": True, "development_domain_count": 4, "train_action_count": int(len(train["qmean"]))},
    )
    cache_path = Path(config["confirmation_materialization"]["cache_path"])
    materialization = {"cache_path": str(cache_path), "cache_reused": cache_path.is_file()}
    if not cache_path.is_file():
        materialization.update(materialize_quantiles(config["confirmation_materialization"]["data"], runs_root, cache_path))
    selection = _load(cache_path)
    selection_scores = score_listwise_compiler(model, selection, mean, scale)
    train_metrics = _metrics(train, train_scores, config)
    selection_metrics = _metrics(selection, selection_scores, config)
    qmean_metrics = _metrics(selection, np.asarray(selection["qmean"], dtype=np.float32), config)
    improvement = {
        "selected_cost_reduction_delta_over_qmean": float(selection_metrics["within_case_selection"]["relative_cost_reduction"] - qmean_metrics["within_case_selection"]["relative_cost_reduction"]),
        "pairwise_delta_over_qmean": float(selection_metrics["pairwise"]["concordance"] - qmean_metrics["pairwise"]["concordance"]),
    }
    gate_config = config["gates"]
    gates = {
        "minimum_selected_cost_reduction": selection_metrics["within_case_selection"]["relative_cost_reduction"] >= float(gate_config["minimum_selected_cost_reduction"]),
        "minimum_selected_cost_reduction_delta_over_qmean": improvement["selected_cost_reduction_delta_over_qmean"] >= float(gate_config["minimum_selected_cost_reduction_delta_over_qmean"]),
        "minimum_pairwise_concordance": selection_metrics["pairwise"]["concordance"] >= float(gate_config["minimum_pairwise_concordance"]),
        "minimum_nonincreasing_scene_support": selection_metrics["within_case_selection"]["scene_nonincreasing_count"] >= int(gate_config["minimum_nonincreasing_scene_support"]),
    }
    verdict = "supported_independent_listwise_action_compiler" if all(gates.values()) else "rejected_independent_listwise_action_compiler"
    summary = {
        "schema_version": "worldsim_v67.p20_listwise_action_compiler_summary.v1",
        "task_id": config["task_id"], "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"], "claim_boundary": config["claim_boundary"], "training": training,
        "confirmation_materialization": materialization, "train_metrics": train_metrics,
        "selection_metrics": selection_metrics, "selection_qmean_baseline": qmean_metrics,
        "selection_improvement": improvement, "gate_results": gates, "failure_ledger_delta": "pending_result",
        "resources": {"gpu": torch.cuda.get_device_name(0), "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024**3), "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2), "wall_seconds": time.monotonic() - started},
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
