"""Train a tail-risk-aware listwise action compiler and confirm it once."""

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
    BoundedListwiseCompiler,
    FEATURE_NAMES,
    score_listwise_compiler,
    train_listwise_compiler,
)
from motion_proj.worldsim_v67.trajectory_quantile import materialize_quantiles
from scripts.run_worldsim_v65_p10v_action_visited_state_transfer import _within_case_selection
from scripts.run_worldsim_v67_p15_trajectory_reliability_train import _metrics
from scripts.run_worldsim_v67_p17_quantile_trajectory import _combine, _load


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _load_p20(config: dict, runs_root: Path):
    artifact = torch.load(
        runs_root / config["inputs"]["mean_compiler_run"] / config["inputs"]["mean_compiler_artifact"],
        map_location="cuda", weights_only=False,
    )
    model = BoundedListwiseCompiler(
        len(artifact["feature_names"]), list(artifact["hidden_dimensions"]), float(artifact["maximum_residual_cost"])
    ).cuda()
    model.load_state_dict(artifact["state_dict"])
    return model.eval(), np.asarray(artifact["mean"], dtype=np.float32), np.asarray(artifact["scale"], dtype=np.float32)


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
    torch.save(
        {"state_dict": model.state_dict(), "feature_names": list(FEATURE_NAMES),
         "hidden_dimensions": list(config["model"]["hidden_dimensions"]),
         "maximum_residual_cost": float(config["model"]["maximum_residual_cost"]), "mean": mean, "scale": scale},
        run_dir / "TAIL_RISK_ACTION_COMPILER.pt",
    )
    p20_model, p20_mean, p20_scale = _load_p20(config, runs_root)
    _write_json(
        run_dir / "model_frozen.json",
        {"tail_compiler_frozen_before_confirmation_materialization": True, "p20_baseline_frozen": True,
         "development_domain_count": 6, "train_action_count": int(len(train["qmean"]))},
    )
    cache_path = Path(config["confirmation_materialization"]["cache_path"])
    materialization = {"cache_path": str(cache_path), "cache_reused": cache_path.is_file()}
    if not cache_path.is_file():
        materialization.update(materialize_quantiles(config["confirmation_materialization"]["data"], runs_root, cache_path))
    selection = _load(cache_path)
    scores = score_listwise_compiler(model, selection, mean, scale)
    p20_scores = score_listwise_compiler(p20_model, selection, p20_mean, p20_scale)
    qmean_scores = np.asarray(selection["qmean"], dtype=np.float32)
    metrics = _metrics(selection, scores, config)
    p20_metrics = _metrics(selection, p20_scores, config)
    qmean_metrics = _metrics(selection, qmean_scores, config)
    target_unsafe = np.asarray(selection["unsafe"], dtype=np.float32)
    cases = np.asarray(selection["case_index"])
    scenes = np.asarray(selection["scene_index"])
    fraction = float(config["evaluation"]["within_case_selected_fraction"])
    unsafe_selection = _within_case_selection(target_unsafe, scores, cases, scenes, fraction)
    p20_unsafe = _within_case_selection(target_unsafe, p20_scores, cases, scenes, fraction)
    qmean_unsafe = _within_case_selection(target_unsafe, qmean_scores, cases, scenes, fraction)
    improvement = {
        "cost_reduction_delta_over_p20": float(metrics["within_case_selection"]["relative_cost_reduction"] - p20_metrics["within_case_selection"]["relative_cost_reduction"]),
        "unsafe_reduction_delta_over_p20": float(unsafe_selection["relative_cost_reduction"] - p20_unsafe["relative_cost_reduction"]),
        "unsafe_reduction_delta_over_qmean": float(unsafe_selection["relative_cost_reduction"] - qmean_unsafe["relative_cost_reduction"]),
    }
    gate_config = config["gates"]
    gates = {
        "minimum_selected_cost_reduction": metrics["within_case_selection"]["relative_cost_reduction"] >= float(gate_config["minimum_selected_cost_reduction"]),
        "minimum_selected_unsafe_reduction": unsafe_selection["relative_cost_reduction"] >= float(gate_config["minimum_selected_unsafe_reduction"]),
        "minimum_unsafe_reduction_delta_over_p20": improvement["unsafe_reduction_delta_over_p20"] >= float(gate_config["minimum_unsafe_reduction_delta_over_p20"]),
        "minimum_nonincreasing_scene_support": metrics["within_case_selection"]["scene_nonincreasing_count"] >= int(gate_config["minimum_nonincreasing_scene_support"]),
    }
    verdict = "supported_tail_risk_listwise_action_compiler" if all(gates.values()) else "rejected_tail_risk_listwise_action_compiler"
    summary = {
        "schema_version": "worldsim_v67.p22_tail_risk_action_compiler_summary.v1", "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict, "role": config["role"],
        "claim_boundary": config["claim_boundary"], "training": training, "confirmation_materialization": materialization,
        "selection_metrics": metrics, "selection_p20_baseline": p20_metrics, "selection_qmean_baseline": qmean_metrics,
        "selection_unsafe": unsafe_selection, "selection_p20_unsafe": p20_unsafe, "selection_qmean_unsafe": qmean_unsafe,
        "selection_improvement": improvement, "gate_results": gates, "failure_ledger_delta": "pending_result",
        "resources": {"gpu": torch.cuda.get_device_name(0), "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
                      "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2), "wall_seconds": time.monotonic() - started},
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
