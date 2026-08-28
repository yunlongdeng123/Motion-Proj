"""Train a continuous entropic-risk listwise compiler and confirm it once."""

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

from motion_proj.worldsim_v67.listwise_action_compiler import BoundedListwiseCompiler, FEATURE_NAMES, score_listwise_compiler, train_listwise_compiler
from motion_proj.worldsim_v67.trajectory_quantile import materialize_quantiles
from scripts.run_worldsim_v67_p15_trajectory_reliability_train import _metrics
from scripts.run_worldsim_v67_p17_quantile_trajectory import _combine, _load


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _load_compiler(runs_root: Path, run: str, artifact_name: str):
    artifact = torch.load(runs_root / run / artifact_name, map_location="cuda", weights_only=False)
    model = BoundedListwiseCompiler(
        len(artifact["feature_names"]), list(artifact["hidden_dimensions"]), float(artifact["maximum_residual_cost"])
    ).cuda()
    model.load_state_dict(artifact["state_dict"])
    return model.eval(), np.asarray(artifact["mean"], dtype=np.float32), np.asarray(artifact["scale"], dtype=np.float32)


def _selected_tail_summary(arrays: dict[str, np.ndarray], scores: np.ndarray, fraction: float, tail_fraction: float) -> dict[str, float]:
    target = np.asarray(arrays["target_cost"], dtype=np.float32)
    cases = np.asarray(arrays["case_index"])
    selected = []
    for case in np.unique(cases):
        members = np.flatnonzero(cases == case)
        if len(members) < 2:
            continue
        count = max(1, int(math.floor(float(fraction) * len(members))))
        selected.extend(members[np.argsort(scores[members], kind="stable")[:count]].tolist())
    selected_cost = target[np.asarray(selected, dtype=np.int64)]
    tail_count = max(1, int(math.ceil(float(tail_fraction) * len(selected_cost))))
    tail = np.sort(selected_cost)[-tail_count:]
    return {
        "selected_action_count": int(len(selected_cost)),
        "selected_mean_cost": float(selected_cost.mean()),
        "selected_p90_cost": float(np.quantile(selected_cost, 0.90)),
        "tail_fraction": float(tail_fraction),
        "tail_count": int(tail_count),
        "selected_tail_mean_cost": float(tail.mean()),
    }


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
        {"state_dict": model.state_dict(), "feature_names": list(FEATURE_NAMES), "hidden_dimensions": list(config["model"]["hidden_dimensions"]),
         "maximum_residual_cost": float(config["model"]["maximum_residual_cost"]), "mean": mean, "scale": scale},
        run_dir / "ENTROPIC_ACTION_COMPILER.pt",
    )
    p20_model, p20_mean, p20_scale = _load_compiler(
        runs_root, config["inputs"]["mean_compiler_run"], config["inputs"]["mean_compiler_artifact"]
    )
    p22_model, p22_mean, p22_scale = _load_compiler(
        runs_root, config["inputs"]["binary_tail_compiler_run"], config["inputs"]["binary_tail_compiler_artifact"]
    )
    _write_json(run_dir / "model_frozen.json", {"entropic_compiler_frozen_before_confirmation_materialization": True,
        "p20_and_p22_baselines_frozen": True, "development_domain_count": 7, "train_action_count": int(len(train["qmean"]))})
    cache_path = Path(config["confirmation_materialization"]["cache_path"])
    materialization = {"cache_path": str(cache_path), "cache_reused": cache_path.is_file()}
    if not cache_path.is_file():
        materialization.update(materialize_quantiles(config["confirmation_materialization"]["data"], runs_root, cache_path))
    selection = _load(cache_path)
    scores = score_listwise_compiler(model, selection, mean, scale)
    p20_scores = score_listwise_compiler(p20_model, selection, p20_mean, p20_scale)
    p22_scores = score_listwise_compiler(p22_model, selection, p22_mean, p22_scale)
    qmean_scores = np.asarray(selection["qmean"], dtype=np.float32)
    metrics = _metrics(selection, scores, config)
    p20_metrics = _metrics(selection, p20_scores, config)
    p22_metrics = _metrics(selection, p22_scores, config)
    qmean_metrics = _metrics(selection, qmean_scores, config)
    fraction = float(config["evaluation"]["within_case_selected_fraction"])
    tail_fraction = float(config["evaluation"]["tail_fraction"])
    tail = _selected_tail_summary(selection, scores, fraction, tail_fraction)
    p20_tail = _selected_tail_summary(selection, p20_scores, fraction, tail_fraction)
    p22_tail = _selected_tail_summary(selection, p22_scores, fraction, tail_fraction)
    qmean_tail = _selected_tail_summary(selection, qmean_scores, fraction, tail_fraction)
    improvement = {
        "cost_reduction_delta_over_p20": float(metrics["within_case_selection"]["relative_cost_reduction"] - p20_metrics["within_case_selection"]["relative_cost_reduction"]),
        "tail_mean_ratio_to_p20": float(tail["selected_tail_mean_cost"] / p20_tail["selected_tail_mean_cost"]),
        "tail_mean_ratio_to_p22": float(tail["selected_tail_mean_cost"] / p22_tail["selected_tail_mean_cost"]),
    }
    gate_config = config["gates"]
    gates = {
        "minimum_selected_cost_reduction": metrics["within_case_selection"]["relative_cost_reduction"] >= float(gate_config["minimum_selected_cost_reduction"]),
        "maximum_tail_mean_ratio_to_p20": improvement["tail_mean_ratio_to_p20"] <= float(gate_config["maximum_tail_mean_ratio_to_p20"]),
        "minimum_pairwise_concordance": metrics["pairwise"]["concordance"] >= float(gate_config["minimum_pairwise_concordance"]),
        "minimum_nonincreasing_scene_support": metrics["within_case_selection"]["scene_nonincreasing_count"] >= int(gate_config["minimum_nonincreasing_scene_support"]),
    }
    verdict = "supported_continuous_entropic_action_compiler" if all(gates.values()) else "rejected_continuous_entropic_action_compiler"
    summary = {
        "schema_version": "worldsim_v67.p23_entropic_action_compiler_summary.v1", "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict, "role": config["role"],
        "claim_boundary": config["claim_boundary"], "training": training, "confirmation_materialization": materialization,
        "selection_metrics": metrics, "selection_p20_baseline": p20_metrics, "selection_p22_baseline": p22_metrics,
        "selection_qmean_baseline": qmean_metrics, "selection_tail": tail, "selection_p20_tail": p20_tail,
        "selection_p22_tail": p22_tail, "selection_qmean_tail": qmean_tail, "selection_improvement": improvement,
        "gate_results": gates, "failure_ledger_delta": "pending_result",
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
