"""Train a cross-case offset and confirm adaptive fixed-total action allocation."""

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

from motion_proj.worldsim_v67.adaptive_budget import FEATURE_NAMES, adaptive_fixed_total_selection, case_offset_dataset, score_case_offset, train_case_offset
from motion_proj.worldsim_v67.listwise_action_compiler import BoundedListwiseCompiler, score_listwise_compiler
from motion_proj.worldsim_v67.trajectory_quantile import materialize_quantiles
from scripts.run_worldsim_v65_p10v_action_visited_state_transfer import _within_case_selection
from scripts.run_worldsim_v67_p17_quantile_trajectory import _combine, _load


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _load_p20(config: dict, runs_root: Path):
    artifact = torch.load(runs_root / config["inputs"]["action_compiler_run"] / config["inputs"]["action_compiler_artifact"], map_location="cuda", weights_only=False)
    model = BoundedListwiseCompiler(len(artifact["feature_names"]), list(artifact["hidden_dimensions"]), float(artifact["maximum_residual_cost"])).cuda()
    model.load_state_dict(artifact["state_dict"])
    return model.eval(), np.asarray(artifact["mean"], dtype=np.float32), np.asarray(artifact["scale"], dtype=np.float32)


def run(config_path: Path, runs_root: Path, run_id: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v67" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _write_json(run_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    p20, p20_mean, p20_scale = _load_p20(config, runs_root)
    train_actions = _combine([Path(path) for path in config["inputs"]["train_action_caches"]])
    train_scores = score_listwise_compiler(p20, train_actions, p20_mean, p20_scale)
    fraction = float(config["compiler"]["fixed_selected_fraction"])
    train_cases = case_offset_dataset(train_actions, train_scores, fraction)
    model, mean, scale, training = train_case_offset(train_cases, config["model"], int(config["seed"]))
    torch.save({"state_dict": model.state_dict(), "feature_names": list(FEATURE_NAMES), "hidden_dimension": int(config["model"]["hidden_dimension"]),
                "maximum_case_offset": float(config["model"]["maximum_case_offset"]), "mean": mean, "scale": scale,
                "frozen_action_compiler_run": config["inputs"]["action_compiler_run"]}, run_dir / "ADAPTIVE_BUDGET_COMPILER.pt")
    _write_json(run_dir / "model_frozen.json", {"p20_ranking_frozen": True, "case_offset_frozen_before_confirmation_materialization": True,
                "development_domain_count": 8, "train_case_count": int(len(train_cases["case_index"]))})
    cache_path = Path(config["confirmation_materialization"]["cache_path"])
    materialization = {"cache_path": str(cache_path), "cache_reused": cache_path.is_file()}
    if not cache_path.is_file(): materialization.update(materialize_quantiles(config["confirmation_materialization"]["data"], runs_root, cache_path))
    selection = _load(cache_path)
    selection_scores = score_listwise_compiler(p20, selection, p20_mean, p20_scale)
    selection_cases = case_offset_dataset(selection, selection_scores, fraction)
    offsets = score_case_offset(model, selection_cases, mean, scale)
    adaptive = adaptive_fixed_total_selection(selection, selection_scores, offsets, fraction, int(config["compiler"]["maximum_actions_per_case"]))
    fixed = _within_case_selection(np.asarray(selection["target_cost"], dtype=np.float32), selection_scores,
                                   np.asarray(selection["case_index"]), np.asarray(selection["scene_index"]), fraction)
    qmean = _within_case_selection(np.asarray(selection["target_cost"], dtype=np.float32), np.asarray(selection["qmean"], dtype=np.float32),
                                   np.asarray(selection["case_index"]), np.asarray(selection["scene_index"]), fraction)
    improvement = {"adaptive_reduction_delta_over_fixed_p20": float(adaptive["relative_cost_reduction"] - fixed["relative_cost_reduction"]),
                   "adaptive_reduction_delta_over_qmean": float(adaptive["relative_cost_reduction"] - qmean["relative_cost_reduction"])}
    gate_config = config["gates"]
    gates = {"exact_fixed_total_action_budget": adaptive["selected_action_count"] == adaptive["fixed_total_action_budget"],
             "minimum_adaptive_cost_reduction": adaptive["relative_cost_reduction"] >= float(gate_config["minimum_adaptive_cost_reduction"]),
             "minimum_reduction_delta_over_fixed_p20": improvement["adaptive_reduction_delta_over_fixed_p20"] >= float(gate_config["minimum_reduction_delta_over_fixed_p20"]),
             "minimum_nonincreasing_scene_support": adaptive["scene_nonincreasing_count"] >= int(gate_config["minimum_nonincreasing_scene_support"])}
    verdict = "supported_adaptive_fixed_total_action_budget" if all(gates.values()) else "rejected_adaptive_fixed_total_action_budget"
    summary = {"schema_version": "worldsim_v67.p24_adaptive_budget_summary.v1", "task_id": config["task_id"], "hypothesis_id": config["hypothesis_id"],
               "status": "done", "verdict": verdict, "role": config["role"], "claim_boundary": config["claim_boundary"], "training": training,
               "confirmation_materialization": materialization, "adaptive_budget": adaptive, "fixed_p20": fixed, "fixed_qmean": qmean,
               "selection_improvement": improvement, "gate_results": gates, "failure_ledger_delta": "pending_result",
               "resources": {"gpu": torch.cuda.get_device_name(0), "peak_gpu_memory_gib": torch.cuda.max_memory_allocated()/(1024**3),
                             "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2), "wall_seconds": time.monotonic()-started}}
    _write_json(run_dir / "summary.json", summary); _write_json(run_dir / "status.json", {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    return {"run_dir": str(run_dir), "verdict": verdict, "gate_results": gates}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--runs-root", type=Path, required=True); parser.add_argument("--run-id", required=True)
    args = parser.parse_args(); print(json.dumps(run(args.config.resolve(), args.runs_root.resolve(), args.run_id), indent=2))


if __name__ == "__main__": main()
