"""Train a bounded heteroscedastic case-authority compiler and compare its conservative priority."""

from __future__ import annotations

import argparse, json, resource, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch, yaml

from motion_proj.worldsim_v67.adaptive_budget import (
    BUDGET_HORIZON_CONDITIONED_FEATURE_NAMES, BoundedCaseOffset, BoundedHeteroscedasticCaseOffset,
    budget_horizon_conditioned_case_offset_dataset, group_coverage_constrained_selection,
    score_case_offset, score_heteroscedastic_case_offset, train_heteroscedastic_case_offset,
)
from motion_proj.worldsim_v67.listwise_action_compiler import BoundedListwiseCompiler, score_listwise_compiler
from scripts.run_worldsim_v65_p10v_action_visited_state_transfer import _within_case_selection
from scripts.run_worldsim_v67_p17_quantile_trajectory import _combine, _load


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _load_p20(config, runs_root):
    artifact = torch.load(runs_root / config["inputs"]["action_compiler_run"] / config["inputs"]["action_compiler_artifact"], map_location="cuda", weights_only=False)
    model = BoundedListwiseCompiler(len(artifact["feature_names"]), list(artifact["hidden_dimensions"]), float(artifact["maximum_residual_cost"])).cuda()
    model.load_state_dict(artifact["state_dict"])
    return model.eval(), np.asarray(artifact["mean"], dtype=np.float32), np.asarray(artifact["scale"], dtype=np.float32)


def _load_mean(config, runs_root):
    artifact = torch.load(runs_root / config["inputs"]["mean_compiler_run"] / config["inputs"]["mean_compiler_artifact"], map_location="cuda", weights_only=False)
    model = BoundedCaseOffset(len(artifact["feature_names"]), int(artifact["hidden_dimension"]), float(artifact["maximum_case_offset"])).cuda()
    model.load_state_dict(artifact["state_dict"])
    return model.eval(), np.asarray(artifact["mean"], dtype=np.float32), np.asarray(artifact["scale"], dtype=np.float32)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = np.argsort(np.argsort(left, kind="stable"), kind="stable").astype(np.float64)
    right_rank = np.argsort(np.argsort(right, kind="stable"), kind="stable").astype(np.float64)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def run(config_path: Path, runs_root: Path, run_id: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v67" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _write(run_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    p20, p20_mean, p20_scale = _load_p20(config, runs_root)
    train = _combine([Path(path) for path in config["inputs"]["train_action_caches"]])
    train_scores = score_listwise_compiler(p20, train, p20_mean, p20_scale)
    fractions = [float(value) for value in config["model"]["training_selected_fractions"]]
    horizons = [float(value) for value in config["model"]["training_horizon_seconds_by_domain"]]
    train_cases = budget_horizon_conditioned_case_offset_dataset(train, train_scores, fractions, horizons)
    model, mean, scale, training = train_heteroscedastic_case_offset(train_cases, config["model"], int(config["seed"]))
    torch.save({"state_dict": model.state_dict(), "feature_names": list(BUDGET_HORIZON_CONDITIONED_FEATURE_NAMES),
                "hidden_dimension": int(config["model"]["hidden_dimension"]), "maximum_case_offset": float(config["model"]["maximum_case_offset"]),
                "minimum_scale": float(config["model"]["minimum_scale"]), "maximum_scale": float(config["model"]["maximum_scale"]),
                "mean": mean, "scale": scale}, run_dir / "HETEROSCEDASTIC_AUTHORITY_COMPILER.pt")
    mean_model, mean_norm, mean_scale = _load_mean(config, runs_root)
    selection = _load(Path(config["confirmation"]["cache_path"]))
    scores = score_listwise_compiler(p20, selection, p20_mean, p20_scale)
    fraction = float(config["confirmation"]["selected_fraction"]); horizon = float(config["confirmation"]["horizon_seconds"])
    cases = budget_horizon_conditioned_case_offset_dataset(selection, scores, [fraction], [horizon])
    predicted_mean, predicted_scale = score_heteroscedastic_case_offset(model, cases, mean, scale)
    conservative_offsets = predicted_mean + float(config["compiler"]["uncertainty_weight"]) * predicted_scale
    baseline_offsets = score_case_offset(mean_model, cases, mean_norm, mean_scale)
    scene_to_group = {int(key): int(value) for key, value in config["compiler"]["scene_to_group"].items()}
    args = (selection, scores, fraction, int(config["compiler"]["maximum_actions_per_case"]), float(config["compiler"]["minimum_case_coverage"]), scene_to_group, float(config["compiler"]["minimum_group_case_coverage"]))
    conservative = group_coverage_constrained_selection(args[0], args[1], conservative_offsets, *args[2:])
    mean_baseline = group_coverage_constrained_selection(args[0], args[1], baseline_offsets, *args[2:])
    fixed = _within_case_selection(np.asarray(selection["target_cost"], dtype=np.float32), scores, np.asarray(selection["case_index"]), np.asarray(selection["scene_index"]), fraction)
    absolute_error = np.abs(np.asarray(cases["target_offset"], dtype=np.float32) - predicted_mean)
    uncertainty = {"scale_error_spearman": _spearman(predicted_scale, absolute_error), "mean_scale": float(predicted_scale.mean()),
                   "minimum_scale": float(predicted_scale.min()), "maximum_scale": float(predicted_scale.max())}
    improvement = {"delta_over_mean_compiler": float(conservative["relative_cost_reduction"] - mean_baseline["relative_cost_reduction"]),
                   "delta_over_fixed_p20": float(conservative["relative_cost_reduction"] - fixed["relative_cost_reduction"])}
    gates_cfg = config["gates"]
    gates = {"exact_total_budget": conservative["selected_action_count"] == conservative["fixed_total_action_budget"],
             "minimum_group_coverage": conservative["minimum_group_case_coverage"] >= float(gates_cfg["minimum_group_case_coverage"]),
             "uncertainty_tracks_error": uncertainty["scale_error_spearman"] >= float(gates_cfg["minimum_scale_error_spearman"]),
             "improves_mean_compiler": improvement["delta_over_mean_compiler"] >= float(gates_cfg["minimum_delta_over_mean_compiler"]),
             "minimum_scene_support": conservative["scene_nonincreasing_count"] >= int(gates_cfg["minimum_nonincreasing_scene_support"])}
    verdict = config["verdict_on_pass"] if all(gates.values()) else config["verdict_on_failure"]
    summary = {"schema_version": config["output_schema_version"], "task_id": config["task_id"], "hypothesis_id": config["hypothesis_id"],
               "status": "done", "verdict": verdict, "role": config["role"], "claim_boundary": config["claim_boundary"],
               "training": training, "uncertainty": uncertainty, "heteroscedastic_conservative": conservative,
               "mean_compiler_baseline": mean_baseline, "fixed_p20": fixed, "selection_improvement": improvement, "gate_results": gates,
               "resources": {"gpu": torch.cuda.get_device_name(0), "peak_gpu_memory_gib": torch.cuda.max_memory_allocated()/(1024**3),
                             "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2), "wall_seconds": time.monotonic()-started}}
    _write(run_dir / "summary.json", summary); _write(run_dir / "status.json", {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    return {"run_dir": str(run_dir), "verdict": verdict, "gate_results": gates}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--runs-root", type=Path, required=True); parser.add_argument("--run-id", required=True)
    args = parser.parse_args(); print(json.dumps(run(args.config.resolve(), args.runs_root.resolve(), args.run_id), indent=2))


if __name__ == "__main__": main()
