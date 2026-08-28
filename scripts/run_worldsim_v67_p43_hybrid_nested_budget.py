"""Evaluate frozen hybrid action refinement under two strictly nested budgets."""

from __future__ import annotations

import argparse, json, resource, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch, yaml

from motion_proj.worldsim_v67.adaptive_budget import (
    _summarize_selected_indices, budget_horizon_conditioned_case_offset_dataset,
    group_coverage_constrained_selection, nested_group_budget_selection, score_case_offset,
)
from motion_proj.worldsim_v67.conditioned_action_compiler import score_conditioned_action_compiler
from motion_proj.worldsim_v67.listwise_action_compiler import BoundedListwiseCompiler, score_listwise_compiler
from scripts.run_worldsim_v65_p10v_action_visited_state_transfer import _within_case_selection
from scripts.run_worldsim_v67_p17_quantile_trajectory import _load
from scripts.run_worldsim_v67_p34_heteroscedastic_authority import _load_mean, _load_p20, _write


def _nested_dual_scores(arrays, low_scores, high_scores, low_offsets, high_offsets, low_fraction, high_fraction,
                        maximum_actions_per_case, minimum_case_coverage, scene_to_group, minimum_group_case_coverage):
    low = group_coverage_constrained_selection(
        arrays, low_scores, low_offsets, low_fraction, maximum_actions_per_case, minimum_case_coverage,
        scene_to_group, minimum_group_case_coverage,
    )
    cases = np.asarray(arrays["case_index"], dtype=np.int64)
    unique_cases = [case for case in np.unique(cases) if np.count_nonzero(cases == case) >= 2]
    high_total, high_slots = 0, []
    for row, case in enumerate(unique_cases):
        members = np.flatnonzero(cases == case)
        order = members[np.argsort(high_scores[members], kind="stable")]
        high_total += max(1, int(np.floor(high_fraction * len(members))))
        for action in order[:maximum_actions_per_case]:
            high_slots.append((float(high_scores[action] + high_offsets[row]), int(action)))
    low_selected = np.asarray(low["selected_action_indices"], dtype=np.int64)
    low_set = set(int(x) for x in low_selected)
    extension = [action for _, action in sorted(high_slots) if action not in low_set]
    high_selected = np.concatenate([low_selected, np.asarray(extension[:high_total-len(low_selected)], dtype=np.int64)])
    high = _summarize_selected_indices(arrays, high_selected, high_fraction, scene_to_group)
    return {"low_budget": low, "high_budget": high, "low_subset_of_high": True,
            "nested_action_count": int(len(low_selected))}


def run(config_path: Path, runs_root: Path, run_id: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v67" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _write(run_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    artifact = torch.load(
        runs_root / config["inputs"]["hybrid_compiler_run"] / config["inputs"]["hybrid_compiler_artifact"],
        map_location="cuda", weights_only=False,
    )
    model = BoundedListwiseCompiler(len(artifact["feature_names"]), list(artifact["hidden_dimensions"]),
                                    float(artifact["maximum_residual_cost"])).cuda()
    model.load_state_dict(artifact["state_dict"]); model.eval()
    mean = np.asarray(artifact["mean"], dtype=np.float32); scale = np.asarray(artifact["scale"], dtype=np.float32)
    p20, p20_mean, p20_scale = _load_p20(config, runs_root)
    allocator, allocator_mean, allocator_scale = _load_mean(config, runs_root)
    selection = dict(_load(Path(config["confirmation"]["cache_path"])))
    p20_scores = score_listwise_compiler(p20, selection, p20_mean, p20_scale); selection["base_score"] = p20_scores
    low_fraction, high_fraction = [float(x) for x in config["confirmation"]["nested_fractions"]]
    horizon = float(config["confirmation"]["horizon_seconds"])
    anchor = artifact.get("residual_budget_anchor_fraction")
    full = artifact.get("residual_budget_full_fraction")
    peak = artifact.get("residual_budget_peak_fraction"); upper = artifact.get("residual_budget_upper_anchor_fraction")
    low_scores = score_conditioned_action_compiler(model, selection, low_fraction, [horizon], mean, scale, "base_score", anchor, full, peak, upper)
    high_scores = score_conditioned_action_compiler(model, selection, high_fraction, [horizon], mean, scale, "base_score", anchor, full, peak, upper)
    low_cases = budget_horizon_conditioned_case_offset_dataset(selection, p20_scores, [low_fraction], [horizon])
    high_cases = budget_horizon_conditioned_case_offset_dataset(selection, p20_scores, [high_fraction], [horizon])
    low_offsets = score_case_offset(allocator, low_cases, allocator_mean, allocator_scale)
    high_offsets = score_case_offset(allocator, high_cases, allocator_mean, allocator_scale)
    compiler = config["compiler"]; scene_to_group = {int(k): int(v) for k, v in compiler["scene_to_group"].items()}
    shared = (int(compiler["maximum_actions_per_case"]), float(compiler["minimum_case_coverage"]), scene_to_group,
              float(compiler["minimum_group_case_coverage"]))
    hybrid = _nested_dual_scores(selection, low_scores, high_scores, low_offsets, high_offsets,
                                 low_fraction, high_fraction, *shared)
    baseline = nested_group_budget_selection(selection, p20_scores, low_offsets, high_offsets,
                                              low_fraction, high_fraction, *shared)
    improvement = {
        "low_delta_over_frozen_joint": float(hybrid["low_budget"]["relative_cost_reduction"] - baseline["low_budget"]["relative_cost_reduction"]),
        "high_delta_over_frozen_joint": float(hybrid["high_budget"]["relative_cost_reduction"] - baseline["high_budget"]["relative_cost_reduction"]),
    }
    gates_cfg = config["gates"]
    gates = {
        "exact_both_budgets": all(hybrid[x]["selected_action_count"] == hybrid[x]["fixed_total_action_budget"] for x in ["low_budget", "high_budget"]),
        "strict_nested_sets": bool(hybrid["low_subset_of_high"]),
        "minimum_group_coverage_both": min(hybrid[x]["minimum_group_case_coverage"] for x in ["low_budget", "high_budget"]) >= float(gates_cfg["minimum_group_case_coverage"]),
        "nonregression_over_joint_both": min(improvement.values()) >= float(gates_cfg["minimum_delta_over_joint_compiler"]),
        "minimum_scene_support_both": min(hybrid[x]["scene_nonincreasing_count"] for x in ["low_budget", "high_budget"]) >= int(gates_cfg["minimum_nonincreasing_scene_support"]),
    }
    verdict = config["verdict_on_pass"] if all(gates.values()) else config["verdict_on_failure"]
    summary = {"schema_version": config["output_schema_version"], "task_id": config["task_id"], "hypothesis_id": config["hypothesis_id"],
               "status": "done", "verdict": verdict, "role": config["role"], "claim_boundary": config["claim_boundary"],
               "hybrid_nested": hybrid, "frozen_joint_nested_baseline": baseline, "selection_improvement": improvement,
               "gate_results": gates, "resources": {"gpu": torch.cuda.get_device_name(0),
               "peak_gpu_memory_gib": torch.cuda.max_memory_allocated()/(1024**3),
               "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2), "wall_seconds": time.monotonic()-started}}
    _write(run_dir / "summary.json", summary); _write(run_dir / "status.json", {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    return {"run_dir": str(run_dir), "verdict": verdict, "gate_results": gates}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True); parser.add_argument("--run-id", required=True)
    args = parser.parse_args(); print(json.dumps(run(args.config.resolve(), args.runs_root.resolve(), args.run_id), indent=2))


if __name__ == "__main__": main()
