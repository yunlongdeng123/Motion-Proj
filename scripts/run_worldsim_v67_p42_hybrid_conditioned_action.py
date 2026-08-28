"""Train a case-centered action refinement over frozen P20 and compose it with P31 allocation."""

from __future__ import annotations

import argparse, json, resource, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch, yaml

from motion_proj.worldsim_v67.adaptive_budget import (
    budget_horizon_conditioned_case_offset_dataset, group_coverage_constrained_selection, score_case_offset,
)
from motion_proj.worldsim_v67.conditioned_action_compiler import (
    CONDITIONED_FEATURE_NAMES, score_conditioned_action_compiler, train_conditioned_action_compiler,
)
from motion_proj.worldsim_v67.listwise_action_compiler import score_listwise_compiler
from scripts.run_worldsim_v65_p10v_action_visited_state_transfer import _within_case_selection
from scripts.run_worldsim_v67_p17_quantile_trajectory import _combine, _load
from scripts.run_worldsim_v67_p34_heteroscedastic_authority import _load_mean, _load_p20, _write


def run(config_path: Path, runs_root: Path, run_id: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v67" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _write(run_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    p20, p20_mean, p20_scale = _load_p20(config, runs_root)
    train = _combine([Path(path) for path in config["inputs"]["train_action_caches"]])
    train = dict(train); train["base_score"] = score_listwise_compiler(p20, train, p20_mean, p20_scale)
    model, mean, scale, training = train_conditioned_action_compiler(train, config["model"], int(config["seed"]))
    torch.save({
        "state_dict": model.state_dict(), "feature_names": list(CONDITIONED_FEATURE_NAMES),
        "hidden_dimensions": list(config["model"]["hidden_dimensions"]),
        "maximum_residual_cost": float(config["model"]["maximum_residual_cost"]), "mean": mean, "scale": scale,
        "base_score": "frozen_p20",
        "residual_budget_anchor_fraction": config["model"].get("residual_budget_anchor_fraction"),
        "residual_budget_full_fraction": config["model"].get("residual_budget_full_fraction"),
    }, run_dir / "HYBRID_CONDITIONED_ACTION_COMPILER.pt")

    selection = dict(_load(Path(config["confirmation"]["cache_path"])))
    fraction = float(config["confirmation"]["selected_fraction"]); horizon = float(config["confirmation"]["horizon_seconds"])
    p20_scores = score_listwise_compiler(p20, selection, p20_mean, p20_scale)
    selection["base_score"] = p20_scores
    hybrid_scores = score_conditioned_action_compiler(
        model, selection, fraction, [horizon], mean, scale, base_score_key="base_score",
        residual_budget_anchor_fraction=config["model"].get("residual_budget_anchor_fraction"),
        residual_budget_full_fraction=config["model"].get("residual_budget_full_fraction"),
    )
    allocator, allocator_mean, allocator_scale = _load_mean(config, runs_root)
    cases = budget_horizon_conditioned_case_offset_dataset(selection, p20_scores, [fraction], [horizon])
    case_offsets = score_case_offset(allocator, cases, allocator_mean, allocator_scale)
    scene_to_group = {int(k): int(v) for k, v in config["compiler"]["scene_to_group"].items()}
    shared = (
        fraction, int(config["compiler"]["maximum_actions_per_case"]), float(config["compiler"]["minimum_case_coverage"]),
        scene_to_group, float(config["compiler"]["minimum_group_case_coverage"]),
    )
    hybrid = group_coverage_constrained_selection(selection, hybrid_scores, case_offsets, *shared)
    frozen = group_coverage_constrained_selection(selection, p20_scores, case_offsets, *shared)
    fixed = _within_case_selection(
        np.asarray(selection["target_cost"], dtype=np.float32), p20_scores,
        np.asarray(selection["case_index"]), np.asarray(selection["scene_index"]), fraction,
    )
    improvement = {
        "delta_over_frozen_joint_compiler": float(hybrid["relative_cost_reduction"] - frozen["relative_cost_reduction"]),
        "delta_over_fixed_p20": float(hybrid["relative_cost_reduction"] - fixed["relative_cost_reduction"]),
    }
    gate_cfg = config["gates"]
    gates = {
        "exact_total_budget": hybrid["selected_action_count"] == hybrid["fixed_total_action_budget"],
        "minimum_group_coverage": hybrid["minimum_group_case_coverage"] >= float(gate_cfg["minimum_group_case_coverage"]),
        "improves_frozen_joint_compiler": improvement["delta_over_frozen_joint_compiler"] >= float(gate_cfg["minimum_delta_over_joint_compiler"]),
        "minimum_scene_support": hybrid["scene_nonincreasing_count"] >= int(gate_cfg["minimum_nonincreasing_scene_support"]),
    }
    verdict = config["verdict_on_pass"] if all(gates.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"], "hypothesis_id": config["hypothesis_id"],
        "status": "done", "verdict": verdict, "role": config["role"], "claim_boundary": config["claim_boundary"],
        "training": training, "hybrid_selection": hybrid, "frozen_joint_compiler_baseline": frozen, "fixed_p20": fixed,
        "selection_improvement": improvement, "gate_results": gates,
        "resources": {"gpu": torch.cuda.get_device_name(0), "peak_gpu_memory_gib": torch.cuda.max_memory_allocated()/(1024**3),
                      "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2), "wall_seconds": time.monotonic()-started},
    }
    _write(run_dir / "summary.json", summary); _write(run_dir / "status.json", {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    return {"run_dir": str(run_dir), "verdict": verdict, "gate_results": gates}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True); parser.add_argument("--run-id", required=True)
    args = parser.parse_args(); print(json.dumps(run(args.config.resolve(), args.runs_root.resolve(), args.run_id), indent=2))


if __name__ == "__main__": main()
