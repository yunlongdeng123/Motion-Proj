"""Evaluate the frozen P36 conditioned action compiler on the second consumed cohort."""

from __future__ import annotations

import argparse, json, resource, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch, yaml

from motion_proj.worldsim_v67.adaptive_budget import (
    budget_horizon_conditioned_case_offset_dataset, group_coverage_constrained_selection, score_case_offset,
)
from motion_proj.worldsim_v67.conditioned_action_compiler import score_conditioned_action_compiler
from motion_proj.worldsim_v67.listwise_action_compiler import BoundedListwiseCompiler, score_listwise_compiler
from scripts.run_worldsim_v65_p10v_action_visited_state_transfer import _within_case_selection
from scripts.run_worldsim_v67_p17_quantile_trajectory import _load
from scripts.run_worldsim_v67_p34_heteroscedastic_authority import _load_mean, _load_p20, _write


def run(config_path: Path, runs_root: Path, run_id: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v67" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _write(run_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    artifact = torch.load(
        runs_root / config["inputs"]["conditioned_compiler_run"] / config["inputs"]["conditioned_compiler_artifact"],
        map_location="cuda", weights_only=False,
    )
    model = BoundedListwiseCompiler(
        len(artifact["feature_names"]), list(artifact["hidden_dimensions"]), float(artifact["maximum_residual_cost"])
    ).cuda()
    model.load_state_dict(artifact["state_dict"]); model.eval()
    mean = np.asarray(artifact["mean"], dtype=np.float32); scale = np.asarray(artifact["scale"], dtype=np.float32)
    selection = _load(Path(config["confirmation"]["cache_path"]))
    fraction = float(config["confirmation"]["selected_fraction"]); horizon = float(config["confirmation"]["horizon_seconds"])
    scores = score_conditioned_action_compiler(model, selection, fraction, [horizon], mean, scale)
    p20, p20_mean, p20_scale = _load_p20(config, runs_root)
    p20_scores = score_listwise_compiler(p20, selection, p20_mean, p20_scale)
    frozen_model, frozen_mean, frozen_scale = _load_mean(config, runs_root)
    cases = budget_horizon_conditioned_case_offset_dataset(selection, p20_scores, [fraction], [horizon])
    frozen_offsets = score_case_offset(frozen_model, cases, frozen_mean, frozen_scale)
    scene_to_group = {int(k): int(v) for k, v in config["compiler"]["scene_to_group"].items()}
    shared = (
        fraction, int(config["compiler"]["maximum_actions_per_case"]), float(config["compiler"]["minimum_case_coverage"]),
        scene_to_group, float(config["compiler"]["minimum_group_case_coverage"]),
    )
    conditioned = group_coverage_constrained_selection(selection, scores, np.zeros(len(cases["target_offset"]), dtype=np.float32), *shared)
    frozen = group_coverage_constrained_selection(selection, p20_scores, frozen_offsets, *shared)
    fixed = _within_case_selection(
        np.asarray(selection["target_cost"], dtype=np.float32), p20_scores,
        np.asarray(selection["case_index"]), np.asarray(selection["scene_index"]), fraction,
    )
    improvement = {
        "delta_over_frozen_joint_compiler": float(conditioned["relative_cost_reduction"] - frozen["relative_cost_reduction"]),
        "delta_over_fixed_p20": float(conditioned["relative_cost_reduction"] - fixed["relative_cost_reduction"]),
    }
    gate_cfg = config["gates"]
    gates = {
        "exact_total_budget": conditioned["selected_action_count"] == conditioned["fixed_total_action_budget"],
        "minimum_group_coverage": conditioned["minimum_group_case_coverage"] >= float(gate_cfg["minimum_group_case_coverage"]),
        "improves_frozen_joint_compiler": improvement["delta_over_frozen_joint_compiler"] >= float(gate_cfg["minimum_delta_over_joint_compiler"]),
        "minimum_scene_support": conditioned["scene_nonincreasing_count"] >= int(gate_cfg["minimum_nonincreasing_scene_support"]),
    }
    verdict = config["verdict_on_pass"] if all(gates.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"], "hypothesis_id": config["hypothesis_id"],
        "status": "done", "verdict": verdict, "role": config["role"], "claim_boundary": config["claim_boundary"],
        "conditioned_action_selection": conditioned, "frozen_joint_compiler_baseline": frozen, "fixed_p20": fixed,
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
