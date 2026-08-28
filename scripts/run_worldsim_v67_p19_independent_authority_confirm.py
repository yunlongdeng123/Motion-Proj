"""Confirm the frozen P18 selective authority compiler on an independent cohort."""

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
    MonotoneBenefitHead,
    authority_metrics,
    case_dataset,
    score_benefit_head,
)
from motion_proj.worldsim_v67.trajectory_quantile import materialize_quantiles
from scripts.run_worldsim_v65_p10v_action_visited_state_transfer import _within_case_selection
from scripts.run_worldsim_v67_p17_quantile_trajectory import _load


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
    artifact = torch.load(
        runs_root / config["inputs"]["compiler_run"] / config["inputs"]["compiler_artifact"],
        map_location="cuda",
        weights_only=False,
    )
    model = MonotoneBenefitHead(len(artifact["feature_names"])).cuda()
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    mean = np.asarray(artifact["mean"], dtype=np.float32)
    scale = np.asarray(artifact["scale"], dtype=np.float32)
    _write_json(
        run_dir / "model_frozen.json",
        {"p18_compiler_loaded_before_confirmation_materialization": True, "qmean_action_order_frozen": True},
    )
    cache_path = Path(config["confirmation_materialization"]["cache_path"])
    materialization = {"cache_path": str(cache_path), "cache_reused": cache_path.is_file()}
    if not cache_path.is_file():
        materialization.update(
            materialize_quantiles(config["confirmation_materialization"]["data"], runs_root, cache_path)
        )
    actions = _load(cache_path)
    within_fraction = float(config["compiler"]["within_case_selected_fraction"])
    cases = case_dataset(actions, within_fraction)
    scores = score_benefit_head(model, cases, mean, scale)
    authority = authority_metrics(
        actions, cases, scores, float(config["compiler"]["authority_case_fraction"])
    )
    ungated = _within_case_selection(
        np.asarray(actions["target_cost"], dtype=np.float32),
        np.asarray(actions["qmean"], dtype=np.float32),
        np.asarray(actions["case_index"]),
        np.asarray(actions["scene_index"]),
        within_fraction,
    )
    improvement = {
        "authority_reduction_delta_over_ungated_qmean": float(
            authority["relative_cost_reduction"] - ungated["relative_cost_reduction"]
        )
    }
    gates_config = config["gates"]
    gates = {
        "minimum_authority_fraction": authority["authority_fraction"] >= float(gates_config["minimum_authority_fraction"]),
        "minimum_authority_cost_reduction": authority["relative_cost_reduction"] >= float(gates_config["minimum_authority_cost_reduction"]),
        "minimum_reduction_delta_over_ungated_qmean": improvement["authority_reduction_delta_over_ungated_qmean"] >= float(gates_config["minimum_reduction_delta_over_ungated_qmean"]),
        "minimum_nonincreasing_scene_support": authority["scene_nonincreasing_count"] >= int(gates_config["minimum_nonincreasing_scene_support"]),
    }
    verdict = "supported_independent_selective_authority_confirmation" if all(gates.values()) else "rejected_independent_selective_authority_confirmation"
    summary = {
        "schema_version": "worldsim_v67.p19_independent_authority_summary.v1",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "role": config["role"],
        "claim_boundary": config["claim_boundary"],
        "materialization": materialization,
        "benefit_spearman": float(spearmanr(cases["benefit"], scores).statistic),
        "authority_metrics": authority,
        "ungated_qmean": ungated,
        "improvement": improvement,
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
