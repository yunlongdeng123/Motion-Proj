"""Train selective authority over the frozen P20 listwise action compiler."""

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

from motion_proj.worldsim_v67.listwise_action_compiler import BoundedListwiseCompiler, score_listwise_compiler
from motion_proj.worldsim_v67.selective_authority import FEATURE_NAMES, authority_metrics, case_dataset, score_benefit_head, train_benefit_head
from motion_proj.worldsim_v67.trajectory_quantile import materialize_quantiles
from scripts.run_worldsim_v65_p10v_action_visited_state_transfer import _within_case_selection
from scripts.run_worldsim_v67_p17_quantile_trajectory import _combine, _load


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _load_compiler(config: dict, runs_root: Path):
    artifact = torch.load(
        runs_root / config["inputs"]["action_compiler_run"] / config["inputs"]["action_compiler_artifact"],
        map_location="cuda", weights_only=False,
    )
    model = BoundedListwiseCompiler(
        len(artifact["feature_names"]), list(artifact["hidden_dimensions"]), float(artifact["maximum_residual_cost"])
    ).cuda()
    model.load_state_dict(artifact["state_dict"])
    return model.eval(), np.asarray(artifact["mean"], dtype=np.float32), np.asarray(artifact["scale"], dtype=np.float32)


def _compiled_arrays(actions: dict[str, np.ndarray], scores: np.ndarray) -> dict[str, np.ndarray]:
    compiled = {name: np.asarray(value).copy() for name, value in actions.items()}
    compiled["qmean"] = np.asarray(scores, dtype=np.float32)
    return compiled


def run(config_path: Path, runs_root: Path, run_id: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v67" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _write_json(run_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    action_model, action_mean, action_scale = _load_compiler(config, runs_root)
    train_actions = _combine([Path(path) for path in config["inputs"]["train_action_caches"]])
    train_action_scores = score_listwise_compiler(action_model, train_actions, action_mean, action_scale)
    train_compiled = _compiled_arrays(train_actions, train_action_scores)
    selected_fraction = float(config["compiler"]["within_case_selected_fraction"])
    train_cases = case_dataset(train_compiled, selected_fraction)
    authority_model, authority_mean, authority_scale, training = train_benefit_head(
        train_cases, config["model"], int(config["seed"])
    )
    train_authority_scores = score_benefit_head(authority_model, train_cases, authority_mean, authority_scale)
    torch.save(
        {"state_dict": authority_model.state_dict(), "feature_names": list(FEATURE_NAMES), "mean": authority_mean, "scale": authority_scale,
         "frozen_action_compiler_run": config["inputs"]["action_compiler_run"]},
        run_dir / "SELECTIVE_LISTWISE_AUTHORITY.pt",
    )
    _write_json(
        run_dir / "model_frozen.json",
        {"p20_action_compiler_frozen": True, "p21_authority_frozen_before_confirmation_materialization": True,
         "train_case_count": int(len(train_cases["case_index"])), "development_domain_count": 5},
    )
    cache_path = Path(config["confirmation_materialization"]["cache_path"])
    materialization = {"cache_path": str(cache_path), "cache_reused": cache_path.is_file()}
    if not cache_path.is_file():
        materialization.update(materialize_quantiles(config["confirmation_materialization"]["data"], runs_root, cache_path))
    selection_actions = _load(cache_path)
    selection_action_scores = score_listwise_compiler(action_model, selection_actions, action_mean, action_scale)
    selection_compiled = _compiled_arrays(selection_actions, selection_action_scores)
    selection_cases = case_dataset(selection_compiled, selected_fraction)
    selection_authority_scores = score_benefit_head(
        authority_model, selection_cases, authority_mean, authority_scale
    )
    authority = authority_metrics(
        selection_actions, selection_cases, selection_authority_scores,
        float(config["compiler"]["authority_case_fraction"]),
    )
    target = np.asarray(selection_actions["target_cost"], dtype=np.float32)
    cases = np.asarray(selection_actions["case_index"])
    scenes = np.asarray(selection_actions["scene_index"])
    ungated_compiled = _within_case_selection(target, selection_action_scores, cases, scenes, selected_fraction)
    ungated_qmean = _within_case_selection(
        target, np.asarray(selection_actions["qmean"], dtype=np.float32), cases, scenes, selected_fraction
    )
    improvement = {
        "authority_delta_over_ungated_compiled": float(authority["relative_cost_reduction"] - ungated_compiled["relative_cost_reduction"]),
        "authority_delta_over_ungated_qmean": float(authority["relative_cost_reduction"] - ungated_qmean["relative_cost_reduction"]),
        "compiled_delta_over_qmean": float(ungated_compiled["relative_cost_reduction"] - ungated_qmean["relative_cost_reduction"]),
    }
    gate_config = config["gates"]
    gates = {
        "minimum_authority_fraction": authority["authority_fraction"] >= float(gate_config["minimum_authority_fraction"]),
        "minimum_authority_cost_reduction": authority["relative_cost_reduction"] >= float(gate_config["minimum_authority_cost_reduction"]),
        "minimum_authority_delta_over_ungated_qmean": improvement["authority_delta_over_ungated_qmean"] >= float(gate_config["minimum_authority_delta_over_ungated_qmean"]),
        "minimum_nonincreasing_scene_support": authority["scene_nonincreasing_count"] >= int(gate_config["minimum_nonincreasing_scene_support"]),
    }
    verdict = "supported_selective_listwise_trajectory_authority" if all(gates.values()) else "rejected_selective_listwise_trajectory_authority"
    summary = {
        "schema_version": "worldsim_v67.p21_selective_listwise_authority_summary.v1",
        "task_id": config["task_id"], "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"], "claim_boundary": config["claim_boundary"], "training": training,
        "train_benefit_spearman": float(spearmanr(train_cases["benefit"], train_authority_scores).statistic),
        "confirmation_materialization": materialization,
        "selection_benefit_spearman": float(spearmanr(selection_cases["benefit"], selection_authority_scores).statistic),
        "authority_metrics": authority, "ungated_compiled": ungated_compiled, "ungated_qmean": ungated_qmean,
        "improvement": improvement, "gate_results": gates, "failure_ledger_delta": "pending_result",
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
