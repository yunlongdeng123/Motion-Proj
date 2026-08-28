"""Train a fixed deep ensemble and consume disagreement as an authority priority."""

from __future__ import annotations

import argparse, json, resource, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch, yaml

from motion_proj.worldsim_v67.adaptive_budget import (
    BUDGET_HORIZON_CONDITIONED_FEATURE_NAMES,
    budget_horizon_conditioned_case_offset_dataset,
    group_coverage_constrained_selection,
    score_case_offset,
    train_case_offset,
)
from motion_proj.worldsim_v67.listwise_action_compiler import score_listwise_compiler
from scripts.run_worldsim_v65_p10v_action_visited_state_transfer import _within_case_selection
from scripts.run_worldsim_v67_p17_quantile_trajectory import _combine, _load
from scripts.run_worldsim_v67_p34_heteroscedastic_authority import _load_mean, _load_p20, _spearman, _write


def run(config_path: Path, runs_root: Path, run_id: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v67" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _write(run_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()

    p20, p20_mean, p20_scale = _load_p20(config, runs_root)
    train = _combine([Path(path) for path in config["inputs"]["train_action_caches"]])
    train_scores = score_listwise_compiler(p20, train, p20_mean, p20_scale)
    train_cases = budget_horizon_conditioned_case_offset_dataset(
        train,
        train_scores,
        [float(value) for value in config["model"]["training_selected_fractions"]],
        [float(value) for value in config["model"]["training_horizon_seconds_by_domain"]],
    )

    models = []
    training = []
    normalizer_mean = normalizer_scale = None
    for seed in [int(value) for value in config["model"]["member_seeds"]]:
        model, member_mean, member_scale, member_training = train_case_offset(train_cases, config["model"], seed)
        models.append(model)
        training.append({"seed": seed, **member_training})
        if normalizer_mean is None:
            normalizer_mean, normalizer_scale = member_mean, member_scale

    torch.save(
        {
            "state_dicts": [model.state_dict() for model in models],
            "member_seeds": [int(value) for value in config["model"]["member_seeds"]],
            "feature_names": list(BUDGET_HORIZON_CONDITIONED_FEATURE_NAMES),
            "hidden_dimension": int(config["model"]["hidden_dimension"]),
            "maximum_case_offset": float(config["model"]["maximum_case_offset"]),
            "mean": normalizer_mean,
            "scale": normalizer_scale,
        },
        run_dir / "ENSEMBLE_AUTHORITY_COMPILER.pt",
    )

    selection = _load(Path(config["confirmation"]["cache_path"]))
    scores = score_listwise_compiler(p20, selection, p20_mean, p20_scale)
    fraction = float(config["confirmation"]["selected_fraction"])
    horizon = float(config["confirmation"]["horizon_seconds"])
    cases = budget_horizon_conditioned_case_offset_dataset(selection, scores, [fraction], [horizon])
    member_predictions = np.stack(
        [score_case_offset(model, cases, normalizer_mean, normalizer_scale) for model in models], axis=0
    )
    ensemble_mean = member_predictions.mean(axis=0)
    ensemble_std = member_predictions.std(axis=0)
    conservative_offsets = ensemble_mean + float(config["compiler"]["uncertainty_weight"]) * ensemble_std

    frozen_model, frozen_mean, frozen_scale = _load_mean(config, runs_root)
    frozen_offsets = score_case_offset(frozen_model, cases, frozen_mean, frozen_scale)
    scene_to_group = {int(key): int(value) for key, value in config["compiler"]["scene_to_group"].items()}
    selection_args = (
        selection, scores, fraction, int(config["compiler"]["maximum_actions_per_case"]),
        float(config["compiler"]["minimum_case_coverage"]), scene_to_group,
        float(config["compiler"]["minimum_group_case_coverage"]),
    )
    conservative = group_coverage_constrained_selection(
        selection_args[0], selection_args[1], conservative_offsets, *selection_args[2:]
    )
    frozen = group_coverage_constrained_selection(
        selection_args[0], selection_args[1], frozen_offsets, *selection_args[2:]
    )
    fixed = _within_case_selection(
        np.asarray(selection["target_cost"], dtype=np.float32), scores,
        np.asarray(selection["case_index"]), np.asarray(selection["scene_index"]), fraction,
    )
    absolute_error = np.abs(np.asarray(cases["target_offset"], dtype=np.float32) - ensemble_mean)
    uncertainty = {
        "disagreement_error_spearman": _spearman(ensemble_std, absolute_error),
        "mean_disagreement": float(ensemble_std.mean()),
        "maximum_disagreement": float(ensemble_std.max()),
    }
    improvement = {
        "delta_over_frozen_mean_compiler": float(
            conservative["relative_cost_reduction"] - frozen["relative_cost_reduction"]
        ),
        "delta_over_fixed_p20": float(conservative["relative_cost_reduction"] - fixed["relative_cost_reduction"]),
    }
    gates_cfg = config["gates"]
    gates = {
        "exact_total_budget": conservative["selected_action_count"] == conservative["fixed_total_action_budget"],
        "minimum_group_coverage": conservative["minimum_group_case_coverage"] >= float(gates_cfg["minimum_group_case_coverage"]),
        "disagreement_tracks_error": uncertainty["disagreement_error_spearman"] >= float(gates_cfg["minimum_disagreement_error_spearman"]),
        "improves_frozen_mean_compiler": improvement["delta_over_frozen_mean_compiler"] >= float(gates_cfg["minimum_delta_over_mean_compiler"]),
        "minimum_scene_support": conservative["scene_nonincreasing_count"] >= int(gates_cfg["minimum_nonincreasing_scene_support"]),
    }
    verdict = config["verdict_on_pass"] if all(gates.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"], "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"], "status": "done", "verdict": verdict,
        "role": config["role"], "claim_boundary": config["claim_boundary"], "training": training,
        "uncertainty": uncertainty, "ensemble_conservative": conservative,
        "frozen_mean_compiler_baseline": frozen, "fixed_p20": fixed,
        "selection_improvement": improvement, "gate_results": gates,
        "resources": {
            "gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2),
            "wall_seconds": time.monotonic() - started,
        },
    }
    _write(run_dir / "summary.json", summary)
    _write(run_dir / "status.json", {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
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
