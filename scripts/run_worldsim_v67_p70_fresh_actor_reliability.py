"""Evaluate frozen trajectory-conditioned Actor reliability on fresh processed scenes."""

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

from motion_proj.worldsim_v67.actor_state_reliability import (
    ACTOR_FEATURE_NAMES,
    ReliabilityMLP,
    evaluate_reliability,
    materialize_actor_query_rows,
    predict_reliability,
)


def _select_by_scene(score: np.ndarray, scenes: np.ndarray, fraction: float) -> np.ndarray:
    selected: list[int] = []
    for scene in np.unique(scenes):
        members = np.flatnonzero(scenes == scene)
        count = max(1, int(np.floor(len(members) * fraction)))
        selected.extend(members[np.argsort(score[members], kind="mergesort")[:count]].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_dir = args.runs_root / "worldsim_v67" / config["task_id"] / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    started = time.monotonic()

    processed_root = Path(config["data"]["processed_root"])
    scene_dirs = [processed_root / f"{int(scene):03d}" for scene in config["data"]["scene_indices"]]
    arrays = materialize_actor_query_rows(
        scene_dirs, [float(config["data"]["horizon_seconds"])], config["data"]
    )
    np.savez_compressed(run_dir / "FRESH_ACTOR_QUERY_ROWS.npz", **arrays)

    source = args.runs_root / config["source"]["run"]
    artifact = torch.load(source / config["source"]["artifact"], map_location="cuda")
    hidden = artifact["hidden_dimensions"]
    query_model = ReliabilityMLP(len(artifact["feature_names"]), hidden).cuda()
    query_model.load_state_dict(artifact["query_model_state_dict"])
    actor_model = ReliabilityMLP(len(ACTOR_FEATURE_NAMES), hidden).cuda()
    actor_model.load_state_dict(artifact["actor_only_model_state_dict"])
    mean = np.asarray(artifact["feature_mean"], dtype=np.float32)
    scale = np.asarray(artifact["feature_scale"], dtype=np.float32)
    query_score = predict_reliability(
        query_model.eval(), arrays["features"], mean, scale, actor_only=False
    )
    actor_score = predict_reliability(
        actor_model.eval(), arrays["features"], mean, scale, actor_only=True
    )
    evaluation = evaluate_reliability(arrays, query_score, actor_score, config["evaluation"])

    scenes = np.asarray(arrays["scene_index"])
    target = np.asarray(arrays["target_cost"], dtype=np.float64)
    unreliable = (
        np.asarray(arrays["raw_actor_state_error_m"])
        > float(config["evaluation"]["unreliable_actor_state_error_m"])
    ) & (
        np.asarray(arrays["predicted_minimum_separation_m"])
        <= float(config["evaluation"]["unreliable_exposure_radius_m"])
    )
    coverage = float(config["selection"]["coverage_fraction"])
    query_selected = _select_by_scene(query_score, scenes, coverage)
    actor_selected = _select_by_scene(actor_score, scenes, coverage)
    all_cost = float(target.mean())
    all_prevalence = float(unreliable.mean())
    query_cost = float(target[query_selected].mean())
    actor_cost = float(target[actor_selected].mean())
    query_prevalence = float(unreliable[query_selected].mean())
    actor_prevalence = float(unreliable[actor_selected].mean())
    scene_rows = []
    for scene in np.unique(scenes):
        members = np.flatnonzero(scenes == scene)
        chosen = query_selected[np.isin(query_selected, members)]
        scene_rows.append({
            "scene_index": int(scene),
            "row_count": int(len(members)),
            "selected_count": int(len(chosen)),
            "all_mean_cost": float(target[members].mean()),
            "selected_mean_cost": float(target[chosen].mean()),
        })
    selection = {
        "row_count": int(len(target)),
        "selected_row_count": int(len(query_selected)),
        "achieved_coverage": float(len(query_selected) / len(target)),
        "all_mean_cost": all_cost,
        "query_selected_mean_cost": query_cost,
        "actor_only_selected_mean_cost": actor_cost,
        "query_cost_reduction": (all_cost - query_cost) / max(all_cost, 1e-12),
        "query_cost_delta_below_actor_only": actor_cost - query_cost,
        "all_unreliable_prevalence": all_prevalence,
        "query_selected_unreliable_prevalence": query_prevalence,
        "actor_only_selected_unreliable_prevalence": actor_prevalence,
        "query_unreliable_prevalence_reduction": (
            (all_prevalence - query_prevalence) / max(all_prevalence, 1e-12)
        ),
        "scene_nonincreasing_count": int(sum(
            row["selected_mean_cost"] <= row["all_mean_cost"] for row in scene_rows
        )),
        "scene_count": int(len(scene_rows)),
        "scene_rows": scene_rows,
    }
    gates = {
        "minimum_query_conditioned_spearman": (
            evaluation["query_conditioned_spearman"]
            >= float(config["gates"]["minimum_query_conditioned_spearman"])
        ),
        "minimum_mae_reduction_over_actor_only": (
            evaluation["mae_reduction_over_actor_only"]
            >= float(config["gates"]["minimum_mae_reduction_over_actor_only"])
        ),
        "minimum_selective_cost_reduction": (
            selection["query_cost_reduction"]
            >= float(config["gates"]["minimum_selective_cost_reduction"])
        ),
        "minimum_selective_unreliable_prevalence_reduction": (
            selection["query_unreliable_prevalence_reduction"]
            >= float(config["gates"]["minimum_selective_unreliable_prevalence_reduction"])
        ),
    }
    verdict = config["verdict_on_pass"] if all(gates.values()) else config["verdict_on_failure"]
    summary = {
        "schema_version": config["output_schema_version"],
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "role": config["role"],
        "data": {
            "processed_root": str(processed_root),
            "scene_indices": [int(scene) for scene in config["data"]["scene_indices"]],
            "source_overlap_scene_indices_excluded": [
                int(scene) for scene in config["data"]["source_overlap_scene_indices_excluded"]
            ],
            "horizon_seconds": float(config["data"]["horizon_seconds"]),
        },
        "fresh_population_evaluation": evaluation,
        "selection": selection,
        "gate_results": gates,
        "resources": {
            "gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "wall_seconds": time.monotonic() - started,
        },
        "claim_boundary": config["claim_boundary"],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (run_dir / "status.json").write_text(
        json.dumps({
            "status": "done",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "run_dir": str(run_dir), "verdict": verdict, "gate_results": gates
    }, indent=2))


if __name__ == "__main__":
    main()
