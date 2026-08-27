"""Compare mean and one preregistered smooth-tail visited-state risk aggregator."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.run_worldsim_v65_p1r4_trajectory_visited_state import (
    _compare_selected,
    _continuous_metrics,
    _load_remaining,
    _q0_probabilities,
    _selected_cost,
    _unsafe_metrics,
    _write_json,
)


def _aggregate_units(
    arrays: dict[str, np.ndarray],
    q0: np.ndarray,
    *,
    minimum_visited_points: int,
    temperature: float,
) -> dict[str, np.ndarray]:
    rows: dict[str, list] = {
        name: []
        for name in (
            "mean_score",
            "smooth_tail_score",
            "target_cost",
            "unsafe",
            "visited_count",
            "hidden_free_count",
            "scene_index",
            "unit_index",
            "is_train",
        )
    }
    scene_index = arrays["scene_index"]
    unit_index = arrays["unit_index"]
    for scene in np.unique(scene_index):
        for unit in np.unique(unit_index[scene_index == scene]):
            members = (scene_index == scene) & (unit_index == unit)
            visited = members & arrays["route"]
            visited_count = int(np.count_nonzero(visited))
            if visited_count < minimum_visited_points:
                continue
            scores = q0[visited].astype(np.float64)
            scaled = scores / temperature
            weights = np.exp(scaled - scaled.max())
            smooth_tail = float(np.sum(weights * scores) / np.sum(weights))
            hidden_free_count = int(np.count_nonzero(arrays["hidden_free"][visited]))
            roles = np.unique(arrays["is_train"][members])
            if roles.shape[0] != 1:
                raise RuntimeError("mixed train/evaluation role inside one unit")
            rows["mean_score"].append(float(scores.mean()))
            rows["smooth_tail_score"].append(smooth_tail)
            rows["target_cost"].append(hidden_free_count / visited_count)
            rows["unsafe"].append(hidden_free_count > 0)
            rows["visited_count"].append(visited_count)
            rows["hidden_free_count"].append(hidden_free_count)
            rows["scene_index"].append(int(scene))
            rows["unit_index"].append(int(unit))
            rows["is_train"].append(bool(roles[0]))
    if not rows["mean_score"]:
        raise RuntimeError("no eligible trajectory-level units")
    return {
        "mean_score": np.asarray(rows["mean_score"], dtype=np.float32),
        "smooth_tail_score": np.asarray(rows["smooth_tail_score"], dtype=np.float32),
        "target_cost": np.asarray(rows["target_cost"], dtype=np.float32),
        "unsafe": np.asarray(rows["unsafe"], dtype=bool),
        "visited_count": np.asarray(rows["visited_count"], dtype=np.int32),
        "hidden_free_count": np.asarray(rows["hidden_free_count"], dtype=np.int32),
        "scene_index": np.asarray(rows["scene_index"], dtype=np.uint8),
        "unit_index": np.asarray(rows["unit_index"], dtype=np.uint8),
        "is_train": np.asarray(rows["is_train"], dtype=bool),
    }


def run(config_path: Path, runs_root: Path, run_id: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v65" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(
        run_dir / "status.json",
        {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    cache_path = Path(config["inputs"]["compact_cache"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        remaining_future = executor.submit(_load_remaining, cache_path)
        q0 = _q0_probabilities(cache_path)
        arrays = remaining_future.result()
    units = _aggregate_units(
        arrays,
        q0,
        minimum_visited_points=int(config["trajectory_contract"]["minimum_visited_points_per_unit"]),
        temperature=float(config["aggregators"]["softmax_temperature_probability_units"]),
    )
    evaluate = ~units["is_train"]
    target = units["target_cost"][evaluate]
    unsafe = units["unsafe"][evaluate]
    scenes = units["scene_index"][evaluate]
    mean_score = units["mean_score"][evaluate]
    tail_score = units["smooth_tail_score"][evaluate]
    continuous = {
        "qmean": _continuous_metrics(target, mean_score),
        "qsoft_tail": _continuous_metrics(target, tail_score),
    }
    continuous["soft_tail_minus_mean_spearman"] = (
        continuous["qsoft_tail"]["spearman"] - continuous["qmean"]["spearman"]
    )
    unsafe_ranking = {
        "qmean": _unsafe_metrics(unsafe, mean_score),
        "qsoft_tail": _unsafe_metrics(unsafe, tail_score),
    }
    unsafe_ranking["soft_tail_minus_mean_auroc"] = (
        unsafe_ranking["qsoft_tail"]["auroc"] - unsafe_ranking["qmean"]["auroc"]
    )
    coverage = float(config["evaluation"]["matched_safe_coverage"])
    mean_selected = _selected_cost(target, mean_score, scenes, coverage=coverage)
    tail_selected = _selected_cost(target, tail_score, scenes, coverage=coverage)
    comparison = _compare_selected(mean_selected, tail_selected)
    thresholds = config["incremental_gates"]
    gates = {
        "minimum_selected_cost_reduction_vs_mean": comparison[
            "relative_selected_cost_reduction_vs_q0"
        ]
        >= float(thresholds["minimum_selected_cost_reduction_vs_mean"]),
        "minimum_unsafe_auroc_gain": unsafe_ranking["soft_tail_minus_mean_auroc"]
        >= float(thresholds["minimum_unsafe_auroc_gain"]),
        "maximum_spearman_regression": continuous["soft_tail_minus_mean_spearman"]
        >= -float(thresholds["maximum_spearman_regression"]),
        "scene_direction_support": comparison["scene_lower_count"]
        > comparison["scene_higher_count"],
    }
    verdict = (
        "positive_train_only_smooth_tail_visited_state_aggregation"
        if all(gates.values())
        else "no_clear_train_only_smooth_tail_visited_state_increment"
    )
    summary = {
        "schema_version": "worldsim_v65.p1r6_smooth_tail_visited_state_summary.v1",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "claim_boundary": config["claim_boundary"],
        "source_cache": str(cache_path),
        "native_hidden_loaded": False,
        "train_unit_count": int(np.count_nonzero(units["is_train"])),
        "evaluation_unit_count": int(np.count_nonzero(evaluate)),
        "evaluation_visited_point_count": int(units["visited_count"][evaluate].sum()),
        "evaluation_hidden_free_count": int(units["hidden_free_count"][evaluate].sum()),
        "continuous_metrics": continuous,
        "unsafe_ranking": unsafe_ranking,
        "selected_cost": {
            "qmean": mean_selected,
            "qsoft_tail": tail_selected,
            "comparison": comparison,
        },
        "incremental_gate_results": gates,
        "formal_v65_selection_read": False,
        "resources": {
            "gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
            "wall_seconds": time.monotonic() - started,
        },
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(
        run_dir / "status.json",
        {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    return {
        "run_dir": str(run_dir),
        "verdict": verdict,
        "incremental_gate_results": gates,
        "resources": summary["resources"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.runs_root.resolve(), args.run_id), indent=2))


if __name__ == "__main__":
    main()
