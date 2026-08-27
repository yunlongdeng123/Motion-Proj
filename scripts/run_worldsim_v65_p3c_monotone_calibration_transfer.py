"""Run one independent transfer read of the frozen R7 monotone Qmean calibrator."""

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

from scripts.run_worldsim_v65_p1r4_trajectory_visited_state import (
    _continuous_metrics,
    _unsafe_metrics,
    _write_json,
)
from scripts.run_worldsim_v65_p1r7_monotone_visited_state_calibration import (
    _calibration_error,
    _selected_indices,
)
from scripts.run_worldsim_v65_p2v_visited_state_transfer import _materialize


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
    cache_reused = cache_path.is_file()
    materialization = {"source_unit_count": 72, "excluded_unit_count": 0}
    if not cache_reused:
        materialization = _materialize(config, runs_root, cache_path)
    with np.load(cache_path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}

    target = arrays["target_cost"]
    base = arrays["qagg"]
    unsafe = arrays["unsafe"]
    scenes = arrays["scene_index"]
    calibrator = config["calibrator"]
    clip = float(calibrator["probability_clip"])
    clipped = np.clip(base, clip, 1.0 - clip)
    logit = np.log(clipped / (1.0 - clipped))
    calibrated = 1.0 / (
        1.0
        + np.exp(
            -(
                float(calibrator["slope"]) * logit
                + float(calibrator["bias"])
            )
        )
    )
    calibrated = calibrated.astype(np.float32)

    continuous = {
        "qmean": _continuous_metrics(target, base),
        "frozen_monotone": _continuous_metrics(target, calibrated),
    }
    base_mse = continuous["qmean"]["mse"]
    calibrated_mse = continuous["frozen_monotone"]["mse"]
    continuous["relative_mse_reduction"] = (
        (base_mse - calibrated_mse) / base_mse if base_mse > 0 else 0.0
    )
    continuous["spearman_delta"] = (
        continuous["frozen_monotone"]["spearman"]
        - continuous["qmean"]["spearman"]
    )
    unsafe_ranking = {
        "qmean": _unsafe_metrics(unsafe, base),
        "frozen_monotone": _unsafe_metrics(unsafe, calibrated),
    }
    unsafe_ranking["auroc_delta"] = (
        unsafe_ranking["frozen_monotone"]["auroc"]
        - unsafe_ranking["qmean"]["auroc"]
    )

    bins = int(config["evaluation"]["calibration_bins"])
    calibration = {
        "qmean": _calibration_error(target, base, bins),
        "frozen_monotone": _calibration_error(target, calibrated, bins),
    }
    base_error = calibration["qmean"]["absolute_calibration_error"]
    calibrated_error = calibration["frozen_monotone"]["absolute_calibration_error"]
    calibration["relative_error_reduction"] = (
        (base_error - calibrated_error) / base_error if base_error > 0 else 0.0
    )

    scene_rows = []
    for scene in np.unique(scenes):
        members = scenes == scene
        before = float(np.mean(np.square(base[members] - target[members])))
        after = float(np.mean(np.square(calibrated[members] - target[members])))
        scene_rows.append(
            {
                "scene_index": int(scene),
                "qmean_mse": before,
                "calibrated_mse": after,
                "delta": after - before,
            }
        )
    scene_lower = sum(row["delta"] < 0 for row in scene_rows)
    scene_equal = sum(row["delta"] == 0 for row in scene_rows)
    scene_higher = sum(row["delta"] > 0 for row in scene_rows)

    coverage = float(config["evaluation"]["matched_safe_coverage"])
    selected_base = _selected_indices(base, coverage)
    selected_calibrated = _selected_indices(calibrated, coverage)
    selected_exact = bool(np.array_equal(selected_base, selected_calibrated))
    thresholds = config["gates"]
    tolerance = float(thresholds["ranking_tolerance"])
    gates = {
        "minimum_mse_reduction": continuous["relative_mse_reduction"]
        >= float(thresholds["minimum_mse_reduction"]),
        "minimum_calibration_error_reduction": calibration[
            "relative_error_reduction"
        ]
        >= float(thresholds["minimum_calibration_error_reduction"]),
        "minimum_scene_mse_support": scene_lower
        >= int(thresholds["minimum_scene_mse_support"]),
        "spearman_nonregression": continuous["spearman_delta"] >= -tolerance,
        "unsafe_auroc_nonregression": unsafe_ranking["auroc_delta"]
        >= -tolerance,
        "exact_selected_set": selected_exact,
    }
    verdict = (
        "supported_independent_monotone_visited_state_calibration_transfer"
        if all(gates.values())
        else "rejected_independent_monotone_visited_state_calibration_transfer"
    )
    summary = {
        "schema_version": "worldsim_v65.p3c_monotone_calibration_transfer_summary.v1",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "claim_boundary": config["claim_boundary"],
        "cache_reused": cache_reused,
        **materialization,
        "eligible_unit_count": int(target.shape[0]),
        "unsafe_unit_count": int(np.count_nonzero(unsafe)),
        "visited_point_count": int(arrays["visited_count"].sum()),
        "hidden_free_count": int(arrays["hidden_free_count"].sum()),
        "calibrator": dict(calibrator),
        "continuous_metrics": continuous,
        "unsafe_ranking": unsafe_ranking,
        "calibration": calibration,
        "scene_mse": {
            "lower_equal_higher": [scene_lower, scene_equal, scene_higher],
            "rows": scene_rows,
        },
        "selected_40_percent": {
            "exact_same_indices": selected_exact,
            "selected_count": int(selected_base.size),
            "mean_actual_cost": float(target[selected_base].mean()),
        },
        "gate_results": gates,
        "formal_v65_calibration_read": True,
        "calibrator_refit": False,
        "resources": {
            "gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024**2),
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
        "gate_results": gates,
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
