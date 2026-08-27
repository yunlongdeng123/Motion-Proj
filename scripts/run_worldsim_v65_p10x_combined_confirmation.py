"""Run the one-shot combined visited-state confirmation."""

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
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.run_worldsim_v65_p10v_action_visited_state_transfer import (
    _materialize,
    _pairwise_concordance,
    _within_case_selection,
)
from scripts.run_worldsim_v65_p1r4_trajectory_visited_state import _continuous_metrics
from scripts.run_worldsim_v65_p1r7_monotone_visited_state_calibration import (
    _calibration_error,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _unsafe_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, scores))
        if np.unique(labels).size == 2
        else float("nan"),
        "auprc": float(average_precision_score(labels, scores)),
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
    cache_reused = cache_path.is_file()
    materialization = {
        "source_case_count": 72,
        "source_action_count": 864,
        "excluded_action_count": 0,
    }
    if not cache_reused:
        materialization = _materialize(config, runs_root, cache_path)
    with np.load(cache_path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}

    target = arrays["target_cost"]
    scores = arrays["qmean"]
    unsafe = arrays["unsafe"]
    cases = arrays["case_index"]
    scenes = arrays["scene_index"]
    evaluation = config["evaluation"]

    action_spearman = float(spearmanr(target, scores).statistic)
    action_unsafe = _unsafe_metrics(unsafe, scores)
    pairwise = _pairwise_concordance(
        target,
        scores,
        cases,
        float(evaluation["pairwise_minimum_target_gap"]),
    )
    selected = _within_case_selection(
        target,
        scores,
        cases,
        scenes,
        float(evaluation["within_case_selected_fraction"]),
    )

    nominal = arrays["action_index"] == int(evaluation["nominal_action_index"])
    route_target = target[nominal]
    route_raw = scores[nominal]
    route_unsafe_labels = unsafe[nominal]
    route_scenes = scenes[nominal]
    calibrator = config["calibrator"]
    clip = float(calibrator["probability_clip"])
    clipped = np.clip(route_raw, clip, 1.0 - clip)
    logit = np.log(clipped / (1.0 - clipped))
    route_calibrated = 1.0 / (
        1.0
        + np.exp(
            -(
                float(calibrator["slope"]) * logit
                + float(calibrator["bias"])
            )
        )
    )
    route_calibrated = route_calibrated.astype(np.float32)
    route_metrics = {
        "raw": _continuous_metrics(route_target, route_raw),
        "frozen_monotone": _continuous_metrics(route_target, route_calibrated),
    }
    raw_mse = route_metrics["raw"]["mse"]
    calibrated_mse = route_metrics["frozen_monotone"]["mse"]
    route_metrics["relative_mse_reduction"] = (
        (raw_mse - calibrated_mse) / raw_mse if raw_mse > 0 else 0.0
    )
    calibration = {
        "raw": _calibration_error(
            route_target, route_raw, int(evaluation["calibration_bins"])
        ),
        "frozen_monotone": _calibration_error(
            route_target, route_calibrated, int(evaluation["calibration_bins"])
        ),
    }
    raw_error = calibration["raw"]["absolute_calibration_error"]
    calibrated_error = calibration["frozen_monotone"]["absolute_calibration_error"]
    calibration["relative_error_reduction"] = (
        (raw_error - calibrated_error) / raw_error if raw_error > 0 else 0.0
    )
    route_scene_rows = []
    for scene in np.unique(route_scenes):
        members = route_scenes == scene
        before = float(np.mean(np.square(route_raw[members] - route_target[members])))
        after = float(
            np.mean(np.square(route_calibrated[members] - route_target[members]))
        )
        route_scene_rows.append(
            {
                "scene_index": int(scene),
                "raw_mse": before,
                "calibrated_mse": after,
                "delta": after - before,
            }
        )
    route_scene_support = [
        sum(row["delta"] < 0 for row in route_scene_rows),
        sum(row["delta"] == 0 for row in route_scene_rows),
        sum(row["delta"] > 0 for row in route_scene_rows),
    ]

    thresholds = config["gates"]
    gates = {
        "minimum_route_spearman": route_metrics["raw"]["spearman"]
        >= float(thresholds["minimum_route_spearman"]),
        "minimum_route_mse_reduction": route_metrics["relative_mse_reduction"]
        >= float(thresholds["minimum_route_mse_reduction"]),
        "minimum_action_spearman": action_spearman
        >= float(thresholds["minimum_action_spearman"]),
        "minimum_action_unsafe_auroc": action_unsafe["auroc"]
        >= float(thresholds["minimum_action_unsafe_auroc"]),
        "minimum_action_pairwise_concordance": pairwise["concordance"]
        >= float(thresholds["minimum_action_pairwise_concordance"]),
        "minimum_action_selected_cost_reduction": selected["relative_cost_reduction"]
        >= float(thresholds["minimum_action_selected_cost_reduction"]),
    }
    verdict = (
        "supported_one_shot_combined_visited_state_confirmation"
        if all(gates.values())
        else "rejected_one_shot_combined_visited_state_confirmation"
    )
    summary = {
        "schema_version": "worldsim_v65.p10x_combined_confirmation_summary.v1",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "claim_boundary": config["claim_boundary"],
        "cache_reused": cache_reused,
        **materialization,
        "eligible_action_count": int(target.size),
        "unsafe_action_count": int(np.count_nonzero(unsafe)),
        "visited_point_count": int(arrays["visited_count"].sum()),
        "hidden_free_count": int(arrays["hidden_free_count"].sum()),
        "nominal_route": {
            "eligible_case_count": int(route_target.size),
            "unsafe_case_count": int(np.count_nonzero(route_unsafe_labels)),
            "continuous_metrics": route_metrics,
            "unsafe_ranking": _unsafe_metrics(route_unsafe_labels, route_raw),
            "calibration": calibration,
            "scene_mse_lower_equal_higher": route_scene_support,
            "scene_rows": route_scene_rows,
        },
        "fixed_action": {
            "pooled_spearman": action_spearman,
            "unsafe_ranking": action_unsafe,
            "pairwise": pairwise,
            "within_case_selection": selected,
        },
        "gate_results": gates,
        "formal_v65_confirmation_read": True,
        "model_refit": False,
        "calibrator_refit": False,
        "new_critic_trained": False,
        "second_confirmation_allowed": False,
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
