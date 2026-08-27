"""Fit one preregistered monotone scalar calibration of trajectory-level Qmean."""

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
import torch.nn.functional as F
import yaml

from scripts.run_worldsim_v65_p1r4_trajectory_visited_state import (
    _continuous_metrics,
    _load_remaining,
    _q0_probabilities,
    _unsafe_metrics,
    _write_json,
)
from scripts.run_worldsim_v65_p1r6_smooth_tail_visited_state import _aggregate_units


def _calibration_error(target: np.ndarray, prediction: np.ndarray, bins: int) -> dict[str, object]:
    order = np.argsort(prediction, kind="stable")
    rows = []
    weighted = 0.0
    for members in np.array_split(order, bins):
        if members.size == 0:
            continue
        predicted_mean = float(prediction[members].mean())
        target_mean = float(target[members].mean())
        gap = abs(predicted_mean - target_mean)
        weighted += gap * members.size
        rows.append(
            {
                "count": int(members.size),
                "predicted_mean": predicted_mean,
                "target_mean": target_mean,
                "absolute_gap": float(gap),
            }
        )
    return {"equal_count_bin_count": len(rows), "absolute_calibration_error": weighted / target.size, "rows": rows}


def _selected_indices(prediction: np.ndarray, coverage: float) -> np.ndarray:
    count = max(1, int(math.floor(coverage * prediction.shape[0])))
    return np.argsort(prediction, kind="stable")[:count]


def run(config_path: Path, runs_root: Path, run_id: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v65" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(run_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    started = time.monotonic()
    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
        temperature=0.10,
    )
    train = units["is_train"]
    evaluate = ~train
    clip = float(config["calibrator"]["probability_clip"])
    device = torch.device("cuda")
    train_score = torch.from_numpy(np.clip(units["mean_score"][train], clip, 1.0 - clip)).to(device)
    train_target = torch.from_numpy(units["target_cost"][train]).to(device)
    train_logit = torch.logit(train_score)
    raw_slope = torch.nn.Parameter(torch.tensor(0.54132485, device=device))
    bias = torch.nn.Parameter(torch.tensor(0.0, device=device))
    optimizer = torch.optim.Adam([raw_slope, bias], lr=float(config["calibrator"]["learning_rate"]))
    losses = []
    for _ in range(int(config["calibrator"]["epochs"])):
        slope = F.softplus(raw_slope) + 1e-6
        prediction = torch.sigmoid(slope * train_logit + bias)
        loss = torch.mean(torch.square(prediction - train_target))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    slope = float((F.softplus(raw_slope) + 1e-6).detach().cpu())
    intercept = float(bias.detach().cpu())
    base = units["mean_score"][evaluate]
    base_clipped = np.clip(base, clip, 1.0 - clip)
    calibrated = 1.0 / (1.0 + np.exp(-(slope * np.log(base_clipped / (1.0 - base_clipped)) + intercept)))
    calibrated = calibrated.astype(np.float32)
    target = units["target_cost"][evaluate]
    unsafe = units["unsafe"][evaluate]
    scenes = units["scene_index"][evaluate]
    metrics = {"qmean": _continuous_metrics(target, base), "monotone_calibrated": _continuous_metrics(target, calibrated)}
    base_mse = metrics["qmean"]["mse"]
    calibrated_mse = metrics["monotone_calibrated"]["mse"]
    metrics["relative_mse_reduction"] = (base_mse - calibrated_mse) / base_mse
    metrics["spearman_delta"] = metrics["monotone_calibrated"]["spearman"] - metrics["qmean"]["spearman"]
    ranking = {"qmean": _unsafe_metrics(unsafe, base), "monotone_calibrated": _unsafe_metrics(unsafe, calibrated)}
    ranking["auroc_delta"] = ranking["monotone_calibrated"]["auroc"] - ranking["qmean"]["auroc"]
    bins = int(config["evaluation"]["calibration_bins"])
    calibration = {"qmean": _calibration_error(target, base, bins), "monotone_calibrated": _calibration_error(target, calibrated, bins)}
    base_ece = calibration["qmean"]["absolute_calibration_error"]
    calibrated_ece = calibration["monotone_calibrated"]["absolute_calibration_error"]
    calibration["relative_error_reduction"] = (base_ece - calibrated_ece) / base_ece if base_ece > 0 else 0.0
    scene_rows = []
    for scene in np.unique(scenes):
        members = scenes == scene
        before = float(np.mean(np.square(base[members] - target[members])))
        after = float(np.mean(np.square(calibrated[members] - target[members])))
        scene_rows.append({"scene_index": int(scene), "qmean_mse": before, "calibrated_mse": after, "delta": after - before})
    scene_lower = sum(row["delta"] < 0 for row in scene_rows)
    scene_equal = sum(row["delta"] == 0 for row in scene_rows)
    scene_higher = sum(row["delta"] > 0 for row in scene_rows)
    coverage = float(config["evaluation"]["matched_safe_coverage"])
    selected_base = _selected_indices(base, coverage)
    selected_calibrated = _selected_indices(calibrated, coverage)
    selected_exact = bool(np.array_equal(selected_base, selected_calibrated))
    thresholds = config["incremental_gates"]
    tolerance = float(thresholds["ranking_tolerance"])
    gates = {
        "minimum_mse_reduction": metrics["relative_mse_reduction"] >= float(thresholds["minimum_mse_reduction"]),
        "minimum_calibration_error_reduction": calibration["relative_error_reduction"] >= float(thresholds["minimum_calibration_error_reduction"]),
        "minimum_scene_mse_direction_support": scene_lower >= int(thresholds["minimum_scene_mse_direction_support"]),
        "spearman_nonregression": metrics["spearman_delta"] >= -tolerance,
        "unsafe_auroc_nonregression": ranking["auroc_delta"] >= -tolerance,
        "exact_selected_set": selected_exact,
    }
    verdict = "positive_train_only_monotone_visited_state_calibration" if all(gates.values()) else "no_clear_train_only_monotone_visited_state_calibration"
    artifact = {"form": config["calibrator"]["form"], "slope": slope, "bias": intercept, "probability_clip": clip}
    _write_json(run_dir / "monotone_calibrator.json", artifact)
    summary = {
        "schema_version": "worldsim_v65.p1r7_monotone_visited_state_calibration_summary.v1",
        "task_id": config["task_id"],
        "hypothesis_id": config["hypothesis_id"],
        "status": "done",
        "verdict": verdict,
        "claim_boundary": config["claim_boundary"],
        "train_unit_count": int(np.count_nonzero(train)),
        "evaluation_unit_count": int(np.count_nonzero(evaluate)),
        "calibrator": artifact,
        "continuous_metrics": metrics,
        "unsafe_ranking": ranking,
        "calibration": calibration,
        "scene_mse": {"lower_equal_higher": [scene_lower, scene_equal, scene_higher], "rows": scene_rows},
        "selected_40_percent": {"exact_same_indices": selected_exact, "selected_count": int(selected_base.size), "mean_actual_cost": float(target[selected_base].mean())},
        "incremental_gate_results": gates,
        "epoch_losses": losses,
        "formal_v65_calibration_read": False,
        "formal_v65_selection_read": False,
        "resources": {
            "gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
            "wall_seconds": time.monotonic() - started,
        },
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "status.json", {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    return {"run_dir": str(run_dir), "verdict": verdict, "incremental_gate_results": gates, "resources": summary["resources"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.runs_root.resolve(), args.run_id), indent=2))


if __name__ == "__main__":
    main()
