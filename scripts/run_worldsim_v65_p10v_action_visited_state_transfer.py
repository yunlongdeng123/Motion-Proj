"""Run one fresh fixed-lattice action-level visited-state reliability read."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import resource
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from motion_proj.worldsim_v64.bounded_collision_critic import _action_rows
from motion_proj.worldsim_v64.native_voxel_uq import _native_unit_dir, _unit_dirs
from scripts.run_worldsim_v65_p2v_visited_state_transfer import (
    _frozen_q0_scores,
    _load_unit,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _visited_masks(centers: np.ndarray, paths: np.ndarray, radius_m: float) -> np.ndarray:
    points = torch.from_numpy(centers[:, :2].astype(np.float32)).cuda()
    trajectories = torch.from_numpy(paths.astype(np.float32)).cuda()
    with torch.inference_mode():
        distance2 = (
            points[:, None, None, :] - trajectories[None, :, :, :]
        ).square().sum(dim=-1).amin(dim=-1)
    return (distance2 <= float(radius_m) ** 2).cpu().numpy().T


def _materialize(config: dict, runs_root: Path, cache_path: Path) -> dict[str, int]:
    inputs = config["inputs"]
    evidence_root = runs_root / inputs["evidence_run"]
    native_root = runs_root / inputs["native_run"]
    processed_root = Path(inputs["processed_root"])
    q0 = joblib.load(runs_root / inputs["risk_run"] / inputs["risk_model_relative_path"])
    origin = np.asarray(config["native_grid"]["origin_m"], dtype=np.float64)
    voxel_size = float(config["native_grid"]["voxel_size_m"])
    future_frames = int(config["trajectory"]["future_frame_count"])
    radius = float(config["trajectory"]["visited_corridor_radius_m"])
    minimum_visited = int(config["trajectory"]["minimum_visited_points_per_action"])
    point_limit = int(config["sampling"]["evaluation_points_per_unit"])
    rng = np.random.default_rng(int(config["sampling"]["seed"]))

    descriptors = []
    for scene_index, scene in enumerate(config["scenes"]):
        name = str(scene["name"])
        partitions = {name: str(inputs["native_partition"])}
        for unit_index, evidence_unit in enumerate(_unit_dirs(evidence_root, name)):
            descriptors.append(
                (
                    scene_index,
                    unit_index,
                    evidence_unit,
                    _native_unit_dir(native_root, name, evidence_unit.name, partitions),
                    processed_root / f"{int(scene['processed_index']):03d}",
                )
            )
    if not descriptors:
        raise RuntimeError("no fresh P10V units found")

    names = (
        "qmean",
        "target_cost",
        "unsafe",
        "visited_count",
        "hidden_free_count",
        "scene_index",
        "unit_index",
        "case_index",
        "action_index",
        "progress_ratio",
        "lateral_offset_m",
    )
    parts: dict[str, list] = {name: [] for name in names}
    source_action_count = 0
    excluded_action_count = 0
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def submit(row: tuple):
        return executor.submit(
            _load_unit,
            row[2],
            row[3],
            row[4],
            origin=origin,
            voxel_size=voxel_size,
            future_frame_count=future_frames,
        )

    future = submit(descriptors[0])
    try:
        for case_index, descriptor in enumerate(descriptors):
            features, centers, labels, logged_route = future.result()
            if case_index + 1 < len(descriptors):
                future = submit(descriptors[case_index + 1])
            if features.shape[0] > point_limit:
                chosen = rng.choice(features.shape[0], size=point_limit, replace=False)
                features, centers, labels = features[chosen], centers[chosen], labels[chosen]
            scores = _frozen_q0_scores(q0, features)
            actions = [
                row
                for row in _action_rows(logged_route, config["action_lattice"])
                if row["source_role"] != "stop"
            ]
            paths = np.stack([np.asarray(row["path"], dtype=np.float32) for row in actions])
            visited_masks = _visited_masks(centers, paths, radius)
            source_action_count += len(actions)
            eligible_here = 0
            for action_index, (action, visited) in enumerate(zip(actions, visited_masks)):
                visited_count = int(np.count_nonzero(visited))
                if visited_count < minimum_visited:
                    excluded_action_count += 1
                    continue
                eligible_here += 1
                hidden_free_count = int(np.count_nonzero(labels[visited]))
                parts["qmean"].append(float(scores[visited].mean()))
                parts["target_cost"].append(hidden_free_count / visited_count)
                parts["unsafe"].append(hidden_free_count > 0)
                parts["visited_count"].append(visited_count)
                parts["hidden_free_count"].append(hidden_free_count)
                parts["scene_index"].append(descriptor[0])
                parts["unit_index"].append(descriptor[1])
                parts["case_index"].append(case_index)
                parts["action_index"].append(action_index)
                parts["progress_ratio"].append(float(action["progress_ratio"]))
                parts["lateral_offset_m"].append(float(action["lateral_offset_m"]))
            print(
                f"P10V actions {case_index + 1}/{len(descriptors)} "
                f"scene={descriptor[0]} unit={descriptor[1]} eligible={eligible_here}/{len(actions)}",
                flush=True,
            )
    finally:
        executor.shutdown(wait=True)

    payload = {
        "qmean": np.asarray(parts["qmean"], dtype=np.float32),
        "target_cost": np.asarray(parts["target_cost"], dtype=np.float32),
        "unsafe": np.asarray(parts["unsafe"], dtype=bool),
        "visited_count": np.asarray(parts["visited_count"], dtype=np.int32),
        "hidden_free_count": np.asarray(parts["hidden_free_count"], dtype=np.int32),
        "scene_index": np.asarray(parts["scene_index"], dtype=np.uint8),
        "unit_index": np.asarray(parts["unit_index"], dtype=np.uint8),
        "case_index": np.asarray(parts["case_index"], dtype=np.uint8),
        "action_index": np.asarray(parts["action_index"], dtype=np.uint8),
        "progress_ratio": np.asarray(parts["progress_ratio"], dtype=np.float32),
        "lateral_offset_m": np.asarray(parts["lateral_offset_m"], dtype=np.float32),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp.npz")
    np.savez(temporary, **payload)
    os.replace(temporary, cache_path)
    return {
        "source_case_count": len(descriptors),
        "source_action_count": source_action_count,
        "excluded_action_count": excluded_action_count,
    }


def _pairwise_concordance(
    target: np.ndarray,
    scores: np.ndarray,
    cases: np.ndarray,
    minimum_gap: float,
) -> dict[str, float | int]:
    concordance = []
    for case in np.unique(cases):
        members = np.flatnonzero(cases == case)
        for left in range(members.size):
            for right in range(left + 1, members.size):
                i, j = int(members[left]), int(members[right])
                target_delta = float(target[i] - target[j])
                if abs(target_delta) < float(minimum_gap):
                    continue
                score_delta = float(scores[i] - scores[j])
                if score_delta == 0.0:
                    concordance.append(0.5)
                else:
                    concordance.append(float(np.sign(score_delta) == np.sign(target_delta)))
    return {
        "minimum_target_gap": float(minimum_gap),
        "pair_count": len(concordance),
        "concordance": float(np.mean(concordance)) if concordance else float("nan"),
    }


def _within_case_selection(
    target: np.ndarray,
    scores: np.ndarray,
    cases: np.ndarray,
    scenes: np.ndarray,
    fraction: float,
) -> dict[str, object]:
    selected_indices = []
    evaluable_indices = []
    eligible_case_count = 0
    case_rows = []
    for case in np.unique(cases):
        members = np.flatnonzero(cases == case)
        if members.size < 2:
            continue
        eligible_case_count += 1
        evaluable_indices.extend(members.tolist())
        count = max(1, int(math.floor(float(fraction) * members.size)))
        selected = members[np.argsort(scores[members], kind="stable")[:count]]
        selected_indices.extend(selected.tolist())
        all_cost = float(target[members].mean())
        selected_cost = float(target[selected].mean())
        case_rows.append(
            {
                "case_index": int(case),
                "scene_index": int(scenes[members[0]]),
                "eligible_action_count": int(members.size),
                "selected_action_count": int(count),
                "all_mean_cost": all_cost,
                "selected_mean_cost": selected_cost,
                "delta": selected_cost - all_cost,
            }
        )
    if not selected_indices:
        raise RuntimeError("no P10V case has at least two eligible actions")
    selected_array = np.asarray(selected_indices, dtype=np.int64)
    evaluable_array = np.asarray(evaluable_indices, dtype=np.int64)
    all_mean = float(target[evaluable_array].mean())
    selected_mean = float(target[selected_array].mean())
    scene_rows = []
    for scene in np.unique(scenes[evaluable_array]):
        members = evaluable_array[scenes[evaluable_array] == scene]
        scene_selected = selected_array[scenes[selected_array] == scene]
        scene_all = float(target[members].mean())
        scene_selected_cost = float(target[scene_selected].mean())
        scene_rows.append(
            {
                "scene_index": int(scene),
                "all_mean_cost": scene_all,
                "selected_mean_cost": scene_selected_cost,
                "delta": scene_selected_cost - scene_all,
            }
        )
    return {
        "evaluable_case_count": eligible_case_count,
        "selected_action_count": int(selected_array.size),
        "all_mean_cost": all_mean,
        "selected_mean_cost": selected_mean,
        "relative_cost_reduction": float(
            (all_mean - selected_mean) / all_mean if all_mean > 0 else 0.0
        ),
        "scene_lower_equal_higher": [
            sum(row["delta"] < 0 for row in scene_rows),
            sum(row["delta"] == 0 for row in scene_rows),
            sum(row["delta"] > 0 for row in scene_rows),
        ],
        "scene_nonincreasing_count": sum(row["delta"] <= 0 for row in scene_rows),
        "scene_rows": scene_rows,
        "case_rows": case_rows,
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
    pooled_spearman = float(spearmanr(target, scores).statistic)
    unsafe_metrics = {
        "auroc": float(roc_auc_score(unsafe, scores))
        if np.unique(unsafe).size == 2
        else float("nan"),
        "auprc": float(average_precision_score(unsafe, scores)),
    }
    pairwise = _pairwise_concordance(
        target,
        scores,
        cases,
        float(config["evaluation"]["pairwise_minimum_target_gap"]),
    )
    selected = _within_case_selection(
        target,
        scores,
        cases,
        scenes,
        float(config["evaluation"]["within_case_selected_fraction"]),
    )
    gates_config = config["gates"]
    gates = {
        "minimum_pooled_spearman": pooled_spearman
        >= float(gates_config["minimum_pooled_spearman"]),
        "minimum_unsafe_auroc": unsafe_metrics["auroc"]
        >= float(gates_config["minimum_unsafe_auroc"]),
        "minimum_pairwise_concordance": pairwise["concordance"]
        >= float(gates_config["minimum_pairwise_concordance"]),
        "minimum_within_case_selected_cost_reduction": selected[
            "relative_cost_reduction"
        ]
        >= float(gates_config["minimum_within_case_selected_cost_reduction"]),
        "minimum_scene_support": selected["scene_nonincreasing_count"]
        >= int(gates_config["minimum_scene_support"]),
        "minimum_evaluable_case_count": selected["evaluable_case_count"]
        >= int(gates_config["minimum_evaluable_case_count"]),
    }
    verdict = (
        "supported_fresh_fixed_action_visited_state_ranking"
        if all(gates.values())
        else "rejected_fresh_fixed_action_visited_state_ranking"
    )
    rows = []
    for index in range(target.size):
        rows.append(
            {
                "scene_index": int(scenes[index]),
                "unit_index": int(arrays["unit_index"][index]),
                "case_index": int(cases[index]),
                "action_index": int(arrays["action_index"][index]),
                "progress_ratio": float(arrays["progress_ratio"][index]),
                "lateral_offset_m": float(arrays["lateral_offset_m"][index]),
                "visited_count": int(arrays["visited_count"][index]),
                "hidden_free_count": int(arrays["hidden_free_count"][index]),
                "qmean": float(scores[index]),
                "target_cost": float(target[index]),
                "unsafe": bool(unsafe[index]),
            }
        )
    (run_dir / "ACTION_ROWS.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "worldsim_v65.p10v_action_visited_state_transfer_summary.v1",
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
        "stop_actions_excluded_by_contract": int(materialization["source_case_count"]),
        "pooled_qmean_target_spearman": pooled_spearman,
        "unsafe_ranking": unsafe_metrics,
        "pairwise": pairwise,
        "within_case_selection": selected,
        "gate_results": gates,
        "formal_v65_action_quality_read": True,
        "new_critic_trained": False,
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
