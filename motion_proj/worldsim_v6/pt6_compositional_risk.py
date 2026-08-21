"""V6 PT6 冻结风险策略的多 actor 组合鲁棒性实验。"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from motion_proj.worldsim_v6.pt2_risk_policy import (
    _actor_geometry,
    _actors_by_frame,
    _balanced_accuracy,
    _canonical,
    _git,
    _load_scene_sources,
    _predict_logistic_head,
    _sha256,
    _write_json,
    _write_jsonl,
)


TASK_ID = "WS-V6-PT6-COMPOSITIONAL-RISK-ROBUSTNESS-01"


class PT6CompositionError(RuntimeError):
    """PT6 compositional experiment 正式合同失败。"""


def _case_geometry(
    ego_pose: np.ndarray,
    case: Mapping[str, Any],
    ego_half: tuple[float, float],
) -> tuple[float, float, float, float, float, float, float]:
    local = np.eye(4, dtype=float)
    yaw = math.radians(float(case["yaw_deg"]))
    local[:2, :2] = np.asarray(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=float,
    )
    local[0, 3] = float(case["forward_m"])
    local[1, 3] = float(case["lateral_m"])
    return _actor_geometry(
        ego_pose,
        ego_pose @ local,
        np.asarray(case["size_m"], dtype=float),
        ego_half,
    )


def _predict_actor(model: Mapping[str, Any], geometry: tuple[float, ...]) -> bool:
    _, forward, lateral, half_forward, half_lateral, _, _ = geometry
    if model["policy_type"] == "constant":
        return bool(model["constant_hazard_prediction"])
    if model["policy_type"] in {"factorized_logistic", "factorized_linear_svm"}:
        forward_features = np.asarray([[forward, half_forward]], dtype=float)
        lateral_features = np.asarray([[lateral, half_lateral]], dtype=float)
        return bool(
            _predict_logistic_head(model["forward_head"], forward_features)[0]
            and _predict_logistic_head(model["lateral_head"], lateral_features)[0]
        )
    return bool(
        forward <= float(model["forward_threshold_m"])
        and lateral <= float(model["lateral_threshold_m"])
    )


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    positive = labels == 1
    negative = labels == 0
    false_safe = int(((predictions == 0) & positive).sum())
    false_brake = int(((predictions == 1) & negative).sum())
    return {
        "episode_count": int(len(labels)),
        "hazard_count": int(positive.sum()),
        "safe_count": int(negative.sum()),
        "balanced_accuracy": _balanced_accuracy(labels, predictions),
        "hazard_recall": float((predictions[positive] == 1).mean()),
        "false_safe_rate": false_safe / int(positive.sum()),
        "safe_route_completion": float((predictions[negative] == 0).mean()),
        "false_brake_rate": false_brake / int(negative.sum()),
    }


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    repo_root = repo_root.resolve()
    config_path = (repo_root / config_path).resolve() if not config_path.is_absolute() else config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["task_id"] != TASK_ID:
        raise PT6CompositionError("task_id 不匹配")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{stamp}__compositional-risk-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        if shutil.disk_usage(run_root).free / (1024**3) < float(config["resources"]["minimum_disk_free_gib"]):
            raise PT6CompositionError("磁盘资源不足")
        source_commit = _git(repo_root, "rev-parse", "HEAD")
        if _git(repo_root, "status", "--short"):
            raise PT6CompositionError("正式运行要求 clean worktree")
        frozen = config["frozen_policy"]
        policy_path = Path(frozen["policy_arms_path"])
        if _sha256(policy_path) != frozen["policy_arms_sha256"]:
            raise PT6CompositionError("冻结 policy hash 漂移")
        policy_arms = json.loads(policy_path.read_text(encoding="utf-8"))
        spec = config["development_scene"]
        instances, poses, paths = _load_scene_sources(
            Path(config["dataset_root"]),
            spec,
            int(config["partition"]["expected_frame_count"]),
        )
        actors_by_frame = _actors_by_frame(instances)
        ego_half = (
            0.5 * float(config["geometry"]["ego_length_m"]),
            0.5 * float(config["geometry"]["ego_width_m"]),
        )
        label_decimals = int(config["geometry"]["factor_label_canonicalization_decimals"])
        frozen_paths = [config_path, policy_path, *paths]
        hashes_before = {str(path): _sha256(path) for path in frozen_paths}
        rows: list[dict[str, Any]] = []
        arm_predictions: dict[str, list[int]] = {arm: [] for arm in policy_arms}
        labels: list[int] = []
        for frame in sorted(poses):
            start = int(config["partition"]["sample_start"])
            stride = int(config["partition"]["sample_stride"])
            if frame < start or (frame - start) % stride:
                continue
            logged_geometries = [
                _actor_geometry(poses[frame], pose, size, ego_half)
                for pose, size in actors_by_frame.get(frame, [])
            ]
            for case_a, case_b in product(config["clone_a_cases"], config["clone_b_cases"]):
                geometries = [
                    *logged_geometries,
                    _case_geometry(poses[frame], case_a, ego_half),
                    _case_geometry(poses[frame], case_b, ego_half),
                ]
                actor_labels = [
                    int(round(value[5], label_decimals) <= 0.0 and round(value[6], label_decimals) <= 0.0)
                    for value in geometries
                ]
                label = int(any(actor_labels))
                labels.append(label)
                predictions = {}
                for arm, value in policy_arms.items():
                    prediction = int(any(_predict_actor(value["model"], geometry) for geometry in geometries))
                    arm_predictions[arm].append(prediction)
                    predictions[arm] = prediction
                rows.append(
                    {
                        "scene": spec["scene"],
                        "frame": frame,
                        "case_a": case_a["id"],
                        "case_b": case_b["id"],
                        "actor_count": len(geometries),
                        "hazard_label": label,
                        "hazard_actor_count": int(sum(actor_labels)),
                        "predictions": predictions,
                    }
                )
        label_array = np.asarray(labels, dtype=int)
        arms1 = {
            arm: _metrics(label_array, np.asarray(predictions, dtype=int))
            for arm, predictions in arm_predictions.items()
        }
        arms2 = {
            arm: _metrics(label_array, np.asarray(predictions, dtype=int))
            for arm, predictions in arm_predictions.items()
        }
        _write_json(run_dir / "COMPOSITION_ARMS.json", arms1)
        _write_jsonl(run_dir / "COMPOSITION_EPISODES.jsonl", rows)
        _write_json(
            run_dir / "SOURCE_AUDIT.json",
            {"scene": spec["scene"], "source_sha256": hashes_before},
        )
        v6 = arms1["real_plus_v6_verified_compiled"]
        real = arms1["real_only"]
        naive = arms1["real_plus_naive_synthetic"]
        reduction_real = real["false_safe_rate"] - v6["false_safe_rate"]
        reduction_naive = naive["false_safe_rate"] - v6["false_safe_rate"]
        cfg = config["gate"]
        wall_seconds = time.monotonic() - started
        checks = {
            "frozen_policy_exact": True,
            "new_development_scene": spec["scene"] not in set(config["prior_scenes"]),
            "two_clone_cartesian_denominator": len(rows) == int(config["expected_episode_count"]),
            "both_outcomes_present": v6["hazard_count"] > 0 and v6["safe_count"] > 0,
            "v6_balanced_accuracy": v6["balanced_accuracy"] >= float(cfg["require_v6_balanced_accuracy_at_least"]),
            "v6_false_safe": v6["false_safe_rate"] <= float(cfg["require_v6_false_safe_rate_at_most"]),
            "v6_safe_route_completion": v6["safe_route_completion"] >= float(cfg["require_v6_safe_route_completion_at_least"]),
            "false_safe_reduction_vs_real_only": reduction_real >= float(cfg["require_false_safe_reduction_vs_real_only_at_least"]),
            "false_safe_reduction_vs_naive": reduction_naive >= float(cfg["require_false_safe_reduction_vs_naive_at_least"]),
            "evaluation_repeat_exact": _canonical(arms1) == _canonical(arms2),
            "source_immutable": hashes_before == {str(path): _sha256(path) for path in frozen_paths},
            "unsupported_metrics_abstain": all(str(value).startswith("ABSTAIN") for value in config["unsupported_metrics"].values()),
            "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]),
        }
        checks["passed"] = all(checks.values())
        gate = {
            "schema_version": "worldsim_v6.pt6_compositional_risk_gate.v1",
            "checks": checks,
            "decision": "accept_frozen_policy_multi_actor_composition" if checks["passed"] else "reject_frozen_policy_multi_actor_composition",
            "false_safe_reduction_vs_real_only": reduction_real,
            "false_safe_reduction_vs_naive": reduction_naive,
            "unsupported_metrics": config["unsupported_metrics"],
        }
        _write_json(run_dir / "PT6_COMPOSITION_GATE.json", gate)
        summary = {
            "schema_version": "worldsim_v6.pt6_compositional_risk_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "source_commit": source_commit,
            "development_scene": spec["scene"],
            "method_arms": arms1,
            "wall_seconds": wall_seconds,
            "claim_boundary": config["claim_boundary"],
            "unsupported_metrics": config["unsupported_metrics"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = ["COMPOSITION_ARMS.json", "COMPOSITION_EPISODES.jsonl", "SOURCE_AUDIT.json", "PT6_COMPOSITION_GATE.json", "SUMMARY.json"]
        _write_json(
            run_dir / "MANIFEST.json",
            {
                "schema_version": "worldsim_v6.pt6_compositional_risk_manifest.v1",
                "source_commit": source_commit,
                "config": str(config_path),
                "files": {name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)} for name in tracked},
            },
        )
        _write_json(
            run_dir / "TERMINAL.json",
            {"schema_version": "worldsim_v6.terminal.v1", "status": summary["status"], "task_id": TASK_ID, "hypothesis_id": config["hypothesis_id"], "manifest_sha256": _sha256(run_dir / "MANIFEST.json")},
        )
        return run_dir
    except Exception as error:
        _write_json(
            run_dir / "TERMINAL.json",
            {"schema_version": "worldsim_v6.terminal.v1", "status": "blocked", "task_id": TASK_ID, "error_type": type(error).__name__, "error": str(error)},
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    print(run_experiment(args.repo_root, args.config, args.run_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
