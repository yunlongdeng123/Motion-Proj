#!/usr/bin/env python3
"""从冻结 V5 observations 重放 V5.1 A0 Bayesian unary 基线。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v4.evidence_metrics import probability_metrics
from motion_proj.worldsim_v5.bayesian_unary import (
    UNARY_ARM_NAMES,
    accumulate_unary_arm_statistics,
    empty_unary_arm_statistics,
    finalize_unary_arms,
)
from motion_proj.worldsim_v51.protocol import (
    ProtocolError,
    V51_BRANCH,
    load_yaml,
    sha256_file,
    verify_canonical_run,
)


TASK_ID = "WS-V51-M1-A-UNARY-OBSERVABILITY-01"
SCHEMA_VERSION = "worldsim_v51_m1_a0_replay_v1"
REPLAY_FIELDS = (
    "unary_posterior",
    "unary_uncertainty",
    "effective_evidence_count",
    "multi_view_disagreement",
    "boundary_ambiguity",
    "depth_support",
)
CORE_SNAPSHOT_FILES = (
    "motion_proj/resim/drivestudio_adapter.py",
    "motion_proj/worldsim_v4/evidence_metrics.py",
    "motion_proj/worldsim_v5/bayesian_unary.py",
    "motion_proj/worldsim_v5/evidence_schema.py",
    "motion_proj/worldsim_v5/geometry_evidence.py",
    "motion_proj/worldsim_v5/observation_aggregation.py",
    "motion_proj/worldsim_v5/observation_builder.py",
    "motion_proj/worldsim_v5/ownership_renderer.py",
    "motion_proj/worldsim_v5/renderer_intersections.py",
    "scripts/eval_worldsim_v3_a3_r1_heldout.py",
    "scripts/run_worldsim_v5_m1_unary_diagnostic.py",
    "scripts/worldsim_v5_forensics_common.py",
)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT), *args], text=True
    ).strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _inventory(run_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "status.json"}:
            continue
        records.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def _bit_mismatch_count(left: np.ndarray, right: np.ndarray) -> int:
    lhs = np.asarray(left)
    rhs = np.asarray(right)
    if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
        return max(lhs.size, rhs.size, 1)
    if lhs.dtype.hasobject:
        raise ProtocolError("bit exact 比较不接受 object dtype")
    item_bytes = max(lhs.dtype.itemsize, 1)
    byte_difference = lhs.view(np.uint8).reshape(lhs.size, item_bytes) != rhs.view(
        np.uint8
    ).reshape(rhs.size, item_bytes)
    return int(np.any(byte_difference, axis=1).sum())


def _binary_iou(predicted: np.ndarray, target: np.ndarray) -> float:
    left = np.asarray(predicted, dtype=bool)
    right = np.asarray(target, dtype=bool)
    union = left | right
    if not union.any():
        return 1.0
    return float((left & right).sum() / union.sum())


def _negative_log_likelihood(probability: np.ndarray, target: np.ndarray) -> float:
    prediction = np.clip(
        np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6
    )
    label = np.asarray(target, dtype=np.float64)
    return float(
        -(label * np.log(prediction) + (1.0 - label) * np.log(1.0 - prediction)).mean()
    )


def _gaussian_metrics(
    posterior: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float,
    ece_bins: int,
) -> dict[str, float]:
    metrics = probability_metrics(posterior, target, bins=ece_bins)
    metrics.update(
        iou_at_frozen_threshold=_binary_iou(posterior >= threshold, target),
        nll=_negative_log_likelihood(posterior, target),
    )
    return metrics


def _verify_core_source_snapshot(run_dir: Path) -> dict[str, Any]:
    snapshot_root = run_dir / "source_snapshot"
    records = []
    for relative in CORE_SNAPSHOT_FILES:
        frozen = snapshot_root / relative
        current = PROJECT / relative
        if not frozen.is_file() or not current.is_file():
            raise ProtocolError(f"A0 core source snapshot 缺失: {relative}")
        frozen_sha = sha256_file(frozen)
        current_sha = sha256_file(current)
        if frozen_sha != current_sha:
            raise ProtocolError(f"A0 core source 漂移: {relative}")
        records.append({"path": relative, "sha256": current_sha})
    return {"exact_file_count": len(records), "files": records}


def _replay_scene(
    scene: str,
    spec: Mapping[str, Any],
    source_config_path: Path,
) -> dict[str, Any]:
    canonical = verify_canonical_run(scene, spec)
    source_config = load_yaml(source_config_path)
    if source_config.get("schema_version") != "worldsim_v5_m1_unary_diagnostic_v1":
        raise ProtocolError(f"V5 unary source config schema 漂移: {scene}")
    unary_config = source_config["unary"]
    run_dir = Path(spec["path"])
    canonical_b0 = _load_npz(run_dir / "artifacts/gaussians/B0.npz")
    base_model = canonical_b0["base_model"]
    gaussian_count = int(base_model.size)
    unassigned = float(unary_config["unassigned_probability"])
    prior = np.full(gaussian_count, unassigned, dtype=np.float64)
    prior[base_model == "RigidNodes"] = 1.0 - unassigned
    statistics = empty_unary_arm_statistics(gaussian_count)
    observation_paths = sorted((run_dir / "artifacts/observations").glob("*.npz"))
    if len(observation_paths) != int(spec["evidence_view_count"]):
        raise ProtocolError(f"A0 evidence view 分母漂移: {scene}")
    for path in observation_paths:
        observations = _load_npz(path)
        accumulate_unary_arm_statistics(
            statistics,
            observations=observations,
            gaussian_count=gaussian_count,
            sam_confidence_floor=float(unary_config["sam_confidence_floor"]),
            boundary_distance_scale_px=float(
                unary_config["boundary_distance_scale_px"]
            ),
            depth_residual_scale_m=float(unary_config["depth_residual_scale_m"]),
        )
    replay = finalize_unary_arms(
        prior_probability=prior,
        prior_strength=float(unary_config["prior_strength"]),
        statistics=statistics,
    )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    arm_reports: dict[str, Any] = {}
    for arm in UNARY_ARM_NAMES:
        canonical_table = _load_npz(run_dir / f"artifacts/gaussians/{arm}.npz")
        mismatches = {
            field: _bit_mismatch_count(replay[arm][field], canonical_table[field])
            for field in REPLAY_FIELDS
        }
        target = (canonical_table["base_model"] == "RigidNodes").astype(np.float32)
        observed_metrics = _gaussian_metrics(
            replay[arm]["unary_posterior"],
            target,
            threshold=float(source_config["evaluation"]["probability_threshold"]),
            ece_bins=int(source_config["evaluation"]["ece_bins"]),
        )
        expected_metrics = manifest["arm_artifacts"][arm]["metrics"]
        metric_delta = {
            name: float(observed_metrics[name] - expected_metrics[name])
            for name in sorted(expected_metrics)
        }
        if any(mismatches.values()) or any(delta != 0.0 for delta in metric_delta.values()):
            raise ProtocolError(f"A0 replay 不 exact: {scene}/{arm}")
        evaluation_records = [
            row
            for row in manifest["inventory"]
            if row["path"].startswith(f"artifacts/evaluation/{arm}/")
        ]
        if len(evaluation_records) != int(spec["accepted_evaluation_view_count"]):
            raise ProtocolError(f"A0 evaluation artifact 分母漂移: {scene}/{arm}")
        arm_reports[arm] = {
            "array_bit_mismatch_count": mismatches,
            "metric_delta": metric_delta,
            "canonical_gaussian_table_sha256": manifest["arm_artifacts"][arm][
                "sha256"
            ],
            "evaluation_artifact_count": len(evaluation_records),
            "evaluation_artifact_bytes": sum(
                int(row["bytes"]) for row in evaluation_records
            ),
            "evaluation_artifact_identity": "manifest_sha256_exact",
        }
    return {
        **canonical,
        "observation_file_count": len(observation_paths),
        "core_source_snapshot": _verify_core_source_snapshot(run_dir),
        "arms": arm_reports,
    }


def run(config_path: Path) -> dict[str, Any]:
    branch = _git("branch", "--show-current")
    source_commit = _git("rev-parse", "HEAD")
    if branch != V51_BRANCH:
        raise ProtocolError(f"A0 必须在 {V51_BRANCH} 执行")
    if _git("status", "--short"):
        raise ProtocolError("A0 formal replay 要求 clean worktree")
    config = load_yaml(config_path)
    if config.get("schema_version") != "worldsim_v51_m1_unary_baselines_v1":
        raise ProtocolError("A0 config schema 漂移")
    if config.get("task_id") != TASK_ID or config.get("phase") != "a0_exact_replay":
        raise ProtocolError("A0 task/phase 漂移")
    if tuple(config.get("arms", ())) != UNARY_ARM_NAMES:
        raise ProtocolError("A0 arm 集合或顺序漂移")
    scene_reports = [
        _replay_scene(
            scene,
            spec,
            PROJECT / config["source_configs"][scene],
        )
        for scene, spec in config["canonical_runs"].items()
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "status": "done",
        "phase": "a0_exact_replay",
        "conclusion": "a0_v5_b0_b1_b3_posterior_and_gaussian_metrics_bit_exact",
        "source_commit": source_commit,
        "source_branch": branch,
        "worktree_clean": True,
        "config_sha256": sha256_file(config_path),
        "scene_count": len(scene_reports),
        "arm_scene_unit_count": len(scene_reports) * len(UNARY_ARM_NAMES),
        "scene_reports": scene_reports,
        "posterior_recomputed": True,
        "gaussian_metrics_recomputed": True,
        "gpu_renderer_reexecuted": False,
        "two_d_evaluation_evidence": (
            "canonical_artifact_bytes_and_generation_source_sha_exact"
        ),
        "method_inference_started": False,
        "parameter_search_performed": False,
        "validation_quality_read": False,
        "test_quality_read": False,
        "kitti_method_tuning": False,
        "failure_ledger_refs": list(config["failure_ledger_refs"]),
        "failure_ledger_delta": "none",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs/worldsim_v51/m1_unary_baselines_v1.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    events = [{"event": "run_started", "at_utc": _utc_now()}]
    _write_json(run_dir / "events.json", {"events": events})
    try:
        summary = run(args.config.resolve())
        summary["created_at_utc"] = _utc_now()
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "fingerprint.json",
            {
                "schema_version": "worldsim_v51_m1_a0_fingerprint_v1",
                "task_id": TASK_ID,
                "source_commit": summary["source_commit"],
                "source_branch": summary["source_branch"],
                "worktree_clean": True,
                "config": {
                    "path": str(args.config.resolve()),
                    "sha256": summary["config_sha256"],
                },
                "canonical_runs": [
                    {
                        "scene": row["scene"],
                        "run_id": row["run_id"],
                        "summary_sha256": row["summary_sha256"],
                        "manifest_sha256": row["manifest_sha256"],
                        "checkpoint_sha256": row["checkpoint_sha256"],
                    }
                    for row in summary["scene_reports"]
                ],
            },
        )
        events.append({"event": "run_done", "at_utc": _utc_now()})
        _write_json(run_dir / "events.json", {"events": events})
        _write_json(
            run_dir / "manifest.json",
            {
                "schema_version": "worldsim_v51_m1_a0_manifest_v1",
                "task_id": TASK_ID,
                "status": "done",
                "inventory": _inventory(run_dir),
            },
        )
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_m1_a0_status_v1",
                "task_id": TASK_ID,
                "status": "done",
                "source_commit": summary["source_commit"],
                "summary_sha256": sha256_file(run_dir / "summary.json"),
                "manifest_sha256": sha256_file(run_dir / "manifest.json"),
                "finished_at_utc": _utc_now(),
            },
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    except Exception as error:
        events.append(
            {
                "event": "run_blocked",
                "at_utc": _utc_now(),
                "reason": f"{type(error).__name__}: {error}",
            }
        )
        _write_json(run_dir / "events.json", {"events": events})
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_m1_a0_status_v1",
                "task_id": TASK_ID,
                "status": "blocked",
                "reason": f"{type(error).__name__}: {error}",
                "finished_at_utc": _utc_now(),
            },
        )
        raise


if __name__ == "__main__":
    main()
