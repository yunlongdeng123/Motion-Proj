"""真实 nuScenes mini 几何链路的可恢复诊断实验。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

import torch
from omegaconf import OmegaConf

from ..auditor import MotionAuditor
from ..config import config_fingerprint, load_config, save_resolved_config
from ..data import NuScenesFutureVideoDataset
from ..projector import DynamicsProjector
from ..runtime.atomic import atomic_write_json, atomic_write_text
from ..runtime.experiment import ExperimentRegistry, JsonlMetrics, RunManifest, utc_now
from ..runtime.fingerprint import environment_fingerprint, git_state, sha256_json
from .diagnostics import corruption_sensitivity

PROTOCOL_VERSION = "nuscenes-mini-geometry-v1"


def _pearson_at_lidar(depth: torch.Tensor, lidar: torch.Tensor) -> list[float]:
    values = []
    for index in range(depth.shape[0]):
        valid = torch.isfinite(lidar[index]) & (lidar[index] > 0) & torch.isfinite(depth[index])
        if int(valid.sum()) < 2:
            continue
        corr = torch.corrcoef(torch.stack([depth[index][valid], lidar[index][valid]]))[0, 1]
        if torch.isfinite(corr):
            values.append(float(corr))
    return values


def evaluate_clip(index: int, dataset, auditor, projector, prior_weight: float) -> dict[str, Any]:
    sample = dataset[index]
    state = auditor.audit(sample)
    result = projector.project(sample["frames"].to(state.depth.device), state)
    lidar = sample.get("lidar_depth")
    correlations = _pearson_at_lidar(state.depth, lidar.to(state.depth.device)) if lidar is not None else []
    sensitivity = corruption_sensitivity(state)
    depth_diagnostics = state.meta.get("depth_diagnostics") or []
    track_before = float(result.energy_before["obj"]) + prior_weight * float(result.energy_before["prior"])
    track_after = float(result.energy_after["obj"]) + prior_weight * float(result.energy_after["prior"])
    finite = bool(
        torch.isfinite(state.depth).all()
        and torch.isfinite(state.u_ego).all()
        and torch.isfinite(result.target).all()
        and torch.isfinite(result.valid_mask).all()
    )
    return {
        "clip_index": index,
        "sample_id": sample["sample_id"],
        "static_drift": auditor.static_drift_score(state),
        "corrupted_static_drift": float(sensitivity["corrupted_drift"]),
        "corruption_detected": bool(sensitivity["detected"]),
        "depth_lidar_pearson_mean": mean(correlations) if correlations else None,
        "lidar_points_min": min((int(item["lidar_points"]) for item in depth_diagnostics), default=0),
        "depth_clamped_fraction_mean": mean(
            float(item["clamped_fraction"]) for item in depth_diagnostics
        ) if depth_diagnostics else None,
        "depth_min": float(state.depth.min()),
        "depth_max": float(state.depth.max()),
        "ego_valid_fraction": float(state.meta["ego_valid_fraction"]),
        "flow_confident_fraction": float((state.flow_conf >= 0.5).float().mean()),
        "static_mask_fraction": float(state.static_mask.mean()),
        "eligible_fraction": float(result.diagnostics["eligible_fraction"]),
        "track_energy_before": track_before,
        "track_energy_after": track_after,
        "track_energy_decreased": track_after < track_before,
        "finite": finite,
    }


def summarize(rows: list[dict[str, Any]], settings: Mapping[str, Any]) -> dict[str, Any]:
    if not rows:
        raise ValueError("没有可汇总的真实 clip")
    drift = [float(row["static_drift"]) for row in rows]
    correlations = [float(row["depth_lidar_pearson_mean"]) for row in rows
                    if row["depth_lidar_pearson_mean"] is not None]
    eligible = [float(row["eligible_fraction"]) for row in rows]
    finite_rate = sum(bool(row["finite"]) for row in rows) / len(rows)
    detection_rate = sum(bool(row["corruption_detected"]) for row in rows) / len(rows)
    thresholds = {
        "maximum_static_drift": float(settings["maximum_static_drift"]),
        "minimum_depth_lidar_pearson": float(settings["minimum_depth_lidar_pearson"]),
        "minimum_corruption_detection_rate": float(settings["minimum_corruption_detection_rate"]),
        "minimum_mean_eligible_fraction": float(settings["minimum_mean_eligible_fraction"]),
        "minimum_finite_rate": 1.0,
    }
    checks = {
        "static_drift_bounded": max(drift) <= thresholds["maximum_static_drift"],
        "depth_direction_correct": bool(correlations) and min(correlations) >= thresholds["minimum_depth_lidar_pearson"],
        "corruption_detected": detection_rate >= thresholds["minimum_corruption_detection_rate"],
        "eligible_gate": mean(eligible) >= thresholds["minimum_mean_eligible_fraction"],
        "all_finite": finite_rate >= thresholds["minimum_finite_rate"],
    }
    return {
        "protocol": PROTOCOL_VERSION,
        "clips": len(rows),
        "static_drift_mean": mean(drift),
        "static_drift_max": max(drift),
        "depth_lidar_pearson_mean": mean(correlations) if correlations else None,
        "depth_lidar_pearson_min": min(correlations) if correlations else None,
        "corruption_detection_rate": detection_rate,
        "ego_valid_fraction_mean": mean(float(row["ego_valid_fraction"]) for row in rows),
        "static_mask_fraction_mean": mean(float(row["static_mask_fraction"]) for row in rows),
        "eligible_fraction_mean": mean(eligible),
        "eligible_fraction_min": min(eligible),
        "clips_eligible_ge_threshold": sum(
            value >= thresholds["minimum_mean_eligible_fraction"] for value in eligible
        ),
        "finite_rate": finite_rate,
        "track_energy_improvement_rate": sum(bool(row["track_energy_decreased"]) for row in rows) / len(rows),
        "acceptance": {
            "thresholds": thresholds,
            "checks": checks,
            "accepted": all(checks.values()),
            "failed_checks": [name for name, passed in checks.items() if not passed],
        },
    }


def _load_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
            rows[int(row["clip_index"])] = row
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return rows


def run_experiment(cfg: Any, run_id: str | None = None) -> tuple[Path, dict[str, Any]]:
    settings = OmegaConf.to_container(cfg.experiment, resolve=True)
    assert isinstance(settings, dict)
    git = git_state(".")
    if git.get("dirty"):
        raise RuntimeError("正式真实几何诊断拒绝在 dirty worktree 上运行")
    cfg_fingerprint = config_fingerprint(cfg)
    experiment_fingerprint = sha256_json(
        {"protocol": PROTOCOL_VERSION, "config": cfg_fingerprint, "git_commit": git["commit"]}
    )
    if run_id is None:
        run_id = (
            f"p0-geometry-mini{int(settings['num_clips'])}-"
            f"{str(git['commit'])[:8]}-{cfg_fingerprint[:8]}"
        )
    root = Path(str(cfg.work_dir))
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    complete_path = run_dir / "COMPLETE"
    summary_path = run_dir / "summary.json"
    if complete_path.is_file() and complete_path.read_text(encoding="utf-8").strip() == experiment_fingerprint:
        return run_dir, json.loads(summary_path.read_text(encoding="utf-8"))

    resolved_path = run_dir / "resolved.yaml"
    resolved_text = OmegaConf.to_yaml(cfg, resolve=True)
    if resolved_path.exists() and resolved_path.read_text(encoding="utf-8") != resolved_text:
        raise RuntimeError(f"run 目录已有不同配置: {run_dir}")
    if not resolved_path.exists():
        save_resolved_config(cfg, str(resolved_path))

    registry = ExperimentRegistry(str(root / "experiments.sqlite3"))
    known = {row["run_id"]: row for row in registry.list()}
    if run_id not in known:
        registry.register(run_id, "running", experiment_fingerprint, str(run_dir))
    else:
        registry.update(run_id, "running", exit_reason="resume")
    environment = environment_fingerprint()
    environment.update({name: os.environ.get(name) for name in ("HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")})
    manifest = RunManifest(
        run_id=run_id,
        command=list(sys.argv),
        config_fingerprint=experiment_fingerprint,
        cache_fingerprint="not-applicable:online-audit",
        seed=int(cfg.seed),
        git=git,
        environment=environment,
        data_split=f"{cfg.data.version}:{','.join(cfg.data.cameras)}:first-{int(settings['num_clips'])}",
    )
    manifest.save(str(run_dir / "manifest.json"))

    completed = _load_rows(run_dir / "metrics.jsonl")
    writer = JsonlMetrics(str(run_dir / "metrics.jsonl"))
    dataset = NuScenesFutureVideoDataset(cfg.data)
    num_clips = min(int(settings["num_clips"]), len(dataset))
    missing = [index for index in range(num_clips) if index not in completed]
    if missing:
        auditor = MotionAuditor(device=str(cfg.device), enable_depth=True)
        projector = DynamicsProjector(smooth_lambda=float(settings["smooth_lambda"]))
        for index in missing:
            row = evaluate_clip(index, dataset, auditor, projector, float(settings["prior_weight"]))
            writer.append(index, row)
            completed[index] = row

    rows = [completed[index] for index in range(num_clips)]
    summary = summarize(rows, settings)
    summary.update({
        "run_id": run_id,
        "task_id": str(settings["task_id"]),
        "git_commit": git["commit"],
        "config_fingerprint": cfg_fingerprint,
        "experiment_fingerprint": experiment_fingerprint,
        "completed_at": utc_now(),
    })
    atomic_write_json(str(summary_path), summary)
    manifest.status = "completed"
    manifest.ended_at = utc_now()
    manifest.exit_reason = "acceptance_passed" if summary["acceptance"]["accepted"] else "acceptance_failed"
    manifest.save(str(run_dir / "manifest.json"))
    atomic_write_text(str(complete_path), experiment_fingerprint + "\n")
    registry.update(run_id, "completed", exit_reason=manifest.exit_reason, summary=summary)
    return run_dir, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    run_dir, summary = run_experiment(cfg, args.run_id)
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
