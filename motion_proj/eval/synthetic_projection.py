"""P0 几何可信度的确定性合成投影验收。

该实验只验证投影器能够降低其明确建模的目标轨迹能量，不把结果外推为
RGB 生成质量或可控驾驶能力。每个 case 都经过完整 ``DynamicsProjector``，
并保存逐 case 指标，便于中断后跳过已经完成的样本。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch
from omegaconf import OmegaConf

from ..auditor.state import MotionState, Track
from ..config import config_fingerprint, load_config, save_resolved_config
from ..projector import DynamicsProjector
from ..runtime.atomic import atomic_write_json, atomic_write_text
from ..runtime.experiment import (
    ExperimentRegistry,
    JsonlMetrics,
    RunManifest,
    utc_now,
)
from ..runtime.fingerprint import environment_fingerprint, git_state, sha256_json

PROTOCOL_VERSION = "synthetic-object-track-v1"
CORRUPTIONS = (
    "center_impulse",
    "center_jitter",
    "constant_acceleration",
    "scale_impulse",
    "temporal_gap",
)


def _rand(generator: torch.Generator, low: float, high: float, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.rand(shape, generator=generator) * (high - low) + low


def _make_track(case_index: int, settings: Mapping[str, Any], seed: int) -> tuple[Track, str]:
    """从平滑轨迹构造一种可复现的动力学错误。"""
    k = int(settings["num_frames"])
    generator = torch.Generator().manual_seed(seed + case_index)
    corruption = CORRUPTIONS[case_index % len(CORRUPTIONS)]

    time = torch.arange(k, dtype=torch.float32)
    center0 = _rand(generator, 0.42, 0.58, (2,)) * torch.tensor(
        [float(settings["width"]), float(settings["height"])]
    )
    velocity = _rand(generator, -1.2, 1.2, (2,))
    centers = center0 + time[:, None] * velocity
    base_scale = _rand(generator, 14.0, 22.0, (2,))
    scale_velocity = _rand(generator, -0.015, 0.015, (2,))
    log_scales = torch.log(base_scale)[None] + time[:, None] * scale_velocity
    present = torch.ones(k, dtype=torch.bool)

    inner = 1 + int(torch.randint(0, k - 2, (1,), generator=generator))
    direction = _rand(generator, -1.0, 1.0, (2,))
    direction = direction / direction.norm().clamp_min(1e-6)
    if corruption == "center_impulse":
        centers[inner] += direction * float(_rand(generator, 9.0, 16.0, ()).item())
    elif corruption == "center_jitter":
        centers += torch.randn((k, 2), generator=generator) * 4.5
    elif corruption == "constant_acceleration":
        curve = (time - (k - 1) / 2.0).square()
        centers += curve[:, None] * direction[None] * float(_rand(generator, 0.7, 1.2, ()).item())
    elif corruption == "scale_impulse":
        log_scales[inner] += direction * float(_rand(generator, 0.45, 0.75, ()).item())
    elif corruption == "temporal_gap":
        present[inner] = False
        spike_index = inner - 1 if inner > 1 else inner + 1
        centers[spike_index] += direction * float(_rand(generator, 8.0, 13.0, ()).item())

    scales = torch.exp(log_scales)
    half = scales * 0.5
    xyxy = torch.cat([centers - half, centers + half], dim=-1)
    xyxy[~present] = float("nan")
    depth = torch.full((k,), 15.0)
    depth[~present] = float("nan")
    track = Track(
        instance_token=f"synthetic-{case_index:03d}",
        category="vehicle.car",
        xyxy=xyxy,
        depth=depth,
        present=present,
    )
    return track, corruption


def _make_frames(track: Track, settings: Mapping[str, Any]) -> torch.Tensor:
    """生成带静态纹理和单个彩色目标的轻量视频，供完整投影路径使用。"""
    k = int(settings["num_frames"])
    h = int(settings["height"])
    w = int(settings["width"])
    ys = torch.linspace(-1.0, 1.0, h)[:, None].expand(h, w)
    xs = torch.linspace(-1.0, 1.0, w)[None, :].expand(h, w)
    background = torch.stack([0.25 * xs, 0.25 * ys, 0.15 * (xs + ys)], dim=0)
    frames = background.unsqueeze(0).repeat(k, 1, 1, 1)
    color = torch.tensor([0.85, -0.15, -0.35])[:, None, None]
    for frame_index in range(k):
        if not bool(track.present[frame_index]):
            continue
        u0, v0, u1, v1 = [int(round(float(value))) for value in track.xyxy[frame_index]]
        u0, u1 = max(0, u0), min(w, u1)
        v0, v1 = max(0, v0), min(h, v1)
        if u1 > u0 and v1 > v0:
            frames[frame_index, :, v0:v1, u0:u1] = color
    return frames


def _make_state(
    track: Track,
    settings: Mapping[str, Any],
    seed: int,
    case_index: int,
) -> MotionState:
    k = int(settings["num_frames"])
    h = int(settings["height"])
    w = int(settings["width"])
    generator = torch.Generator().manual_seed(seed * 1009 + case_index)
    reliable = (
        torch.rand((k - 1, h, w), generator=generator)
        < float(settings["reliability_probability"])
    ).float()
    intrinsics = torch.tensor(
        [[120.0, 0.0, w / 2.0], [0.0, 120.0, h / 2.0], [0.0, 0.0, 1.0]]
    )
    return MotionState(
        u_static=torch.zeros(k - 1, h, w, 2),
        u_ego=torch.zeros(k - 1, h, w, 2),
        static_mask=reliable,
        flow_conf=reliable,
        depth=torch.full((k, h, w), 15.0),
        tracks=[track],
        meta={
            "intrinsics": intrinsics,
            "cam2ego": torch.eye(4),
            "ego2global": torch.eye(4).unsqueeze(0).repeat(k, 1, 1),
            "hw": (h, w),
            "sample_id": f"synthetic-{case_index:03d}",
        },
    )


def evaluate_case(case_index: int, settings: Mapping[str, Any], seed: int) -> dict[str, Any]:
    track, corruption = _make_track(case_index, settings, seed)
    frames = _make_frames(track, settings)
    state = _make_state(track, settings, seed, case_index)
    result = DynamicsProjector(smooth_lambda=float(settings["smooth_lambda"])).project(frames, state)

    prior_weight = float(settings.get("prior_weight", 0.1))
    before = float(result.energy_before["obj"]) + prior_weight * float(result.energy_before["prior"])
    after = float(result.energy_after["obj"]) + prior_weight * float(result.energy_after["prior"])
    finite = bool(
        torch.isfinite(result.target).all()
        and torch.isfinite(result.valid_mask).all()
        and all(torch.isfinite(torch.tensor(value)) for value in (before, after))
    )
    mask_in_range = bool((result.valid_mask >= 0).all() and (result.valid_mask <= 1).all())
    decreased = finite and after < before - float(settings.get("energy_tolerance", 1e-6))
    return {
        "case_id": f"synthetic-{case_index:03d}",
        "case_index": case_index,
        "case_seed": seed + case_index,
        "corruption": corruption,
        "energy_scope": "E_obj + prior_weight * E_prior",
        "prior_weight": prior_weight,
        "energy_before": before,
        "energy_after": after,
        "relative_reduction": (before - after) / max(abs(before), 1e-12),
        "energy_decreased": decreased,
        "obj_decreased": float(result.energy_after["obj"]) < float(result.energy_before["obj"]),
        "prior_decreased": float(result.energy_after["prior"]) < float(result.energy_before["prior"]),
        "eligible_fraction": float(result.diagnostics["eligible_fraction"]),
        "finite": finite,
        "mask_in_range": mask_in_range,
    }


def run_cases(settings: Mapping[str, Any], seed: int) -> list[dict[str, Any]]:
    return [evaluate_case(index, settings, seed) for index in range(int(settings["num_cases"]))]


def summarize(rows: list[dict[str, Any]], settings: Mapping[str, Any]) -> dict[str, Any]:
    if not rows:
        raise ValueError("没有可汇总的合成 case")
    rows = sorted(rows, key=lambda row: int(row["case_index"]))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["corruption"])].append(row)

    def group_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "cases": len(items),
            "improved": sum(bool(item["energy_decreased"]) for item in items),
            "improvement_rate": sum(bool(item["energy_decreased"]) for item in items) / len(items),
            "mean_relative_reduction": sum(float(item["relative_reduction"]) for item in items) / len(items),
            "mean_eligible_fraction": sum(float(item["eligible_fraction"]) for item in items) / len(items),
        }

    overall = group_summary(rows)
    finite_rate = sum(bool(row["finite"] and row["mask_in_range"]) for row in rows) / len(rows)
    min_eligible = min(float(row["eligible_fraction"]) for row in rows)
    threshold = float(settings["minimum_improvement_rate"])
    eligible_threshold = float(settings["minimum_eligible_fraction"])
    accepted = (
        overall["improvement_rate"] >= threshold
        and finite_rate == 1.0
        and min_eligible >= eligible_threshold
    )
    return {
        "protocol": PROTOCOL_VERSION,
        "energy_scope": "仅目标轨迹：E_obj + 0.1 * E_prior；不代表 RGB/FVD 或可控性结论",
        **overall,
        "finite_and_mask_valid_rate": finite_rate,
        "minimum_eligible_fraction": min_eligible,
        "acceptance": {
            "minimum_improvement_rate": threshold,
            "minimum_eligible_fraction": eligible_threshold,
            "accepted": accepted,
        },
        "by_corruption": {name: group_summary(items) for name, items in sorted(grouped.items())},
    }


def _load_completed_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
            rows[int(row["case_index"])] = row
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return rows


def run_experiment(cfg: Any, run_id: str | None = None) -> tuple[Path, dict[str, Any]]:
    settings = OmegaConf.to_container(cfg.experiment, resolve=True)
    assert isinstance(settings, dict)
    seed = int(cfg.seed)
    git = git_state(".")
    if git.get("dirty"):
        raise RuntimeError("正式合成验收拒绝在 dirty worktree 上运行")
    cfg_fingerprint = config_fingerprint(cfg)
    experiment_fingerprint = sha256_json(
        {"protocol": PROTOCOL_VERSION, "config": cfg_fingerprint, "git_commit": git["commit"]}
    )
    if run_id is None:
        run_id = (
            f"p0-geometry-synth{int(settings['num_cases'])}-s{seed}-"
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

    manifest = RunManifest(
        run_id=run_id,
        command=list(sys.argv),
        config_fingerprint=experiment_fingerprint,
        cache_fingerprint=f"not-applicable:{PROTOCOL_VERSION}",
        seed=seed,
        git=git,
        environment=environment_fingerprint(),
        data_split=PROTOCOL_VERSION,
    )
    manifest.save(str(run_dir / "manifest.json"))
    metrics_path = run_dir / "metrics.jsonl"
    completed = _load_completed_rows(metrics_path)
    writer = JsonlMetrics(str(metrics_path))
    for case_index in range(int(settings["num_cases"])):
        if case_index in completed:
            continue
        row = evaluate_case(case_index, settings, seed)
        writer.append(case_index, row)
        completed[case_index] = row

    rows = [completed[index] for index in range(int(settings["num_cases"]))]
    summary = summarize(rows, settings)
    summary.update(
        {
            "run_id": run_id,
            "task_id": str(settings["task_id"]),
            "seed": seed,
            "git_commit": git["commit"],
            "config_fingerprint": cfg_fingerprint,
            "experiment_fingerprint": experiment_fingerprint,
            "completed_at": utc_now(),
        }
    )
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
