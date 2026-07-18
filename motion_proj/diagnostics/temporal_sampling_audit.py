"""RP-R1：真实时间采样与 SVD fps micro-conditioning 审计。

本模块把两个容易混淆的量分开：

1. nuScenes CAM_FRONT 关键帧的真实时间戳与真实运动尺度；
2. SVD ``fps`` added-time-id 对冻结 Base rollout 的影响。

生成对照只读取 validation clip 的首帧。future ego、box 与 track 只用于独立的真实视频
时间统计，绝不进入生成、候选选择或 generated-rollout evaluator。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import time
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf

from ..backbones import build_backbone
from ..backbones.svd_backbone import SVDBackbone
from ..config import config_fingerprint, load_config, save_resolved_config
from ..data import NuScenesFutureVideoDataset
from ..eval.independent_tracks import (
    CoTracker3IndependentEvaluator,
    aggregate_dynamics,
    camera_compensated_velocity,
    summarize_camera_compensated_dynamics,
)
from ..runtime.atomic import atomic_write_json, atomic_write_text
from ..runtime.experiment import JsonlMetrics, RunManifest, utc_now
from ..runtime.fingerprint import environment_fingerprint, file_fingerprint, git_state, sha256_json
from ..train.trainer import seed_everything
from ..utils.io import to_uint8_video, write_video
from .svd_conditioning_parity import _base_model_fingerprint


MOVABLE_CATEGORY_PREFIXES = ("vehicle.", "human.pedestrian.", "cycle.")


class TemporalSamplingAuditError(RuntimeError):
    """R1 provenance、配对或预注册决策不成立。"""


def _tensor_fingerprint(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.flatten().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _finite(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def finite_summary(values: Sequence[float]) -> dict[str, Any]:
    """返回不以零填充缺失值的稳定分布摘要。"""
    data = np.asarray(_finite(values), dtype=np.float64)
    if data.size == 0:
        return {"status": "invalid", "count": 0}
    return {
        "status": "valid",
        "count": int(data.size),
        "min": float(data.min()),
        "p05": float(np.quantile(data, 0.05)),
        "p25": float(np.quantile(data, 0.25)),
        "median": float(np.quantile(data, 0.50)),
        "mean": float(data.mean()),
        "p75": float(np.quantile(data, 0.75)),
        "p95": float(np.quantile(data, 0.95)),
        "max": float(data.max()),
    }


def select_scene_distinct_clip_records(
    records: Sequence[Mapping[str, Any]], *, count: int,
) -> list[dict[str, Any]]:
    """按 scene 名称稳定选择每个 scene 的首个 clip。"""
    if int(count) <= 0:
        raise ValueError("count 必须大于 0")
    ordered = sorted(
        (dict(row) for row in records),
        key=lambda row: (
            str(row.get("scene_name", "")),
            int(row.get("start_index", -1)),
            str(row.get("sample_id", "")),
        ),
    )
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in ordered:
        scene = str(row.get("scene_token") or row.get("scene_name") or "")
        if not scene or scene in seen:
            continue
        seen.add(scene)
        selected.append(row)
        if len(selected) == int(count):
            break
    if len(selected) != int(count):
        raise TemporalSamplingAuditError(
            f"跨 scene clip 不足: required={count}, available={len(selected)}"
        )
    return selected


def quaternion_angle_rad(first: Sequence[float], second: Sequence[float]) -> float:
    """nuScenes ``wxyz`` 单位四元数间最短旋转角。"""
    q0 = np.asarray(first, dtype=np.float64)
    q1 = np.asarray(second, dtype=np.float64)
    if q0.shape != (4,) or q1.shape != (4,):
        raise ValueError("quaternion 必须是长度 4 的 wxyz")
    n0, n1 = float(np.linalg.norm(q0)), float(np.linalg.norm(q1))
    if n0 <= 0.0 or n1 <= 0.0:
        raise ValueError("quaternion 范数必须大于 0")
    dot = float(np.clip(abs(np.dot(q0 / n0, q1 / n1)), 0.0, 1.0))
    return float(2.0 * math.acos(dot))


def _movable_category(category: str) -> bool:
    return any(str(category).startswith(prefix) for prefix in MOVABLE_CATEGORY_PREFIXES)


def temporal_record_from_nuscenes(
    nusc: Any,
    record: Mapping[str, Any],
    *,
    camera: str,
) -> dict[str, Any]:
    """只读 metadata，计算一个真实 clip 的时间与运动统计。"""
    timestamps: list[int] = []
    ego_translation: list[np.ndarray] = []
    ego_rotation: list[list[float]] = []
    tracks: dict[str, list[dict[str, Any]]] = defaultdict(list)

    sample_tokens = [str(value) for value in record["sample_tokens"]]
    for frame_index, sample_token in enumerate(sample_tokens):
        sample = nusc.get("sample", sample_token)
        camera_token = sample["data"][camera]
        sample_data = nusc.get("sample_data", camera_token)
        ego = nusc.get("ego_pose", sample_data["ego_pose_token"])
        timestamp = int(sample_data["timestamp"])
        timestamps.append(timestamp)
        ego_translation.append(np.asarray(ego["translation"], dtype=np.float64))
        ego_rotation.append([float(value) for value in ego["rotation"]])
        for annotation_token in sample["anns"]:
            annotation = nusc.get("sample_annotation", annotation_token)
            category = str(annotation["category_name"])
            if not _movable_category(category):
                continue
            tracks[str(annotation["instance_token"])].append(
                {
                    "frame_index": int(frame_index),
                    "timestamp_us": timestamp,
                    "translation_global": np.asarray(annotation["translation"], dtype=np.float64),
                    "category": category,
                }
            )

    timestamp_delta_s: list[float] = []
    effective_fps: list[float] = []
    ego_translation_per_frame_m: list[float] = []
    ego_rotation_per_frame_rad: list[float] = []
    ego_translation_mps: list[float] = []
    ego_rotation_radps: list[float] = []
    for index in range(len(timestamps) - 1):
        delta_s = float(timestamps[index + 1] - timestamps[index]) / 1.0e6
        if not math.isfinite(delta_s) or delta_s <= 0.0:
            raise TemporalSamplingAuditError(
                f"非正 CAM_FRONT timestamp delta: {record.get('sample_id')} pair={index}"
            )
        translation = float(np.linalg.norm(ego_translation[index + 1] - ego_translation[index]))
        rotation = quaternion_angle_rad(ego_rotation[index], ego_rotation[index + 1])
        timestamp_delta_s.append(delta_s)
        effective_fps.append(1.0 / delta_s)
        ego_translation_per_frame_m.append(translation)
        ego_rotation_per_frame_rad.append(rotation)
        ego_translation_mps.append(translation / delta_s)
        ego_rotation_radps.append(rotation / delta_s)

    actor_track_length_frames: list[float] = []
    actor_center_speed_global_mps: list[float] = []
    valid_actor_pairs = 0
    for observations in tracks.values():
        observations.sort(key=lambda row: int(row["frame_index"]))
        actor_track_length_frames.append(float(len(observations)))
        for first, second in zip(observations, observations[1:]):
            if int(second["frame_index"]) != int(first["frame_index"]) + 1:
                continue
            delta_s = float(int(second["timestamp_us"]) - int(first["timestamp_us"])) / 1.0e6
            if delta_s <= 0.0:
                continue
            displacement = float(
                np.linalg.norm(second["translation_global"] - first["translation_global"])
            )
            actor_center_speed_global_mps.append(displacement / delta_s)
            valid_actor_pairs += 1

    return {
        "scene_name": str(record["scene_name"]),
        "scene_token": str(record["scene_token"]),
        "sample_id": str(record["sample_id"]),
        "sample_tokens": sample_tokens,
        "timestamps_us": timestamps,
        "timestamp_delta_s": timestamp_delta_s,
        "effective_fps": effective_fps,
        "clip_duration_s": float(timestamps[-1] - timestamps[0]) / 1.0e6,
        "ego_translation_per_frame_m": ego_translation_per_frame_m,
        "ego_rotation_per_frame_rad": ego_rotation_per_frame_rad,
        "ego_translation_mps": ego_translation_mps,
        "ego_rotation_radps": ego_rotation_radps,
        "actor_track_length_frames": actor_track_length_frames,
        "actor_center_speed_global_mps": actor_center_speed_global_mps,
        "valid_actor_track_count": int(sum(length >= 2 for length in actor_track_length_frames)),
        "valid_actor_pair_count": int(valid_actor_pairs),
    }


def summarize_real_temporal_records(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = (
        "timestamp_delta_s",
        "effective_fps",
        "ego_translation_per_frame_m",
        "ego_rotation_per_frame_rad",
        "ego_translation_mps",
        "ego_rotation_radps",
        "actor_track_length_frames",
        "actor_center_speed_global_mps",
    )
    summary = {
        field: finite_summary(
            [float(value) for row in rows for value in row.get(field, [])]
        )
        for field in fields
    }
    summary.update(
        {
            "clip_count": len(rows),
            "scene_count": len({str(row["scene_token"]) for row in rows}),
            "clip_duration_s": finite_summary([float(row["clip_duration_s"]) for row in rows]),
            "valid_actor_track_count": int(sum(int(row["valid_actor_track_count"]) for row in rows)),
            "valid_actor_pair_count": int(sum(int(row["valid_actor_pair_count"]) for row in rows)),
            "timestamp_unit": "microseconds",
            "camera": "CAM_FRONT",
        }
    )
    return summary


def _copy_data_config(data_cfg: Any, *, split: str) -> Any:
    copied = OmegaConf.create(copy.deepcopy(OmegaConf.to_container(data_cfg, resolve=True)))
    copied.split = str(split)
    copied.use_lidar_depth = False
    return copied


def _condition_frame(dataset: NuScenesFutureVideoDataset, sample_token: str) -> torch.Tensor:
    """只读取首帧 RGB；不触发 dataset future frame/box/LiDAR 加载。"""
    from PIL import Image

    sample = dataset.nusc.get("sample", str(sample_token))
    camera_token = sample["data"][dataset.camera]
    path = dataset.nusc.get_sample_data_path(camera_token)
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB").resize((dataset.W, dataset.H), Image.BILINEAR)).copy()
    return torch.from_numpy(array).float().permute(2, 0, 1).div(127.5).sub(1.0)


class _SamplingIdentityTrace:
    """只捕获 R1 配对所需的 condition noise、initial latent 与 time-id。"""

    def __init__(self, pipe: Any):
        self.pipe = pipe
        self.condition_noise: torch.Tensor | None = None
        self.initial_latents: torch.Tensor | None = None
        self.added_time_ids: torch.Tensor | None = None
        self._random_draws: list[torch.Tensor] = []
        self._restore: list[tuple[Any, str, Any]] = []

    def _replace(self, owner: Any, name: str, replacement: Any) -> None:
        self._restore.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    def __enter__(self):
        original_prepare = self.pipe.prepare_latents
        original_time_ids = self.pipe._get_add_time_ids

        def prepare_latents(pipeline, *args, **kwargs):
            value = original_prepare(*args, **kwargs)
            self.initial_latents = value.detach().cpu().clone()
            return value

        def time_ids(pipeline, *args, **kwargs):
            value = original_time_ids(*args, **kwargs)
            self.added_time_ids = value.detach().cpu().clone()
            return value

        self._replace(self.pipe, "prepare_latents", types.MethodType(prepare_latents, self.pipe))
        self._replace(self.pipe, "_get_add_time_ids", types.MethodType(time_ids, self.pipe))

        import diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion as module

        original_randn = module.randn_tensor

        def randn_tensor(*args, **kwargs):
            value = original_randn(*args, **kwargs)
            self._random_draws.append(value.detach().cpu().clone())
            return value

        self._replace(module, "randn_tensor", randn_tensor)
        return self

    def __exit__(self, exc_type, exc, traceback):
        for owner, name, original in reversed(self._restore):
            setattr(owner, name, original)
        if exc_type is None:
            if not self._random_draws or self.initial_latents is None or self.added_time_ids is None:
                raise TemporalSamplingAuditError("SVD sampling identity trace 不完整")
            self.condition_noise = self._random_draws[0]


def basic_video_metrics(frames: torch.Tensor, condition: torch.Tensor) -> dict[str, float]:
    """不使用 future GT 的像素级 quality/motion guard。"""
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError("frames 必须为 [T,3,H,W]")
    if condition.shape != frames.shape[1:]:
        raise ValueError("condition 与生成帧空间形状不一致")
    video = frames.detach().float().cpu().clamp(-1.0, 1.0)
    cond = condition.detach().float().cpu().clamp(-1.0, 1.0)
    unit = (video + 1.0) / 2.0
    cond_unit = (cond + 1.0) / 2.0
    difference = unit[1:] - unit[:-1]
    dynamic_degree = float(difference.abs().mean()) if int(difference.numel()) else 0.0
    luma = 0.299 * unit[:, 0] + 0.587 * unit[:, 1] + 0.114 * unit[:, 2]
    global_luma = luma.mean(dim=(1, 2))
    global_delta = (global_luma[1:] - global_luma[:-1]).abs()
    flicker_p95 = float(torch.quantile(global_delta, 0.95)) if int(global_delta.numel()) else 0.0
    laplacian = (
        -4.0 * luma[:, 1:-1, 1:-1]
        + luma[:, :-2, 1:-1]
        + luma[:, 2:, 1:-1]
        + luma[:, 1:-1, :-2]
        + luma[:, 1:-1, 2:]
    )
    sharpness = float(laplacian.square().mean()) if int(laplacian.numel()) else 0.0
    first_mse = float((unit[0] - cond_unit).square().mean())
    first_psnr = float(-10.0 * math.log10(max(first_mse, 1.0e-12)))
    return {
        "dynamic_degree_mean_abs_rgb": dynamic_degree,
        "global_luma_flicker_p95": flicker_p95,
        "spatial_laplacian_energy": sharpness,
        "first_frame_mse": first_mse,
        "first_frame_psnr_db": first_psnr,
        "first_frame_mae": float((unit[0] - cond_unit).abs().mean()),
        "finite_fraction": float(torch.isfinite(frames).float().mean()),
    }


def acceleration_metrics(
    points: torch.Tensor,
    visibility: torch.Tensor,
    affine: torch.Tensor,
    *,
    outlier_threshold_px: float,
) -> dict[str, Any]:
    velocity, valid_velocity = camera_compensated_velocity(points, visibility, affine)
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    valid = valid_velocity[:, 1:] & valid_velocity[:, :-1]
    magnitude = torch.linalg.vector_norm(acceleration, dim=-1)[valid]
    magnitude = magnitude[torch.isfinite(magnitude)]
    if not int(magnitude.numel()):
        return {"status": "invalid", "count": 0}
    return {
        "status": "valid",
        "count": int(magnitude.numel()),
        "median_px_per_frame2": float(torch.quantile(magnitude, 0.5)),
        "p95_px_per_frame2": float(torch.quantile(magnitude, 0.95)),
        "outlier_threshold_px_per_frame2": float(outlier_threshold_px),
        "outlier_fraction": float((magnitude > float(outlier_threshold_px)).float().mean()),
    }


def _paired_groups(rows: Sequence[Mapping[str, Any]], fps_values: Sequence[int]) -> dict[tuple[str, int], dict[int, Mapping[str, Any]]]:
    groups: dict[tuple[str, int], dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if not bool(row.get("valid", False)):
            continue
        key = (str(row["condition_id"]), int(row["generation_seed"]))
        fps = int(row["fps_input"])
        if fps in groups[key]:
            raise TemporalSamplingAuditError(f"重复 R1 case: {key} fps={fps}")
        groups[key][fps] = row
    required = set(int(value) for value in fps_values)
    return {key: value for key, value in groups.items() if set(value) == required}


def _metric(row: Mapping[str, Any], name: str) -> float:
    value = row.get("metrics", {}).get(name)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise TemporalSamplingAuditError(f"R1 metric invalid: {name}")
    return float(value)


def paired_relative_effect(
    pairs: Sequence[tuple[float, float]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """alt 对 fps=7 的配对相对变化；CI 对配对中位数 bootstrap。"""
    if not pairs:
        return {"status": "invalid", "count": 0}
    values = np.asarray(
        [(alt - reference) / max(abs(reference), 1.0e-8) for alt, reference in pairs],
        dtype=np.float64,
    )
    rng = np.random.default_rng(int(seed))
    boot = np.empty(int(bootstrap_samples), dtype=np.float64)
    for index in range(int(bootstrap_samples)):
        sampled = rng.integers(0, len(values), size=len(values))
        boot[index] = np.median(values[sampled])
    return {
        "status": "valid",
        "count": int(len(values)),
        "median_relative_change": float(np.median(values)),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
    }


def _median_ratio(groups: Mapping[tuple[str, int], Mapping[int, Mapping[str, Any]]], alt: int, reference: int, metric: str) -> float:
    ratios = [
        _metric(group[alt], metric) / max(abs(_metric(group[reference], metric)), 1.0e-8)
        for group in groups.values()
    ]
    return float(np.median(np.asarray(ratios, dtype=np.float64)))


def _median_delta(groups: Mapping[tuple[str, int], Mapping[int, Mapping[str, Any]]], alt: int, reference: int, metric: str) -> float:
    values = [_metric(group[alt], metric) - _metric(group[reference], metric) for group in groups.values()]
    return float(np.median(np.asarray(values, dtype=np.float64)))


def decide_temporal_conditioning(
    rows: Sequence[Mapping[str, Any]],
    *,
    fps_values: Sequence[int],
    reference_fps: int,
    real_effective_fps: float,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """预注册 R1 machine decision；不使用训练结果或 future GT。"""
    values = [int(value) for value in fps_values]
    reference = int(reference_fps)
    if reference not in values:
        raise ValueError("reference_fps 必须属于 fps_values")
    groups = _paired_groups(rows, values)
    minimum_groups = int(thresholds["minimum_paired_groups"])
    if len(groups) < minimum_groups:
        return {
            "status": "blocked",
            "reason": "insufficient_valid_paired_groups",
            "paired_group_count": len(groups),
            "minimum_paired_groups": minimum_groups,
            "selected_fps": None,
        }

    comparisons: dict[str, Any] = {}
    eligible: list[int] = []
    for alt in values:
        if alt == reference:
            continue
        effects: dict[str, Any] = {}
        motion_significant = False
        for offset, metric_name in enumerate(
            ("dynamic_degree_mean_abs_rgb", "image_plane_velocity_rms_px")
        ):
            pairs = [
                (_metric(group[alt], metric_name), _metric(group[reference], metric_name))
                for group in groups.values()
            ]
            effect = paired_relative_effect(
                pairs,
                bootstrap_samples=int(thresholds["bootstrap_samples"]),
                seed=int(thresholds["bootstrap_seed"]) + alt * 17 + offset,
            )
            effect_size = abs(float(effect["median_relative_change"]))
            ci_excludes_zero = float(effect["ci95_low"]) > 0.0 or float(effect["ci95_high"]) < 0.0
            effect["significant"] = bool(
                effect_size >= float(thresholds["minimum_motion_relative_change"])
                and ci_excludes_zero
            )
            motion_significant = motion_significant or bool(effect["significant"])
            effects[metric_name] = effect

        quality = {
            "first_frame_psnr_delta_db": _median_delta(groups, alt, reference, "first_frame_psnr_db"),
            "sharpness_ratio": _median_ratio(groups, alt, reference, "spatial_laplacian_energy"),
            "flicker_ratio": _median_ratio(groups, alt, reference, "global_luma_flicker_p95"),
            "survival_ratio": _median_ratio(groups, alt, reference, "survival_rate"),
            "acceleration_p95_ratio": _median_ratio(groups, alt, reference, "acceleration_p95_px_per_frame2"),
            "dynamic_degree_ratio": _median_ratio(groups, alt, reference, "dynamic_degree_mean_abs_rgb"),
            "velocity_ratio": _median_ratio(groups, alt, reference, "image_plane_velocity_rms_px"),
        }
        checks = {
            "first_frame": quality["first_frame_psnr_delta_db"] >= -float(thresholds["maximum_first_frame_psnr_drop_db"]),
            "sharpness": quality["sharpness_ratio"] >= float(thresholds["minimum_sharpness_ratio"]),
            "flicker": quality["flicker_ratio"] <= float(thresholds["maximum_flicker_ratio"]),
            "track_survival": quality["survival_ratio"] >= float(thresholds["minimum_survival_ratio"]),
            "acceleration": quality["acceleration_p95_ratio"] <= float(thresholds["maximum_acceleration_p95_ratio"]),
            "dynamic_motion_floor": quality["dynamic_degree_ratio"] >= float(thresholds["minimum_motion_floor_ratio"]),
            "velocity_motion_floor": quality["velocity_ratio"] >= float(thresholds["minimum_motion_floor_ratio"]),
        }
        quality_preserved = all(bool(value) for value in checks.values())
        candidate_eligible = bool(motion_significant and quality_preserved)
        if candidate_eligible:
            eligible.append(alt)
        comparisons[str(alt)] = {
            "motion_effects": effects,
            "motion_significant": motion_significant,
            "quality": quality,
            "quality_checks": checks,
            "quality_preserved": quality_preserved,
            "eligible": candidate_eligible,
        }

    if eligible:
        selected = min(eligible, key=lambda fps: (abs(float(fps) - float(real_effective_fps)), fps))
        reason = "lower_fps_changes_motion_with_quality_and_motion_floor_preserved"
    else:
        selected = reference
        reason = "no_lower_fps_met_significance_and_safeguards"
    return {
        "status": "done",
        "reason": reason,
        "selected_fps": int(selected),
        "reference_fps": reference,
        "real_effective_fps_median": float(real_effective_fps),
        "paired_group_count": len(groups),
        "comparisons": comparisons,
        "selection_rule": "eligible fps closest to real median effective fps; otherwise keep reference",
    }


def _preflight_model_path(pretrained: str) -> dict[str, Any]:
    root = Path(str(pretrained))
    required = [
        root / "model_index.json",
        root / "unet" / "config.json",
        root / "vae" / "config.json",
        root / "scheduler" / "scheduler_config.json",
        root / "image_encoder" / "config.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    return {
        "ready": not missing,
        "root": str(root),
        "missing": missing,
        "model_index_sha256": file_fingerprint(str(root / "model_index.json")) if not missing else None,
    }


def preflight_temporal_sampling_audit(cfg: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "task_id": str(cfg.temporal.task_id),
        "status": "ready",
        "uses_gpu": False,
        "uses_future_gt_for_generation": False,
        "blockers": [],
    }
    try:
        train = NuScenesFutureVideoDataset(_copy_data_config(cfg.data, split="train"))
        val = NuScenesFutureVideoDataset(_copy_data_config(cfg.data, split="val"))
        train_selected = select_scene_distinct_clip_records(
            train.clip_records, count=int(cfg.temporal.real_clip_count),
        )
        val_selected = select_scene_distinct_clip_records(
            val.clip_records, count=int(cfg.temporal.validation_condition_count),
        )
        result["data"] = {
            "train_clip_count": len(train),
            "val_clip_count": len(val),
            "selected_train_scene_count": len(train_selected),
            "selected_val_scene_count": len(val_selected),
            "train_sample_ids": [str(row["sample_id"]) for row in train_selected],
            "val_sample_ids": [str(row["sample_id"]) for row in val_selected],
        }
    except Exception as exc:
        result["blockers"].append({"kind": "nuscenes", "error": repr(exc)})
    model = _preflight_model_path(str(cfg.model.pretrained))
    result["model"] = model
    if not bool(model["ready"]):
        result["blockers"].append({"kind": "svd", "missing": model["missing"]})
    evaluator = CoTracker3IndependentEvaluator(dict(cfg.temporal.evaluator)).preflight()
    result["evaluator"] = evaluator
    if not bool(evaluator.get("available")):
        result["blockers"].append({"kind": "cotracker3", "reasons": evaluator.get("reasons", [])})
    usage = shutil.disk_usage(Path(str(cfg.work_dir)).parent)
    free_gb = float(usage.free) / float(1024**3)
    result["disk"] = {"free_gb": free_gb, "minimum_free_gb": float(cfg.temporal.minimum_free_disk_gb)}
    if free_gb < float(cfg.temporal.minimum_free_disk_gb):
        result["blockers"].append({"kind": "disk", "free_gb": free_gb})
    if result["blockers"]:
        result["status"] = "blocked"
    return result


def _validate_protocol(cfg: Any) -> None:
    temporal = cfg.temporal
    fps_values = [int(value) for value in temporal.fps_values]
    if fps_values != sorted(set(fps_values)) or fps_values != [2, 4, 7]:
        raise TemporalSamplingAuditError("R1 fps_values 必须恰为 [2, 4, 7]")
    if int(temporal.reference_fps) != 7 or int(cfg.model.generation.fps) != 7:
        raise TemporalSamplingAuditError("R1 reference/config fps 必须为 7")
    if str(cfg.model.generation.protocol) != "svd_official_v1" or bool(cfg.model.lora.enable):
        raise TemporalSamplingAuditError("R1 只允许冻结 svd_official_v1 Base")
    if int(temporal.num_inference_steps) != 25 or int(temporal.num_frames) != 8:
        raise TemporalSamplingAuditError("R1 必须使用 25 steps、8 frames")
    if len([int(value) for value in temporal.generation_seeds]) != 2:
        raise TemporalSamplingAuditError("R1 必须恰有 2 个 generation seeds")
    if int(cfg.data.num_frames) != 8 or int(cfg.model.num_frames) != 8:
        raise TemporalSamplingAuditError("R1 data/model num_frames 必须为 8")


def _case_condition_id(record: Mapping[str, Any]) -> str:
    payload = f"{record['scene_token']}:{record['sample_id']}"
    return f"r1-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _sync_cuda(device: str) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _generate_case(
    backbone: SVDBackbone,
    condition: torch.Tensor,
    *,
    fps: int,
    seed: int,
    num_frames: int,
    num_inference_steps: int,
    height: int,
    width: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    pipe = backbone._generation_pipeline()
    pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=backbone.device).manual_seed(int(seed))
    started = time.perf_counter()
    with _SamplingIdentityTrace(pipe) as trace:
        frames = backbone.generate(
            condition.to(backbone.device),
            num_frames=int(num_frames),
            num_inference_steps=int(num_inference_steps),
            height=int(height),
            width=int(width),
            fps=int(fps),
            generator=generator,
            decode_chunk_size=4,
        )
    _sync_cuda(str(backbone.device))
    assert trace.condition_noise is not None
    assert trace.initial_latents is not None
    assert trace.added_time_ids is not None
    if int(round(float(trace.added_time_ids[0, 0]))) != int(fps) - 1:
        raise TemporalSamplingAuditError("SVD fps time-id 未等于 fps-1")
    return frames.detach().cpu(), {
        "generation_seconds": float(time.perf_counter() - started),
        "condition_noise_sha256": _tensor_fingerprint(trace.condition_noise),
        "initial_video_latents_sha256": _tensor_fingerprint(trace.initial_latents),
        "added_time_ids": trace.added_time_ids.float().tolist(),
    }


def _validate_paired_identity(rows: Sequence[Mapping[str, Any]], fps_values: Sequence[int]) -> None:
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["condition_id"]), int(row["generation_seed"]))].append(row)
    required = set(int(value) for value in fps_values)
    for key, group in groups.items():
        actual = {int(row["fps_input"]) for row in group}
        if actual != required:
            raise TemporalSamplingAuditError(f"R1 paired fps 不完整: {key} actual={sorted(actual)}")
        initial = {str(row["initial_video_latents_sha256"]) for row in group}
        condition_noise = {str(row["condition_noise_sha256"]) for row in group}
        if len(initial) != 1 or len(condition_noise) != 1:
            raise TemporalSamplingAuditError(f"R1 同 condition/seed 未共享 exact initial noise: {key}")


def _score_case(
    evaluator: CoTracker3IndependentEvaluator,
    frames: torch.Tensor,
    *,
    outlier_threshold_px: float,
) -> tuple[dict[str, Any], dict[str, float]]:
    started = time.perf_counter()
    state = evaluator.track(frames)
    dynamics = summarize_camera_compensated_dynamics(state)
    aggregate = aggregate_dynamics(dynamics)
    acceleration = acceleration_metrics(
        state.points,
        state.visibility,
        state.affine_background,
        outlier_threshold_px=float(outlier_threshold_px),
    )
    valid = bool(state.valid) and aggregate is not None and acceleration.get("status") == "valid"
    detail = {
        "valid": valid,
        "seconds": float(time.perf_counter() - started),
        "query_count": int(state.visibility.shape[0]),
        "track_survival": float(state.visibility[:, -1].float().mean()),
        "track_coverage": float(state.visibility.float().mean()),
        "dynamics": dynamics,
        "aggregate": aggregate,
        "acceleration": acceleration,
        "provider_diagnostics": state.diagnostics,
    }
    if not valid:
        return detail, {}
    assert aggregate is not None
    return detail, {
        "survival_rate": float(aggregate["survival_rate"]),
        "image_plane_velocity_rms_px": float(aggregate["camera_compensated_image_plane_velocity_rms_px"]),
        "image_plane_acceleration_rms_px": float(aggregate["camera_compensated_image_plane_acceleration_rms_px"]),
        "image_plane_jerk_rms_px": float(aggregate["camera_compensated_image_plane_jerk_rms_px"]),
        "acceleration_p95_px_per_frame2": float(acceleration["p95_px_per_frame2"]),
        "acceleration_outlier_fraction": float(acceleration["outlier_fraction"]),
    }


def _make_review_material(
    work_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    reference_fps: int,
) -> dict[str, Any]:
    review_dir = work_dir / "review"
    videos_dir = review_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=False)
    by_group: dict[tuple[str, int], dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_group[(str(row["condition_id"]), int(row["generation_seed"]))][int(row["fps_input"])] = row
    public_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    template_rows: list[dict[str, Any]] = []
    index = 0
    for group_key in sorted(by_group):
        group = by_group[group_key]
        for alternate in sorted(fps for fps in group if fps != int(reference_fps)):
            index += 1
            case_id = f"r1-review-{index:03d}"
            flip = int(hashlib.sha256(case_id.encode("utf-8")).hexdigest(), 16) % 2 == 1
            ordered = [group[alternate], group[int(reference_fps)]]
            if flip:
                ordered.reverse()
            case_dir = videos_dir / case_id
            case_dir.mkdir()
            side_paths = []
            for side, row in zip(("a", "b"), ordered):
                source = work_dir / str(row["video_path"])
                destination = case_dir / f"{side}.mp4"
                shutil.copy2(source, destination)
                side_paths.append(str(destination.relative_to(work_dir)))
            public_rows.append(
                {
                    "case_id": case_id,
                    "video_a": side_paths[0],
                    "video_b": side_paths[1],
                    "question": "哪一侧在保持画质和首帧一致性的同时呈现更可信的驾驶运动？",
                }
            )
            key_rows.append(
                {
                    "case_id": case_id,
                    "condition_id": group_key[0],
                    "generation_seed": group_key[1],
                    "side_a_fps": int(ordered[0]["fps_input"]),
                    "side_b_fps": int(ordered[1]["fps_input"]),
                }
            )
            template_rows.append(
                {
                    "case_id": case_id,
                    "motion_preference": None,
                    "quality_a": None,
                    "quality_b": None,
                    "first_frame_a": None,
                    "first_frame_b": None,
                    "notes": "",
                }
            )
    _write_jsonl(review_dir / "cases.jsonl", public_rows)
    _write_jsonl(review_dir / "reviews.template.jsonl", template_rows)
    _write_jsonl(review_dir / "review_key.jsonl", key_rows)
    prompt = """# R1 fps 对照人工复核

每个 case 的 A/B 使用相同首帧、相同 SVD 权重、相同 initial noise、相同 25-step 协议；
只改变 SVD fps micro-conditioning。所有视频以相同 playback fps 编码，避免播放速度泄漏。

请分别检查：

1. 首帧是否忠实且无明显偏移；
2. 车辆、行人、背景视差与相机运动是否自然；
3. 是否出现跳变、形变、闪烁、纹理漂移或近静止投机；
4. 在两侧都可接受时选择 motion 更可信的一侧；无法区分填 tie；任一侧不可审填 invalid。

`motion_preference` 只允许 `a / b / tie / invalid`；质量与首帧字段只允许 `pass / fail / invalid`。
不要打开 `review_key.jsonl`，该文件只用于聚合解盲。不得根据运动多少直接判优。
"""
    atomic_write_text(str(review_dir / "REVIEW_PROMPT.md"), prompt)
    return {
        "status": "awaiting_reviews",
        "case_count": len(public_rows),
        "cases_path": str((review_dir / "cases.jsonl").relative_to(work_dir)),
        "template_path": str((review_dir / "reviews.template.jsonl").relative_to(work_dir)),
        "prompt_path": str((review_dir / "REVIEW_PROMPT.md").relative_to(work_dir)),
        "key_path": str((review_dir / "review_key.jsonl").relative_to(work_dir)),
    }


def run_temporal_sampling_audit(cfg: Any) -> dict[str, Any]:
    _validate_protocol(cfg)
    git = git_state(".")
    if git.get("dirty"):
        raise TemporalSamplingAuditError("正式 R1 拒绝在 dirty worktree 上运行")
    work_dir = Path(str(cfg.work_dir))
    if work_dir.exists():
        raise FileExistsError(f"R1 run directory 已存在: {work_dir}")
    preflight = preflight_temporal_sampling_audit(cfg)
    if preflight["status"] != "ready":
        raise TemporalSamplingAuditError(f"R1 preflight blocked: {preflight['blockers']}")

    temporal = cfg.temporal
    config_fp = config_fingerprint(cfg)
    work_dir.mkdir(parents=True, exist_ok=False)
    (work_dir / "videos").mkdir()
    manifest = RunManifest(
        run_id=str(cfg.run_id),
        command=list(sys.argv),
        config_fingerprint=config_fp,
        cache_fingerprint="not-applicable:route-pivot-r1-temporal",
        seed=int(cfg.seed),
        git=git,
        environment=environment_fingerprint(),
        data_split="nuScenes official train timing + val first-frame generation",
    )
    manifest_data = manifest.__dict__ | {
        "task_id": str(temporal.task_id),
        "preflight": preflight,
        "uses_future_gt_for_real_video_timing_statistics": True,
        "uses_future_gt_for_generation": False,
        "uses_future_gt_for_generated_evaluation": False,
        "generation_protocol": "svd_official_v1/fps_sweep_r1",
        "fps_values": [int(value) for value in temporal.fps_values],
        "generation_seeds": [int(value) for value in temporal.generation_seeds],
        "playback_fps": int(temporal.playback_fps),
        "base_model_fingerprint": _base_model_fingerprint(str(cfg.model.pretrained)),
    }
    atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
    save_resolved_config(cfg, str(work_dir / "resolved.yaml"))
    metrics_log = JsonlMetrics(str(work_dir / "metrics.jsonl"))

    try:
        train_dataset = NuScenesFutureVideoDataset(_copy_data_config(cfg.data, split="train"))
        real_selected = select_scene_distinct_clip_records(
            train_dataset.clip_records, count=int(temporal.real_clip_count),
        )
        real_rows = [
            temporal_record_from_nuscenes(train_dataset.nusc, row, camera=train_dataset.camera)
            for row in real_selected
        ]
        real_summary = summarize_real_temporal_records(real_rows)
        if real_summary["effective_fps"].get("status") != "valid":
            raise TemporalSamplingAuditError("真实 effective fps 统计 invalid")
        _write_jsonl(work_dir / "real_temporal_clips.jsonl", real_rows)
        atomic_write_json(str(work_dir / "real_temporal_summary.json"), real_summary)
        metrics_log.append(0, {
            "event": "real_temporal_summary",
            "clip_count": real_summary["clip_count"],
            "effective_fps_median": real_summary["effective_fps"]["median"],
            "clip_duration_median_s": real_summary["clip_duration_s"]["median"],
            "valid_actor_track_count": real_summary["valid_actor_track_count"],
        })

        val_dataset = NuScenesFutureVideoDataset(_copy_data_config(cfg.data, split="val"))
        val_selected = select_scene_distinct_clip_records(
            val_dataset.clip_records, count=int(temporal.validation_condition_count),
        )
        conditions: list[dict[str, Any]] = []
        condition_tensors: dict[str, torch.Tensor] = {}
        for row in val_selected:
            condition_id = _case_condition_id(row)
            frame = _condition_frame(val_dataset, str(row["sample_tokens"][0]))
            condition_tensors[condition_id] = frame
            conditions.append(
                {
                    "condition_id": condition_id,
                    "scene_name": str(row["scene_name"]),
                    "scene_token": str(row["scene_token"]),
                    "sample_id": str(row["sample_id"]),
                    "first_sample_token": str(row["sample_tokens"][0]),
                    "condition_frame_sha256": _tensor_fingerprint(frame),
                    "uses_future_gt": False,
                }
            )
        _write_jsonl(work_dir / "conditions.jsonl", conditions)
        manifest_data["conditions"] = conditions
        atomic_write_json(str(work_dir / "manifest.json"), manifest_data)

        seed_everything(int(cfg.seed), deterministic=bool(cfg.train.deterministic))
        backbone = build_backbone(cfg.model, load=True, device=str(cfg.device))
        if not isinstance(backbone, SVDBackbone):
            raise TemporalSamplingAuditError("R1 当前只支持 SVDBackbone")
        backbone.unet.eval()
        backbone.vae.eval()
        backbone.image_encoder.eval()
        case_rows: list[dict[str, Any]] = []
        frames_by_case: dict[str, torch.Tensor] = {}
        condition_by_id = {str(row["condition_id"]): row for row in conditions}
        for condition_row in conditions:
            condition_id = str(condition_row["condition_id"])
            condition = condition_tensors[condition_id]
            for generation_seed in [int(value) for value in temporal.generation_seeds]:
                for fps in [int(value) for value in temporal.fps_values]:
                    case_id = f"{condition_id}-s{generation_seed}-fps{fps}"
                    frames, trace = _generate_case(
                        backbone,
                        condition,
                        fps=fps,
                        seed=generation_seed,
                        num_frames=int(temporal.num_frames),
                        num_inference_steps=int(temporal.num_inference_steps),
                        height=int(cfg.data.height),
                        width=int(cfg.data.width),
                    )
                    if not bool(torch.isfinite(frames).all()):
                        raise TemporalSamplingAuditError(f"R1 generated RGB NaN/Inf: {case_id}")
                    relative = Path("videos") / condition_id / f"s{generation_seed}" / f"fps{fps}.mp4"
                    destination = work_dir / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    write_video(to_uint8_video(frames), str(destination), fps=int(temporal.playback_fps))
                    if not destination.is_file() and destination.with_suffix(".npy").is_file():
                        destination = destination.with_suffix(".npy")
                        relative = destination.relative_to(work_dir)
                    if not destination.is_file():
                        raise TemporalSamplingAuditError(f"R1 video writer 未产生 artifact: {case_id}")
                    row = {
                        "case_id": case_id,
                        "condition_id": condition_id,
                        "scene_name": str(condition_by_id[condition_id]["scene_name"]),
                        "sample_id": str(condition_by_id[condition_id]["sample_id"]),
                        "generation_seed": generation_seed,
                        "fps_input": fps,
                        "fps_time_id": fps - 1,
                        "playback_fps": int(temporal.playback_fps),
                        "video_path": str(relative),
                        "uses_future_gt": False,
                        **trace,
                        "pixel_metrics": basic_video_metrics(frames, condition),
                    }
                    case_rows.append(row)
                    frames_by_case[case_id] = frames
                    metrics_log.append(len(case_rows), {
                        "event": "generated",
                        "case_id": case_id,
                        "fps_input": fps,
                        "generation_seconds": trace["generation_seconds"],
                    })
        _validate_paired_identity(case_rows, [int(value) for value in temporal.fps_values])
        _write_jsonl(work_dir / "generation_cases.jsonl", case_rows)

        del backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            _sync_cuda(str(cfg.device))
        evaluator = CoTracker3IndependentEvaluator(dict(temporal.evaluator))
        evaluator._load()
        scored_rows: list[dict[str, Any]] = []
        for index, row in enumerate(case_rows, start=1):
            score_detail, track_metrics = _score_case(
                evaluator,
                frames_by_case[str(row["case_id"])],
                outlier_threshold_px=float(temporal.thresholds.acceleration_outlier_threshold_px),
            )
            metrics = {**dict(row["pixel_metrics"]), **track_metrics}
            valid = bool(score_detail["valid"]) and float(metrics.get("finite_fraction", 0.0)) == 1.0
            scored = {**row, "valid": valid, "metrics": metrics, "track_evaluation": score_detail}
            scored_rows.append(scored)
            metrics_log.append(index, {
                "event": "scored",
                "case_id": row["case_id"],
                "fps_input": row["fps_input"],
                "valid": valid,
                **{key: value for key, value in metrics.items() if isinstance(value, (int, float))},
            })
        _write_jsonl(work_dir / "scored_cases.jsonl", scored_rows)

        decision = decide_temporal_conditioning(
            scored_rows,
            fps_values=[int(value) for value in temporal.fps_values],
            reference_fps=int(temporal.reference_fps),
            real_effective_fps=float(real_summary["effective_fps"]["median"]),
            thresholds=dict(temporal.thresholds),
        )
        review = _make_review_material(
            work_dir, scored_rows, reference_fps=int(temporal.reference_fps),
        )
        valid_count = sum(bool(row["valid"]) for row in scored_rows)
        result = {
            "task_id": str(temporal.task_id),
            "real_temporal_summary": real_summary,
            "generation_case_count": len(scored_rows),
            "valid_generation_case_count": valid_count,
            "fps_values": [int(value) for value in temporal.fps_values],
            "decision": decision,
            "review": review,
            "same_initial_noise_verified": True,
            "generated_evaluator_uses_future_gt": False,
        }
        atomic_write_json(str(work_dir / "result.json"), result)
        status = str(decision["status"])
        summary = {
            "status": status,
            "task_id": str(temporal.task_id),
            "run_id": str(cfg.run_id),
            "config_fingerprint": config_fp,
            "real_clip_count": int(real_summary["clip_count"]),
            "real_effective_fps_median": float(real_summary["effective_fps"]["median"]),
            "generation_case_count": len(scored_rows),
            "valid_generation_case_count": valid_count,
            "paired_group_count": decision.get("paired_group_count"),
            "selected_fps": decision.get("selected_fps"),
            "decision_reason": decision.get("reason"),
            "review_status": review["status"],
            "review_case_count": review["case_count"],
            "same_initial_noise_verified": True,
            "uses_future_gt_for_generation": False,
            "result_fingerprint": sha256_json(result),
            "next_gate": "RP-A0-03" if status == "done" else None,
        }
        atomic_write_json(str(work_dir / "summary.json"), summary)
        atomic_write_text(str(work_dir / "COMPLETE"), sha256_json(summary) + "\n")
        manifest_data.update(
            {
                "status": "completed" if status == "done" else "failed",
                "ended_at": utc_now(),
                "exit_reason": str(decision.get("reason")),
                "selected_fps": decision.get("selected_fps"),
            }
        )
        atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
        return summary
    except Exception as exc:
        failure = {
            "status": "failed",
            "task_id": str(cfg.temporal.task_id),
            "run_id": str(cfg.run_id),
            "error": repr(exc),
        }
        atomic_write_json(str(work_dir / "summary.json"), failure)
        manifest_data.update({"status": "failed", "ended_at": utc_now(), "exit_reason": repr(exc)})
        atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="audit real temporal sampling and SVD fps conditioning")
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, list(args.overrides))
    result = preflight_temporal_sampling_audit(cfg) if args.preflight else run_temporal_sampling_audit(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
