"""RP-A1-SCAN：冻结 SVD 的真实 ego / actor-residual feature probe。

只训练固定容量 linear ridge heads；SVD、VAE、image encoder 全冻结。完整 feature map
只存在于单次 forward 内，落盘仅保存 query 处随机投影 feature、5x5 cost window 与 target。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf

from ..backbones import build_backbone
from ..backbones.base import Conditioning
from ..backbones.svd_backbone import SVDBackbone
from ..config import config_fingerprint, load_config, save_resolved_config
from ..data import NuScenesFutureVideoDataset
from ..data.motion_feature_records import (
    MotionFeatureRecordError,
    angular_error_deg,
    deterministic_subsample_indices,
    fit_ridge,
    local_correlation_window,
    permute_learned_features,
    permute_instance_targets,
    projection_fingerprint,
    random_projection_matrix,
    relative_improvement,
    sample_temporal_features,
    split_fingerprint,
    stable_scene_split,
    vector_epe,
)
from ..data.real_motion_targets import (
    REAL_TARGET_SCOPE,
    boxes_background_mask,
    build_actor_residual_targets,
    sparse_ego_flow_target,
    timestamps_to_seconds,
)
from ..runtime.atomic import atomic_write_json, atomic_write_text
from ..runtime.experiment import JsonlMetrics, RunManifest, utc_now
from ..runtime.fingerprint import environment_fingerprint, file_fingerprint, git_state, sha256_json
from ..train.trainer import seed_everything
from .svd_conditioning_parity import _base_model_fingerprint


class MotionFeatureProbeError(RuntimeError):
    """A1 split、feature、control 或 gate 不合法。"""


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(dict(row)), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _tensor_fingerprint(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _first_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for value in output:
            try:
                return _first_tensor(value)
            except TypeError:
                continue
    raise TypeError(f"hook output contains no tensor: {type(output)!r}")


class _FeatureCapture:
    def __init__(self, module: torch.nn.Module, layer_paths: Mapping[str, str]):
        self.outputs: dict[str, torch.Tensor] = {}
        self.handles = []
        for alias, path in layer_paths.items():
            target = module.get_submodule(str(path))
            self.handles.append(target.register_forward_hook(self._hook(str(alias))))

    def _hook(self, alias: str):
        def save(_module, _inputs, output):
            self.outputs[alias] = _first_tensor(output).detach()
        return save

    def clear(self) -> None:
        self.outputs.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _copy_data_config(cfg: Any) -> Any:
    copied = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    copied.split = "train"
    copied.use_lidar_depth = True
    return copied


def _official_conditioning(
    backbone: SVDBackbone,
    cond_frame: torch.Tensor,
    *,
    seed: int,
    height: int,
    width: int,
) -> tuple[Conditioning, dict[str, Any]]:
    generator = torch.Generator(device=backbone.device).manual_seed(int(seed))
    device_type = "cuda" if str(backbone.device).startswith("cuda") else "cpu"
    # diffusers 0.30 的 official SVD pipeline 依赖 autocast 将 fp32 preprocess
    # 输入适配到 bf16 VAE；直接调用私有 encode helper 会触发 dtype mismatch。
    with torch.autocast(device_type=device_type, dtype=backbone.dtype):
        packed = backbone.build_official_generation_conditioning(
            cond_frame,
            generator=generator,
            num_frames=int(backbone.cfg.num_frames),
            height=int(height),
            width=int(width),
        )

    def conditional_half(value: torch.Tensor) -> torch.Tensor:
        if value.shape[0] == 2:
            return value[-1:].contiguous()
        if value.shape[0] != 1:
            raise MotionFeatureProbeError(f"official conditioning batch 非 1/2: {tuple(value.shape)}")
        return value

    image_embeds = conditional_half(packed["image_embeds"])
    image_latents = conditional_half(packed["image_latents"])
    added_time_ids = conditional_half(packed["added_time_ids"])
    if int(round(float(added_time_ids[0, 0]))) != int(backbone.fps) - 1:
        raise MotionFeatureProbeError("A1 official fps time-id 未等于 fps-1")
    return Conditioning(
        data={
            "image_embeds": image_embeds,
            "image_latents": image_latents,
            "added_time_ids": added_time_ids,
        }
    ), {
        "condition_noise_sha256": _tensor_fingerprint(packed["condition_noise"]),
        "fps_input": int(packed["fps_input"]),
        "fps_time_id": int(packed["fps_time_id"]),
        "do_classifier_free_guidance": bool(packed["do_classifier_free_guidance"]),
    }


def _balanced_actor_rows(rows: Sequence[Mapping[str, Any]], maximum: int) -> list[dict[str, Any]]:
    ordered = sorted(
        (dict(row) for row in rows if bool(row.get("localizable_common_support"))),
        key=lambda row: (
            {"moving": 0, "stationary": 1}.get(str(row.get("motion_label")), 2),
            str(row.get("instance_token")),
            int(row.get("frame_index", -1)),
        ),
    )
    if len(ordered) <= int(maximum):
        return ordered
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ordered:
        groups[str(row.get("motion_label", "unknown"))].append(row)
    selected: list[dict[str, Any]] = []
    primary_budget = int(maximum) // 2
    for label in ("moving", "stationary"):
        group = groups.get(label, [])
        indices = deterministic_subsample_indices(len(group), min(primary_budget, len(group)))
        selected.extend(group[int(index)] for index in indices)
    used = {(row["instance_token"], row["frame_index"]) for row in selected}
    remaining = [
        row for row in ordered
        if (row["instance_token"], row["frame_index"]) not in used
    ]
    if len(selected) < int(maximum):
        indices = deterministic_subsample_indices(len(remaining), int(maximum) - len(selected))
        selected.extend(remaining[int(index)] for index in indices)
    return sorted(selected, key=lambda row: (int(row["frame_index"]), str(row["instance_token"])))


def prepare_motion_queries(
    sample: Mapping[str, Any],
    *,
    sample_id: str,
    scene_name: str,
    split_name: str,
    maximum_actor_queries: int,
    maximum_ego_queries: int,
    box_dilation_px: int,
) -> dict[str, Any]:
    frames = torch.as_tensor(sample["frames"])
    _, _, height, width = frames.shape
    dt = timestamps_to_seconds(sample["timestamps"]).float()
    actor_rows = _balanced_actor_rows(
        build_actor_residual_targets(sample, min_visibility=2),
        int(maximum_actor_queries),
    )
    actor = {
        "time": torch.tensor([int(row["frame_index"]) for row in actor_rows], dtype=torch.long),
        "source": torch.tensor([row["actual_uv_t"] for row in actor_rows], dtype=torch.float32).reshape(-1, 2),
        "static": torch.tensor([row["static_uv_tp1"] for row in actor_rows], dtype=torch.float32).reshape(-1, 2),
        "actual_next": torch.tensor([row["actual_uv_tp1"] for row in actor_rows], dtype=torch.float32).reshape(-1, 2),
        "residual_target": torch.tensor(
            [[row["residual_px"][0] / width, row["residual_px"][1] / height] for row in actor_rows],
            dtype=torch.float32,
        ).reshape(-1, 2),
        "absolute_target": torch.tensor(
            [[row["actual_uv_tp1"][0] / width, row["actual_uv_tp1"][1] / height] for row in actor_rows],
            dtype=torch.float32,
        ).reshape(-1, 2),
        "res_aux": torch.tensor(
            [[row["static_uv_tp1"][0] / width, row["static_uv_tp1"][1] / height, row["dt_s"]] for row in actor_rows],
            dtype=torch.float32,
        ).reshape(-1, 3),
        "abs_aux": torch.tensor(
            [[row["actual_uv_t"][0] / width, row["actual_uv_t"][1] / height, row["dt_s"]] for row in actor_rows],
            dtype=torch.float32,
        ).reshape(-1, 3),
        "labels": [str(row["motion_label"]) for row in actor_rows],
        "categories": [str(row["category"]) for row in actor_rows],
        "instance_ids": [str(row["instance_token"]) for row in actor_rows],
        "query_ids": [
            f"{sample_id}:actor:{row['instance_token']}:{row['frame_index']}"
            for row in actor_rows
        ],
        "metadata": actor_rows,
    }

    intrinsics = torch.as_tensor(sample["intrinsics_frames"], dtype=torch.float64)
    cam2ego = torch.as_tensor(sample["cam2ego_frames"], dtype=torch.float64)
    ego2global = torch.as_tensor(sample["ego2global"], dtype=torch.float64)
    lidar = torch.as_tensor(sample["lidar_depth"], dtype=torch.float64)
    per_pair = max(int(math.ceil(maximum_ego_queries / max(frames.shape[0] - 1, 1))), 1)
    ego_time: list[int] = []
    ego_points: list[list[float]] = []
    ego_targets: list[list[float]] = []
    ego_aux: list[list[float]] = []
    ego_ids: list[str] = []
    for time in range(frames.shape[0] - 1):
        flow, valid = sparse_ego_flow_target(
            lidar[time], intrinsics[time], intrinsics[time + 1],
            cam2ego[time], cam2ego[time + 1], ego2global[time], ego2global[time + 1],
        )
        valid &= boxes_background_mask(
            height, width, sample["boxes"][time], dilation_px=int(box_dilation_px),
        )
        valid &= boxes_background_mask(
            height, width, sample["boxes"][time + 1], dilation_px=int(box_dilation_px),
        )
        indices = torch.nonzero(valid, as_tuple=False)
        selected = deterministic_subsample_indices(int(indices.shape[0]), per_pair)
        for index in indices[selected].tolist():
            v, u = int(index[0]), int(index[1])
            vector = flow[v, u]
            ego_time.append(time)
            ego_points.append([float(u), float(v)])
            ego_targets.append([float(vector[0]) / width, float(vector[1]) / height])
            ego_aux.append([float(u) / width, float(v) / height, float(dt[time])])
            ego_ids.append(f"{sample_id}:ego:{time}:{u}:{v}")
    if len(ego_time) > int(maximum_ego_queries):
        selected = deterministic_subsample_indices(len(ego_time), int(maximum_ego_queries)).tolist()
        ego_time = [ego_time[index] for index in selected]
        ego_points = [ego_points[index] for index in selected]
        ego_targets = [ego_targets[index] for index in selected]
        ego_aux = [ego_aux[index] for index in selected]
        ego_ids = [ego_ids[index] for index in selected]
    ego = {
        "time": torch.tensor(ego_time, dtype=torch.long),
        "source": torch.tensor(ego_points, dtype=torch.float32).reshape(-1, 2),
        "target": torch.tensor(ego_targets, dtype=torch.float32).reshape(-1, 2),
        "aux": torch.tensor(ego_aux, dtype=torch.float32).reshape(-1, 3),
        "query_ids": ego_ids,
    }
    query_rows = []
    for row, query_id in zip(actor_rows, actor["query_ids"]):
        query_rows.append(
            {
                "query_id": query_id,
                "query_type": "actor",
                "sample_id": sample_id,
                "scene_name": scene_name,
                "split": split_name,
                "frame_index": int(row["frame_index"]),
                "instance_token": str(row["instance_token"]),
                "category": str(row["category"]),
                "attributes": list(row["attributes"]),
                "visibility": [int(row["visibility_t"]), int(row["visibility_tp1"])],
                "dt_s": float(row["dt_s"]),
                "actual_position": list(row["actual_uv_tp1"]),
                "ego_expected_position": list(row["static_uv_tp1"]),
                "actor_residual": list(row["residual_px"]),
                "validity": "localizable_common_support",
            }
        )
    for query_id, time, point, target in zip(ego_ids, ego_time, ego_points, ego_targets):
        query_rows.append(
            {
                "query_id": query_id,
                "query_type": "ego_background",
                "sample_id": sample_id,
                "scene_name": scene_name,
                "split": split_name,
                "frame_index": int(time),
                "query_position": point,
                "ego_flow_normalized": target,
                "dt_s": float(dt[int(time)]),
                "validity": "lidar_box_excluded_in_frame",
            }
        )
    return {"actor": actor, "ego": ego, "query_rows": query_rows, "image_hw": (height, width)}


def _layer_projection_seed(base_seed: int, alias: str) -> int:
    digest = int(hashlib.sha256(str(alias).encode("utf-8")).hexdigest()[:8], 16)
    return int(base_seed) + digest % 1_000_000


def extract_feature_bundle(
    features: torch.Tensor,
    queries: Mapping[str, Any],
    projection: torch.Tensor,
    *,
    radius_cells: int,
) -> dict[str, Any]:
    image_hw = tuple(int(value) for value in queries["image_hw"])
    actor = queries["actor"]
    ego = queries["ego"]
    feature_device = features.device
    source_actor = sample_temporal_features(
        features,
        actor["time"].to(feature_device),
        actor["source"].to(feature_device),
        image_hw=image_hw,
    ) if int(actor["time"].numel()) else torch.empty((0, features.shape[1]), device=feature_device)
    source_ego = sample_temporal_features(
        features,
        ego["time"].to(feature_device),
        ego["source"].to(feature_device),
        image_hw=image_hw,
    ) if int(ego["time"].numel()) else torch.empty((0, features.shape[1]), device=feature_device)
    res_cost = local_correlation_window(
        features,
        actor["time"].to(feature_device),
        actor["source"].to(feature_device),
        actor["static"].to(feature_device),
        image_hw=image_hw,
        radius_cells=int(radius_cells),
    ) if int(actor["time"].numel()) else torch.empty((0, (2 * radius_cells + 1) ** 2), device=feature_device)
    abs_cost = local_correlation_window(
        features,
        actor["time"].to(feature_device),
        actor["source"].to(feature_device),
        actor["source"].to(feature_device),
        image_hw=image_hw,
        radius_cells=int(radius_cells),
    ) if int(actor["time"].numel()) else torch.empty_like(res_cost)
    projected_actor = source_actor.float().cpu() @ projection
    projected_ego = source_ego.float().cpu() @ projection
    actor_res_x = torch.cat([projected_actor, res_cost.float().cpu(), actor["res_aux"]], dim=1)
    actor_abs_x = torch.cat([projected_actor, abs_cost.float().cpu(), actor["abs_aux"]], dim=1)
    ego_x = torch.cat([projected_ego, ego["aux"]], dim=1)
    return {
        "actor_res_x": actor_res_x,
        "actor_abs_x": actor_abs_x,
        "actor_res_target": actor["residual_target"].clone(),
        "actor_abs_target": actor["absolute_target"].clone(),
        "actor_labels": list(actor["labels"]),
        "actor_categories": list(actor["categories"]),
        "actor_instance_ids": list(actor["instance_ids"]),
        "actor_query_ids": list(actor["query_ids"]),
        "ego_x": ego_x,
        "ego_target": ego["target"].clone(),
        "ego_query_ids": list(ego["query_ids"]),
        "actor_learned_width": int(projected_actor.shape[1] + res_cost.shape[1]),
        "ego_learned_width": int(projected_ego.shape[1]),
        "image_hw": image_hw,
    }


def _append_bundle(storage: dict[str, Any], bundle: Mapping[str, Any]) -> None:
    tensor_keys = (
        "actor_res_x", "actor_abs_x", "actor_res_target", "actor_abs_target", "ego_x", "ego_target",
    )
    list_keys = (
        "actor_labels", "actor_categories", "actor_instance_ids", "actor_query_ids", "ego_query_ids",
    )
    for key in tensor_keys:
        storage.setdefault(key, []).append(torch.as_tensor(bundle[key]).cpu())
    for key in list_keys:
        storage.setdefault(key, []).extend(list(bundle[key]))
    for key in ("actor_learned_width", "ego_learned_width", "image_hw"):
        if key in storage and storage[key] != bundle[key]:
            raise MotionFeatureProbeError(f"bundle {key} 跨 clip 不一致")
        storage[key] = bundle[key]


def _finalize_bundle(storage: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(storage)
    for key in ("actor_res_x", "actor_abs_x", "actor_res_target", "actor_abs_target", "ego_x", "ego_target"):
        rows = list(storage.get(key, []))
        if not rows:
            raise MotionFeatureProbeError(f"feature bundle 缺少 {key}")
        output[key] = torch.cat(rows, dim=0)
    return output


def _pixel_epe(prediction: torch.Tensor, target: torch.Tensor, image_hw: tuple[int, int]) -> float:
    height, width = image_hw
    scale = torch.tensor([width, height], dtype=torch.float32)
    return vector_epe(prediction * scale, target * scale)


def _category_mean_prediction(
    train_target: torch.Tensor,
    train_categories: Sequence[str],
    dev_categories: Sequence[str],
) -> torch.Tensor:
    global_mean = train_target.mean(dim=0)
    means = {}
    for category in sorted(set(train_categories)):
        mask = torch.tensor([value == category for value in train_categories], dtype=torch.bool)
        means[category] = train_target[mask].mean(dim=0)
    return torch.stack([means.get(category, global_mean) for category in dev_categories])


def evaluate_probe_config(
    train: Mapping[str, Any],
    dev: Mapping[str, Any],
    *,
    ridge_regularization: float,
    control_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    res_model = fit_ridge(train["actor_res_x"], train["actor_res_target"], regularization=ridge_regularization)
    abs_model = fit_ridge(train["actor_abs_x"], train["actor_abs_target"], regularization=ridge_regularization)
    ego_model = fit_ridge(train["ego_x"], train["ego_target"], regularization=ridge_regularization)
    res_prediction = res_model.predict(dev["actor_res_x"])
    abs_prediction = abs_model.predict(dev["actor_abs_x"])
    ego_prediction = ego_model.predict(dev["ego_x"])
    moving = torch.tensor([label == "moving" for label in dev["actor_labels"]], dtype=torch.bool)
    stationary = torch.tensor([label == "stationary" for label in dev["actor_labels"]], dtype=torch.bool)
    if not bool(moving.any() and stationary.any()):
        raise MotionFeatureProbeError("dev actor moving/stationary support 不足")
    image_hw = tuple(int(value) for value in dev["image_hw"])
    res_error = _pixel_epe(res_prediction[moving], dev["actor_res_target"][moving], image_hw)
    zero_error = _pixel_epe(torch.zeros_like(dev["actor_res_target"][moving]), dev["actor_res_target"][moving], image_hw)
    category_prediction = _category_mean_prediction(
        train["actor_res_target"], train["actor_categories"], dev["actor_categories"],
    )
    category_error = _pixel_epe(category_prediction[moving], dev["actor_res_target"][moving], image_hw)
    abs_error = _pixel_epe(abs_prediction[moving], dev["actor_abs_target"][moving], image_hw)
    ego_error = _pixel_epe(ego_prediction, dev["ego_target"], image_hw)
    ego_zero = _pixel_epe(torch.zeros_like(dev["ego_target"]), dev["ego_target"], image_hw)
    ego_mean_prediction = train["ego_target"].mean(dim=0, keepdim=True).expand_as(dev["ego_target"])
    ego_mean = _pixel_epe(ego_mean_prediction, dev["ego_target"], image_hw)

    time_actor_x = permute_learned_features(
        dev["actor_res_x"], learned_width=int(dev["actor_learned_width"]), seed=int(control_seed),
    )
    time_ego_x = permute_learned_features(
        dev["ego_x"], learned_width=int(dev["ego_learned_width"]), seed=int(control_seed) + 1,
    )
    time_actor_error = _pixel_epe(res_model.predict(time_actor_x)[moving], dev["actor_res_target"][moving], image_hw)
    time_ego_error = _pixel_epe(ego_model.predict(time_ego_x), dev["ego_target"], image_hw)
    shuffled_target_model = fit_ridge(
        train["actor_res_x"],
        permute_instance_targets(
            train["actor_res_target"], train["actor_instance_ids"], seed=int(control_seed) + 2,
        ),
        regularization=ridge_regularization,
    )
    shuffled_target_error = _pixel_epe(
        shuffled_target_model.predict(dev["actor_res_x"])[moving],
        dev["actor_res_target"][moving],
        image_hw,
    )
    scale = torch.tensor([image_hw[1], image_hw[0]], dtype=torch.float32)
    stationary_prediction_magnitude = torch.linalg.vector_norm(res_prediction[stationary] * scale, dim=-1)
    moving_target_magnitude = torch.linalg.vector_norm(dev["actor_res_target"][moving] * scale, dim=-1)
    stationary_median = float(stationary_prediction_magnitude.median())
    moving_target_median = float(moving_target_magnitude.median())
    metrics = {
        "valid": True,
        "train_actor_count": int(train["actor_res_x"].shape[0]),
        "dev_actor_count": int(dev["actor_res_x"].shape[0]),
        "dev_moving_count": int(moving.sum()),
        "dev_stationary_count": int(stationary.sum()),
        "train_ego_count": int(train["ego_x"].shape[0]),
        "dev_ego_count": int(dev["ego_x"].shape[0]),
        "actor_res_epe_moving_px": res_error,
        "actor_zero_epe_moving_px": zero_error,
        "actor_category_mean_epe_moving_px": category_error,
        "actor_abs_epe_moving_px": abs_error,
        "actor_res_vs_zero_improvement": relative_improvement(res_error, zero_error),
        "actor_res_vs_category_mean_improvement": relative_improvement(res_error, category_error),
        "actor_res_vs_abs_improvement": relative_improvement(res_error, abs_error),
        "ego_epe_px": ego_error,
        "ego_zero_epe_px": ego_zero,
        "ego_mean_epe_px": ego_mean,
        "ego_vs_best_baseline_improvement": relative_improvement(ego_error, min(ego_zero, ego_mean)),
        "ego_angular_error_deg": angular_error_deg(ego_prediction, dev["ego_target"]),
        "actor_time_shuffled_epe_px": time_actor_error,
        "actor_time_shuffle_degradation": time_actor_error / max(res_error, 1.0e-8) - 1.0,
        "ego_time_shuffled_epe_px": time_ego_error,
        "ego_time_shuffle_degradation": time_ego_error / max(ego_error, 1.0e-8) - 1.0,
        "actor_target_shuffled_epe_px": shuffled_target_error,
        "actor_target_shuffle_degradation": shuffled_target_error / max(res_error, 1.0e-8) - 1.0,
        "stationary_prediction_median_px": stationary_median,
        "moving_target_median_px": moving_target_median,
        "stationary_prediction_to_moving_target_ratio": stationary_median / max(moving_target_median, 1.0e-8),
        "actor_res_head_parameter_count": int((train["actor_res_x"].shape[1] + 1) * 2),
        "actor_abs_head_parameter_count": int((train["actor_abs_x"].shape[1] + 1) * 2),
        "same_actor_head_capacity": bool(train["actor_res_x"].shape[1] == train["actor_abs_x"].shape[1]),
    }
    models = {
        "actor_res": res_model,
        "actor_abs": abs_model,
        "ego": ego_model,
        "moving_mask": moving,
    }
    return metrics, models


def primary_scan_checks(metrics: Mapping[str, Any], thresholds: Mapping[str, Any]) -> dict[str, bool]:
    def number(name: str) -> float | None:
        value = metrics.get(name)
        if value is None:
            return None
        result = float(value)
        return result if math.isfinite(result) else None

    ego = number("ego_vs_best_baseline_improvement")
    actor = number("actor_res_vs_zero_improvement")
    residual = number("actor_res_vs_abs_improvement")
    time_actor = number("actor_time_shuffle_degradation")
    time_ego = number("ego_time_shuffle_degradation")
    target_actor = number("actor_target_shuffle_degradation")
    stationary = number("stationary_prediction_to_moving_target_ratio")

    return {
        "sample_support": (
            int(metrics.get("train_actor_count", 0)) >= int(thresholds["minimum_train_actor_queries"])
            and int(metrics.get("dev_actor_count", 0)) >= int(thresholds["minimum_dev_actor_queries"])
            and int(metrics.get("train_ego_count", 0)) >= int(thresholds["minimum_train_ego_queries"])
            and int(metrics.get("dev_ego_count", 0)) >= int(thresholds["minimum_dev_ego_queries"])
        ),
        "ego_signal": ego is not None and ego >= float(thresholds["scan_minimum_ego_improvement"]),
        "actor_signal": actor is not None and actor >= float(thresholds["scan_minimum_actor_improvement"]),
        "residual_parameterization": residual is not None and residual >= float(thresholds["scan_minimum_res_vs_abs_improvement"]),
        "time_shuffle_actor": time_actor is not None and time_actor >= float(thresholds["minimum_control_degradation"]),
        "time_shuffle_ego": time_ego is not None and time_ego >= float(thresholds["minimum_ego_control_degradation"]),
        "target_shuffle_actor": target_actor is not None and target_actor >= float(thresholds["minimum_control_degradation"]),
        "stationary_safeguard": stationary is not None and stationary <= float(thresholds["maximum_stationary_to_moving_ratio"]),
        "matched_actor_capacity": bool(metrics.get("same_actor_head_capacity")),
    }


def rank_primary_configs(
    rows: Sequence[Mapping[str, Any]], thresholds: Mapping[str, Any], *, top_k: int,
) -> dict[str, Any]:
    annotated = []
    for row in rows:
        checks = primary_scan_checks(row, thresholds)
        score_values = [
            row.get("ego_vs_best_baseline_improvement"),
            row.get("actor_res_vs_zero_improvement"),
            row.get("actor_res_vs_abs_improvement"),
            row.get("actor_time_shuffle_degradation"),
            row.get("actor_target_shuffle_degradation"),
        ]
        finite = [float(value) for value in score_values if value is not None and math.isfinite(float(value))]
        annotated.append({**dict(row), "primary_checks": checks, "primary_eligible": all(checks.values()), "primary_score": min(finite) if finite else float("-inf")})
    eligible_by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in annotated:
        if bool(row["primary_eligible"]):
            eligible_by_layer[str(row["layer"])].append(row)
    stable_layers = sorted(
        layer for layer, values in eligible_by_layer.items()
        if len({float(row["sigma"]) for row in values}) >= int(thresholds["minimum_stable_sigmas_per_layer"])
    )
    stable_candidates = [row for row in annotated if row["primary_eligible"] and row["layer"] in stable_layers]
    stable_candidates.sort(
        key=lambda row: (-float(row["primary_score"]), str(row["layer"]), float(row["sigma"]))
    )
    selected = [str(row["config_id"]) for row in stable_candidates[: int(top_k)]]
    return {
        "rows": annotated,
        "stable_layers": stable_layers,
        "primary_selected_configs": selected,
        "primary_candidate_count": len(stable_candidates),
    }


def _control_metrics(
    normal: Mapping[str, Any],
    control: Mapping[str, Any],
    models: Mapping[str, Any],
) -> dict[str, Any]:
    if normal["actor_query_ids"] != control["actor_query_ids"] or normal["ego_query_ids"] != control["ego_query_ids"]:
        raise MotionFeatureProbeError("control query order 与 normal 不一致")
    if not torch.equal(normal["actor_res_target"], control["actor_res_target"]):
        raise MotionFeatureProbeError("control actor target 被改写")
    if not torch.equal(normal["ego_target"], control["ego_target"]):
        raise MotionFeatureProbeError("control ego target 被改写")
    moving = models["moving_mask"]
    image_hw = tuple(int(value) for value in normal["image_hw"])
    actor_error = _pixel_epe(
        models["actor_res"].predict(control["actor_res_x"])[moving],
        normal["actor_res_target"][moving],
        image_hw,
    )
    ego_error = _pixel_epe(
        models["ego"].predict(control["ego_x"]), normal["ego_target"], image_hw,
    )
    return {"actor_epe_px": actor_error, "ego_epe_px": ego_error}


def _model_files_ready(pretrained: str) -> dict[str, Any]:
    root = Path(str(pretrained))
    required = [
        root / "model_index.json", root / "unet" / "config.json", root / "vae" / "config.json",
        root / "image_encoder" / "config.json", root / "scheduler" / "scheduler_config.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    return {
        "ready": not missing,
        "missing": missing,
        "model_index_sha256": file_fingerprint(str(root / "model_index.json")) if not missing else None,
    }


def preflight_motion_feature_probe(cfg: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "task_id": str(cfg.a1.task_id),
        "status": "ready",
        "uses_gpu": True,
        "trains_diffusion_backbone": False,
        "target_scope": REAL_TARGET_SCOPE,
        "blockers": [],
    }
    parent = Path(str(cfg.a1.parent_a0_run))
    try:
        parent_summary = json.loads((parent / "summary.json").read_text(encoding="utf-8"))
        if not (parent / "COMPLETE").is_file() or not bool(parent_summary.get("machine_pass")):
            raise MotionFeatureProbeError("A0 parent 未 machine pass/COMPLETE")
        result["parent_a0"] = {
            "run": str(parent),
            "machine_pass": True,
            "result_fingerprint": parent_summary.get("result_fingerprint"),
        }
    except Exception as exc:
        result["blockers"].append({"kind": "a0_parent", "error": repr(exc)})
    try:
        dataset = NuScenesFutureVideoDataset(_copy_data_config(cfg.data))
        split = stable_scene_split(
            dataset.clip_records,
            train_count=int(cfg.a1.train_clip_count),
            dev_count=int(cfg.a1.dev_clip_count),
            holdout_count=0,
        )
        result["data"] = {
            "dataset_clip_count": len(dataset),
            "train_count": len(split["train"]),
            "dev_count": len(split["dev"]),
            "split_fingerprint": split_fingerprint(split),
            "train_sample_ids": [str(row["sample_id"]) for row in split["train"]],
            "dev_sample_ids": [str(row["sample_id"]) for row in split["dev"]],
        }
    except Exception as exc:
        result["blockers"].append({"kind": "nuscenes", "error": repr(exc)})
    model = _model_files_ready(str(cfg.model.pretrained))
    result["model"] = model
    if not model["ready"]:
        result["blockers"].append({"kind": "svd", "missing": model["missing"]})
    usage = shutil.disk_usage(Path(str(cfg.work_dir)).parent)
    free_gb = float(usage.free) / float(1024**3)
    result["disk"] = {"free_gb": free_gb, "minimum_free_gb": float(cfg.a1.minimum_free_disk_gb)}
    if free_gb < float(cfg.a1.minimum_free_disk_gb):
        result["blockers"].append({"kind": "disk", "free_gb": free_gb})
    if result["blockers"]:
        result["status"] = "blocked"
    return result


def _validate_protocol(cfg: Any) -> None:
    if str(cfg.a1.task_id) != "RP-A1-SCAN-04A" or str(cfg.a1.mode) != "scan":
        raise MotionFeatureProbeError("当前模块只允许 RP-A1-SCAN-04A / scan")
    if int(cfg.a1.train_clip_count) != 24 or int(cfg.a1.dev_clip_count) != 8:
        raise MotionFeatureProbeError("A1-SCAN 必须为 24 train / 8 dev clips")
    if [float(value) for value in cfg.a1.sigmas] != [0.05, 0.2, 1.0]:
        raise MotionFeatureProbeError("A1-SCAN sigmas 必须为 [0.05,0.2,1.0]")
    if len(dict(cfg.a1.feature_layers)) != 7:
        raise MotionFeatureProbeError("A1-SCAN 必须使用 7 个冻结 layer hooks")
    if int(cfg.data.num_frames) != 8 or int(cfg.model.num_frames) != 8:
        raise MotionFeatureProbeError("A1-SCAN 必须使用 8 frames")
    if bool(cfg.model.lora.enable) or int(cfg.model.generation.fps) != 7:
        raise MotionFeatureProbeError("A1-SCAN 必须冻结 Base 且继承 R1 fps=7")
    if int(cfg.a1.local_cost_radius_cells) != 2:
        raise MotionFeatureProbeError("A1-SCAN local cost 必须为 5x5")


def _forward_capture(
    backbone: SVDBackbone,
    capture: _FeatureCapture,
    latent: torch.Tensor,
    noise: torch.Tensor,
    sigma: float,
    condition: Conditioning,
) -> dict[str, torch.Tensor]:
    z = latent + float(sigma) * noise
    sigma_tensor = torch.tensor([float(sigma)], device=latent.device, dtype=torch.float32)
    capture.clear()
    with torch.no_grad():
        backbone.predict_model_output(z, sigma_tensor, condition)
    outputs = dict(capture.outputs)
    if not outputs:
        raise MotionFeatureProbeError("feature hooks 未捕获任何输出")
    for alias, tensor in outputs.items():
        if tensor.ndim != 4 or tensor.shape[0] != latent.shape[1]:
            raise MotionFeatureProbeError(f"{alias} feature shape 非 [T,C,H,W]: {tuple(tensor.shape)}")
    return outputs


def run_motion_feature_probe(cfg: Any) -> dict[str, Any]:
    _validate_protocol(cfg)
    git = git_state(".")
    if git.get("dirty"):
        raise MotionFeatureProbeError("正式 A1-SCAN 拒绝 dirty worktree")
    work_dir = Path(str(cfg.work_dir))
    if work_dir.exists():
        raise FileExistsError(f"A1-SCAN run directory 已存在: {work_dir}")
    preflight = preflight_motion_feature_probe(cfg)
    if preflight["status"] != "ready":
        raise MotionFeatureProbeError(f"A1 preflight blocked: {preflight['blockers']}")
    config_fp = config_fingerprint(cfg)
    work_dir.mkdir(parents=True, exist_ok=False)
    feature_dir = work_dir / "feature_records"
    control_dir = work_dir / "control_records"
    feature_dir.mkdir()
    control_dir.mkdir()
    layer_paths = {str(key): str(value) for key, value in dict(cfg.a1.feature_layers).items()}
    sigmas = [float(value) for value in cfg.a1.sigmas]
    manifest = RunManifest(
        run_id=str(cfg.run_id), command=list(sys.argv), config_fingerprint=config_fp,
        cache_fingerprint=str(preflight["data"]["split_fingerprint"]), seed=int(cfg.seed),
        git=git, environment=environment_fingerprint(), data_split="nuScenes train 24/8 scene-disjoint scan",
    )
    target_builder_files = [
        Path("motion_proj/data/real_motion_targets.py"), Path("motion_proj/data/motion_feature_records.py"),
    ]
    manifest_data = manifest.__dict__ | {
        "task_id": str(cfg.a1.task_id),
        "preflight": preflight,
        "parent_a0_run": str(cfg.a1.parent_a0_run),
        "trains_diffusion_backbone": False,
        "uses_future_gt_for_real_representation_probe": True,
        "uses_future_gt_for_generated_evaluation": False,
        "feature_layers": layer_paths,
        "sigmas": sigmas,
        "base_model_fingerprint": _base_model_fingerprint(str(cfg.model.pretrained)),
        "target_builder_fingerprint": sha256_json([
            (str(path), file_fingerprint(str(path))) for path in target_builder_files
        ]),
        "preregistered_thresholds": _json_safe(dict(cfg.a1.thresholds)),
    }
    atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
    save_resolved_config(cfg, str(work_dir / "resolved.yaml"))
    metrics_log = JsonlMetrics(str(work_dir / "metrics.jsonl"))

    try:
        seed_everything(int(cfg.seed), deterministic=bool(cfg.train.deterministic))
        dataset = NuScenesFutureVideoDataset(_copy_data_config(cfg.data))
        split = stable_scene_split(
            dataset.clip_records,
            train_count=int(cfg.a1.train_clip_count),
            dev_count=int(cfg.a1.dev_clip_count),
            holdout_count=0,
        )
        _write_jsonl(
            work_dir / "scene_split.jsonl",
            [
                {"split": name, **row}
                for name in ("train", "dev")
                for row in split[name]
            ],
        )
        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        model_cfg.lora.enable = False
        model_cfg.gradient_checkpointing = False
        model_cfg.enable_xformers = False
        backbone = build_backbone(model_cfg, load=True, device=str(cfg.device))
        if not isinstance(backbone, SVDBackbone):
            raise MotionFeatureProbeError("A1 当前只支持 SVDBackbone")
        backbone.unet.eval().requires_grad_(False)
        backbone.vae.eval().requires_grad_(False)
        backbone.image_encoder.eval().requires_grad_(False)
        capture = _FeatureCapture(backbone.unet, layer_paths)
        projection_by_layer: dict[str, torch.Tensor] = {}
        record_storage: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
        query_rows: list[dict[str, Any]] = []
        condition_rows: list[dict[str, Any]] = []
        query_cache: dict[tuple[str, str], dict[str, Any]] = {}
        latent_cache: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor, Conditioning]] = {}

        for split_name in ("train", "dev"):
            for clip_position, record in enumerate(split[split_name]):
                sample = dataset[int(record["dataset_index"])]
                sample_id = str(record["sample_id"])
                scene_name = str(record["scene_name"])
                queries = prepare_motion_queries(
                    sample,
                    sample_id=sample_id,
                    scene_name=scene_name,
                    split_name=split_name,
                    maximum_actor_queries=int(cfg.a1.maximum_actor_queries_per_clip),
                    maximum_ego_queries=int(cfg.a1.maximum_ego_queries_per_clip),
                    box_dilation_px=int(cfg.a1.box_dilation_px),
                )
                if not int(queries["ego"]["time"].numel()):
                    raise MotionFeatureProbeError(f"A1 clip ego query 为空: {sample_id}")
                query_cache[(split_name, sample_id)] = queries
                query_rows.extend(queries["query_rows"])
                frames = torch.as_tensor(sample["frames"]).unsqueeze(0)
                latent = backbone.encode(frames.to(backbone.device)).detach()
                noise_generator = torch.Generator(device="cpu").manual_seed(
                    int(cfg.a1.noise_seed) + int(record["dataset_index"])
                )
                noise = torch.randn(latent.shape, generator=noise_generator, dtype=torch.float32).to(
                    backbone.device, latent.dtype,
                )
                condition, condition_meta = _official_conditioning(
                    backbone,
                    torch.as_tensor(sample["cond_frame"]),
                    seed=int(cfg.a1.condition_seed) + int(record["dataset_index"]),
                    height=int(cfg.data.height),
                    width=int(cfg.data.width),
                )
                if split_name == "dev":
                    latent_cache[(split_name, sample_id)] = (latent, noise, condition)
                condition_rows.append(
                    {
                        "split": split_name,
                        "sample_id": sample_id,
                        "scene_name": scene_name,
                        "latent_sha256": _tensor_fingerprint(latent),
                        "noise_sha256": _tensor_fingerprint(noise),
                        **condition_meta,
                    }
                )
                for sigma in sigmas:
                    outputs = _forward_capture(backbone, capture, latent, noise, sigma, condition)
                    if set(outputs) != set(layer_paths):
                        raise MotionFeatureProbeError(
                            f"feature hooks 不完整: expected={sorted(layer_paths)}, actual={sorted(outputs)}"
                        )
                    for alias, features in outputs.items():
                        if alias not in projection_by_layer:
                            projection_by_layer[alias] = random_projection_matrix(
                                int(features.shape[1]),
                                int(cfg.a1.projection_dim),
                                seed=_layer_projection_seed(int(cfg.a1.projection_seed), alias),
                            )
                        projection = projection_by_layer[alias]
                        if projection.shape[0] != features.shape[1]:
                            raise MotionFeatureProbeError(f"{alias} channel 数变化")
                        bundle = extract_feature_bundle(
                            features,
                            queries,
                            projection,
                            radius_cells=int(cfg.a1.local_cost_radius_cells),
                        )
                        config_id = f"{alias}-sigma{sigma:g}"
                        _append_bundle(record_storage[config_id][split_name], bundle)
                    metrics_log.append(
                        len(condition_rows),
                        {
                            "event": "feature_extracted",
                            "split": split_name,
                            "sample_id": sample_id,
                            "sigma": sigma,
                            "actor_queries": int(queries["actor"]["time"].numel()),
                            "ego_queries": int(queries["ego"]["time"].numel()),
                        },
                    )
                    del outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        _write_jsonl(work_dir / "queries.jsonl", query_rows)
        _write_jsonl(work_dir / "conditioning.jsonl", condition_rows)

        finalized: dict[str, dict[str, Any]] = {}
        feature_index_rows = []
        for alias in layer_paths:
            for sigma in sigmas:
                config_id = f"{alias}-sigma{sigma:g}"
                train_bundle = _finalize_bundle(record_storage[config_id]["train"])
                dev_bundle = _finalize_bundle(record_storage[config_id]["dev"])
                finalized[config_id] = {"train": train_bundle, "dev": dev_bundle}
                path = feature_dir / f"{config_id}.pt"
                payload = {
                    "config_id": config_id,
                    "layer": alias,
                    "module_path": layer_paths[alias],
                    "sigma": sigma,
                    "projection": projection_by_layer[alias],
                    "projection_fingerprint": projection_fingerprint(projection_by_layer[alias]),
                    "train": train_bundle,
                    "dev": dev_bundle,
                    "contains_full_feature_maps": False,
                }
                torch.save(payload, path)
                feature_index_rows.append(
                    {
                        "config_id": config_id,
                        "path": str(path.relative_to(work_dir)),
                        "sha256": file_fingerprint(str(path)),
                        "projection_fingerprint": payload["projection_fingerprint"],
                        "train_actor_count": int(train_bundle["actor_res_x"].shape[0]),
                        "dev_actor_count": int(dev_bundle["actor_res_x"].shape[0]),
                        "train_ego_count": int(train_bundle["ego_x"].shape[0]),
                        "dev_ego_count": int(dev_bundle["ego_x"].shape[0]),
                    }
                )
        _write_jsonl(work_dir / "feature_record_index.jsonl", feature_index_rows)

        primary_rows = []
        models_by_config: dict[str, dict[str, Any]] = {}
        for alias in layer_paths:
            for sigma in sigmas:
                config_id = f"{alias}-sigma{sigma:g}"
                probe, models = evaluate_probe_config(
                    finalized[config_id]["train"],
                    finalized[config_id]["dev"],
                    ridge_regularization=float(cfg.a1.ridge_regularization),
                    control_seed=int(cfg.a1.control_seed),
                )
                row = {
                    "config_id": config_id,
                    "layer": alias,
                    "module_path": layer_paths[alias],
                    "sigma": sigma,
                    **probe,
                }
                primary_rows.append(row)
                models_by_config[config_id] = models
                metrics_log.append(len(primary_rows), {"event": "primary_probe", **row})
        ranking = rank_primary_configs(
            primary_rows,
            dict(cfg.a1.thresholds),
            top_k=int(cfg.a1.top_k),
        )
        _write_jsonl(work_dir / "primary_probe_metrics.jsonl", ranking["rows"])

        selected_ids = list(ranking["primary_selected_configs"])
        control_storage: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
        selected_by_sigma: dict[float, set[str]] = defaultdict(set)
        for config_id in selected_ids:
            row = next(item for item in primary_rows if item["config_id"] == config_id)
            selected_by_sigma[float(row["sigma"])].add(str(row["layer"]))
        if selected_ids:
            for record in split["dev"]:
                sample_id = str(record["sample_id"])
                queries = query_cache[("dev", sample_id)]
                latent, noise, condition = latent_cache[("dev", sample_id)]
                variants = {
                    "single_frame": (
                        latent[:, :1].repeat(1, latent.shape[1], 1, 1, 1),
                        noise[:, :1].repeat(1, noise.shape[1], 1, 1, 1),
                    ),
                    "future_reversed": (
                        torch.cat([latent[:, :1], latent[:, 1:].flip(1)], dim=1),
                        torch.cat([noise[:, :1], noise[:, 1:].flip(1)], dim=1),
                    ),
                }
                for sigma, aliases in selected_by_sigma.items():
                    for variant, (variant_latent, variant_noise) in variants.items():
                        outputs = _forward_capture(
                            backbone, capture, variant_latent, variant_noise, sigma, condition,
                        )
                        for alias in aliases:
                            config_id = f"{alias}-sigma{sigma:g}"
                            bundle = extract_feature_bundle(
                                outputs[alias], queries, projection_by_layer[alias],
                                radius_cells=int(cfg.a1.local_cost_radius_cells),
                            )
                            _append_bundle(control_storage[config_id][variant], bundle)
                        del outputs
            control_rows = []
            final_selected = []
            thresholds = dict(cfg.a1.thresholds)
            for config_id in selected_ids:
                normal = finalized[config_id]["dev"]
                single = _finalize_bundle(control_storage[config_id]["single_frame"])
                reversed_bundle = _finalize_bundle(control_storage[config_id]["future_reversed"])
                single_metrics = _control_metrics(normal, single, models_by_config[config_id])
                reversed_metrics = _control_metrics(normal, reversed_bundle, models_by_config[config_id])
                primary = next(row for row in primary_rows if row["config_id"] == config_id)
                row = {
                    "config_id": config_id,
                    "single_frame": single_metrics,
                    "future_reversed": reversed_metrics,
                    "actor_single_frame_degradation": single_metrics["actor_epe_px"] / max(float(primary["actor_res_epe_moving_px"]), 1.0e-8) - 1.0,
                    "actor_future_reversed_degradation": reversed_metrics["actor_epe_px"] / max(float(primary["actor_res_epe_moving_px"]), 1.0e-8) - 1.0,
                    "ego_single_frame_degradation": single_metrics["ego_epe_px"] / max(float(primary["ego_epe_px"]), 1.0e-8) - 1.0,
                    "ego_future_reversed_degradation": reversed_metrics["ego_epe_px"] / max(float(primary["ego_epe_px"]), 1.0e-8) - 1.0,
                }
                checks = {
                    "actor_single_frame": row["actor_single_frame_degradation"] >= float(thresholds["minimum_control_degradation"]),
                    "actor_future_reversed": row["actor_future_reversed_degradation"] >= float(thresholds["minimum_control_degradation"]),
                    "ego_single_frame": row["ego_single_frame_degradation"] >= float(thresholds["minimum_ego_control_degradation"]),
                    "ego_future_reversed": row["ego_future_reversed_degradation"] >= float(thresholds["minimum_ego_control_degradation"]),
                }
                row["checks"] = checks
                row["passed"] = all(checks.values())
                if row["passed"]:
                    final_selected.append(config_id)
                control_rows.append(row)
                torch.save(
                    {"normal_query_ids": normal["actor_query_ids"], "single_frame": single, "future_reversed": reversed_bundle},
                    control_dir / f"{config_id}.pt",
                )
            _write_jsonl(work_dir / "control_metrics.jsonl", control_rows)
        else:
            control_rows = []
            final_selected = []
            _write_jsonl(work_dir / "control_metrics.jsonl", [])
        capture.close()

        scan_pass = len(final_selected) == int(cfg.a1.top_k)
        reason = (
            "two_stable_configs_pass_primary_and_temporal_controls"
            if scan_pass
            else "insufficient_stable_configs_or_temporal_control_failure"
        )
        decision = {
            "status": "scan_pass" if scan_pass else "rejected",
            "scan_pass": scan_pass,
            "reason": reason,
            "stable_layers": ranking["stable_layers"],
            "primary_candidate_count": ranking["primary_candidate_count"],
            "primary_selected_configs": selected_ids,
            "final_selected_configs": final_selected,
            "top_k_required": int(cfg.a1.top_k),
            "next_gate": "RP-A1-CONFIRM-04B" if scan_pass else "RP-B0-05",
        }
        result = {
            "task_id": str(cfg.a1.task_id),
            "split_fingerprint": str(preflight["data"]["split_fingerprint"]),
            "query_count": len(query_rows),
            "feature_record_count": len(feature_index_rows),
            "contains_full_feature_maps": False,
            "primary_metrics": ranking["rows"],
            "controls": control_rows,
            "decision": decision,
            "uses_future_gt_for_generated_evaluation": False,
        }
        atomic_write_json(str(work_dir / "result.json"), _json_safe(result))
        summary = {
            "status": "done" if scan_pass else "rejected",
            "task_id": str(cfg.a1.task_id),
            "run_id": str(cfg.run_id),
            "config_fingerprint": config_fp,
            "split_fingerprint": str(preflight["data"]["split_fingerprint"]),
            "scan_pass": scan_pass,
            "reason": reason,
            "stable_layers": ranking["stable_layers"],
            "primary_candidate_count": ranking["primary_candidate_count"],
            "selected_configs": final_selected,
            "query_count": len(query_rows),
            "feature_record_count": len(feature_index_rows),
            "result_fingerprint": sha256_json(_json_safe(result)),
            "next_gate": decision["next_gate"],
        }
        atomic_write_json(str(work_dir / "summary.json"), summary)
        atomic_write_text(str(work_dir / "COMPLETE"), sha256_json(summary) + "\n")
        manifest_data.update(
            {"status": "completed", "ended_at": utc_now(), "exit_reason": reason, "selected_configs": final_selected}
        )
        atomic_write_json(str(work_dir / "manifest.json"), _json_safe(manifest_data))
        return summary
    except Exception as exc:
        failure = {
            "status": "failed", "task_id": str(cfg.a1.task_id), "run_id": str(cfg.run_id), "error": repr(exc),
        }
        atomic_write_json(str(work_dir / "summary.json"), failure)
        manifest_data.update({"status": "failed", "ended_at": utc_now(), "exit_reason": repr(exc)})
        atomic_write_json(str(work_dir / "manifest.json"), _json_safe(manifest_data))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="scan frozen SVD ego/actor motion features")
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, list(args.overrides))
    result = preflight_motion_feature_probe(cfg) if args.preflight else run_motion_feature_probe(cfg)
    print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
