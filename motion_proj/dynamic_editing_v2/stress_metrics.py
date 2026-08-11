"""M5 压力测试使用的冻结轨迹调度与指标。"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch


FAILURE_PRIORITY = (
    "BASELINE_RUNTIME_FAILURE",
    "ACTOR_IDENTITY_MISMATCH",
    "ACTOR_GEOMETRY_DEFORMATION",
    "TRAJECTORY_NOT_FOLLOWED",
    "SOURCE_RESIDUAL",
    "SHADOW_RESIDUAL",
    "BACKGROUND_HOLE",
    "DEPTH_ORDERING_ERROR",
    "NEW_OCCLUSION_ERROR",
    "MULTICAMERA_INCONSISTENCY",
    "TEMPORAL_FLICKER",
    "NON_TARGET_VISUAL_DRIFT",
    "NON_TARGET_PERCEPTION_DRIFT",
    "UNSUPPORTED_DISOCCLUSION",
    "INSUFFICIENT_EVIDENCE",
)


def smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def time_warp_ordinals(
    valid_count: int, edit: str, *, fps: int = 10
) -> np.ndarray:
    """把目标有效帧序号映射到源轨迹序号。"""
    if valid_count <= 0:
        return np.empty((0,), dtype=np.float64)
    destination = np.arange(valid_count, dtype=np.float64)
    if edit == "speed":
        return 0.75 * destination
    if edit != "stop_restart":
        return destination
    middle = valid_count // 2
    duration = min(fps, max(1, valid_count - middle - 1))
    restart = middle + duration
    mapping = destination.copy()
    stopped = (destination >= middle) & (destination < restart)
    mapping[stopped] = float(middle)
    after = destination >= restart
    if after.any() and valid_count - 1 > middle:
        denominator = max(1.0, float(valid_count - 1 - restart))
        progress = (destination[after] - restart) / denominator
        mapping[after] = middle + (valid_count - 1 - middle) * smoothstep(progress)
    return np.clip(mapping, 0.0, valid_count - 1.0)


def interpolate_trajectory(
    translations: torch.Tensor,
    quaternions_wxyz: torch.Tensor,
    source_ordinals: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """线性插值平移，并按最短路径插值后归一化四元数。"""
    if translations.shape[0] != quaternions_wxyz.shape[0]:
        raise ValueError("translation/quaternion trajectory lengths differ")
    source = torch.as_tensor(
        source_ordinals, dtype=translations.dtype, device=translations.device
    )
    lower = torch.floor(source).long()
    upper = torch.ceil(source).long().clamp_max(translations.shape[0] - 1)
    alpha = (source - lower).unsqueeze(-1)
    output_translations = translations[lower] * (1 - alpha) + translations[upper] * alpha
    first = quaternions_wxyz[lower]
    second = quaternions_wxyz[upper]
    sign = torch.where(
        (first * second).sum(dim=-1, keepdim=True) < 0,
        -torch.ones_like(alpha),
        torch.ones_like(alpha),
    )
    second = second * sign
    output_quaternions = first * (1 - alpha) + second * alpha
    output_quaternions = output_quaternions / output_quaternions.norm(
        dim=-1, keepdim=True
    ).clamp_min(1e-12)
    return output_translations, output_quaternions


def apply_time_warp(
    rigid, model_index: int, edit: str, *, fps: int = 10
) -> dict:
    valid = rigid.instances_fv[:, model_index].bool()
    frame_indices = torch.nonzero(valid, as_tuple=False).flatten()
    mapping = time_warp_ordinals(len(frame_indices), edit, fps=fps)
    if len(frame_indices) == 0:
        return {"valid_frames": [], "source_ordinals": []}
    translations = rigid.instances_trans[frame_indices, model_index].detach().clone()
    quaternions = rigid.instances_quats[frame_indices, model_index].detach().clone()
    edited_translations, edited_quaternions = interpolate_trajectory(
        translations, quaternions, mapping
    )
    with torch.no_grad():
        rigid.instances_trans[frame_indices, model_index] = edited_translations
        rigid.instances_quats[frame_indices, model_index] = edited_quaternions
    return {
        "valid_frames": frame_indices.detach().cpu().tolist(),
        "source_ordinals": mapping.tolist(),
        "expected_translations": edited_translations.detach().cpu().tolist(),
        "expected_quaternions_wxyz": edited_quaternions.detach().cpu().tolist(),
    }


def masked_ssim(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    """在冻结 mask 上计算全局 RGB SSIM；NaN 表示支持不足。"""
    selected = np.asarray(mask, dtype=bool)
    if selected.sum() < 2:
        return float("nan")
    first_values = np.asarray(first, dtype=np.float64)[selected] / 255.0
    second_values = np.asarray(second, dtype=np.float64)[selected] / 255.0
    means_first = first_values.mean(axis=0)
    means_second = second_values.mean(axis=0)
    variance_first = first_values.var(axis=0)
    variance_second = second_values.var(axis=0)
    covariance = (
        (first_values - means_first) * (second_values - means_second)
    ).mean(axis=0)
    c1, c2 = 0.01**2, 0.03**2
    values = (
        (2 * means_first * means_second + c1) * (2 * covariance + c2)
    ) / (
        (means_first**2 + means_second**2 + c1)
        * (variance_first + variance_second + c2)
    )
    return float(np.mean(values))


def safe_mean(values: Iterable[float | None]) -> float | None:
    array = np.asarray(
        [value for value in values if value is not None], dtype=np.float64
    )
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else None


def primary_failure(codes: Iterable[str]) -> str | None:
    present = set(codes)
    unknown = present.difference(FAILURE_PRIORITY)
    if unknown:
        raise ValueError(f"unknown failure codes: {sorted(unknown)}")
    return next((code for code in FAILURE_PRIORITY if code in present), None)


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    top_left = np.maximum(first[:2], second[:2])
    bottom_right = np.minimum(first[2:], second[2:])
    size = np.maximum(bottom_right - top_left, 0.0)
    intersection = float(size[0] * size[1])
    area_first = float(np.prod(np.maximum(first[2:] - first[:2], 0.0)))
    area_second = float(np.prod(np.maximum(second[2:] - second[:2], 0.0)))
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0
