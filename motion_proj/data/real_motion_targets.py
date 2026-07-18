"""真实驾驶视频上的 ego-induced 与 actor-residual motion target。

本模块只允许用于真实训练视频的 representation probe / auxiliary supervision。
它读取真实时间戳、相机标定、ego pose、LiDAR depth 与 3D annotation；这些
future 信息不得进入自由生成 rollout 的条件、scorer 或 evaluator。
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ..utils.geometry import backproject, pixel_grid, project_points, se3_inverse


MOVABLE_CATEGORY_PREFIXES = ("vehicle.", "human.pedestrian.", "cycle.")
MOVING_ATTRIBUTES = {
    "vehicle.moving",
    "cycle.with_rider",
    "pedestrian.moving",
}
STATIONARY_ATTRIBUTES = {
    "vehicle.parked",
    "vehicle.stopped",
    "cycle.without_rider",
    "pedestrian.standing",
    "pedestrian.sitting_lying_down",
}
REAL_TARGET_SCOPE = "real_training_representation_only"
GENERATED_EVALUATION_SCOPE = "generated_rollout_evaluation"


class RealMotionTargetError(RuntimeError):
    """真实运动 target 的几何、单位或使用范围不合法。"""


def assert_target_scope(scope: str) -> None:
    """阻止真实 future annotation 泄漏到 generated-rollout evaluator。"""
    if str(scope) != REAL_TARGET_SCOPE:
        raise RealMotionTargetError(
            "真实 ego/box/LiDAR future target 只允许 real-training representation；"
            f"收到 scope={scope!r}"
        )


def is_movable_category(category: str) -> bool:
    return any(str(category).startswith(prefix) for prefix in MOVABLE_CATEGORY_PREFIXES)


def attribute_motion_label(attributes: Sequence[str]) -> str:
    values = {str(value) for value in attributes}
    if values & MOVING_ATTRIBUTES:
        return "moving"
    if values & STATIONARY_ATTRIBUTES:
        return "stationary"
    return "unknown"


def timestamps_to_seconds(timestamps_us: torch.Tensor | Sequence[int]) -> torch.Tensor:
    """验证微秒时间戳并返回相邻帧秒数。"""
    values = torch.as_tensor(timestamps_us, dtype=torch.int64)
    if values.ndim != 1 or values.numel() < 2:
        raise RealMotionTargetError("timestamps 必须是一维且至少包含两帧")
    delta_us = values[1:] - values[:-1]
    if bool((delta_us <= 0).any()):
        raise RealMotionTargetError("camera timestamps 必须严格递增")
    # nuScenes keyframe delta 约 5e5 us。这个宽门只拦截把微秒误当秒/毫秒。
    if bool((delta_us < 10_000).any() or (delta_us > 10_000_000).any()):
        raise RealMotionTargetError("timestamp delta 不在微秒单位的合理视频范围")
    return delta_us.to(torch.float64) / 1.0e6


def _as_transform_sequence(sample: Mapping[str, Any], key: str, fallback: str, count: int) -> torch.Tensor:
    value = sample.get(key)
    if value is None:
        base = torch.as_tensor(sample[fallback], dtype=torch.float64)
        value = base.unsqueeze(0).repeat(count, 1, 1)
    result = torch.as_tensor(value, dtype=torch.float64)
    if result.shape != (count, 4, 4):
        raise RealMotionTargetError(f"{key} 形状必须为 {(count, 4, 4)}，实际 {tuple(result.shape)}")
    return result


def _as_intrinsics_sequence(sample: Mapping[str, Any], count: int) -> torch.Tensor:
    value = sample.get("intrinsics_frames")
    if value is None:
        base = torch.as_tensor(sample["intrinsics"], dtype=torch.float64)
        value = base.unsqueeze(0).repeat(count, 1, 1)
    result = torch.as_tensor(value, dtype=torch.float64)
    if result.shape != (count, 3, 3):
        raise RealMotionTargetError(
            f"intrinsics_frames 形状必须为 {(count, 3, 3)}，实际 {tuple(result.shape)}"
        )
    return result


def transform_points(transform: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """列向量约定的齐次 SE(3) 点变换，支持任意前导维。"""
    transform = torch.as_tensor(transform)
    points = torch.as_tensor(points, dtype=transform.dtype, device=transform.device)
    ones = torch.ones_like(points[..., :1])
    homogeneous = torch.cat([points, ones], dim=-1)
    return (transform @ homogeneous.unsqueeze(-1)).squeeze(-1)[..., :3]


def camera_to_camera_transform(
    cam2ego_t: torch.Tensor,
    ego2global_t: torch.Tensor,
    cam2ego_tp1: torch.Tensor,
    ego2global_tp1: torch.Tensor,
) -> torch.Tensor:
    """把 cam@t 中的世界静止点映射到 cam@(t+1)。"""
    cam2global_t = torch.as_tensor(ego2global_t) @ torch.as_tensor(cam2ego_t)
    cam2global_tp1 = torch.as_tensor(ego2global_tp1) @ torch.as_tensor(cam2ego_tp1)
    return se3_inverse(cam2global_tp1) @ cam2global_t


def _project_single(point_cam: torch.Tensor, intrinsics: torch.Tensor) -> tuple[torch.Tensor, bool]:
    point = torch.as_tensor(point_cam, dtype=torch.float64)
    valid = bool(torch.isfinite(point).all() and float(point[2]) > 0.1)
    if not valid:
        return torch.full((2,), float("nan"), dtype=torch.float64), False
    uv = project_points(point, torch.as_tensor(intrinsics, dtype=torch.float64))
    valid = bool(torch.isfinite(uv).all())
    return uv, valid


def _inside_xyxy(uv: torch.Tensor, xyxy: Sequence[float], tolerance: float = 0.5) -> bool:
    box = torch.as_tensor(xyxy, dtype=torch.float64)
    if box.shape != (4,) or not bool(torch.isfinite(box).all() and torch.isfinite(uv).all()):
        return False
    return bool(
        float(uv[0]) >= float(box[0]) - tolerance
        and float(uv[0]) <= float(box[2]) + tolerance
        and float(uv[1]) >= float(box[1]) - tolerance
        and float(uv[1]) <= float(box[3]) + tolerance
    )


def actor_residual_target(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    frame_index: int,
    dt_s: float,
    intrinsics_t: torch.Tensor,
    intrinsics_tp1: torch.Tensor,
    cam2ego_t: torch.Tensor,
    cam2ego_tp1: torch.Tensor,
    ego2global_t: torch.Tensor,
    ego2global_tp1: torch.Tensor,
) -> dict[str, Any]:
    """构造一个相邻帧实例的相机补偿 residual target。"""
    if str(first.get("instance_token")) != str(second.get("instance_token")):
        raise RealMotionTargetError("actor pair 的 instance_token 不一致")
    if not math.isfinite(float(dt_s)) or float(dt_s) <= 0.0:
        raise RealMotionTargetError("actor pair dt 必须为正秒数")
    first_center = torch.as_tensor(first["center_cam"], dtype=torch.float64)
    second_center = torch.as_tensor(second["center_cam"], dtype=torch.float64)
    actual_t, valid_t = _project_single(first_center, intrinsics_t)
    actual_tp1, valid_tp1 = _project_single(second_center, intrinsics_tp1)
    relative = camera_to_camera_transform(
        cam2ego_t, ego2global_t, cam2ego_tp1, ego2global_tp1,
    )
    static_center_tp1 = transform_points(relative, first_center)
    static_tp1, valid_static = _project_single(static_center_tp1, intrinsics_tp1)
    valid = bool(valid_t and valid_tp1 and valid_static)
    residual = actual_tp1 - static_tp1 if valid else torch.full((2,), float("nan"), dtype=torch.float64)
    residual_velocity = residual / float(dt_s)

    cam2global_t = torch.as_tensor(ego2global_t) @ torch.as_tensor(cam2ego_t)
    cam2global_tp1 = torch.as_tensor(ego2global_tp1) @ torch.as_tensor(cam2ego_tp1)
    center_global_t = transform_points(cam2global_t, first_center)
    center_global_tp1 = transform_points(cam2global_tp1, second_center)
    global_displacement = center_global_tp1 - center_global_t
    global_speed = float(torch.linalg.vector_norm(global_displacement) / float(dt_s))

    velocity = torch.as_tensor(first.get("velocity_global", [float("nan")] * 3), dtype=torch.float64)
    velocity_projection = torch.full((2,), float("nan"), dtype=torch.float64)
    velocity_direction_cosine = None
    if velocity.shape == (3,) and bool(torch.isfinite(velocity).all()) and valid_static:
        predicted_global = center_global_t + velocity * float(dt_s)
        predicted_cam = transform_points(se3_inverse(cam2global_tp1), predicted_global)
        predicted_uv, predicted_valid = _project_single(predicted_cam, intrinsics_tp1)
        if predicted_valid:
            velocity_projection = predicted_uv - static_tp1
            first_norm = float(torch.linalg.vector_norm(residual))
            second_norm = float(torch.linalg.vector_norm(velocity_projection))
            if first_norm > 1.0e-6 and second_norm > 1.0e-6:
                velocity_direction_cosine = float(
                    torch.dot(residual, velocity_projection) / (first_norm * second_norm)
                )

    ego_translation = torch.as_tensor(ego2global_tp1, dtype=torch.float64)[:3, 3] - torch.as_tensor(
        ego2global_t, dtype=torch.float64,
    )[:3, 3]
    attributes = sorted({str(value) for value in first.get("attributes", [])})
    category = str(first.get("category", ""))
    return {
        "frame_index": int(frame_index),
        "next_frame_index": int(frame_index + 1),
        "annotation_token_t": str(first.get("annotation_token", "")),
        "annotation_token_tp1": str(second.get("annotation_token", "")),
        "instance_token": str(first["instance_token"]),
        "category": category,
        "attributes": attributes,
        "motion_label": attribute_motion_label(attributes),
        "visibility_t": int(first.get("visibility", 0)),
        "visibility_tp1": int(second.get("visibility", 0)),
        "dt_s": float(dt_s),
        "actual_uv_t": actual_t.tolist(),
        "actual_uv_tp1": actual_tp1.tolist(),
        "static_uv_tp1": static_tp1.tolist(),
        "residual_px": residual.tolist(),
        "residual_speed_px_per_s": float(torch.linalg.vector_norm(residual_velocity)),
        "static_center_cam_tp1": static_center_tp1.tolist(),
        "actual_center_depth_t": float(first_center[2]),
        "actual_center_depth_tp1": float(second_center[2]),
        "center_global_displacement_m": global_displacement.tolist(),
        "center_global_speed_mps": global_speed,
        "velocity_global_mps": velocity.tolist(),
        "velocity_projected_residual_px": velocity_projection.tolist(),
        "velocity_direction_cosine": velocity_direction_cosine,
        "ego_translation_speed_mps": float(torch.linalg.vector_norm(ego_translation) / float(dt_s)),
        "center_projection_in_box_t": _inside_xyxy(actual_t, first["xyxy"]),
        "center_projection_in_box_tp1": _inside_xyxy(actual_tp1, second["xyxy"]),
        "finite": valid and bool(torch.isfinite(residual_velocity).all()),
        "target_scope": REAL_TARGET_SCOPE,
    }


def build_actor_residual_targets(
    sample: Mapping[str, Any], *, min_visibility: int = 2,
) -> list[dict[str, Any]]:
    """按 instance token 构建所有相邻帧 actor residual rows。"""
    assert_target_scope(REAL_TARGET_SCOPE)
    boxes = sample["boxes"]
    count = len(boxes)
    if count < 2:
        raise RealMotionTargetError("真实 clip 至少需要两帧")
    dt = timestamps_to_seconds(sample["timestamps"])
    intrinsics = _as_intrinsics_sequence(sample, count)
    cam2ego = _as_transform_sequence(sample, "cam2ego_frames", "cam2ego", count)
    ego2global = torch.as_tensor(sample["ego2global"], dtype=torch.float64)
    if ego2global.shape != (count, 4, 4):
        raise RealMotionTargetError("ego2global 必须为 [K,4,4]")
    rows: list[dict[str, Any]] = []
    for frame_index in range(count - 1):
        left = {
            str(box["instance_token"]): box
            for box in boxes[frame_index]
            if is_movable_category(str(box.get("category", "")))
            and int(box.get("visibility", 0)) >= int(min_visibility)
        }
        right = {
            str(box["instance_token"]): box
            for box in boxes[frame_index + 1]
            if is_movable_category(str(box.get("category", "")))
            and int(box.get("visibility", 0)) >= int(min_visibility)
        }
        for token in sorted(left.keys() & right.keys()):
            rows.append(
                actor_residual_target(
                    left[token], right[token], frame_index=frame_index, dt_s=float(dt[frame_index]),
                    intrinsics_t=intrinsics[frame_index], intrinsics_tp1=intrinsics[frame_index + 1],
                    cam2ego_t=cam2ego[frame_index], cam2ego_tp1=cam2ego[frame_index + 1],
                    ego2global_t=ego2global[frame_index], ego2global_tp1=ego2global[frame_index + 1],
                )
            )
    return rows


def sparse_ego_flow_target(
    depth_t: torch.Tensor,
    intrinsics_t: torch.Tensor,
    intrinsics_tp1: torch.Tensor,
    cam2ego_t: torch.Tensor,
    cam2ego_tp1: torch.Tensor,
    ego2global_t: torch.Tensor,
    ego2global_tp1: torch.Tensor,
    *,
    minimum_depth: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从稀疏 LiDAR z-depth 构造世界静止点的 ego-induced flow。"""
    depth = torch.as_tensor(depth_t, dtype=torch.float64)
    height, width = depth.shape
    uv = pixel_grid(height, width, dtype=torch.float64)
    source_valid = torch.isfinite(depth) & (depth >= float(minimum_depth))
    safe_depth = torch.where(source_valid, depth, torch.full_like(depth, float(minimum_depth)))
    points_t = backproject(uv, safe_depth, torch.as_tensor(intrinsics_t, dtype=torch.float64))
    relative = camera_to_camera_transform(
        cam2ego_t, ego2global_t, cam2ego_tp1, ego2global_tp1,
    )
    points_tp1 = transform_points(relative, points_t)
    destination_valid = torch.isfinite(points_tp1).all(dim=-1) & (points_tp1[..., 2] >= minimum_depth)
    safe_points = torch.where(
        destination_valid[..., None], points_tp1, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
    )
    uv_tp1 = project_points(safe_points, torch.as_tensor(intrinsics_tp1, dtype=torch.float64))
    in_image = (
        torch.isfinite(uv_tp1).all(dim=-1)
        & (uv_tp1[..., 0] >= 0)
        & (uv_tp1[..., 0] <= width - 1)
        & (uv_tp1[..., 1] >= 0)
        & (uv_tp1[..., 1] <= height - 1)
    )
    valid = source_valid & destination_valid & in_image
    flow = torch.where(valid[..., None], uv_tp1 - uv, torch.zeros_like(uv))
    return flow.to(torch.float32), valid


def boxes_background_mask(
    height: int, width: int, boxes: Sequence[Mapping[str, Any]], *, dilation_px: int = 2,
) -> torch.Tensor:
    """返回 GT box 外的保守背景 mask。"""
    mask = torch.ones((int(height), int(width)), dtype=torch.bool)
    for box in boxes:
        xyxy = np.asarray(box.get("xyxy"), dtype=np.float64)
        if xyxy.shape != (4,) or not np.isfinite(xyxy).all():
            continue
        u0 = max(int(math.floor(xyxy[0])) - int(dilation_px), 0)
        v0 = max(int(math.floor(xyxy[1])) - int(dilation_px), 0)
        u1 = min(int(math.ceil(xyxy[2])) + int(dilation_px) + 1, int(width))
        v1 = min(int(math.ceil(xyxy[3])) + int(dilation_px) + 1, int(height))
        if u1 > u0 and v1 > v0:
            mask[v0:v1, u0:u1] = False
    return mask


def flow_direction_agreement(
    target: torch.Tensor,
    observed: torch.Tensor,
    valid: torch.Tensor,
    *,
    minimum_magnitude_px: float = 0.25,
    maximum_angle_deg: float = 45.0,
) -> dict[str, Any]:
    """计算稀疏 target 与观测光流的方向一致率，不以 0 填充无效点。"""
    target = torch.as_tensor(target, dtype=torch.float64)
    observed = torch.as_tensor(observed, dtype=torch.float64)
    valid = torch.as_tensor(valid, dtype=torch.bool)
    if target.shape != observed.shape or target.shape[-1] != 2 or valid.shape != target.shape[:-1]:
        raise ValueError("flow/valid 形状不一致")
    target_norm = torch.linalg.vector_norm(target, dim=-1)
    observed_norm = torch.linalg.vector_norm(observed, dim=-1)
    usable = (
        valid
        & torch.isfinite(target).all(dim=-1)
        & torch.isfinite(observed).all(dim=-1)
        & (target_norm >= float(minimum_magnitude_px))
        & (observed_norm >= float(minimum_magnitude_px))
    )
    if not bool(usable.any()):
        return {"status": "invalid", "count": 0}
    cosine = (
        (target[usable] * observed[usable]).sum(dim=-1)
        / (target_norm[usable] * observed_norm[usable]).clamp_min(1.0e-12)
    ).clamp(-1.0, 1.0)
    threshold = math.cos(math.radians(float(maximum_angle_deg)))
    return {
        "status": "valid",
        "count": int(cosine.numel()),
        "agreement_fraction": float((cosine >= threshold - 1.0e-12).double().mean()),
        "mean_cosine": float(cosine.mean()),
        "median_cosine": float(cosine.median()),
        "median_angle_deg": float(torch.rad2deg(torch.acos(cosine)).median()),
        "maximum_angle_deg": float(maximum_angle_deg),
    }


def binary_roc_auc(positive: Sequence[float], negative: Sequence[float]) -> float | None:
    """Mann–Whitney 定义的 AUC，ties 计 0.5。"""
    pos = [float(value) for value in positive if math.isfinite(float(value))]
    neg = [float(value) for value in negative if math.isfinite(float(value))]
    if not pos or not neg:
        return None
    wins = 0.0
    for left in pos:
        for right in neg:
            wins += 1.0 if left > right else 0.5 if left == right else 0.0
    return wins / float(len(pos) * len(neg))


def spearman_correlation(first: Sequence[float], second: Sequence[float]) -> float | None:
    """无 SciPy 依赖的平均秩 Spearman；常数序列返回 None。"""
    pairs = [
        (float(left), float(right))
        for left, right in zip(first, second)
        if math.isfinite(float(left)) and math.isfinite(float(right))
    ]
    if len(pairs) < 3:
        return None

    def ranks(values: Sequence[float]) -> np.ndarray:
        order = np.argsort(np.asarray(values, dtype=np.float64), kind="mergesort")
        result = np.empty(len(values), dtype=np.float64)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and values[order[end]] == values[order[start]]:
                end += 1
            result[order[start:end]] = (start + end - 1) / 2.0
            start = end
        return result

    left_rank = ranks([row[0] for row in pairs])
    right_rank = ranks([row[1] for row in pairs])
    if float(left_rank.std()) <= 0.0 or float(right_rank.std()) <= 0.0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])
