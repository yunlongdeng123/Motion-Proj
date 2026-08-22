"""WorldSim V6.1 ME-3：GaussianWorld 类别网格到 SceneIR-O 的保守适配。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from motion_proj.worldsim_v61.occupancy import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    Transform,
    VoxelGridSpec,
    load_frame_boxes,
    voxelize_oriented_box,
)


def resample_gaussianworld_classes(
    class_label: np.ndarray,
    target_spec: VoxelGridSpec,
    *,
    source_origin_m: tuple[float, float, float] = (-50.0, -50.0, -5.0),
    source_voxel_size_m: float = 0.5,
    noise_class: int = 0,
    occupied_class_min: int = 1,
    occupied_class_max: int = 16,
    empty_class: int = 17,
) -> dict[str, np.ndarray]:
    """按 target cell center 直接重采样；不插值、不设 confidence threshold。"""
    source = np.asarray(class_label, dtype=np.uint8)
    if source.ndim != 3 or source.shape != (200, 200, 16):
        raise ValueError(f"GaussianWorld class grid 尺寸非法: {source.shape}")
    source_origin = np.asarray(source_origin_m, dtype=np.float64)
    target_origin = target_spec.origin
    axes = []
    valid_axes = []
    for axis, size in enumerate(target_spec.shape):
        centers = target_origin[axis] + (np.arange(size, dtype=np.float64) + 0.5) * target_spec.voxel_size_m
        indices = np.floor((centers - source_origin[axis]) / source_voxel_size_m).astype(np.int64)
        valid = (indices >= 0) & (indices < source.shape[axis])
        axes.append(np.clip(indices, 0, source.shape[axis] - 1))
        valid_axes.append(valid)
    sampled = source[np.ix_(axes[0], axes[1], axes[2])]
    valid = (
        valid_axes[0][:, None, None]
        & valid_axes[1][None, :, None]
        & valid_axes[2][None, None, :]
    )
    semantics = np.full(target_spec.shape, UNKNOWN, dtype=np.uint8)
    semantics[valid & (sampled == int(empty_class))] = FREE
    occupied = valid & (sampled >= int(occupied_class_min)) & (
        sampled <= int(occupied_class_max)
    )
    semantics[occupied] = OCCUPIED
    # noise class 与 source extent 外一律 UNKNOWN；不把低置信区域写成 FREE。
    predicted_class = np.full(target_spec.shape, 255, dtype=np.uint8)
    predicted_class[valid] = sampled[valid]
    if np.any(valid & ~(
        (sampled == int(noise_class))
        | (sampled == int(empty_class))
        | ((sampled >= int(occupied_class_min)) & (sampled <= int(occupied_class_max)))
    )):
        raise ValueError("GaussianWorld 出现冻结类别合同外的 label")
    return {
        "semantics": semantics,
        "predicted_class": predicted_class,
        "source_valid": valid,
    }


def bind_native_actor_identity_without_geometry_fill(
    semantics: np.ndarray,
    target_spec: VoxelGridSpec,
    scene_root: Path,
    target_frame: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """native OBB 只标注模型已预测 occupied 的体素，不用 box 补占据。"""
    values = np.asarray(semantics, dtype=np.uint8)
    if values.shape != target_spec.shape:
        raise ValueError("predicted semantics 与 target spec 漂移")
    t_global_target = Transform(
        dst="global",
        src=target_spec.frame,
        matrix=np.loadtxt(scene_root / f"lidar_pose/{int(target_frame):03d}.txt"),
    )
    boxes = load_frame_boxes(scene_root, int(target_frame), t_global_target.inverse())
    actor_grid = np.full(target_spec.shape, -1, dtype=np.int32)
    rows: list[dict[str, Any]] = []
    for box in boxes:
        indices, _ = voxelize_oriented_box(target_spec, box["transform"], box["size_lwh"])
        if indices.size:
            predicted = values[indices[:, 0], indices[:, 1], indices[:, 2]] == OCCUPIED
            selected = indices[predicted]
            actor_grid[selected[:, 0], selected[:, 1], selected[:, 2]] = int(box["actor_id"])
        else:
            selected = np.empty((0, 3), dtype=np.int32)
        rows.append(
            {
                "actor_id": int(box["actor_id"]),
                "class_name": box["class_name"],
                "native_obb_voxel_count": int(indices.shape[0]),
                "predicted_occupied_inside_native_obb": int(selected.shape[0]),
                "geometry_created_by_identity_binding": False,
            }
        )
    return actor_grid, rows


def raycast_predicted_conservative(
    grid: dict[str, Any],
    t_lidar_camera: np.ndarray,
    intrinsics: np.ndarray,
    projection: dict[str, Any],
) -> dict[str, np.ndarray]:
    """只接受 FREE 前缀后的首个 OCCUPIED；UNKNOWN 会封住该条射线。"""
    device = torch.device("cuda")
    height, width = int(projection["height"]), int(projection["width"])
    ys, xs = np.indices((height, width), dtype=np.float32)
    directions_camera = np.stack(
        (
            (xs - intrinsics[0, 2]) / intrinsics[0, 0],
            (ys - intrinsics[1, 2]) / intrinsics[1, 1],
            np.ones_like(xs),
        ),
        axis=-1,
    ).reshape(-1, 3)
    directions_lidar = directions_camera @ t_lidar_camera[:3, :3].T
    directions_lidar /= np.linalg.norm(directions_lidar, axis=1, keepdims=True)
    semantics = np.asarray(grid["semantics"], dtype=np.uint8)
    actor_grid = np.asarray(grid["actor_grid"], dtype=np.int32)
    semantics_gpu = torch.from_numpy(semantics.astype(np.int16)).to(device)
    origin_gpu = torch.as_tensor(grid["origin"], dtype=torch.float32, device=device)
    camera_origin_gpu = torch.as_tensor(
        t_lidar_camera[:3, 3], dtype=torch.float32, device=device
    )
    shape_gpu = torch.as_tensor(semantics.shape, dtype=torch.long, device=device)
    distances = torch.arange(
        float(projection["near_m"]),
        float(projection["far_m"]),
        float(projection["step_m"]),
        device=device,
    )
    depth = torch.full((height * width,), float("nan"), dtype=torch.float32, device=device)
    linear = torch.full((height * width,), -1, dtype=torch.long, device=device)
    unknown_blocked = torch.zeros((height * width,), dtype=torch.bool, device=device)
    batch_size = int(projection["ray_batch_size"])
    stride_yz = int(semantics.shape[1] * semantics.shape[2])
    stride_z = int(semantics.shape[2])
    for start in range(0, directions_lidar.shape[0], batch_size):
        directions = torch.as_tensor(
            directions_lidar[start : start + batch_size], dtype=torch.float32, device=device
        )
        points = camera_origin_gpu[None, None, :] + directions[:, None, :] * distances[None, :, None]
        indices = torch.floor(
            (points - origin_gpu[None, None, :]) / float(grid["voxel_size"])
        ).long()
        valid = torch.all((indices >= 0) & (indices < shape_gpu[None, None, :]), dim=-1)
        clipped = torch.minimum(
            torch.maximum(indices, torch.zeros_like(indices)), shape_gpu[None, None, :] - 1
        )
        labels = semantics_gpu[clipped[..., 0], clipped[..., 1], clipped[..., 2]]
        first_evidence = valid & (labels != int(FREE))
        has_evidence = torch.any(first_evidence, dim=1)
        first = torch.argmax(first_evidence.to(torch.int8), dim=1)
        row = torch.arange(directions.shape[0], device=device)
        first_labels = labels[row, first]
        first_indices = clipped[row, first]
        accepted = has_evidence & (first_labels == int(OCCUPIED))
        blocked = has_evidence & (first_labels == int(UNKNOWN))
        local_depth = distances[first]
        local_linear = (
            first_indices[:, 0] * stride_yz
            + first_indices[:, 1] * stride_z
            + first_indices[:, 2]
        )
        local_depth[~accepted] = float("nan")
        local_linear[~accepted] = -1
        depth[start : start + directions.shape[0]] = local_depth
        linear[start : start + directions.shape[0]] = local_linear
        unknown_blocked[start : start + directions.shape[0]] = blocked
    depth_cpu = depth.reshape(height, width).cpu().numpy()
    linear_cpu = linear.reshape(height, width).cpu().numpy()
    actor = np.full_like(linear_cpu, -1, dtype=np.int32)
    hit = linear_cpu >= 0
    actor[hit] = actor_grid.reshape(-1)[linear_cpu[hit]]
    return {
        "depth_m": depth_cpu,
        "voxel_linear": linear_cpu,
        "actor_instance_id": actor,
        "unknown_blocked": unknown_blocked.reshape(height, width).cpu().numpy(),
    }


def predicted_method_factors(
    mask: np.ndarray,
    raycast: dict[str, np.ndarray],
    case: dict[str, Any],
    minimum_coverage: float,
) -> tuple[dict[str, Any], np.ndarray, list[int]]:
    """只在 predicted OCCUPIED 上形成 proposal；predicted FREE 不冒充观测 FREE。"""
    linear = np.asarray(raycast["voxel_linear"], dtype=np.int64)
    valid = linear >= 0
    candidates = np.unique(linear[mask & valid])
    actor_ids = sorted(
        int(value)
        for value in np.unique(
            np.asarray(raycast["actor_instance_id"], dtype=np.int32)[mask & valid]
        )
        if int(value) >= 0
    )
    coverage = float(np.mean(valid[mask]))
    unknown_fraction = float(
        np.mean(np.asarray(raycast["unknown_blocked"], dtype=bool)[mask])
    )
    needs_actor = case["hole_type"] == "actor_removal_hole"
    passed = bool(
        candidates.size
        and coverage >= float(minimum_coverage)
        and (not needs_actor or actor_ids)
    )
    factors = {
        "predicted_surface_coverage": coverage,
        "predicted_unknown_blocked_fraction": unknown_fraction,
        "candidate_occupied_voxel_count": int(candidates.size),
        "native_actor_identity_count": len(actor_ids),
        "actor_identity_required": needs_actor,
        "predicted_free_is_not_observed_free": True,
        "observed_free_space_conflict_evaluated_at_method_time": False,
        "passed": passed,
    }
    return factors, candidates, actor_ids
