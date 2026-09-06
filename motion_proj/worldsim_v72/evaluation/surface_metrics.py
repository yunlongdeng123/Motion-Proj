"""低内存、target 仅在评测端可见的点表面指标。"""

from __future__ import annotations

from typing import Any

import numpy as np


def _points(value: np.ndarray, *, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} 必须为 [N,3]")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} 必须为有限数")
    return points


def deterministic_farthest_point_sample(points: np.ndarray, budget: int) -> np.ndarray:
    """只依赖输入点的确定性 FPS；不得读取 target 或 target 范围。"""
    points = _points(points, name="points")
    budget = int(budget)
    if budget <= 0:
        raise ValueError("点数预算必须为正数")
    if len(points) <= budget:
        return points.copy()
    center = points.mean(axis=0, dtype=np.float64)
    first = int(np.argmax(np.sum((points - center) ** 2, axis=1)))
    selected = np.empty(budget, dtype=np.int64)
    selected[0] = first
    minimum_squared = np.sum((points - points[first]) ** 2, axis=1)
    for index in range(1, budget):
        current = int(np.argmax(minimum_squared))
        selected[index] = current
        squared = np.sum((points - points[current]) ** 2, axis=1)
        minimum_squared = np.minimum(minimum_squared, squared)
    return points[selected]


def _directed_nearest_distances(
    left: np.ndarray, right: np.ndarray, *, chunk_size: int
) -> np.ndarray:
    if len(left) == 0 or len(right) == 0:
        return np.full(len(left), np.inf, dtype=np.float32)
    right64 = right.astype(np.float64, copy=False)
    right_norm = np.sum(right64 * right64, axis=1)[None, :]
    outputs: list[np.ndarray] = []
    for start in range(0, len(left), int(chunk_size)):
        chunk = left[start : start + int(chunk_size)].astype(np.float64, copy=False)
        squared = (
            np.sum(chunk * chunk, axis=1)[:, None]
            + right_norm
            - 2.0 * chunk @ right64.T
        )
        outputs.append(np.sqrt(np.maximum(squared.min(axis=1), 0.0)).astype(np.float32))
    return np.concatenate(outputs)


def _geometry_metrics(
    surface: np.ndarray,
    target: np.ndarray,
    *,
    fscore_threshold_m: float,
    distance_chunk_size: int,
) -> dict[str, float | int]:
    surface_to_target = _directed_nearest_distances(
        surface, target, chunk_size=distance_chunk_size
    )
    target_to_surface = _directed_nearest_distances(
        target, surface, chunk_size=distance_chunk_size
    )
    precision = float(np.mean(surface_to_target <= float(fscore_threshold_m)))
    recall = float(np.mean(target_to_surface <= float(fscore_threshold_m)))
    fscore = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    surface_mean = float(np.mean(surface_to_target))
    target_mean = float(np.mean(target_to_surface))
    return {
        "surface_point_count": int(len(surface)),
        "target_point_count": int(len(target)),
        "surface_to_target_mean_m": surface_mean,
        "target_to_surface_mean_m": target_mean,
        "symmetric_cd_l1_m": 0.5 * (surface_mean + target_mean),
        "precision_at_threshold": precision,
        "recall_at_threshold": recall,
        "fscore_at_threshold": fscore,
        "fscore_threshold_m": float(fscore_threshold_m),
    }


def _literal_return_metrics(
    surface: np.ndarray,
    target: np.ndarray,
    origins: np.ndarray,
    *,
    lateral_tolerance_m: float,
    depth_tolerance_m: float,
    ray_chunk_size: int,
    point_chunk_size: int,
) -> dict[str, float | int | bool | None]:
    if len(target) != len(origins):
        raise ValueError("target 与 origins 数量不一致")
    vectors = target - origins
    target_depth = np.linalg.norm(vectors, axis=1)
    if np.any(target_depth <= 1.0e-6):
        raise ValueError("target ray 深度必须为正")
    directions = vectors / target_depth[:, None]
    first_depth = np.full(len(target), np.inf, dtype=np.float32)
    lateral_sq_limit = float(lateral_tolerance_m) ** 2
    for ray_start in range(0, len(target), int(ray_chunk_size)):
        ray_slice = slice(ray_start, ray_start + int(ray_chunk_size))
        chunk_origins = origins[ray_slice].astype(np.float64, copy=False)
        chunk_directions = directions[ray_slice].astype(np.float64, copy=False)
        best = np.full(len(chunk_origins), np.inf, dtype=np.float64)
        for point_start in range(0, len(surface), int(point_chunk_size)):
            points = surface[point_start : point_start + int(point_chunk_size)].astype(
                np.float64, copy=False
            )
            displacement = points[None, :, :] - chunk_origins[:, None, :]
            depth = np.einsum("rpc,rc->rp", displacement, chunk_directions)
            squared_norm = np.einsum("rpc,rpc->rp", displacement, displacement)
            lateral_squared = np.maximum(squared_norm - depth * depth, 0.0)
            valid = (depth > 0.0) & (lateral_squared <= lateral_sq_limit)
            local = np.where(valid, depth, np.inf).min(axis=1)
            best = np.minimum(best, local)
        first_depth[ray_slice] = best.astype(np.float32)
    observable = np.isfinite(first_depth)
    early = observable & (first_depth < target_depth - float(depth_tolerance_m))
    hit = observable & (np.abs(first_depth - target_depth) <= float(depth_tolerance_m))
    late = observable & (first_depth > target_depth + float(depth_tolerance_m))
    miss = ~observable
    observed_error = np.abs(first_depth[observable] - target_depth[observable])
    ray_count = len(target)
    return {
        "ray_count": int(ray_count),
        "observable_count": int(observable.sum()),
        "early_count": int(early.sum()),
        "hit_count": int(hit.sum()),
        "late_count": int(late.sum()),
        "miss_count": int(miss.sum()),
        "observable_rate": float(observable.mean()),
        "early_rate": float(early.mean()),
        "hit_recall": float(hit.mean()),
        "late_rate": float(late.mean()),
        "miss_rate": float(miss.mean()),
        "observable_range_mae_m": (
            float(observed_error.mean()) if len(observed_error) else None
        ),
        "full_return_metrics_supported": False,
        "full_return_metrics_reason": "legacy cache contains positive target returns only",
    }


def evaluate_point_surface(
    surface: np.ndarray,
    target: np.ndarray,
    origins: np.ndarray,
    *,
    fscore_threshold_m: float = 0.2,
    lateral_tolerance_m: float = 0.2,
    depth_tolerance_m: float = 0.2,
    distance_chunk_size: int = 256,
    ray_chunk_size: int = 128,
    point_chunk_size: int = 512,
) -> dict[str, Any]:
    """评测已预测 surface；target/origins 不参与 surface 生成。"""
    surface = _points(surface, name="surface")
    target = _points(target, name="target")
    origins = _points(origins, name="origins")
    if not len(surface) or not len(target):
        raise ValueError("surface 与 target 均不能为空")
    return {
        "geometry": _geometry_metrics(
            surface,
            target,
            fscore_threshold_m=fscore_threshold_m,
            distance_chunk_size=distance_chunk_size,
        ),
        "conditional_return": _literal_return_metrics(
            surface,
            target,
            origins,
            lateral_tolerance_m=lateral_tolerance_m,
            depth_tolerance_m=depth_tolerance_m,
            ray_chunk_size=ray_chunk_size,
            point_chunk_size=point_chunk_size,
        ),
    }
