"""由同一 WorldState/RenderRequest 生成有限标签并审计同步不变量。"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import numpy as np

SEMANTIC_UNKNOWN = 0
SEMANTIC_STATIC_BACKGROUND = 1
SEMANTIC_VEHICLE = 2
SEMANTIC_IGNORE = 255
LIMITED_SEMANTIC_VALUES = {
    SEMANTIC_UNKNOWN,
    SEMANTIC_STATIC_BACKGROUND,
    SEMANTIC_VEHICLE,
    SEMANTIC_IGNORE,
}


@dataclass(frozen=True)
class TypedDepth:
    value: np.ndarray
    valid: np.ndarray
    name: str
    truth_tier: str
    definition: str

    def __post_init__(self) -> None:
        if self.value.shape != self.valid.shape:
            raise ValueError("depth value/valid shape 不一致")
        if self.name == "depth_render_expected" and self.truth_tier == "T0":
            raise ValueError("Gaussian expected depth 不得登记为 T0")
        if self.name == "depth_lidar_measured" and self.truth_tier != "T0":
            raise ValueError("LiDAR measured depth 必须登记为 T0")


def cumulative_first_hit(
    depths: np.ndarray,
    alphas: np.ndarray,
    *,
    threshold: float,
) -> TypedDepth:
    """输入按 near→far 排序的 per-ray contributions，求累计 alpha 首次过阈值。"""
    depths = np.asarray(depths, dtype=np.float32)
    alphas = np.asarray(alphas, dtype=np.float32)
    if depths.shape != alphas.shape or depths.ndim < 2:
        raise ValueError("depths/alphas 必须为 [..., contributions] 同形数组")
    if not 0 < threshold < 1:
        raise ValueError("first-hit threshold 必须在 (0,1)")
    cumulative = 1.0 - np.cumprod(1.0 - np.clip(alphas, 0, 1), axis=-1)
    reached = cumulative >= threshold
    valid = np.any(reached, axis=-1)
    index = np.argmax(reached, axis=-1)
    value = np.take_along_axis(depths, index[..., None], axis=-1)[..., 0]
    value = np.where(valid, value, 0.0).astype(np.float32)
    return TypedDepth(
        value, valid, "depth_surface_first_hit", "T1",
        f"cumulative_alpha_first_crossing_{threshold:g}",
    )


def render_expected_depth(value: np.ndarray, opacity: np.ndarray) -> TypedDepth:
    valid = np.asarray(opacity) > 0
    depth = np.asarray(value, dtype=np.float32)
    return TypedDepth(
        np.where(valid, depth, 0).astype(np.float32),
        valid,
        "depth_render_expected",
        "diagnostic",
        "Gaussian alpha-weighted expected z-depth",
    )


def lidar_measured_depth(value: np.ndarray) -> TypedDepth:
    depth = np.asarray(value, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    return TypedDepth(
        np.where(valid, depth, 0).astype(np.float32),
        valid,
        "depth_lidar_measured",
        "T0",
        "sparse calibrated LiDAR projection",
    )


def vehicle_instance_mask(
    actor_opacities: dict[int, np.ndarray],
    actor_depths: dict[int, np.ndarray],
    *,
    alpha_threshold: float,
) -> np.ndarray:
    if not actor_opacities:
        raise ValueError("actor opacity layers 不能为空")
    shape = next(iter(actor_opacities.values())).squeeze().shape
    best_depth = np.full(shape, np.inf, dtype=np.float32)
    instance = np.zeros(shape, dtype=np.int32)
    for actor_id in sorted(actor_opacities):
        alpha = np.asarray(actor_opacities[actor_id]).squeeze()
        depth = np.asarray(actor_depths[actor_id]).squeeze()
        visible = (alpha >= alpha_threshold) & np.isfinite(depth) & (depth > 0)
        nearer = visible & (depth < best_depth)
        best_depth[nearer] = depth[nearer]
        instance[nearer] = int(actor_id)
    return instance


def limited_semantic_mask(
    background_opacity: np.ndarray,
    instance_mask: np.ndarray,
    *,
    alpha_threshold: float,
    ignore_mask: np.ndarray | None = None,
) -> np.ndarray:
    background = np.asarray(background_opacity).squeeze()
    instance = np.asarray(instance_mask)
    semantic = np.full(instance.shape, SEMANTIC_UNKNOWN, dtype=np.uint8)
    semantic[background >= alpha_threshold] = SEMANTIC_STATIC_BACKGROUND
    semantic[instance > 0] = SEMANTIC_VEHICLE
    if ignore_mask is not None:
        semantic[np.asarray(ignore_mask, dtype=bool)] = SEMANTIC_IGNORE
    if not set(np.unique(semantic)).issubset(LIMITED_SEMANTIC_VALUES):
        raise RuntimeError("limited semantic scope 越界")
    return semantic


def visible_box_from_mask(mask: np.ndarray) -> dict:
    value = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(value)
    height, width = value.shape
    if len(xs) == 0:
        return {"status": "invisible", "xyxy": None, "pixel_count": 0, "truncated": False}
    xyxy = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    truncated = xyxy[0] == 0 or xyxy[1] == 0 or xyxy[2] == width or xyxy[3] == height
    return {
        "status": "truncated" if truncated else "visible",
        "xyxy": xyxy,
        "pixel_count": int(len(xs)),
        "truncated": truncated,
    }


def raw_projected_box(points_camera: np.ndarray, intrinsics: np.ndarray, resolution: tuple[int, int]) -> dict:
    points = np.asarray(points_camera, dtype=float)
    depth = points[:, 2]
    if not np.any(depth > 1e-6):
        return {"status": "behind_camera", "unclipped_xyxy": None, "clipped_xyxy": None}
    projected = (np.asarray(intrinsics) @ points.T).T
    xy = projected[:, :2] / projected[:, 2:3]
    finite = np.isfinite(xy).all(axis=1) & (depth > 1e-6)
    if not np.any(finite):
        return {"status": "missing", "unclipped_xyxy": None, "clipped_xyxy": None}
    xy = xy[finite]
    raw = [float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()), float(xy[:, 1].max())]
    width, height = resolution
    clipped = [
        max(0.0, min(float(width), raw[0])),
        max(0.0, min(float(height), raw[1])),
        max(0.0, min(float(width), raw[2])),
        max(0.0, min(float(height), raw[3])),
    ]
    intersects = clipped[2] > clipped[0] and clipped[3] > clipped[1]
    return {
        "status": "projected" if intersects else "outside_image",
        "unclipped_xyxy": raw,
        "clipped_xyxy": clipped if intersects else None,
    }


def audit_label_bundle(bundle: dict, *, box_tolerance_px: float = 1.0) -> dict:
    required = {
        "world_state_hash", "render_request_hash", "typed_depths",
        "instance_mask", "limited_semantic_mask", "visible_boxes",
    }
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError(f"label bundle 缺字段: {missing}")
    depths: Iterable[TypedDepth] = bundle["typed_depths"]
    names = {value.name for value in depths}
    typed_ok = names == {
        "depth_render_expected", "depth_surface_first_hit", "depth_lidar_measured"
    } and all(
        not (value.name == "depth_render_expected" and value.truth_tier == "T0")
        for value in depths
    )
    semantic_ok = set(np.unique(bundle["limited_semantic_mask"])).issubset(
        LIMITED_SEMANTIC_VALUES
    )
    containment = True
    for actor_id, visible in bundle["visible_boxes"].items():
        actor_pixels = bundle["instance_mask"] == int(actor_id)
        if not actor_pixels.any():
            containment &= visible["status"] == "invisible"
            continue
        ys, xs = np.nonzero(actor_pixels)
        x0, y0, x1, y1 = visible["xyxy"]
        containment &= (
            xs.min() >= x0 - box_tolerance_px
            and ys.min() >= y0 - box_tolerance_px
            and xs.max() < x1 + box_tolerance_px
            and ys.max() < y1 + box_tolerance_px
        )
    checks = {
        "typed_depth_names_and_truth_tiers": bool(typed_ok),
        "limited_semantic_scope": bool(semantic_ok),
        "visible_box_contains_instance": bool(containment),
        "world_state_hash_present": len(bundle["world_state_hash"]) == 64,
        "render_request_hash_present": len(bundle["render_request_hash"]) == 64,
    }
    checks["pass"] = all(checks.values())
    return checks


def bytes_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()
