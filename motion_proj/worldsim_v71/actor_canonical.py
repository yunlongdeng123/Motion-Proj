"""Actor-local 坐标合同与固定帧角色。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ActorCanonicalFrame:
    """使用 row-vector 约定的 Actor canonical frame。"""

    center_world_m: np.ndarray
    rotation_world_from_actor: np.ndarray

    def __post_init__(self) -> None:
        center = np.asarray(self.center_world_m, dtype=np.float64).reshape(3)
        rotation = np.asarray(self.rotation_world_from_actor, dtype=np.float64).reshape(3, 3)
        object.__setattr__(self, "center_world_m", center)
        object.__setattr__(self, "rotation_world_from_actor", rotation)


def world_to_actor(points_world_m: np.ndarray, frame: ActorCanonicalFrame) -> np.ndarray:
    points = np.asarray(points_world_m, dtype=np.float64).reshape(-1, 3)
    return (points - frame.center_world_m[None, :]) @ frame.rotation_world_from_actor


def actor_to_world(points_actor_m: np.ndarray, frame: ActorCanonicalFrame) -> np.ndarray:
    points = np.asarray(points_actor_m, dtype=np.float64).reshape(-1, 3)
    return points @ frame.rotation_world_from_actor.T + frame.center_world_m[None, :]


def normalized_actor_coordinates(points_actor_m: np.ndarray, size_lwh_m: Sequence[float]) -> np.ndarray:
    points = np.asarray(points_actor_m, dtype=np.float32).reshape(-1, 3)
    half_size = np.maximum(np.asarray(size_lwh_m, dtype=np.float32).reshape(3) * 0.5, 1.0e-6)
    return points / half_size[None, :]


def denormalized_actor_coordinates(points_normalized: np.ndarray, size_lwh_m: Sequence[float]) -> np.ndarray:
    points = np.asarray(points_normalized, dtype=np.float32).reshape(-1, 3)
    half_size = np.maximum(np.asarray(size_lwh_m, dtype=np.float32).reshape(3) * 0.5, 1.0e-6)
    return points * half_size[None, :]


def split_frame_ranks(frame_ranks: Sequence[int], modulo: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """固定 `{0,1}` 为 build、`2` 为 held-out，不读取几何质量。"""
    ranks = np.asarray(frame_ranks, dtype=np.int64)
    residue = np.mod(ranks, int(modulo))
    return residue != modulo - 1, residue == modulo - 1


def immutable_actor_state(actor: Any) -> tuple[Any, ...]:
    """提取编译前后必须逐项相同的 Actor/hazard 状态。"""
    if isinstance(actor, Mapping):
        return (
            actor.get("track_id", actor.get("actor_id")),
            actor.get("trajectory_xyz_m"),
            actor.get("size_lwh_m"),
            actor.get("hazardous", actor.get("hazard_label")),
        )
    return (
        getattr(actor, "track_id", getattr(actor, "actor_id", None)),
        getattr(actor, "city_centers_m", getattr(actor, "trajectory_xyz_m", None)),
        getattr(actor, "size_lwh_m", None),
        getattr(actor, "hazardous", getattr(actor, "hazard_label", None)),
    )


def assert_actor_state_immutable(before: Any, after: Any) -> None:
    left = immutable_actor_state(before)
    right = immutable_actor_state(after)
    if left[0] != right[0] or left[3] != right[3]:
        raise ValueError("Actor identity 或 hazard state 被表面更新改变")
    for before_value, after_value, name in zip(left[1:3], right[1:3], ("trajectory", "size")):
        if before_value is None and after_value is None:
            continue
        if not np.array_equal(np.asarray(before_value), np.asarray(after_value)):
            raise ValueError(f"Actor {name} 被表面更新改变")
