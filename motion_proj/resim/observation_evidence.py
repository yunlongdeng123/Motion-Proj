"""V7.1 分层 LiDAR 观测证据。"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping

import numpy as np

from .canonical_hash import canonical_sha256
from .safety_geometry import GridSpec, OrientedBox, voxelize_oriented_box

UNKNOWN, RAY_FREE, STATIC_OCCUPIED, DYNAMIC_OCCUPIED = 0, 1, 2, 3


def _array_hash(value: np.ndarray) -> str:
    data = np.ascontiguousarray(value)
    digest = sha256()
    digest.update(str(data.dtype).encode())
    digest.update(str(data.shape).encode())
    digest.update(data.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ObservationEvidenceFrame:
    grid: GridSpec
    base_state: np.ndarray
    observed_count: np.ndarray
    observation_age_frames: np.ndarray
    dynamic_instance_layers: Mapping[int, np.ndarray]
    source: str

    def __post_init__(self) -> None:
        if self.base_state.shape != self.grid.shape:
            raise ValueError("base_state shape 与 grid 不一致")
        if self.observed_count.shape != self.grid.shape:
            raise ValueError("observed_count shape 与 grid 不一致")
        if self.observation_age_frames.shape != self.grid.shape:
            raise ValueError("observation_age_frames shape 与 grid 不一致")
        if not set(np.unique(self.base_state)).issubset({UNKNOWN, RAY_FREE, STATIC_OCCUPIED}):
            raise ValueError("base_state 不得包含动态占据")
        for actor_id, mask in self.dynamic_instance_layers.items():
            if int(actor_id) <= 0 or mask.shape != self.grid.shape or mask.dtype != bool:
                raise ValueError("动态 layer 的 id/shape/dtype 非法")

    @classmethod
    def from_legacy_o0(
        cls,
        *,
        semantics: np.ndarray,
        mask_lidar: np.ndarray,
        instance_id: np.ndarray,
        grid: GridSpec,
        age_frames: int = 0,
    ) -> "ObservationEvidenceFrame":
        """拆分旧扁平 O0；动态 box 下的 base 恢复 UNKNOWN，绝不恢复 FREE。"""
        base = np.full(semantics.shape, UNKNOWN, dtype=np.uint8)
        base[semantics == RAY_FREE] = RAY_FREE
        base[semantics == STATIC_OCCUPIED] = STATIC_OCCUPIED
        dynamic = {
            int(actor_id): np.asarray(instance_id == actor_id, dtype=bool)
            for actor_id in np.unique(instance_id)
            if int(actor_id) > 0
        }
        counts = np.asarray(mask_lidar > 0, dtype=np.uint16)
        age = np.full(semantics.shape, np.iinfo(np.uint16).max, dtype=np.uint16)
        age[counts > 0] = np.uint16(age_frames)
        return cls(grid, base, counts, age, dynamic, "legacy-o0-separated-v2")

    def without_actor(self, actor_id: int) -> "ObservationEvidenceFrame":
        layers = {key: value.copy() for key, value in self.dynamic_instance_layers.items()}
        layers.pop(int(actor_id), None)
        return ObservationEvidenceFrame(
            self.grid,
            self.base_state.copy(),
            self.observed_count.copy(),
            self.observation_age_frames.copy(),
            layers,
            self.source,
        )

    def with_actor_box(self, actor_id: int, box: OrientedBox) -> "ObservationEvidenceFrame":
        if int(actor_id) <= 0:
            raise ValueError("actor_id 必须大于 0")
        layers = {key: value.copy() for key, value in self.dynamic_instance_layers.items()}
        layers[int(actor_id)] = voxelize_oriented_box(box, self.grid)
        return ObservationEvidenceFrame(
            self.grid,
            self.base_state.copy(),
            self.observed_count.copy(),
            self.observation_age_frames.copy(),
            layers,
            self.source,
        )

    def composite(self) -> tuple[np.ndarray, np.ndarray]:
        semantics = self.base_state.copy()
        instance_id = np.zeros(self.grid.shape, dtype=np.int32)
        # id 排序使重叠时的失败可复现；重叠本身必须由 safety geometry 报告。
        for actor_id in sorted(self.dynamic_instance_layers):
            mask = self.dynamic_instance_layers[actor_id]
            semantics[mask] = DYNAMIC_OCCUPIED
            instance_id[mask] = actor_id
        return semantics, instance_id

    def content_hash(self) -> str:
        return canonical_sha256(
            {
                "schema": "observation-evidence-v2",
                "grid": {
                    "minimum": self.grid.minimum,
                    "maximum": self.grid.maximum,
                    "voxel_size": self.grid.voxel_size,
                },
                "base_state": _array_hash(self.base_state),
                "observed_count": _array_hash(self.observed_count),
                "observation_age_frames": _array_hash(self.observation_age_frames),
                "dynamic_instance_layers": {
                    str(key): _array_hash(value)
                    for key, value in sorted(self.dynamic_instance_layers.items())
                },
                "source": self.source,
            }
        )
