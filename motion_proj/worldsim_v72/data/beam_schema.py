"""原生 LiDAR beam 与回波目标合同。

查询射线和监督保持分离。无回波只在 ``valid_emission_mask`` 为真时成立；
结构性无效或未知 firing 不能由零点云记录反推。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

import numpy as np

from motion_proj.worldsim_v72.data.schema import QueryRayBatch


BEAM_RETURN_SCHEMA_VERSION = "worldsim_v72.beam_returns.v1"
INVALID_RETURN_COUNT = np.int8(-1)
MAX_RETURNS = 2


def _vector(value: Any, dtype: Any, length: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if length is not None and len(array) != length:
        raise ValueError(f"向量长度 {len(array)} 与预期 {length} 不一致")
    return array


@dataclass(frozen=True)
class BeamReturnTargets:
    """逐 firing opportunity 的 0/1/2 回波监督。"""

    ray_id: np.ndarray
    target_known_mask: np.ndarray
    return_count: np.ndarray
    range_m: np.ndarray
    intensity: np.ndarray
    intensity_valid: np.ndarray
    elongation: np.ndarray
    no_label_zone: np.ndarray
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        ray_id = _vector(self.ray_id, str)
        count = len(ray_id)
        known = _vector(self.target_known_mask, bool, count)
        returns = _vector(self.return_count, np.int8, count)
        if np.any(known & ((returns < 0) | (returns > MAX_RETURNS))):
            raise ValueError("已知 beam 的 return_count 必须为 0/1/2")
        if np.any(~known & (returns != INVALID_RETURN_COUNT)):
            raise ValueError("未知或无效 beam 的 return_count 必须为 -1")
        arrays: dict[str, np.ndarray] = {}
        for name, value, dtype in (
            ("range_m", self.range_m, np.float32),
            ("intensity", self.intensity, np.float32),
            ("intensity_valid", self.intensity_valid, bool),
            ("elongation", self.elongation, np.float32),
            ("no_label_zone", self.no_label_zone, bool),
        ):
            array = np.asarray(value, dtype=dtype)
            if array.shape != (count, MAX_RETURNS):
                raise ValueError(f"{name} 必须为 (N,{MAX_RETURNS})")
            arrays[name] = array
        slot = np.arange(MAX_RETURNS)[None, :]
        active = known[:, None] & (slot < returns[:, None])
        ranges = arrays["range_m"]
        if np.any(active & (~np.isfinite(ranges) | (ranges <= 0.0))):
            raise ValueError("有效回波距离必须有限且大于零")
        if np.any(~active & np.isfinite(ranges)):
            raise ValueError("无回波 slot 的距离必须为 NaN")
        if np.any(arrays["intensity_valid"] & ~active):
            raise ValueError("无回波 slot 不能具有有效强度")
        object.__setattr__(self, "ray_id", ray_id)
        object.__setattr__(self, "target_known_mask", known)
        object.__setattr__(self, "return_count", returns)
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "provenance", dict(self.provenance))

    def __len__(self) -> int:
        return len(self.ray_id)

    @property
    def no_return_mask(self) -> np.ndarray:
        return self.target_known_mask & (self.return_count == 0)

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256(BEAM_RETURN_SCHEMA_VERSION.encode("utf-8"))
        for array in (
            self.ray_id.astype("U"),
            self.target_known_mask,
            self.return_count,
            self.range_m,
            self.intensity,
            self.intensity_valid,
            self.elongation,
            self.no_label_zone,
        ):
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
            digest.update(contiguous.tobytes())
        return digest.hexdigest()


def decode_waymo_range_images(
    *,
    range_image_return1: np.ndarray,
    range_image_return2: np.ndarray,
    ray_origins_world_m: np.ndarray,
    ray_directions_world: np.ndarray,
    valid_emission_mask: np.ndarray,
    beam_parameters: np.ndarray,
    query_pose: np.ndarray,
    log_id: str,
    frame_id: str,
    sensor_id: str,
    time_ns: int,
    range_min_m: float,
    range_max_m: float,
    firing_validity_source: str,
) -> tuple[QueryRayBatch, BeamReturnTargets]:
    """把 Waymo 的两个原生 range images 转为分离的查询和监督。

    Waymo 官方点云工具用 ``range > 0`` 选择真实回波。这里沿用该定义，
    但要求调用方单独提供 firing validity；因此 ``range <= 0`` 只有在有效
    firing 上才标为 no-return。
    """

    return1 = np.asarray(range_image_return1, dtype=np.float32)
    return2 = np.asarray(range_image_return2, dtype=np.float32)
    if return1.ndim != 3 or return1.shape[-1] < 4 or return2.shape != return1.shape:
        raise ValueError("Waymo range images 必须具有相同的 (H,W,C>=4) 形状")
    height, width = return1.shape[:2]
    origins = np.asarray(ray_origins_world_m, dtype=np.float32)
    directions = np.asarray(ray_directions_world, dtype=np.float32)
    valid = np.asarray(valid_emission_mask, dtype=bool)
    parameters = np.asarray(beam_parameters, dtype=np.float32)
    if origins.shape != (height, width, 3) or directions.shape != origins.shape:
        raise ValueError("逐 beam origin/direction 必须为 (H,W,3)")
    if valid.shape != (height, width):
        raise ValueError("valid_emission_mask 必须为 (H,W)")
    if parameters.ndim != 3 or parameters.shape[:2] != (height, width):
        raise ValueError("beam_parameters 必须为 (H,W,P)")
    if not firing_validity_source.strip():
        raise ValueError("必须记录独立于回波值的 firing_validity_source")

    count = height * width
    norm = np.linalg.norm(directions, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norm)) or np.any(norm <= 0.0):
        raise ValueError("ray_directions_world 必须有限且非零")
    directions = directions / norm
    first = valid & np.isfinite(return1[..., 0]) & (return1[..., 0] > 0.0)
    second = valid & np.isfinite(return2[..., 0]) & (return2[..., 0] > 0.0)
    if np.any(second & ~first):
        raise ValueError("Waymo 第二回波不能在第一回波缺失时单独存在")
    return_count = np.full((height, width), INVALID_RETURN_COUNT, dtype=np.int8)
    return_count[valid] = 0
    return_count[first] = 1
    return_count[second] = 2

    row, column = np.indices((height, width))
    ray_id = np.char.add(
        np.char.add(
            np.char.add(np.asarray([f"{frame_id}:{sensor_id}:"] * count), row.reshape(-1).astype(str)),
            ":",
        ),
        column.reshape(-1).astype(str),
    )
    pose = np.asarray(query_pose, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError("query_pose 必须为 (4,4)")
    queries = QueryRayBatch(
        ray_id=ray_id,
        log_id=log_id,
        frame_id=np.full(count, frame_id),
        time_ns=np.full(count, time_ns, dtype=np.int64),
        sensor_id=np.full(count, sensor_id),
        origin_m=origins.reshape(count, 3),
        unit_direction=directions.reshape(count, 3),
        range_min_m=np.full(count, range_min_m, dtype=np.float32),
        range_max_m=np.full(count, range_max_m, dtype=np.float32),
        query_pose=np.broadcast_to(pose, (count, 4, 4)).copy(),
        beam_parameters=parameters.reshape(count, parameters.shape[-1]),
        valid_emission_mask=valid.reshape(-1),
    )

    ranges = np.full((height, width, MAX_RETURNS), np.nan, dtype=np.float32)
    intensity = np.zeros_like(ranges)
    intensity_valid = np.zeros_like(ranges, dtype=bool)
    elongation = np.zeros_like(ranges)
    no_label_zone = np.zeros_like(ranges, dtype=bool)
    for slot, (image, mask) in enumerate(((return1, first), (return2, second))):
        ranges[..., slot][mask] = image[..., 0][mask]
        finite_intensity = mask & np.isfinite(image[..., 1])
        intensity[..., slot][finite_intensity] = image[..., 1][finite_intensity]
        intensity_valid[..., slot] = finite_intensity
        finite_elongation = mask & np.isfinite(image[..., 2])
        elongation[..., slot][finite_elongation] = image[..., 2][finite_elongation]
        no_label_zone[..., slot][mask] = image[..., 3][mask] > 0.0
    targets = BeamReturnTargets(
        ray_id=ray_id,
        target_known_mask=valid.reshape(-1),
        return_count=return_count.reshape(-1),
        range_m=ranges.reshape(count, MAX_RETURNS),
        intensity=intensity.reshape(count, MAX_RETURNS),
        intensity_valid=intensity_valid.reshape(count, MAX_RETURNS),
        elongation=elongation.reshape(count, MAX_RETURNS),
        no_label_zone=no_label_zone.reshape(count, MAX_RETURNS),
        provenance={
            "dataset": "Waymo Open Dataset Perception v2",
            "range_hit_rule": "finite_range_gt_0",
            "firing_validity_source": firing_validity_source,
            "target_separated_from_query": True,
        },
    )
    return queries, targets
