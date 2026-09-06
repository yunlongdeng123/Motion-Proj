"""V7.2 的 Actor、查询射线与监督目标数据合同。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


ACTOR_BUNDLE_SCHEMA_VERSION = "worldsim_v72.actor_bundle.v2"


def _array(value: Any, dtype: Any, trailing_shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 1 + len(trailing_shape) or array.shape[1:] != trailing_shape:
        raise ValueError(f"数组形状必须为 (N, {', '.join(map(str, trailing_shape))})")
    return array


def _vector(value: Any, dtype: Any, length: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if length is not None and len(array) != length:
        raise ValueError(f"向量长度 {len(array)} 与预期 {length} 不一致")
    return array


def _validate_unit_vectors(vectors: np.ndarray, *, name: str) -> None:
    if len(vectors) == 0:
        return
    norms = np.linalg.norm(vectors, axis=1)
    if not np.all(np.isfinite(norms)) or not np.allclose(norms, 1.0, atol=1.0e-4):
        raise ValueError(f"{name} 必须是有限单位向量")


@dataclass(frozen=True)
class QueryRayBatch:
    """部署可见的查询射线；类型中不包含任何 target 字段。"""

    ray_id: np.ndarray
    log_id: str
    frame_id: np.ndarray
    time_ns: np.ndarray
    sensor_id: np.ndarray
    origin_m: np.ndarray
    unit_direction: np.ndarray
    range_min_m: np.ndarray
    range_max_m: np.ndarray
    query_pose: np.ndarray
    beam_parameters: np.ndarray
    valid_emission_mask: np.ndarray

    def __post_init__(self) -> None:
        origins = _array(self.origin_m, np.float32, (3,))
        count = len(origins)
        directions = _array(self.unit_direction, np.float32, (3,))
        poses = _array(self.query_pose, np.float64, (4, 4))
        beams = np.asarray(self.beam_parameters, dtype=np.float32)
        if beams.ndim != 2:
            raise ValueError("beam_parameters 必须为二维数组")
        for name, array in (
            ("unit_direction", directions),
            ("query_pose", poses),
            ("beam_parameters", beams),
        ):
            if len(array) != count:
                raise ValueError(f"{name} 的射线数量不一致")
        range_min = _vector(self.range_min_m, np.float32, count)
        range_max = _vector(self.range_max_m, np.float32, count)
        if np.any(range_min < 0.0) or np.any(range_max <= range_min):
            raise ValueError("查询量程必须满足 0 <= range_min < range_max")
        _validate_unit_vectors(directions, name="unit_direction")
        object.__setattr__(self, "ray_id", _vector(self.ray_id, str, count))
        object.__setattr__(self, "frame_id", _vector(self.frame_id, str, count))
        object.__setattr__(self, "time_ns", _vector(self.time_ns, np.int64, count))
        object.__setattr__(self, "sensor_id", _vector(self.sensor_id, str, count))
        object.__setattr__(self, "origin_m", origins)
        object.__setattr__(self, "unit_direction", directions)
        object.__setattr__(self, "range_min_m", range_min)
        object.__setattr__(self, "range_max_m", range_max)
        object.__setattr__(self, "query_pose", poses)
        object.__setattr__(self, "beam_parameters", beams)
        object.__setattr__(
            self,
            "valid_emission_mask",
            _vector(self.valid_emission_mask, bool, count),
        )

    def __len__(self) -> int:
        return len(self.ray_id)


@dataclass(frozen=True)
class RayTargets:
    """训练与评估专用的回波目标，与 QueryRayBatch 分离。"""

    return_valid: np.ndarray
    return_index: np.ndarray
    range_m: np.ndarray
    intensity: np.ndarray
    intensity_valid: np.ndarray
    censoring_or_missing_mask: np.ndarray
    object_id_for_evaluation_only: np.ndarray

    def __post_init__(self) -> None:
        valid = _vector(self.return_valid, bool)
        count = len(valid)
        indices = _vector(self.return_index, np.int32, count)
        ranges = _vector(self.range_m, np.float32, count)
        intensity = _vector(self.intensity, np.float32, count)
        intensity_valid = _vector(self.intensity_valid, bool, count)
        censored = _vector(self.censoring_or_missing_mask, bool, count)
        object_ids = _vector(self.object_id_for_evaluation_only, str, count)
        if np.any(valid & censored):
            raise ValueError("有效回波不能同时标为裁剪或缺测")
        if np.any(valid & (~np.isfinite(ranges) | (ranges < 0.0))):
            raise ValueError("有效回波必须具有有限非负距离")
        if np.any(valid & (indices < 0)):
            raise ValueError("有效回波必须具有非负 return_index")
        object.__setattr__(self, "return_valid", valid)
        object.__setattr__(self, "return_index", indices)
        object.__setattr__(self, "range_m", ranges)
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "intensity_valid", intensity_valid)
        object.__setattr__(self, "censoring_or_missing_mask", censored)
        object.__setattr__(self, "object_id_for_evaluation_only", object_ids)

    def __len__(self) -> int:
        return len(self.return_valid)


@dataclass(frozen=True)
class SurfaceTargets:
    """仅监督端可见的表面与已知空间标签。"""

    surface_points_m: np.ndarray
    frame_id: np.ndarray
    normals: np.ndarray
    normal_valid: np.ndarray
    free_space_queries_m: np.ndarray
    occupied_queries_m: np.ndarray
    known_label_mask: np.ndarray
    label_provenance: np.ndarray
    confidence_or_sensor_tolerance_m: np.ndarray

    def __post_init__(self) -> None:
        points = _array(self.surface_points_m, np.float32, (3,))
        count = len(points)
        normals = _array(self.normals, np.float32, (3,))
        if len(normals) != count:
            raise ValueError("surface normals 与 points 数量不一致")
        normal_valid = _vector(self.normal_valid, bool, count)
        if np.any(normal_valid):
            _validate_unit_vectors(normals[normal_valid], name="有效 surface normals")
        free = _array(self.free_space_queries_m, np.float32, (3,))
        occupied = _array(self.occupied_queries_m, np.float32, (3,))
        known = _vector(self.known_label_mask, bool, len(free) + len(occupied))
        object.__setattr__(self, "surface_points_m", points)
        object.__setattr__(self, "frame_id", _vector(self.frame_id, str, count))
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "normal_valid", normal_valid)
        object.__setattr__(self, "free_space_queries_m", free)
        object.__setattr__(self, "occupied_queries_m", occupied)
        object.__setattr__(self, "known_label_mask", known)
        object.__setattr__(
            self,
            "label_provenance",
            _vector(self.label_provenance, str, len(known)),
        )
        object.__setattr__(
            self,
            "confidence_or_sensor_tolerance_m",
            _vector(self.confidence_or_sensor_tolerance_m, np.float32, len(known)),
        )


@dataclass(frozen=True)
class ActorBundleV2:
    """由真实逐点来源构成的 build-only Actor 资产。"""

    dataset: str
    log_id: str
    scene_id: str
    actor_id: str
    category: str
    size_lwh_m: np.ndarray
    build_frame_ids: np.ndarray
    build_time_ns: np.ndarray
    world_from_sensor: np.ndarray
    world_from_actor: np.ndarray
    build_points_actor_m: np.ndarray
    point_frame_id: np.ndarray
    point_sensor_id: np.ndarray
    build_ray_origin_actor_m: np.ndarray
    build_ray_direction_actor: np.ndarray
    build_range_m: np.ndarray
    build_intensity: np.ndarray
    build_intensity_valid: np.ndarray
    sensor_model_ref: str
    beam_or_ring_id: np.ndarray
    per_point_time_offset_ns: np.ndarray
    evidence_fou: np.ndarray
    conflict_count: np.ndarray
    opportunity_count: np.ndarray
    build_only_surface_m: np.ndarray
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        size = np.asarray(self.size_lwh_m, dtype=np.float32).reshape(3)
        if np.any(~np.isfinite(size)) or np.any(size <= 0.0):
            raise ValueError("Actor 尺寸必须为有限正数")
        frames = _vector(self.build_frame_ids, str)
        frame_count = len(frames)
        times = _vector(self.build_time_ns, np.int64, frame_count)
        sensor_poses = _array(self.world_from_sensor, np.float64, (4, 4))
        actor_poses = _array(self.world_from_actor, np.float64, (4, 4))
        if len(sensor_poses) != frame_count or len(actor_poses) != frame_count:
            raise ValueError("逐帧 pose 与 build_frame_ids 数量不一致")
        points = _array(self.build_points_actor_m, np.float32, (3,))
        point_count = len(points)
        origins = _array(self.build_ray_origin_actor_m, np.float32, (3,))
        directions = _array(self.build_ray_direction_actor, np.float32, (3,))
        evidence = _array(self.evidence_fou, np.float32, (3,))
        for name, array in (
            ("build_ray_origin_actor_m", origins),
            ("build_ray_direction_actor", directions),
            ("evidence_fou", evidence),
        ):
            if len(array) != point_count:
                raise ValueError(f"{name} 与 build points 数量不一致")
        ranges = _vector(self.build_range_m, np.float32, point_count)
        if np.any(~np.isfinite(ranges)) or np.any(ranges < 0.0):
            raise ValueError("build range 必须为有限非负数")
        _validate_unit_vectors(directions, name="build_ray_direction_actor")
        if point_count and not np.allclose(
            np.linalg.norm(points - origins, axis=1), ranges, atol=2.0e-3
        ):
            raise ValueError("build point、ray origin 与 range 不一致")
        if np.any(evidence < 0.0) or not np.allclose(evidence.sum(axis=1), 1.0, atol=1.0e-4):
            raise ValueError("evidence_fou 必须为非负且逐行和为 1")
        object.__setattr__(self, "size_lwh_m", size)
        object.__setattr__(self, "build_frame_ids", frames)
        object.__setattr__(self, "build_time_ns", times)
        object.__setattr__(self, "world_from_sensor", sensor_poses)
        object.__setattr__(self, "world_from_actor", actor_poses)
        object.__setattr__(self, "build_points_actor_m", points)
        object.__setattr__(self, "point_frame_id", _vector(self.point_frame_id, str, point_count))
        object.__setattr__(self, "point_sensor_id", _vector(self.point_sensor_id, str, point_count))
        object.__setattr__(self, "build_ray_origin_actor_m", origins)
        object.__setattr__(self, "build_ray_direction_actor", directions)
        object.__setattr__(self, "build_range_m", ranges)
        object.__setattr__(self, "build_intensity", _vector(self.build_intensity, np.float32, point_count))
        object.__setattr__(self, "build_intensity_valid", _vector(self.build_intensity_valid, bool, point_count))
        object.__setattr__(self, "beam_or_ring_id", _vector(self.beam_or_ring_id, str, point_count))
        object.__setattr__(self, "per_point_time_offset_ns", _vector(self.per_point_time_offset_ns, np.int64, point_count))
        object.__setattr__(self, "evidence_fou", evidence)
        object.__setattr__(self, "conflict_count", _vector(self.conflict_count, np.int32, point_count))
        object.__setattr__(self, "opportunity_count", _vector(self.opportunity_count, np.int32, point_count))
        object.__setattr__(self, "build_only_surface_m", _array(self.build_only_surface_m, np.float32, (3,)))
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def point_count(self) -> int:
        return len(self.build_points_actor_m)
