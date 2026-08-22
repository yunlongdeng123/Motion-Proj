"""SceneIR-O 的坐标、占据三态与 oriented actor volume 核心。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


UNKNOWN = np.uint8(0)
FREE = np.uint8(1)
OCCUPIED = np.uint8(2)


class OccupancyContractError(RuntimeError):
    """SceneIR-O 占据合同被破坏。"""


@dataclass(frozen=True)
class Transform:
    """显式命名的齐次变换 T_dst_src。"""

    dst: str
    src: str
    matrix: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise OccupancyContractError("T_dst_src 必须是有限 4x4 矩阵")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9, rtol=0.0):
            raise OccupancyContractError("T_dst_src 齐次末行非法")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.0):
            raise OccupancyContractError("T_dst_src rotation 非正交")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0.0):
            raise OccupancyContractError("T_dst_src rotation 必须为右手系")
        object.__setattr__(self, "matrix", matrix)

    def inverse(self) -> "Transform":
        return Transform(dst=self.src, src=self.dst, matrix=np.linalg.inv(self.matrix))

    def then(self, next_transform: "Transform") -> "Transform":
        if self.dst != next_transform.src:
            raise OccupancyContractError(
                f"变换不可组合: {self.dst} != {next_transform.src}"
            )
        return Transform(
            dst=next_transform.dst,
            src=self.src,
            matrix=next_transform.matrix @ self.matrix,
        )

    def apply(self, points_src: np.ndarray) -> np.ndarray:
        points = np.asarray(points_src, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise OccupancyContractError("points 必须为 N×3")
        return points @ self.matrix[:3, :3].T + self.matrix[:3, 3]


@dataclass(frozen=True)
class VoxelGridSpec:
    """目标帧中的规则体素合同。"""

    frame: str
    origin_m: tuple[float, float, float]
    voxel_size_m: float
    shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        if self.voxel_size_m <= 0.0 or any(value <= 0 for value in self.shape):
            raise OccupancyContractError("voxel size/shape 非法")
        if len(self.origin_m) != 3 or len(self.shape) != 3:
            raise OccupancyContractError("grid 必须为 3D")

    @property
    def origin(self) -> np.ndarray:
        return np.asarray(self.origin_m, dtype=np.float64)

    @property
    def extent_m(self) -> np.ndarray:
        return self.origin + self.voxel_size_m * np.asarray(self.shape, dtype=np.float64)

    def points_to_indices(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.floor((np.asarray(points, dtype=np.float64) - self.origin) / self.voxel_size_m).astype(np.int64)
        shape = np.asarray(self.shape, dtype=np.int64)
        valid = np.all((values >= 0) & (values < shape), axis=1)
        return values, valid

    def indices_to_centers(self, indices: np.ndarray) -> np.ndarray:
        return self.origin + (np.asarray(indices, dtype=np.float64) + 0.5) * self.voxel_size_m


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def array_bundle_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _box_corners(size_lwh: np.ndarray) -> np.ndarray:
    half = np.asarray(size_lwh, dtype=np.float64) / 2.0
    signs = np.asarray(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    return signs * half


def points_inside_oriented_box(
    points_dst: np.ndarray,
    t_dst_box: Transform,
    size_lwh: np.ndarray,
    margin_m: float = 0.0,
) -> np.ndarray:
    if t_dst_box.dst == t_dst_box.src:
        raise OccupancyContractError("box frame 与目标 frame 必须显式不同")
    local = (np.asarray(points_dst, dtype=np.float64) - t_dst_box.matrix[:3, 3]) @ t_dst_box.matrix[:3, :3]
    half = np.asarray(size_lwh, dtype=np.float64) / 2.0 + float(margin_m)
    return np.all(np.abs(local) <= half[None, :] + 1e-12, axis=1)


def voxelize_oriented_box(
    spec: VoxelGridSpec, t_grid_box: Transform, size_lwh: np.ndarray
) -> tuple[np.ndarray, int]:
    corners = t_grid_box.apply(_box_corners(np.asarray(size_lwh, dtype=np.float64)))
    lower = np.floor((corners.min(axis=0) - spec.origin) / spec.voxel_size_m).astype(np.int64)
    upper = np.floor((corners.max(axis=0) - spec.origin) / spec.voxel_size_m).astype(np.int64)
    lower = np.maximum(lower, 0)
    upper = np.minimum(upper, np.asarray(spec.shape, dtype=np.int64) - 1)
    if np.any(lower > upper):
        return np.empty((0, 3), dtype=np.int32), 0
    axes = [np.arange(lower[axis], upper[axis] + 1, dtype=np.int32) for axis in range(3)]
    candidate = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    centers = spec.indices_to_centers(candidate)
    inside = points_inside_oriented_box(centers, t_grid_box, np.asarray(size_lwh, dtype=np.float64))
    return candidate[inside], int(candidate.shape[0])


def load_lidar(path: Path, record_width: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if record_width < 3 or raw.size % record_width != 0:
        raise OccupancyContractError(f"LiDAR record width 漂移: {path}")
    return raw.reshape(-1, record_width)[:, :3].astype(np.float64)


def load_frame_boxes(
    scene_root: Path,
    frame: int,
    t_target_global: Transform,
    instances: Mapping[str, Any] | None = None,
    frame_instances: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if instances is None:
        instances = json.loads((scene_root / "instances/instances_info.json").read_text(encoding="utf-8"))
    if frame_instances is None:
        frame_instances = json.loads((scene_root / "instances/frame_instances.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for actor_id in sorted((int(value) for value in frame_instances[str(frame)])):
        info = instances[str(actor_id)]
        annotations = info["frame_annotations"]
        indices = [int(value) for value in annotations["frame_idx"]]
        try:
            annotation_index = indices.index(int(frame))
        except ValueError as error:
            raise OccupancyContractError(f"actor {actor_id} lifecycle 与 annotation 不一致") from error
        t_global_box = Transform(
            dst="global",
            src=f"actor_{actor_id:04d}_box",
            matrix=np.asarray(annotations["obj_to_world"][annotation_index], dtype=np.float64),
        )
        t_target_box = t_global_box.then(t_target_global)
        rows.append(
            {
                "actor_id": actor_id,
                "class_name": str(info["class_name"]),
                "size_lwh": np.asarray(annotations["box_size"][annotation_index], dtype=np.float64),
                "transform": t_target_box,
            }
        )
    return rows


def _mark_free_rays(
    semantics: np.ndarray,
    spec: VoxelGridSpec,
    sensor_origin: np.ndarray,
    endpoints: np.ndarray,
    maximum_rays: int,
    maximum_range_m: float,
) -> int:
    if endpoints.shape[0] == 0:
        return 0
    count = min(int(maximum_rays), int(endpoints.shape[0]))
    selection = np.linspace(0, endpoints.shape[0] - 1, num=count, dtype=np.int64)
    selected = endpoints[selection]
    delta = selected - sensor_origin[None, :]
    distances = np.linalg.norm(delta, axis=1)
    valid = (distances > spec.voxel_size_m) & np.isfinite(distances)
    delta = delta[valid]
    distances = np.minimum(distances[valid], float(maximum_range_m))
    directions = delta / np.linalg.norm(delta, axis=1, keepdims=True)
    carved = 0
    maximum_steps = int(np.ceil(float(maximum_range_m) / spec.voxel_size_m))
    for step in range(1, maximum_steps):
        distance = step * spec.voxel_size_m
        active = distance < distances - 0.5 * spec.voxel_size_m
        if not np.any(active):
            break
        points = sensor_origin[None, :] + directions[active] * distance
        indices, inside = spec.points_to_indices(points)
        indices = indices[inside]
        if indices.shape[0] == 0:
            continue
        unique = np.unique(indices, axis=0)
        values = semantics[unique[:, 0], unique[:, 1], unique[:, 2]]
        free_indices = unique[values == UNKNOWN]
        semantics[free_indices[:, 0], free_indices[:, 1], free_indices[:, 2]] = FREE
        carved += int(free_indices.shape[0])
    return carved


def build_observed_occupancy(
    scene_root: Path,
    target_frame: int,
    source_frames: Iterable[int],
    spec: VoxelGridSpec,
    *,
    record_width: int,
    dynamic_box_margin_m: float,
    maximum_free_rays_per_sweep: int,
    maximum_range_m: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    scene_root = scene_root.resolve()
    target_pose_path = scene_root / f"lidar_pose/{int(target_frame):03d}.txt"
    t_global_target = Transform(dst="global", src=spec.frame, matrix=np.loadtxt(target_pose_path))
    t_target_global = t_global_target.inverse()
    semantics = np.zeros(spec.shape, dtype=np.uint8)
    instances = json.loads((scene_root / "instances/instances_info.json").read_text(encoding="utf-8"))
    frame_instances = json.loads((scene_root / "instances/frame_instances.json").read_text(encoding="utf-8"))
    shared_metadata_paths = [
        target_pose_path,
        scene_root / "instances/instances_info.json",
        scene_root / "instances/frame_instances.json",
    ]
    source_rows: list[dict[str, Any]] = []
    transform_errors: list[float] = []
    raw_points = 0
    dynamic_removed = 0
    static_hits = 0
    carved_writes = 0

    for source_frame in source_frames:
        lidar_path = scene_root / f"lidar/{int(source_frame):03d}.bin"
        pose_path = scene_root / f"lidar_pose/{int(source_frame):03d}.txt"
        points_source = load_lidar(lidar_path, record_width)
        t_global_source = Transform(
            dst="global", src=f"lidar_{int(source_frame):03d}", matrix=np.loadtxt(pose_path)
        )
        t_target_source = t_global_source.then(t_target_global)
        points_target = t_target_source.apply(points_source)
        sample = points_source[: min(256, points_source.shape[0])]
        roundtrip = t_target_source.inverse().apply(points_target[: sample.shape[0]])
        transform_errors.append(float(np.max(np.abs(roundtrip - sample))) if sample.size else 0.0)
        boxes = load_frame_boxes(
            scene_root,
            int(source_frame),
            t_target_global,
            instances=instances,
            frame_instances=frame_instances,
        )
        dynamic = np.zeros(points_target.shape[0], dtype=bool)
        for box in boxes:
            dynamic |= points_inside_oriented_box(
                points_target,
                box["transform"],
                box["size_lwh"],
                margin_m=dynamic_box_margin_m,
            )
        carved_writes += _mark_free_rays(
            semantics,
            spec,
            t_target_source.matrix[:3, 3],
            points_target,
            maximum_free_rays_per_sweep,
            maximum_range_m,
        )
        static_points = points_target[~dynamic]
        indices, valid = spec.points_to_indices(static_points)
        unique_hits = np.unique(indices[valid], axis=0)
        semantics[unique_hits[:, 0], unique_hits[:, 1], unique_hits[:, 2]] = OCCUPIED
        raw_points += int(points_target.shape[0])
        dynamic_removed += int(dynamic.sum())
        static_hits += int(unique_hits.shape[0])
        source_rows.extend(
            [
                {
                    "kind": "lidar",
                    "frame": int(source_frame),
                    "path": str(lidar_path),
                    "bytes": lidar_path.stat().st_size,
                    "sha256": sha256_file(lidar_path),
                },
                {
                    "kind": "lidar_pose",
                    "frame": int(source_frame),
                    "path": str(pose_path),
                    "bytes": pose_path.stat().st_size,
                    "sha256": sha256_file(pose_path),
                },
            ]
        )

    target_boxes = load_frame_boxes(
        scene_root,
        int(target_frame),
        t_target_global,
        instances=instances,
        frame_instances=frame_instances,
    )
    actor_indices: list[np.ndarray] = []
    actor_ids: list[np.ndarray] = []
    actor_rows: list[dict[str, Any]] = []
    strict_aabb_reduction_count = 0
    for box in target_boxes:
        indices, aabb_count = voxelize_oriented_box(spec, box["transform"], box["size_lwh"])
        if indices.shape[0] < aabb_count:
            strict_aabb_reduction_count += 1
        if indices.shape[0] > 0:
            semantics[indices[:, 0], indices[:, 1], indices[:, 2]] = UNKNOWN
            actor_indices.append(indices.astype(np.int32))
            actor_ids.append(np.full(indices.shape[0], int(box["actor_id"]), dtype=np.int32))
        actor_rows.append(
            {
                "actor_id": int(box["actor_id"]),
                "class_name": box["class_name"],
                "size_lwh": box["size_lwh"].tolist(),
                "t_grid_box": box["transform"].matrix.tolist(),
                "oriented_voxel_count": int(indices.shape[0]),
                "corner_aabb_voxel_count": int(aabb_count),
                "lifecycle_active": True,
            }
        )

    sparse_indices = (
        np.concatenate(actor_indices, axis=0) if actor_indices else np.empty((0, 3), dtype=np.int32)
    )
    sparse_ids = (
        np.concatenate(actor_ids, axis=0) if actor_ids else np.empty((0,), dtype=np.int32)
    )
    arrays = {
        "static_semantics": semantics,
        "actor_voxel_indices": sparse_indices,
        "actor_instance_ids": sparse_ids,
        "grid_origin_m": np.asarray(spec.origin_m, dtype=np.float64),
        "voxel_size_m": np.asarray(spec.voxel_size_m, dtype=np.float64),
        "grid_shape": np.asarray(spec.shape, dtype=np.int64),
    }
    state_counts = {
        "unknown": int(np.count_nonzero(semantics == UNKNOWN)),
        "free": int(np.count_nonzero(semantics == FREE)),
        "occupied": int(np.count_nonzero(semantics == OCCUPIED)),
    }
    audit = {
        "target_frame": int(target_frame),
        "source_frames": [int(value) for value in source_frames],
        "source_files": source_rows,
        "shared_metadata_files": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in shared_metadata_paths
        ],
        "source_payload_sha256": content_sha256(
            [{"path": row["path"], "sha256": row["sha256"]} for row in source_rows]
        ),
        "content_sha256": array_bundle_sha256(arrays),
        "state_counts": state_counts,
        "raw_point_count": raw_points,
        "dynamic_removed_point_count": dynamic_removed,
        "static_hit_voxel_writes": static_hits,
        "free_voxel_writes": carved_writes,
        "coordinate_roundtrip_max_abs_m": max(transform_errors, default=0.0),
        "actor_count": len(actor_rows),
        "actor_rows": actor_rows,
        "actor_identity_unique_count": len({row["actor_id"] for row in actor_rows}),
        "strict_aabb_reduction_actor_count": strict_aabb_reduction_count,
        "source_removal_unknown_count": int(
            np.count_nonzero(
                semantics[sparse_indices[:, 0], sparse_indices[:, 1], sparse_indices[:, 2]]
                == UNKNOWN
            )
        )
        if sparse_indices.shape[0]
        else 0,
        "actor_sparse_voxel_count": int(sparse_indices.shape[0]),
    }
    return arrays, audit
