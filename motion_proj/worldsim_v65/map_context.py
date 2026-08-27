"""Ego-aligned nuScenes map features for the V6.5 train-only R3 probe."""

from __future__ import annotations

import math
from pathlib import Path
from threading import Lock

import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from scipy.ndimage import distance_transform_edt


MAP_CONTEXT_FEATURE_NAMES = (
    "map_drivable_area",
    "map_road_segment",
    "map_lane",
    "map_ped_crossing",
    "map_walkway",
    "map_carpark_area",
    "map_road_divider",
    "map_lane_divider",
    "signed_drivable_distance",
    "route_mean_curvature",
    "route_max_curvature",
    "route_length",
    "route_drivable_fraction",
    "local_drivable_fraction",
)

_MAP_APIS: dict[tuple[str, str], NuScenesMap] = {}
_MAP_LOCK = Lock()


def _get_map_api(map_root: Path, location: str) -> NuScenesMap:
    key = (str(map_root), location)
    with _MAP_LOCK:
        api = _MAP_APIS.get(key)
        if api is None:
            api = NuScenesMap(dataroot=str(map_root), map_name=location)
            _MAP_APIS[key] = api
    return api


def _route_statistics(
    route_xy: np.ndarray,
    drivable: np.ndarray,
    *,
    origin_xy: np.ndarray,
    voxel_size_m: float,
) -> tuple[float, float, float, float]:
    route = np.asarray(route_xy, dtype=np.float64).reshape(-1, 2)
    route = np.concatenate((np.zeros((1, 2), dtype=np.float64), route), axis=0)
    segments = np.diff(route, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    keep = lengths > 1e-3
    segments = segments[keep]
    lengths = lengths[keep]
    route_length = float(lengths.sum()) if lengths.size else 0.0
    if lengths.size >= 2:
        headings = np.unwrap(np.arctan2(segments[:, 1], segments[:, 0]))
        turns = np.abs(np.diff(headings))
        arc = 0.5 * (lengths[:-1] + lengths[1:])
        curvature = turns / np.maximum(arc, 1e-3)
        mean_curvature = float(np.average(curvature, weights=arc))
        max_curvature = float(curvature.max())
    else:
        mean_curvature = 0.0
        max_curvature = 0.0

    route_indices = np.floor((route - origin_xy[None, :]) / voxel_size_m).astype(np.int64)
    valid = (
        (route_indices[:, 0] >= 0)
        & (route_indices[:, 0] < drivable.shape[1])
        & (route_indices[:, 1] >= 0)
        & (route_indices[:, 1] < drivable.shape[0])
    )
    if np.any(valid):
        inside = route_indices[valid]
        route_drivable = float(drivable[inside[:, 1], inside[:, 0]].mean())
    else:
        route_drivable = 0.0
    return mean_curvature, max_curvature, route_length, route_drivable


def ego_aligned_map_context(
    map_root: Path,
    map_location: str,
    processed_scene: Path,
    frame: int,
    indices: np.ndarray,
    route_xy: np.ndarray,
    *,
    origin_m: np.ndarray,
    voxel_size_m: float,
    grid_shape: tuple[int, int, int],
    layers: tuple[str, ...],
    signed_distance_clip_m: float,
) -> np.ndarray:
    """Return eight point-map channels, signed distance, and five unit features."""

    pose = np.loadtxt(processed_scene / "lidar_pose" / f"{frame:03d}.txt")
    yaw_degrees = math.degrees(math.atan2(float(pose[1, 0]), float(pose[0, 0])))
    size_x = float(grid_shape[0]) * voxel_size_m
    size_y = float(grid_shape[1]) * voxel_size_m
    map_api = _get_map_api(map_root, map_location)
    masks = map_api.get_map_mask(
        (float(pose[0, 3]), float(pose[1, 3]), size_y, size_x),
        yaw_degrees,
        list(layers),
        (int(grid_shape[1]), int(grid_shape[0])),
    ).astype(bool, copy=False)
    if masks.shape[0] != len(layers):
        raise RuntimeError(f"expected {len(layers)} map layers, received {masks.shape[0]}")

    x = np.asarray(indices[:, 0], dtype=np.int64)
    y = np.asarray(indices[:, 1], dtype=np.int64)
    point_layers = masks[:, y, x].T.astype(np.float32)
    drivable = masks[0]
    signed_distance = (
        distance_transform_edt(drivable) - distance_transform_edt(~drivable)
    ) * voxel_size_m
    signed_distance = np.clip(
        signed_distance / float(signed_distance_clip_m), -1.0, 1.0
    ).astype(np.float32)
    route_stats = _route_statistics(
        route_xy,
        drivable,
        origin_xy=np.asarray(origin_m[:2], dtype=np.float64),
        voxel_size_m=voxel_size_m,
    )
    unit_features = np.asarray(
        (*route_stats, float(drivable.mean())), dtype=np.float32
    )
    return np.concatenate(
        (
            point_layers,
            signed_distance[y, x, None],
            np.broadcast_to(unit_features, (indices.shape[0], unit_features.size)),
        ),
        axis=1,
    ).astype(np.float32, copy=False)

