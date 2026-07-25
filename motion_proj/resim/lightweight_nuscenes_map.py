"""N1 挖掘所需的最小 nuScenes map-expansion 读取器。

官方 ``NuScenesMap`` 在导入阶段会同时加载 OpenCV、Matplotlib、Shapely 和
完整渲染 API。N1 只使用 lane/lane_connector 的 arcline centerline 与
connectivity；本模块流式读取这四个字段，保持相同的窄接口并降低内存峰值。
"""
from __future__ import annotations

import math
from pathlib import Path

import ijson

from motion_proj.resim.io_memory import (
    advise_sequential,
    drop_handle_page_cache,
)


MAP_NAMES = {
    "boston-seaport",
    "singapore-hollandvillage",
    "singapore-onenorth",
    "singapore-queenstown",
}


def _items(path: Path, prefix: str) -> list:
    with path.open("rb") as handle:
        advise_sequential(handle)
        try:
            return list(ijson.items(handle, prefix, use_float=True))
        finally:
            drop_handle_page_cache(handle)


def _mapping(path: Path, prefix: str) -> dict:
    with path.open("rb") as handle:
        advise_sequential(handle)
        try:
            return dict(ijson.kvitems(handle, prefix, use_float=True))
        finally:
            drop_handle_page_cache(handle)


def _principal_value(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _segment_sign(path: dict) -> tuple[int, int, int]:
    shape = path["shape"]
    first = 1 if shape in {"LRL", "LSL", "LSR"} else -1
    middle = 1 if shape == "RLR" else (-1 if shape == "LRL" else 0)
    last = 1 if shape in {"LRL", "LSL", "RSL"} else -1
    return first, middle, last


def _transformation_at_step(
    pose: tuple[float, float, float],
    step: float,
) -> tuple[float, float, float]:
    theta = pose[2] * step
    if abs(pose[2]) < 1e-6:
        return pose[0] * step, pose[1] * step, theta
    cosine = math.cos(theta)
    sine = math.sin(theta)
    return (
        (pose[1] * (cosine - 1.0) + pose[0] * sine) / pose[2],
        (pose[0] * (1.0 - cosine) + pose[1] * sine) / pose[2],
        theta,
    )


def _apply_transform(
    pose: tuple[float, float, float],
    transform: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        math.cos(pose[2]) * transform[0]
        - math.sin(pose[2]) * transform[1]
        + pose[0],
        math.sin(pose[2]) * transform[0]
        + math.cos(pose[2]) * transform[1]
        + pose[1],
        _principal_value(pose[2] + transform[2]),
    )


def _lie_algebra(path: dict) -> list[tuple[float, float, float]]:
    signs = _segment_sign(path)
    radius = float(path["radius"])
    return [(1.0, 0.0, sign / radius) for sign in signs]


def _pose_at_length(path: dict, position: float) -> tuple[float, float, float]:
    segment_lengths = [float(value) for value in path["segment_length"]]
    position = max(0.0, min(position, sum(segment_lengths)))
    result = tuple(float(value) for value in path["start_pose"])
    for algebra, length in zip(_lie_algebra(path), segment_lengths):
        step = min(position, length)
        result = _apply_transform(
            result,
            _transformation_at_step(algebra, step),
        )
        if position <= length:
            break
        position -= length
    return result


def _discretize_path(
    path: dict,
    resolution_meters: float,
) -> list[tuple[float, float, float]]:
    path_length = sum(float(value) for value in path["segment_length"])
    point_count = int(max(math.ceil(path_length / resolution_meters) + 1.5, 2))
    step = path_length / (point_count - 1)
    return [_pose_at_length(path, index * step) for index in range(point_count)]


def _discretize_lane(
    lane: list[dict],
    resolution_meters: float,
) -> list[tuple[float, float, float]]:
    return [
        pose
        for path in lane
        for pose in _discretize_path(path, resolution_meters)
    ]


class LightweightNuScenesMap:
    """兼容 ``LaneIndex`` 所需属性和 ``discretize_lanes`` 的轻量地图。"""

    def __init__(self, dataroot: str | Path, map_name: str):
        if map_name not in MAP_NAMES:
            raise ValueError(f"未知 nuScenes map location: {map_name}")
        self.dataroot = str(dataroot)
        self.map_name = map_name
        self.json_path = (
            Path(dataroot) / "maps" / "expansion" / f"{map_name}.json"
        )
        if not self.json_path.is_file():
            raise FileNotFoundError(self.json_path)
        versions = _items(self.json_path, "version")
        self.version = str(versions[0]) if versions else "1.0"
        if self.version < "1.3":
            raise RuntimeError(f"nuScenes map expansion 版本过旧: {self.version}")
        self.lane = _items(self.json_path, "lane.item")
        self.lane_connector = _items(
            self.json_path,
            "lane_connector.item",
        )
        self.arcline_path_3 = _mapping(self.json_path, "arcline_path_3")
        self.connectivity = _mapping(self.json_path, "connectivity")

    def discretize_lanes(
        self,
        tokens: list[str],
        resolution_meters: float,
    ) -> dict[str, list[tuple[float, float, float]]]:
        return {
            token: _discretize_lane(
                self.arcline_path_3.get(token, []),
                resolution_meters,
            )
            for token in tokens
        }
