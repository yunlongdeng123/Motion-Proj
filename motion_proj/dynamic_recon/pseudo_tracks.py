"""M6 对训练前冻结 pseudo-object ID 的只读连续性审计。

本模块不做事后重关联、不生成新轨迹，也不把几何启发式回填为 M6 baseline。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PseudoTrackConfig:
    """权威计划第 11.1 节冻结的对象资格门槛。"""

    n_frames: int = 60
    n_cameras: int = 3
    min_support_frames: int = 20
    min_visible_frames_one_camera: int = 10
    min_median_mask_area_px: float = 500.0

    def fingerprint(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


_PLY_SCALAR_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def read_scalar_vertex_ply(path: str | Path) -> np.ndarray:
    """读取仅含标量 vertex properties 的 PLY，不改变冻结环境。"""

    path = Path(path)
    vertex_count = None
    vertex_properties: list[tuple[str, str]] = []
    active_element = None
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"不是 PLY: {path}")
        format_line = handle.readline().decode("ascii").strip().split()
        header_lines = 2
        if len(format_line) != 3 or format_line[0] != "format":
            raise ValueError(f"PLY format 行非法: {path}")
        ply_format = format_line[1]
        while True:
            raw = handle.readline()
            header_lines += 1
            if not raw:
                raise ValueError(f"PLY header 未结束: {path}")
            line = raw.decode("ascii").strip()
            if line == "end_header":
                data_offset = handle.tell()
                break
            fields = line.split()
            if not fields or fields[0] in {"comment", "obj_info"}:
                continue
            if fields[0] == "element":
                active_element = fields[1]
                if active_element == "vertex":
                    vertex_count = int(fields[2])
            elif fields[0] == "property" and active_element == "vertex":
                if fields[1] == "list":
                    raise ValueError(f"vertex list property 不受支持: {path}")
                if fields[1] not in _PLY_SCALAR_TYPES:
                    raise ValueError(f"未知 PLY scalar type {fields[1]}: {path}")
                vertex_properties.append((fields[2], _PLY_SCALAR_TYPES[fields[1]]))
    if vertex_count is None or not vertex_properties:
        raise ValueError(f"PLY 缺少 vertex 定义: {path}")
    if ply_format == "binary_little_endian":
        dtype = np.dtype([(name, "<" + kind) for name, kind in vertex_properties])
        return np.memmap(
            path, mode="r", dtype=dtype, offset=data_offset, shape=(vertex_count,)
        )
    if ply_format == "binary_big_endian":
        dtype = np.dtype([(name, ">" + kind) for name, kind in vertex_properties])
        return np.memmap(
            path, mode="r", dtype=dtype, offset=data_offset, shape=(vertex_count,)
        )
    if ply_format == "ascii":
        matrix = np.atleast_2d(np.loadtxt(path, skiprows=header_lines))
        dtype = np.dtype(vertex_properties)
        result = np.empty(vertex_count, dtype=dtype)
        for column, (name, _) in enumerate(vertex_properties):
            result[name] = matrix[:vertex_count, column]
        return result
    raise ValueError(f"不支持的 PLY format {ply_format}: {path}")


def _mask_areas(mask: np.ndarray) -> dict[int, int]:
    ids, counts = np.unique(mask, return_counts=True)
    return {
        int(instance_id): int(count)
        for instance_id, count in zip(ids, counts)
        if int(instance_id) > 0
    }


def audit_mask_id_continuity(
    scene_dir: str | Path, config: PseudoTrackConfig | None = None
) -> dict[str, Any]:
    """审计磁盘上冻结的 SAM pseudo ID，不事后改写身份。"""

    config = config or PseudoTrackConfig()
    scene_dir = Path(scene_dir)
    mask_paths = sorted((scene_dir / "semantic").glob("mask_*.npy"))
    expected = config.n_frames * config.n_cameras
    if len(mask_paths) != expected:
        raise ValueError(f"语义 mask 数量错误: {len(mask_paths)} != {expected}")
    support: dict[tuple[int, int], set[int]] = {}
    areas: dict[tuple[int, int], list[int]] = {}
    same_frame_conflicts: list[dict[str, int]] = []
    for image_index, path in enumerate(mask_paths):
        frame, camera = divmod(image_index, config.n_cameras)
        for instance_id, area in _mask_areas(np.load(path)).items():
            key = (camera, instance_id)
            if frame in support.setdefault(key, set()):
                same_frame_conflicts.append(
                    {"camera": camera, "frame": frame, "instance_id": instance_id}
                )
            support[key].add(frame)
            areas.setdefault(key, []).append(area)
    rows = []
    for (camera, instance_id), frames in support.items():
        median_area = float(np.median(areas[(camera, instance_id)]))
        continuity_eligible = (
            len(frames) >= config.min_support_frames
            and len(frames) >= config.min_visible_frames_one_camera
            and median_area >= config.min_median_mask_area_px
        )
        rows.append(
            {
                "camera": camera,
                "instance_id": instance_id,
                "support_frames": len(frames),
                "span": [min(frames), max(frames)],
                "median_mask_area_px": median_area,
                "continuity_eligible_without_class_check": bool(continuity_eligible),
            }
        )
    rows.sort(
        key=lambda item: (
            -item["support_frames"], item["camera"], item["instance_id"]
        )
    )
    eligible = [item for item in rows if item["continuity_eligible_without_class_check"]]
    return {
        "protocol_fingerprint": config.fingerprint(),
        "pseudo_id_namespace": "camera-local SAM instance ID",
        "class_label_available_in_frozen_artifact": False,
        "identity_count": len(rows),
        "max_support_frames": max((item["support_frames"] for item in rows), default=0),
        "continuity_eligible_without_class_check_count": len(eligible),
        "vehicle_eligible_count": 0,
        "same_frame_conflicts": same_frame_conflicts,
        "top_identities": rows[:20],
        "coverage_reason": (
            "冻结 mask 未保存类别标签，且没有 ID 达到连续性门槛"
            if not eligible
            else "冻结 mask 未保存类别标签，不能证明候选为车辆"
        ),
    }
