"""Actor 评测产物的 fail-closed schema 辅助。"""
from __future__ import annotations

import math
from typing import Any


RAW_PROVENANCE = "nuscenes_raw_2hz"
INTERPOLATED_PROVENANCE = "visualization_interpolation_not_truth"


def finite_vector(values: Any) -> bool:
    if not isinstance(values, (list, tuple)):
        return False
    try:
        return all(math.isfinite(float(value)) for value in values)
    except (TypeError, ValueError):
        return False


def validate_raw_annotation(row: dict[str, Any]) -> None:
    required = {
        "sample_token",
        "timestamp_us",
        "translation_global",
        "size_wlh",
        "rotation_quaternion",
        "visibility_token",
        "num_lidar_pts",
        "num_radar_pts",
        "provenance",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"raw annotation 缺字段: {missing}")
    if row["provenance"] != RAW_PROVENANCE:
        raise ValueError("raw annotation provenance 被混写")
    if not finite_vector(row["translation_global"]) or len(row["translation_global"]) != 3:
        raise ValueError("translation_global 无效")
    if not finite_vector(row["size_wlh"]) or len(row["size_wlh"]) != 3:
        raise ValueError("size_wlh 无效")
    if any(float(value) <= 0 for value in row["size_wlh"]):
        raise ValueError("size_wlh 必须为正")
    if not finite_vector(row["rotation_quaternion"]) or len(row["rotation_quaternion"]) != 4:
        raise ValueError("rotation_quaternion 无效")
    if not math.isfinite(float(row["timestamp_us"])):
        raise ValueError("timestamp_us 无效")


def validate_actor_record(actor: dict[str, Any]) -> None:
    for key in ("scene_id", "instance_token", "category_name", "raw_annotations"):
        if key not in actor:
            raise ValueError(f"actor 缺字段: {key}")
    if not actor["raw_annotations"]:
        raise ValueError("actor raw_annotations 为空")
    for row in actor["raw_annotations"]:
        validate_raw_annotation(row)
    for row in actor.get("interpolated_visualization", []):
        if row.get("provenance") != INTERPOLATED_PROVENANCE:
            raise ValueError("interpolated provenance 被混写")
        if row.get("sample_token") is not None:
            raise ValueError("插值记录不得冒用 raw sample_token")
