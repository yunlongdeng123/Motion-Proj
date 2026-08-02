"""nuScenes raw 2 Hz actor chain 构建与三相机投影适配。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .actor_projection import project_box
from .frame_mapping import CAMERAS, nearest_camera_frame, validate_frame_table
from .schema import RAW_PROVENANCE, validate_actor_record


def load_table(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def stream_filter_tokens(
    path: Path, wanted: set[str], token_field: str = "token"
) -> dict[str, dict[str, Any]]:
    result = {}
    for row in iter_table(path):
        token = row[token_field]
        if token in wanted:
            result[token] = row
            if len(result) == len(wanted):
                break
    missing = sorted(wanted - set(result))
    if missing:
        raise ValueError(f"metadata token 缺失 {path.name}: {missing[:5]}")
    return result


def stream_filter_rows(
    path: Path, wanted: set[str], field: str
) -> list[dict[str, Any]]:
    """按非唯一外键流式过滤；不依赖 devkit 运行时反向索引。"""

    result = []
    seen = set()
    for row in iter_table(path):
        if row[field] in wanted:
            result.append(row)
            seen.add(row[field])
    missing = sorted(wanted - seen)
    if missing:
        raise ValueError(f"metadata 外键无记录 {path.name}.{field}: {missing[:5]}")
    return result


def iter_table(path: Path):
    try:
        import ijson
    except ImportError:
        if path.stat().st_size > 64 * 1024 * 1024:
            raise RuntimeError(f"大型 metadata 流式读取需要 ijson: {path}")
        yield from load_table(path)
        return
    with path.open("rb") as handle:
        # ijson 默认把 JSON float 解析为 Decimal，会污染后续运行合同的 JSON。
        yield from ijson.items(handle, "item", use_float=True)


def scene_clip_samples(
    samples: Iterable[dict[str, Any]],
    scene_token: str,
    min_timestamp_us: int,
    max_timestamp_us: int,
) -> list[dict[str, Any]]:
    result = sorted(
        (
            row
            for row in samples
            if row["scene_token"] == scene_token
            and min_timestamp_us <= int(row["timestamp"]) <= max_timestamp_us
        ),
        key=lambda row: int(row["timestamp"]),
    )
    if not result:
        raise ValueError("场景 clip 无 raw sample")
    tokens = {row["token"] for row in result}
    if len(tokens) != len(result):
        raise ValueError("raw sample token 重复")
    return result


def validate_selected_chain(
    rows: list[dict[str, Any]], sample_timestamp: dict[str, int]
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: sample_timestamp[row["sample_token"]])
    for current, following in zip(ordered, ordered[1:]):
        if current["next"] != following["token"] or following["prev"] != current["token"]:
            raise ValueError(
                f"instance_token 链断裂: {current['token']} -> {following['token']}"
            )
    return ordered


def build_actor_candidates(
    scene_name: str,
    scene_token: str,
    frame_rows: list[dict[str, Any]],
    clip_samples: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    instance_by_token: dict[str, dict[str, Any]],
    category_by_token: dict[str, dict[str, Any]],
    sample_data_by_token: dict[str, dict[str, Any]],
    calibrated_by_token: dict[str, dict[str, Any]],
    ego_pose_by_token: dict[str, dict[str, Any]],
    max_camera_delta_us: int,
) -> list[dict[str, Any]]:
    if not annotations:
        raise ValueError(f"{scene_name} clip 缺 sample_annotation")
    sample_by_token = {row["token"]: row for row in clip_samples}
    sample_timestamp = {token: int(row["timestamp"]) for token, row in sample_by_token.items()}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for annotation in annotations:
        if annotation["sample_token"] not in sample_by_token:
            raise ValueError(f"annotation 越出 scene clip: {annotation['token']}")
        grouped.setdefault(annotation["instance_token"], []).append(annotation)
    validate_frame_table(frame_rows)

    actors = []
    for instance_token, instance_annotations in sorted(grouped.items()):
        instance = instance_by_token.get(instance_token)
        if instance is None:
            raise ValueError(f"instance metadata 缺失: {instance_token}")
        category = category_by_token.get(instance["category_token"])
        if category is None:
            raise ValueError(f"category metadata 缺失: {instance['category_token']}")
        ordered = validate_selected_chain(instance_annotations, sample_timestamp)
        raw_annotations = []
        camera_observations = []
        for annotation in ordered:
            raw = {
                "annotation_token": annotation["token"],
                "instance_token": instance_token,
                "sample_token": annotation["sample_token"],
                "timestamp_us": sample_timestamp[annotation["sample_token"]],
                "translation_global": annotation["translation"],
                "size_wlh": annotation["size"],
                "rotation_quaternion": annotation["rotation"],
                "visibility_token": annotation["visibility_token"],
                "num_lidar_pts": int(annotation["num_lidar_pts"]),
                "num_radar_pts": int(annotation["num_radar_pts"]),
                "provenance": RAW_PROVENANCE,
            }
            raw_annotations.append(raw)
            for camera in CAMERAS:
                mapping = nearest_camera_frame(
                    raw["timestamp_us"],
                    raw["sample_token"],
                    camera,
                    frame_rows,
                    sample_data_by_token,
                    max_camera_delta_us,
                )
                calibrated = calibrated_by_token[mapping["calibrated_sensor_token"]]
                ego_pose = ego_pose_by_token[mapping["ego_pose_token"]]
                projection = project_box(raw, calibrated, ego_pose, 1600, 900)
                camera_observations.append(
                    {
                        "annotation_token": raw["annotation_token"],
                        **mapping,
                        "projection": projection,
                    }
                )
        actor = {
            "scene_id": scene_name,
            "scene_token": scene_token,
            "instance_token": instance_token,
            "category_name": category["name"],
            "raw_annotations": raw_annotations,
            "camera_observations": camera_observations,
            "interpolated_visualization": [],
            "support_summary": {},
        }
        validate_actor_record(actor)
        actors.append(actor)
    return actors
