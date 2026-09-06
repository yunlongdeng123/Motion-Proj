"""ActorBundleV2 的分片存储；不读取或覆盖 V7.1 v1 缓存。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from motion_proj.worldsim_v72.data.schema import (
    ACTOR_BUNDLE_SCHEMA_VERSION,
    ActorBundleV2,
)


_ARRAY_FIELDS = (
    "size_lwh_m",
    "build_frame_ids",
    "build_time_ns",
    "world_from_sensor",
    "world_from_actor",
    "build_points_actor_m",
    "point_frame_id",
    "point_sensor_id",
    "build_ray_origin_actor_m",
    "build_ray_direction_actor",
    "build_range_m",
    "build_intensity",
    "build_intensity_valid",
    "beam_or_ring_id",
    "per_point_time_offset_ns",
    "evidence_fou",
    "conflict_count",
    "opportunity_count",
    "build_only_surface_m",
)


def save_actor_bundle_v2(bundle: ActorBundleV2, output_path: Path) -> tuple[Path, Path]:
    output_path = Path(output_path)
    if output_path.suffix != ".npz":
        raise ValueError("V2 Actor cache 路径必须以 .npz 结尾")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path = output_path.with_suffix(".json")
    temporary_npz = output_path.with_suffix(".tmp.npz")
    temporary_json = sidecar_path.with_suffix(".tmp.json")
    np.savez_compressed(
        temporary_npz,
        schema_version=np.asarray(ACTOR_BUNDLE_SCHEMA_VERSION),
        **{name: getattr(bundle, name) for name in _ARRAY_FIELDS},
    )
    metadata = {
        "schema_version": ACTOR_BUNDLE_SCHEMA_VERSION,
        "dataset": bundle.dataset,
        "log_id": bundle.log_id,
        "scene_id": bundle.scene_id,
        "actor_id": bundle.actor_id,
        "category": bundle.category,
        "sensor_model_ref": bundle.sensor_model_ref,
        "provenance": dict(bundle.provenance),
        "point_count": bundle.point_count,
    }
    temporary_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_npz.replace(output_path)
    temporary_json.replace(sidecar_path)
    return output_path, sidecar_path


def load_actor_bundle_v2(input_path: Path) -> ActorBundleV2:
    input_path = Path(input_path)
    sidecar_path = input_path.with_suffix(".json")
    metadata: dict[str, Any] = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != ACTOR_BUNDLE_SCHEMA_VERSION:
        raise ValueError("ActorBundleV2 JSON schema 版本不匹配")
    with np.load(input_path, allow_pickle=False) as payload:
        if str(payload["schema_version"]) != ACTOR_BUNDLE_SCHEMA_VERSION:
            raise ValueError("ActorBundleV2 NPZ schema 版本不匹配")
        arrays = {name: payload[name] for name in _ARRAY_FIELDS}
    return ActorBundleV2(
        dataset=str(metadata["dataset"]),
        log_id=str(metadata["log_id"]),
        scene_id=str(metadata["scene_id"]),
        actor_id=str(metadata["actor_id"]),
        category=str(metadata["category"]),
        sensor_model_ref=str(metadata["sensor_model_ref"]),
        provenance=dict(metadata["provenance"]),
        **arrays,
    )
