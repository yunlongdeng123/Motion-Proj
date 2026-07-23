"""WorldStateSequence/Frame 与 RenderRequest 的最小 V7.1 schema。"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .canonical_hash import CANONICALIZATION_VERSION, canonical_sha256

WORLD_STATE_SCHEMA_VERSION = "v71-world-state-v1"
RENDER_REQUEST_SCHEMA_VERSION = "v71-render-request-v1"

WORLD_REQUIRED = {
    "sequence_id",
    "scene_id",
    "split",
    "data_fingerprint",
    "coordinate_convention",
    "timebase_hz",
    "reconstruction",
    "actor_registry_sha256",
    "safety_geometry_sha256",
    "observation_evidence_sha256",
    "render_support_sha256",
    "edit_spec",
    "frames",
}
FRAME_REQUIRED = {
    "frame_index",
    "timestamp",
    "timestamp_provenance",
    "T_world_ego",
    "camera_models",
    "actor_nodes",
}


class SchemaError(ValueError):
    """WorldState 或 RenderRequest 不满足冻结 schema。"""


def _sort_world_state(payload: dict[str, Any]) -> dict[str, Any]:
    output = deepcopy(payload)
    frames = output["frames"]
    if not isinstance(frames, list):
        raise SchemaError("frames 必须是数组")
    for frame in frames:
        missing = sorted(FRAME_REQUIRED - set(frame))
        if missing:
            raise SchemaError(f"WorldStateFrame 缺少字段: {missing}")
        if frame["timestamp_provenance"] not in {"observed", "interpolated"}:
            raise SchemaError("timestamp_provenance 非法")
        frame["camera_models"] = sorted(
            frame["camera_models"], key=lambda camera: str(camera["camera_id"])
        )
        frame["actor_nodes"] = sorted(
            frame["actor_nodes"], key=lambda actor: int(actor["true_instance_id"])
        )
    indices = [int(frame["frame_index"]) for frame in frames]
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise SchemaError("frames 必须按唯一 frame_index 升序排列")
    return output


def world_state_payload(value: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(WORLD_REQUIRED - set(value))
    if missing:
        raise SchemaError(f"WorldStateSequence 缺少字段: {missing}")
    forbidden = {
        "output_resolution",
        "rasterizer_mode",
        "write_path",
        "world_state_hash",
    } & set(value)
    if forbidden:
        raise SchemaError(f"WorldState 混入 render/runtime 字段: {sorted(forbidden)}")
    payload = _sort_world_state(value)
    payload["schema_version"] = WORLD_STATE_SCHEMA_VERSION
    payload["canonicalization_version"] = CANONICALIZATION_VERSION
    return payload


def world_state_hash(value: dict[str, Any]) -> str:
    return canonical_sha256(world_state_payload(value))


def render_request_payload(value: dict[str, Any]) -> dict[str, Any]:
    required = {
        "world_state_hash",
        "frame_index",
        "camera_id",
        "output_resolution",
        "rasterizer_mode",
        "precision",
        "deterministic_flags",
        "renderer_config_sha256",
        "depth_definitions",
        "alpha_first_hit_threshold",
        "compositing_policy",
        "instance_policy",
        "limited_semantic_policy",
        "color_space",
        "encoding",
    }
    missing = sorted(required - set(value))
    if missing:
        raise SchemaError(f"RenderRequest 缺少字段: {missing}")
    payload = deepcopy(value)
    payload["schema_version"] = RENDER_REQUEST_SCHEMA_VERSION
    payload["canonicalization_version"] = CANONICALIZATION_VERSION
    return payload


def render_request_hash(value: dict[str, Any]) -> str:
    return canonical_sha256(render_request_payload(value))
