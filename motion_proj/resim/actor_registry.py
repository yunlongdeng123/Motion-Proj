"""V7.1 版本化 actor registry。"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any

from .canonical_hash import canonical_sha256

ACTOR_REGISTRY_SCHEMA_VERSION = "v71-actor-registry-v1"
MAPPING_ALGORITHM_VERSION = "drivestudio-get-init-objects-order-v1"


class ActorRegistryError(ValueError):
    """actor identity 映射不满足一一对应约束。"""


def _duplicates(values) -> list:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def build_actor_registry(
    *,
    scene_id: str,
    checkpoint_sha256: str,
    actors: list[dict[str, Any]],
    checkpoint_num_instances: int,
) -> dict[str, Any]:
    if checkpoint_num_instances != len(actors):
        raise ActorRegistryError(
            f"checkpoint actor 数 {checkpoint_num_instances} 与映射数 {len(actors)} 不一致"
        )
    required = {
        "true_instance_id",
        "dataset_instance_column",
        "rigid_model_index",
        "occupancy_instance_id",
        "limited_label_id",
        "class_name",
        "canonical_dimensions_lwh",
        "first_frame",
        "last_frame",
        "active_frame_count",
        "lidar_point_count",
    }
    normalized = []
    for actor in actors:
        missing = sorted(required - set(actor))
        if missing:
            raise ActorRegistryError(f"actor row 缺少字段: {missing}")
        row = deepcopy(actor)
        row["true_instance_id"] = int(row["true_instance_id"])
        row["dataset_instance_column"] = int(row["dataset_instance_column"])
        row["rigid_model_index"] = int(row["rigid_model_index"])
        row["occupancy_instance_id"] = int(row["occupancy_instance_id"])
        row["limited_label_id"] = int(row["limited_label_id"])
        row["canonical_dimensions_lwh"] = [
            float(value) for value in row["canonical_dimensions_lwh"]
        ]
        if len(row["canonical_dimensions_lwh"]) != 3:
            raise ActorRegistryError("canonical_dimensions_lwh 必须有 3 项")
        if any(value <= 0 for value in row["canonical_dimensions_lwh"]):
            raise ActorRegistryError("actor dimensions 必须为正")
        row["mapping_algorithm_version"] = MAPPING_ALGORITHM_VERSION
        row["checkpoint_sha256"] = checkpoint_sha256
        row["mapping_validation"] = "PASS"
        normalized.append(row)
    normalized.sort(key=lambda row: row["rigid_model_index"])

    identity_fields = (
        "true_instance_id",
        "dataset_instance_column",
        "rigid_model_index",
        "occupancy_instance_id",
        "limited_label_id",
    )
    conflicts = {
        field: _duplicates(row[field] for row in normalized)
        for field in identity_fields
    }
    conflicts = {field: values for field, values in conflicts.items() if values}
    if conflicts:
        raise ActorRegistryError(f"actor registry 非一一映射: {conflicts}")
    expected_indices = list(range(checkpoint_num_instances))
    actual_indices = [row["rigid_model_index"] for row in normalized]
    if actual_indices != expected_indices:
        raise ActorRegistryError(
            f"RigidNodes model index 不连续: {actual_indices} != {expected_indices}"
        )
    registry = {
        "schema_version": ACTOR_REGISTRY_SCHEMA_VERSION,
        "mapping_algorithm_version": MAPPING_ALGORITHM_VERSION,
        "scene_id": str(scene_id),
        "checkpoint_sha256": checkpoint_sha256,
        "actor_count": len(normalized),
        "actors": normalized,
    }
    registry["actor_registry_sha256"] = canonical_sha256(registry)
    return registry


def validate_registry_hash(registry: dict[str, Any]) -> None:
    expected = registry.get("actor_registry_sha256")
    payload = {key: value for key, value in registry.items() if key != "actor_registry_sha256"}
    if expected != canonical_sha256(payload):
        raise ActorRegistryError("actor_registry_sha256 不匹配")


def require_actor(registry: dict[str, Any], true_instance_id: int) -> dict[str, Any]:
    validate_registry_hash(registry)
    matches = [
        actor
        for actor in registry["actors"]
        if actor["true_instance_id"] == int(true_instance_id)
    ]
    if len(matches) != 1:
        raise ActorRegistryError(
            f"actor {true_instance_id} 在 registry 中必须恰好映射一次，实际 {len(matches)}"
        )
    return matches[0]
