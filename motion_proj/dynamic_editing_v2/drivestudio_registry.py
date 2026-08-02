"""Fail-closed nuScenes-token registry for DriveStudio RigidNodes checkpoints."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "dr-v2-drivestudio-actor-registry-v2"
MAPPING_ALGORITHM = "processed-token-to-visible-column-to-init-order-to-point-ids-v2"


class DriveStudioRegistryError(ValueError):
    """Raised when any identity link is ambiguous or inconsistent."""


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _duplicates(values: Sequence[Any]) -> list[Any]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _ranges(indices: Sequence[int]) -> list[list[int]]:
    if not indices:
        return []
    result: list[list[int]] = []
    start = previous = int(indices[0])
    for raw in indices[1:]:
        value = int(raw)
        if value != previous + 1:
            result.append([start, previous + 1])
            start = value
        previous = value
    result.append([start, previous + 1])
    return result


def build_drivestudio_registry(
    *,
    scene_id: str,
    scene_name: str,
    checkpoint_sha256: str,
    processed_instances: Mapping[str, Mapping[str, Any]],
    raw_instance_chains: Mapping[str, Mapping[str, Any]],
    dataset_true_ids: Sequence[int],
    ordered_init_columns: Sequence[int],
    checkpoint_point_ids: Sequence[int],
    checkpoint_instance_count: int,
) -> dict[str, Any]:
    """Build the complete token→checkpoint mapping used by an edit run."""
    if checkpoint_instance_count != len(ordered_init_columns):
        raise DriveStudioRegistryError(
            "checkpoint instance count does not match RigidNodes initialization order: "
            f"{checkpoint_instance_count} != {len(ordered_init_columns)}"
        )
    point_id_values = {int(value) for value in checkpoint_point_ids}
    invalid_point_ids = sorted(
        value
        for value in point_id_values
        if value < 0 or value >= checkpoint_instance_count
    )
    if invalid_point_ids:
        raise DriveStudioRegistryError(
            f"checkpoint point_ids are outside the model index range: {invalid_point_ids}"
        )

    actors = []
    for model_index, raw_column in enumerate(ordered_init_columns):
        column = int(raw_column)
        if column < 0 or column >= len(dataset_true_ids):
            raise DriveStudioRegistryError(f"dataset column out of bounds: {column}")
        true_id = int(dataset_true_ids[column])
        processed = processed_instances.get(str(true_id))
        if processed is None:
            raise DriveStudioRegistryError(f"processed true id {true_id} is missing")
        token = str(processed.get("id", ""))
        if not token:
            raise DriveStudioRegistryError(f"processed true id {true_id} has no source token")
        chain = raw_instance_chains.get(token)
        if chain is None:
            raise DriveStudioRegistryError(f"raw annotation chain missing for token {token}")
        indices = [
            index
            for index, point_id in enumerate(checkpoint_point_ids)
            if int(point_id) == model_index
        ]
        frames = [int(value) for value in processed["frame_annotations"]["frame_idx"]]
        actors.append(
            {
                "instance_token": token,
                "raw_annotation_chain": {
                    "first_annotation_token": str(chain["first_annotation_token"]),
                    "last_annotation_token": str(chain["last_annotation_token"]),
                    "nbr_annotations": int(chain["nbr_annotations"]),
                },
                "processed_true_instance_id": true_id,
                "dataset_instance_column": column,
                "rigid_model_index": model_index,
                "availability": (
                    "available" if indices else "unavailable_empty_checkpoint_slice"
                ),
                "checkpoint_tensor_slice": {
                    "selector": f"models.RigidNodes.points_ids[:,0] == {model_index}",
                    "gaussian_count": len(indices),
                    "flat_index_ranges_half_open": _ranges(indices),
                    "flat_indices_sha256": hashlib.sha256(
                        json.dumps(indices, separators=(",", ":")).encode("utf-8")
                    ).hexdigest(),
                },
                "class_name": str(processed["class_name"]),
                "first_processed_frame": min(frames),
                "last_processed_frame": max(frames),
                "processed_frame_count": len(frames),
            }
        )

    identity_fields = (
        "instance_token",
        "processed_true_instance_id",
        "dataset_instance_column",
        "rigid_model_index",
    )
    conflicts = {
        field: _duplicates([actor[field] for actor in actors]) for field in identity_fields
    }
    conflicts = {field: values for field, values in conflicts.items() if values}
    if conflicts:
        raise DriveStudioRegistryError(f"registry is not one-to-one: {conflicts}")

    registry = {
        "schema_version": SCHEMA_VERSION,
        "mapping_algorithm": MAPPING_ALGORITHM,
        "scene_id": str(scene_id),
        "scene_name": str(scene_name),
        "checkpoint_sha256": str(checkpoint_sha256),
        "actor_count": len(actors),
        "available_actor_count": sum(
            actor["availability"] == "available" for actor in actors
        ),
        "empty_checkpoint_actor_count": sum(
            actor["availability"] == "unavailable_empty_checkpoint_slice"
            for actor in actors
        ),
        "actors": actors,
    }
    registry["actor_registry_sha256"] = canonical_sha256(registry)
    return registry


def require_token(
    registry: Mapping[str, Any],
    instance_token: str,
    *,
    require_nonempty: bool = True,
) -> Mapping[str, Any]:
    expected = registry.get("actor_registry_sha256")
    payload = dict(registry)
    payload.pop("actor_registry_sha256", None)
    if expected != canonical_sha256(payload):
        raise DriveStudioRegistryError("actor registry hash mismatch")
    matches = [
        actor for actor in registry["actors"] if actor["instance_token"] == instance_token
    ]
    if len(matches) != 1:
        raise DriveStudioRegistryError(
            f"token {instance_token} must map exactly once, got {len(matches)}"
        )
    if require_nonempty and int(
        matches[0].get("checkpoint_tensor_slice", {}).get("gaussian_count", 0)
    ) <= 0:
        raise DriveStudioRegistryError(
            f"token {instance_token} has an empty checkpoint Gaussian slice"
        )
    return matches[0]
