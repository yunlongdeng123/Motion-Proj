from __future__ import annotations

import pytest

from motion_proj.dynamic_editing_v2.drivestudio_registry import (
    DriveStudioRegistryError,
    build_drivestudio_registry,
    require_token,
)


def inputs():
    return {
        "scene_id": "179",
        "scene_name": "scene-0230",
        "checkpoint_sha256": "abc",
        "processed_instances": {
            "4": {
                "id": "token-a",
                "class_name": "vehicle.car",
                "frame_annotations": {"frame_idx": [1, 2, 3]},
            },
            "7": {
                "id": "token-b",
                "class_name": "vehicle.truck",
                "frame_annotations": {"frame_idx": [2, 3]},
            },
        },
        "raw_instance_chains": {
            "token-a": {
                "first_annotation_token": "a0",
                "last_annotation_token": "a1",
                "nbr_annotations": 2,
            },
            "token-b": {
                "first_annotation_token": "b0",
                "last_annotation_token": "b1",
                "nbr_annotations": 2,
            },
        },
        "dataset_true_ids": [7, 4],
        "ordered_init_columns": [1, 0],
        "checkpoint_point_ids": [0, 0, 1, 0, 1],
        "checkpoint_instance_count": 2,
    }


def test_registry_links_token_to_noncontiguous_checkpoint_slices() -> None:
    registry = build_drivestudio_registry(**inputs())
    actor = require_token(registry, "token-a")
    assert actor["processed_true_instance_id"] == 4
    assert actor["dataset_instance_column"] == 1
    assert actor["rigid_model_index"] == 0
    assert actor["checkpoint_tensor_slice"]["flat_index_ranges_half_open"] == [
        [0, 2],
        [3, 4],
    ]


def test_registry_marks_checkpoint_culled_actor_unavailable() -> None:
    payload = inputs()
    payload["checkpoint_point_ids"] = [0, 0]
    registry = build_drivestudio_registry(**payload)
    assert registry["available_actor_count"] == 1
    assert registry["empty_checkpoint_actor_count"] == 1
    actor = next(row for row in registry["actors"] if row["instance_token"] == "token-b")
    assert actor["availability"] == "unavailable_empty_checkpoint_slice"
    assert actor["checkpoint_tensor_slice"]["gaussian_count"] == 0
    with pytest.raises(DriveStudioRegistryError, match="empty checkpoint"):
        require_token(registry, "token-b")


def test_registry_rejects_out_of_range_checkpoint_point_id() -> None:
    payload = inputs()
    payload["checkpoint_point_ids"] = [0, 2]
    with pytest.raises(DriveStudioRegistryError, match="outside the model index range"):
        build_drivestudio_registry(**payload)


def test_registry_fails_when_raw_chain_is_missing() -> None:
    payload = inputs()
    del payload["raw_instance_chains"]["token-b"]
    with pytest.raises(DriveStudioRegistryError, match="raw annotation chain missing"):
        build_drivestudio_registry(**payload)
