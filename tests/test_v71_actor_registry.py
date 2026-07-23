import copy

import pytest

from motion_proj.resim.actor_registry import (
    ActorRegistryError,
    build_actor_registry,
    require_actor,
    validate_registry_hash,
)


def _actor(true_id, model_idx):
    return {
        "true_instance_id": true_id,
        "dataset_instance_column": true_id,
        "rigid_model_index": model_idx,
        "occupancy_instance_id": true_id + 1,
        "limited_label_id": true_id + 1,
        "class_name": "vehicle.car",
        "canonical_dimensions_lwh": [4.5, 1.8, 1.6],
        "first_frame": 0,
        "last_frame": 79,
        "active_frame_count": 80,
        "lidar_point_count": 100,
    }


def test_registry_is_one_to_one_and_hash_verified():
    registry = build_actor_registry(
        scene_id="003",
        checkpoint_sha256="a" * 64,
        actors=[_actor(35, 1), _actor(21, 0)],
        checkpoint_num_instances=2,
    )
    validate_registry_hash(registry)
    assert require_actor(registry, 35)["rigid_model_index"] == 1


def test_duplicate_model_index_fails_closed():
    with pytest.raises(ActorRegistryError, match="非一一映射"):
        build_actor_registry(
            scene_id="003",
            checkpoint_sha256="a" * 64,
            actors=[_actor(21, 0), _actor(35, 0)],
            checkpoint_num_instances=2,
        )


def test_checkpoint_actor_count_mismatch_fails_closed():
    with pytest.raises(ActorRegistryError, match="checkpoint actor 数"):
        build_actor_registry(
            scene_id="003",
            checkpoint_sha256="a" * 64,
            actors=[_actor(21, 0)],
            checkpoint_num_instances=2,
        )


def test_missing_actor_and_hash_tamper_fail_closed():
    registry = build_actor_registry(
        scene_id="003",
        checkpoint_sha256="a" * 64,
        actors=[_actor(21, 0)],
        checkpoint_num_instances=1,
    )
    with pytest.raises(ActorRegistryError, match="恰好映射一次"):
        require_actor(registry, 35)
    tampered = copy.deepcopy(registry)
    tampered["actors"][0]["true_instance_id"] = 999
    with pytest.raises(ActorRegistryError, match="不匹配"):
        validate_registry_hash(tampered)
