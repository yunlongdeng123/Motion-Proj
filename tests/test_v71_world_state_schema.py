import copy

import pytest

from motion_proj.resim.schema import (
    SchemaError,
    render_request_hash,
    world_state_hash,
)


def _world():
    return {
        "sequence_id": "seq",
        "scene_id": "003",
        "split": "PILOT-3",
        "data_fingerprint": "a" * 64,
        "coordinate_convention": {
            "handedness": "right",
            "units": {"length": "meter", "time": "second"},
            "transform_naming": "T_dst_src",
        },
        "timebase_hz": 10.0,
        "reconstruction": {"checkpoint_sha256": "b" * 64},
        "actor_registry_sha256": "c" * 64,
        "safety_geometry_sha256": "d" * 64,
        "observation_evidence_sha256": "e" * 64,
        "render_support_sha256": "f" * 64,
        "edit_spec": {
            "requested_trajectory": [],
            "realized_trajectory": [],
            "projection_delta": [],
            "scenario_effect": "none",
        },
        "frames": [
            {
                "frame_index": 0,
                "timestamp": 0.0,
                "timestamp_provenance": "observed",
                "T_world_ego": list(range(16)),
                "camera_models": [
                    {"camera_id": "CAM_FRONT_RIGHT"},
                    {"camera_id": "CAM_FRONT"},
                ],
                "actor_nodes": [
                    {"true_instance_id": 8},
                    {"true_instance_id": 4},
                ],
            }
        ],
    }


def _render(world_hash):
    return {
        "world_state_hash": world_hash,
        "frame_index": 0,
        "camera_id": "CAM_FRONT",
        "output_resolution": [900, 1600],
        "rasterizer_mode": "gsplat",
        "precision": "float32",
        "deterministic_flags": {"enabled": True},
        "renderer_config_sha256": "1" * 64,
        "depth_definitions": ["expected", "surface_first_hit"],
        "alpha_first_hit_threshold": 0.5,
        "compositing_policy": "background_actor_z",
        "instance_policy": "vehicle",
        "limited_semantic_policy": "v1",
        "color_space": "srgb",
        "encoding": "png",
    }


def test_unordered_actor_and_camera_collections_hash_stably():
    left = _world()
    right = copy.deepcopy(left)
    right["frames"][0]["actor_nodes"].reverse()
    right["frames"][0]["camera_models"].reverse()
    assert world_state_hash(left) == world_state_hash(right)


def test_renderer_change_only_changes_render_request_hash():
    world = _world()
    state_hash = world_state_hash(world)
    first = _render(state_hash)
    second = copy.deepcopy(first)
    second["output_resolution"] = [450, 800]
    assert world_state_hash(world) == state_hash
    assert render_request_hash(first) != render_request_hash(second)


def test_render_policy_is_forbidden_in_world_state():
    world = _world()
    world["rasterizer_mode"] = "gsplat"
    with pytest.raises(SchemaError, match="render/runtime"):
        world_state_hash(world)
