from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from motion_proj.worldsim_v4.v33_replay import V33ReplayError
from scripts.run_worldsim_v4_v33_abstain_scene import (
    select_development_views,
    validate_no_actor_contract,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path: Path) -> tuple[dict, dict]:
    checkpoint = tmp_path / "checkpoint.pth"
    source_config = tmp_path / "config.yaml"
    registry_path = tmp_path / "actor_registry.json"
    checkpoint.write_bytes(b"checkpoint")
    source_config.write_text("data: {}\n", encoding="utf-8")
    actor = {
        "instance_token": "high-token",
        "dataset_instance_id": 12,
        "class_name": "vehicle.car",
        "availability": "unavailable_initialization_filter",
        "rigid_model_index": None,
    }
    registry = {
        "checkpoint_sha256": sha(checkpoint),
        "actors": [
            {
                "instance_token": actor["instance_token"],
                "processed_true_instance_id": actor["dataset_instance_id"],
                "availability": actor["availability"],
                "rigid_model_index": None,
                "checkpoint_tensor_slice": {
                    "gaussian_count": 0,
                    "flat_index_ranges_half_open": [],
                },
            }
        ],
    }
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    processed = tmp_path / "processed"
    (processed / "images").mkdir(parents=True)
    for frame in range(18):
        (processed / "images" / f"{frame:03d}_0.jpg").write_bytes(b"image")
    replay = {
        "schema_version": "worldsim_v4_v33_replay_v1",
        "algorithm": {"implementation_commit": "e6663e1"},
        "abstain_no_actor": {
            "reason": "ABSTAIN_NO_ACTOR",
            "render_camera_id": 0,
            "development_view_count": 3,
            "view_selection": "first_development_frames",
            "semantic_render_status": "done",
        },
        "frame_partition": {"modulus": 5, "development_remainder": 2},
    }
    bound = {
        "schema_version": "worldsim_v4_v33_bound_scene_v1",
        "partition_contract": "sample_index_mod_5",
        "algorithm_commit": "e6663e1",
        "test_quality_read": False,
        "scene": "scene-0255",
        "processed_scene": str(processed),
        "base_checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha(checkpoint),
            "source_config": str(source_config),
            "source_config_sha256": sha(source_config),
        },
        "actor_registry": {
            "path": str(registry_path),
            "bytes": registry_path.stat().st_size,
            "sha256": sha(registry_path),
        },
        "actors": {"high_support": actor},
    }
    return replay, bound


def test_validates_zero_gaussian_d0_actor_and_selects_development_views(
    tmp_path: Path,
) -> None:
    replay, bound = fixture(tmp_path)

    proof = validate_no_actor_contract(replay, bound)
    views = select_development_views(replay, bound)

    assert proof["reason"] == "ABSTAIN_NO_ACTOR"
    assert proof["actor"]["checkpoint_gaussian_count"] == 0
    assert views == [(2, 0), (7, 0), (12, 0)]


def test_rejects_available_or_nonempty_actor(tmp_path: Path) -> None:
    replay, bound = fixture(tmp_path)
    bound["actors"]["high_support"]["availability"] = "available"

    with pytest.raises(V33ReplayError, match="只允许不可用"):
        validate_no_actor_contract(replay, bound)

