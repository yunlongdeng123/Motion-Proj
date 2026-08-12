from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.materialize_worldsim_v4_v33_semantic_config import (
    build_semantic_config,
    discover_scene_layout,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path: Path, *, boundary_available: bool = True) -> tuple[dict, dict, dict]:
    processed = tmp_path / "processed" / "204"
    (processed / "images").mkdir(parents=True)
    (processed / "instances").mkdir()
    for frame in range(10):
        for camera in range(3):
            (processed / "images" / f"{frame:03d}_{camera}.jpg").write_bytes(b"rgb")
    instances = processed / "instances" / "instances_info.json"
    instances.write_text(json.dumps({"1": {"id": "high"}, "2": {"id": "boundary"}}))
    checkpoint = tmp_path / "checkpoint_final.pth"
    checkpoint.write_bytes(b"checkpoint")
    source_config = tmp_path / "config.yaml"
    source_config.write_text("data: {}\n")
    registry = tmp_path / "actor_registry.json"
    registry.write_text("{}\n")
    replay = {
        "task_id": "WS-V4-B0-MATCHED-BASELINES-01",
        "algorithm": {"implementation_commit": "e6663e1"},
        "semantic_lift": {
            "cameras": [
                {"id": 0, "name": "CAM_FRONT"},
                {"id": 1, "name": "CAM_FRONT_LEFT"},
                {"id": 2, "name": "CAM_FRONT_RIGHT"},
            ]
        },
        "frame_partition": {
            "modulus": 5,
            "train_remainders": [0, 1, 3],
            "development_remainder": 2,
            "heldout_remainder": 4,
        },
        "runtimes": {
            "sam2_python": "/env/sam/bin/python",
            "drivestudio_python": "/env/ds/bin/python",
            "drivestudio_checkout": "/code/drivestudio",
        },
    }
    template = {
        "seed": 7,
        "prompts": {"minimum_depth_m": 0.1},
        "sam2": {"checkpoint": "/sam.pt"},
        "lift": {"first_hit_alpha_threshold": 0.5},
        "outputs": {"model_native_width": 800, "model_native_height": 450},
    }
    bound = {
        "schema_version": "worldsim_v4_v33_bound_scene_v1",
        "partition_contract": "sample_index_mod_5",
        "algorithm_commit": "e6663e1",
        "test_quality_read": False,
        "scene": "scene-0255",
        "processed_scene": str(processed),
        "instances_info": {"path": str(instances), "sha256": sha(instances)},
        "base_checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha(checkpoint),
            "source_config": str(source_config),
            "source_config_sha256": sha(source_config),
        },
        "actor_registry": {
            "path": str(registry),
            "bytes": registry.stat().st_size,
            "sha256": sha(registry),
        },
        "actors": {
            "high_support": {
                "instance_token": "high",
                "dataset_instance_id": 1,
                "class_name": "vehicle.car",
                "availability": "available",
                "rigid_model_index": 5,
            },
            "boundary_support": {
                "instance_token": "boundary",
                "dataset_instance_id": 2,
                "class_name": "vehicle.car",
                "availability": (
                    "available" if boundary_available else "unavailable_empty_slice"
                ),
                "rigid_model_index": 8 if boundary_available else None,
            },
        },
    }
    return replay, template, bound


def test_materializes_exact_mod5_semantic_contract(tmp_path: Path) -> None:
    replay, template, bound = fixture(tmp_path)

    config = build_semantic_config(
        replay=replay, template=template, bound_scene=bound
    )

    assert config["scene"]["frame_count"] == 10
    assert config["split"]["development_frames"] == [2, 7]
    assert config["split"]["heldout_frames"] == [4, 9]
    assert config["inputs"]["checkpoint_sha256"] == bound["base_checkpoint"]["sha256"]
    assert config["provenance"]["test_quality_read"] is False


def test_unavailable_boundary_is_explicitly_omitted(tmp_path: Path) -> None:
    replay, template, bound = fixture(tmp_path, boundary_available=False)

    config = build_semantic_config(
        replay=replay, template=template, bound_scene=bound
    )

    assert set(config["actors"]) == {"high_support"}
    assert config["provenance"]["actor_abstentions"]["boundary_support"][
        "reason"
    ] == "actor_registry_unavailable"


def test_scene_layout_rejects_camera_frame_drift(tmp_path: Path) -> None:
    replay, _, bound = fixture(tmp_path)
    missing = Path(bound["processed_scene"]) / "images" / "009_2.jpg"
    missing.unlink()

    try:
        discover_scene_layout(
            Path(bound["processed_scene"]), replay["semantic_lift"]["cameras"]
        )
    except RuntimeError as error:
        assert "frame 集不一致" in str(error)
    else:
        raise AssertionError("camera frame drift 必须 fail closed")
