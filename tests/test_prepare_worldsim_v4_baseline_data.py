from __future__ import annotations

import copy

import pytest

from scripts.prepare_worldsim_v4_baseline_data import (
    BaselineDataError,
    M1_FRAME_CONTRACT,
    expected_scene_frames,
    scene_directory_name,
    validate_config,
    validate_processed_scene,
)


SCENES = ["scene-0230", "scene-0242", "scene-0255", "scene-0048", "scene-0994", "scene-0139"]
VALIDATION_SCENES = [
    "scene-0071", "scene-1089", "scene-0317",
    "scene-0862", "scene-1012", "scene-0450",
]


def config() -> dict:
    rows = {scene: {"scene_index": index, "state": "reuse_processed"} for index, scene in enumerate(SCENES)}
    for scene in ("scene-0048", "scene-0994", "scene-0139"):
        rows[scene]["state"] = "extract_and_preprocess"
    return {
        "schema_version": "worldsim_v4_baseline_data_v1",
        "task_id": "WS-V4-B0-MATCHED-BASELINES-01",
        "status": "running",
        "protocol": {
            "sensors": ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_BACK", "LIDAR_TOP"],
            "no_download": True,
            "test_quality_read": False,
        },
        "scenes": rows,
        "gates": {"expected_extract_scene_count": 3},
    }


def cohort() -> dict:
    return {
        "freeze": {
            "scene_roles": {
                "development": SCENES,
                "validation": VALIDATION_SCENES,
            }
        }
    }


def test_data_config_matches_frozen_development_and_missing_three() -> None:
    result = validate_config(config(), cohort())
    assert result["extract_scenes"] == ["scene-0048", "scene-0139", "scene-0994"]


def test_m1_validation_data_config_requires_all_six_frozen_scenes() -> None:
    value = config()
    value["task_id"] = "WS-V4-M1-EVIDENCE-FIELD-01"
    value["protocol"]["cohort_role"] = "validation"
    value["scenes"] = {
        scene: {
            "scene_index": index,
            "state": "extract_and_preprocess",
            "expected_frames_10hz": M1_FRAME_CONTRACT[scene],
        }
        for index, scene in enumerate(VALIDATION_SCENES)
    }
    value["gates"]["expected_extract_scene_count"] = 6
    result = validate_config(value, cohort())
    assert result["cohort_role"] == "validation"
    assert result["extract_scenes"] == sorted(VALIDATION_SCENES)
    assert expected_scene_frames(value, "scene-0317") == 191
    assert expected_scene_frames(value, "scene-0071") == 196


def test_m1_validation_data_config_rejects_per_scene_frame_drift() -> None:
    value = config()
    value["task_id"] = "WS-V4-M1-EVIDENCE-FIELD-01"
    value["protocol"]["cohort_role"] = "validation"
    value["scenes"] = {
        scene: {
            "scene_index": index,
            "state": "extract_and_preprocess",
            "expected_frames_10hz": M1_FRAME_CONTRACT[scene],
        }
        for index, scene in enumerate(VALIDATION_SCENES)
    }
    value["scenes"]["scene-0317"]["expected_frames_10hz"] = 196
    value["gates"]["expected_extract_scene_count"] = 6
    with pytest.raises(BaselineDataError, match="per-scene frame"):
        validate_config(value, cohort())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["protocol"].update(no_download=False), "sensor/no-download"),
        (lambda value: value["scenes"].pop("scene-0139"), "精确匹配"),
        (lambda value: value["scenes"]["scene-0230"].update(state="extract_and_preprocess"), "extract scene count"),
    ],
)
def test_data_config_fails_closed(mutation, message: str) -> None:
    value = copy.deepcopy(config())
    mutation(value)
    with pytest.raises(BaselineDataError, match=message):
        validate_config(value, cohort())


def test_processed_scene_validator_checks_exact_counts(tmp_path) -> None:
    root = tmp_path / "45"
    (root / "images").mkdir(parents=True)
    (root / "lidar").mkdir()
    (root / "instances").mkdir()
    for index in range(4):
        (root / "images" / f"{index}.jpg").write_bytes(b"rgb")
    for index in range(2):
        (root / "lidar" / f"{index}.bin").write_bytes(b"lidar")
    (root / "instances/instances_info.json").write_text("{}")
    (root / "instances/frame_instances.json").write_text("{}")
    result = validate_processed_scene(root, expected_frames=2, expected_cameras=2)
    assert result["image_count"] == 4
    assert result["lidar_count"] == 2


def test_upstream_scene_directory_is_zero_padded() -> None:
    assert scene_directory_name(45) == "045"
    assert scene_directory_name(110) == "110"
