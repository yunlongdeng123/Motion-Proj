from __future__ import annotations

import copy

import pytest

from scripts.prepare_worldsim_v4_baseline_data import BaselineDataError, validate_config


SCENES = ["scene-0230", "scene-0242", "scene-0255", "scene-0048", "scene-0994", "scene-0139"]


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
    return {"freeze": {"scene_roles": {"development": SCENES}}}


def test_data_config_matches_frozen_development_and_missing_three() -> None:
    result = validate_config(config(), cohort())
    assert result["extract_scenes"] == ["scene-0048", "scene-0139", "scene-0994"]


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
