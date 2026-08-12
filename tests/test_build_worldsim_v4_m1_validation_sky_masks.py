from __future__ import annotations

import copy

import pytest

from scripts.build_worldsim_v4_sky_masks import (
    SkyMaskError,
    expected_sky_counts,
    source_snapshot_relpaths,
    validate_config,
)


def validation_config() -> dict:
    return {
        "schema_version": "worldsim_v4_sky_masks_v1",
        "task_id": "WS-V4-M1-EVIDENCE-FIELD-01",
        "status": "running",
        "model": {
            "revision": "2c6f153e4c23c229e2fa2b188eb250607e030cd8",
            "local_files_only": True,
        },
        "data": {
            "cameras": [0, 1, 2],
            "expected_timesteps": 196,
            "expected_masks": 588,
            "expected_timesteps_by_scene": {
                "scene-0071": 196,
                "scene-0317": 191,
                "scene-0450": 196,
                "scene-0862": 196,
                "scene-1012": 196,
                "scene-1089": 196,
            },
            "expected_masks_by_scene": {
                "scene-0071": 588,
                "scene-0317": 573,
                "scene-0450": 588,
                "scene-0862": 588,
                "scene-1012": 588,
                "scene-1089": 588,
            },
            "scenes": {
                "scene-0071": 68,
                "scene-0317": 251,
                "scene-0450": 364,
                "scene-0862": 652,
                "scene-1012": 770,
                "scene-1089": 829,
            },
        },
        "runtime": {
            "generation_network_access": False,
            "no_test_quality_read": True,
        },
    }


def test_validation_sky_mask_config_freezes_exact_six_scenes() -> None:
    assert validate_config(validation_config()) == {
        "scene_count": 6,
        "expected_masks": 588,
    }
    assert expected_sky_counts(validation_config(), "scene-0317") == (191, 573)


def test_validation_sky_mask_config_rejects_scene_drift() -> None:
    value = copy.deepcopy(validation_config())
    value["data"]["scenes"]["scene-0071"] = 69
    with pytest.raises(SkyMaskError, match="scene"):
        validate_config(value)


def test_validation_sky_mask_config_rejects_frame_drift() -> None:
    value = copy.deepcopy(validation_config())
    value["data"]["expected_timesteps_by_scene"]["scene-0317"] = 196
    with pytest.raises(SkyMaskError, match="per-scene frame"):
        validate_config(value)


def test_source_snapshot_uses_selected_config(tmp_path) -> None:
    project_root = tmp_path / "project"
    config_path = project_root / "configs" / "worldsim_v4" / "m1_validation.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.touch()

    relpaths = source_snapshot_relpaths(config_path, project_root)

    assert relpaths[0] == "configs/worldsim_v4/m1_validation.yaml"
    assert "scripts/build_worldsim_v4_sky_masks.py" in relpaths
    assert "tests/test_build_worldsim_v4_m1_validation_sky_masks.py" in relpaths


def test_source_snapshot_rejects_config_outside_project(tmp_path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config_path = tmp_path / "outside.yaml"
    config_path.touch()
    with pytest.raises(SkyMaskError, match="inside the project"):
        source_snapshot_relpaths(config_path, project_root)
