from __future__ import annotations

import copy

import pytest

from scripts.build_worldsim_v4_sky_masks import (
    SkyMaskError,
    TASK_FRAME_COUNTS,
    TASK_SCENES,
    expected_sky_counts,
    validate_config,
)


TASK_ID = "WS-V4-M3-TEMPORAL-DELTA-01"


def config() -> dict:
    frames = TASK_FRAME_COUNTS[TASK_ID]
    return {
        "schema_version": "worldsim_v4_sky_masks_v1",
        "task_id": TASK_ID,
        "status": "running",
        "model": {
            "revision": "2c6f153e4c23c229e2fa2b188eb250607e030cd8",
            "local_files_only": True,
        },
        "data": {
            "cameras": [0, 1, 2],
            "expected_timesteps": 196,
            "expected_masks": 588,
            "expected_timesteps_by_scene": frames,
            "expected_masks_by_scene": {
                scene: count * 3 for scene, count in frames.items()
            },
            "scenes": TASK_SCENES[TASK_ID],
        },
        "runtime": {
            "generation_network_access": False,
            "no_test_quality_read": True,
        },
    }


def test_m3_sky_mask_config_freezes_exact_eighteen_local_only() -> None:
    value = config()
    assert validate_config(value)["scene_count"] == 18
    assert expected_sky_counts(value, "scene-0919") == (201, 603)
    assert expected_sky_counts(value, "scene-0100") == (196, 588)


def test_m3_sky_mask_config_rejects_scene_or_count_drift() -> None:
    value = copy.deepcopy(config())
    value["data"]["scenes"]["scene-0919"] = 705
    with pytest.raises(SkyMaskError, match="scene"):
        validate_config(value)

    value = copy.deepcopy(config())
    value["data"]["expected_masks_by_scene"]["scene-0919"] = 588
    with pytest.raises(SkyMaskError, match="mask contract"):
        validate_config(value)
