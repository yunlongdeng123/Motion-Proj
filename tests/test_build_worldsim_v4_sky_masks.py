from __future__ import annotations

import copy

import pytest

from scripts.build_worldsim_v4_sky_masks import SkyMaskError, validate_config, validate_output_paths


def config() -> dict:
    return {
        "schema_version": "worldsim_v4_sky_masks_v1",
        "task_id": "WS-V4-B0-MATCHED-BASELINES-01",
        "status": "running",
        "model": {"revision": "2c6f153e4c23c229e2fa2b188eb250607e030cd8", "local_files_only": True},
        "data": {"cameras": [0, 1, 2], "expected_timesteps": 196, "expected_masks": 588, "scenes": {"scene-0048": 45, "scene-0139": 110, "scene-0994": 752}},
        "runtime": {"generation_network_access": False, "no_test_quality_read": True},
    }


def test_sky_mask_config_freezes_local_snapshot_and_three_scenes() -> None:
    assert validate_config(config()) == {"scene_count": 3, "expected_masks": 588}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["model"].update(local_files_only=False), "revision/local-only"),
        (lambda value: value["data"].update(expected_masks=587), "camera/timestep/mask"),
        (lambda value: value["runtime"].update(generation_network_access=True), "generation-network"),
    ],
)
def test_sky_mask_config_fails_closed(mutation, message: str) -> None:
    value = copy.deepcopy(config())
    mutation(value)
    with pytest.raises(SkyMaskError, match=message):
        validate_config(value)


def test_output_paths_accept_preprocess_created_empty_target(tmp_path) -> None:
    target = tmp_path / "sky_masks"
    target.mkdir()

    assert validate_output_paths(target, tmp_path / "sky_masks.partial.run") is True


def test_output_paths_reject_existing_mask_or_partial(tmp_path) -> None:
    target = tmp_path / "sky_masks"
    target.mkdir()
    (target / "000_0.png").write_bytes(b"mask")
    with pytest.raises(SkyMaskError, match="target 已存在且非空"):
        validate_output_paths(target, tmp_path / "sky_masks.partial.run")

    target.joinpath("000_0.png").unlink()
    partial = tmp_path / "sky_masks.partial.run"
    partial.mkdir()
    with pytest.raises(SkyMaskError, match="partial 已存在"):
        validate_output_paths(target, partial)
