from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.build_worldsim_v4_sky_masks import (
    SkyMaskError,
    V5_TASK_ID,
    output_schema,
    sha256_file,
    source_snapshot_relpaths,
    validate_config,
)


SCENES = {
    "scene-0471": 382,
    "scene-1087": 827,
    "scene-0379": 296,
    "scene-0998": 756,
    "scene-0359": 276,
    "scene-0875": 663,
    "scene-0535": 425,
    "scene-0436": 350,
}
FRAMES = {
    "scene-0471": 196,
    "scene-1087": 196,
    "scene-0379": 191,
    "scene-0998": 196,
    "scene-0359": 196,
    "scene-0875": 196,
    "scene-0535": 201,
    "scene-0436": 196,
}


def config(tmp_path: Path) -> dict:
    reconstruction = tmp_path / "reconstruction.yaml"
    reconstruction.write_text("frozen\n", encoding="utf-8")
    return {
        "schema_version": "worldsim_v5_sky_masks_v1",
        "task_id": V5_TASK_ID,
        "status": "running",
        "preprocess_reconstruction_config": {
            "path": str(reconstruction),
            "sha256": sha256_file(reconstruction),
        },
        "model": {
            "revision": "2c6f153e4c23c229e2fa2b188eb250607e030cd8",
            "local_files_only": True,
        },
        "data": {
            "processed_root": "/data/trainval",
            "cameras": [0, 1, 2],
            "expected_timesteps_by_scene": dict(FRAMES),
            "expected_masks_by_scene": {
                scene: frames * 3 for scene, frames in FRAMES.items()
            },
            "scenes": dict(SCENES),
        },
        "runtime": {
            "generation_network_access": False,
            "no_test_quality_read": True,
        },
    }


def test_v5_sky_masks_preserve_variable_timeline_and_preprocess_binding(
    tmp_path: Path,
) -> None:
    result = validate_config(config(tmp_path))
    assert result == {"scene_count": 8, "expected_masks": 4704}
    assert output_schema(V5_TASK_ID, "summary") == "worldsim_v5_sky_mask_summary_v1"


def test_v5_sky_masks_reject_binding_or_timeline_drift(tmp_path: Path) -> None:
    value = config(tmp_path)
    value["preprocess_reconstruction_config"]["sha256"] = "0" * 64
    with pytest.raises(SkyMaskError, match="binding"):
        validate_config(value)

    value = copy.deepcopy(config(tmp_path))
    value["data"]["expected_masks_by_scene"]["scene-0379"] = 588
    with pytest.raises(SkyMaskError, match="mask contract"):
        validate_config(value)


def test_v5_source_snapshot_routes_to_v5_contract(tmp_path: Path) -> None:
    project = tmp_path / "project"
    config_path = project / "configs/worldsim_v5/sky.yaml"
    config_path.parent.mkdir(parents=True)
    import yaml

    config_path.write_text(yaml.safe_dump(config(tmp_path)), encoding="utf-8")
    relpaths = source_snapshot_relpaths(config_path, project)
    assert "tests/test_build_worldsim_v5_sky_masks.py" in relpaths
    assert "tests/test_build_worldsim_v4_m3_test_sky_masks.py" not in relpaths
