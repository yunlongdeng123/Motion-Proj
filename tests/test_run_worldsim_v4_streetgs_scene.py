from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.run_worldsim_v4_streetgs_scene import StreetGSTrainingError, build_train_command, validate_config


def config(tmp_path: Path) -> dict:
    patch = tmp_path / "patch.diff"
    patch.write_text("patch\n")
    from scripts.run_worldsim_v4_streetgs_scene import sha256_file
    return {
        "schema_version": "worldsim_v4_streetgs_training_v1",
        "task_id": "WS-V4-B0-MATCHED-BASELINES-01",
        "status": "running",
        "implementation": {"compatibility_patch": "patch.diff", "compatibility_patch_sha256": sha256_file(patch), "environment": "/env", "config_file": "configs/streetgs.yaml"},
        "data": {"dataset_config": "nuscenes/3cams", "processed_root": "/data/trainval", "start_timestep": 0, "end_timestep": -1, "load_smpl": False, "test_image_stride": 10},
        "scenes": {"scene-0230": 179, "scene-0242": 191, "scene-0255": 204, "scene-0048": 45, "scene-0994": 752, "scene-0139": 110},
        "training": {"seed": 0, "modes": {"profile100": 100, "formal": 30000}},
    }


def test_config_and_formal_command_freeze_same_split_no_post_render(tmp_path: Path) -> None:
    value = config(tmp_path)
    assert validate_config(value, tmp_path)["scene_count"] == 6
    command, checkpoint, iterations = build_train_command(value, "scene-0048", "formal", tmp_path / "run")
    assert iterations == 30000
    assert "data.scene_idx=45" in command
    assert "data.pixel_source.test_image_stride=10" in command
    assert "trainer.optim.num_iters=30000" in command
    assert "render.render_test=false" in command
    assert checkpoint.name == "checkpoint_final.pth"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["training"].update(seed=1), "seed/iteration"),
        (lambda value: value["training"]["modes"].update(formal=1000), "seed/iteration"),
        (lambda value: value["scenes"].pop("scene-0994"), "六场景"),
    ],
)
def test_training_config_fails_closed(tmp_path: Path, mutation, message: str) -> None:
    value = copy.deepcopy(config(tmp_path))
    mutation(value)
    with pytest.raises(StreetGSTrainingError, match=message):
        validate_config(value, tmp_path)
