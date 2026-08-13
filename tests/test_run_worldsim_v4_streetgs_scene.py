from __future__ import annotations

import copy
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_worldsim_v4_streetgs_scene import (
    FRAME_CONTRACTS,
    SCENE_CONTRACTS,
    StreetGSTrainingError,
    build_train_command,
    expected_scene_frames,
    validate_config,
)


def config(tmp_path: Path) -> dict:
    patch = tmp_path / "patch.diff"
    patch.write_text("patch\n")
    from scripts.run_worldsim_v4_streetgs_scene import sha256_file
    return {
        "schema_version": "worldsim_v4_streetgs_training_v1",
        "task_id": "WS-V4-B0-MATCHED-BASELINES-01",
        "status": "running",
        "implementation": {"compatibility_patch": "patch.diff", "compatibility_patch_sha256": sha256_file(patch), "environment": "/env", "config_file": "configs/streetgs.yaml"},
        "data": {
            "dataset_config": "nuscenes/3cams",
            "processed_root": "/data/trainval",
            "expected_frames": 196,
            "expected_cameras": 6,
            "start_timestep": 0,
            "end_timestep": -1,
            "load_smpl": False,
            "test_image_stride": 0,
            "frame_partition": {
                "modulus": 5,
                "development_remainder": 2,
                "heldout_remainder": 4,
                "train_remainders": [0, 1, 3],
            },
        },
        "scenes": {"scene-0230": 179, "scene-0242": 191, "scene-0255": 204, "scene-0048": 45, "scene-0994": 752, "scene-0139": 110},
        "training": {"seed": 0, "modes": {"profile100": 100, "formal": 30000}},
    }


def test_config_and_formal_command_freeze_same_split_no_post_render(tmp_path: Path) -> None:
    value = config(tmp_path)
    assert validate_config(value, tmp_path)["scene_count"] == 6
    command, checkpoint, iterations = build_train_command(value, "scene-0048", "formal", tmp_path / "run")
    assert iterations == 30000
    assert "data.scene_idx=45" in command
    assert "data.pixel_source.test_image_stride=0" in command
    assert "+data.pixel_source.partition_modulus=5" in command
    assert "+data.pixel_source.excluded_remainders=[2,4]" in command
    assert "trainer.optim.num_iters=30000" in command
    assert "logging.vis_freq=-1" in command
    assert "render.render_test=false" in command
    assert checkpoint.name == "checkpoint_final.pth"


def test_m1_validation_reconstruction_uses_frozen_validation_scene_contract(
    tmp_path: Path,
) -> None:
    value = config(tmp_path)
    value["task_id"] = "WS-V4-M1-EVIDENCE-FIELD-01"
    value["scenes"] = {
        "scene-0071": 68,
        "scene-1089": 829,
        "scene-0317": 251,
        "scene-0862": 652,
        "scene-1012": 770,
        "scene-0450": 364,
    }
    value["data"]["expected_frames_by_scene"] = {
        "scene-0071": 196,
        "scene-0317": 191,
        "scene-0450": 196,
        "scene-0862": 196,
        "scene-1012": 196,
        "scene-1089": 196,
    }
    result = validate_config(value, tmp_path)
    assert result["task_id"] == "WS-V4-M1-EVIDENCE-FIELD-01"
    assert expected_scene_frames(value, "scene-0317") == 191
    command, _, iterations = build_train_command(
        value, "scene-0071", "profile100", tmp_path / "run"
    )
    assert iterations == 100
    assert "data.scene_idx=68" in command


def test_m1_validation_reconstruction_rejects_frame_drift(tmp_path: Path) -> None:
    value = config(tmp_path)
    value["task_id"] = "WS-V4-M1-EVIDENCE-FIELD-01"
    value["scenes"] = {
        "scene-0071": 68,
        "scene-1089": 829,
        "scene-0317": 251,
        "scene-0862": 652,
        "scene-1012": 770,
        "scene-0450": 364,
    }
    value["data"]["expected_frames_by_scene"] = {
        scene: 196 for scene in value["scenes"]
    }
    with pytest.raises(StreetGSTrainingError, match="frame"):
        validate_config(value, tmp_path)


def test_m3_test_reconstruction_uses_exact_eighteen_train_only(tmp_path: Path) -> None:
    value = config(tmp_path)
    value["task_id"] = "WS-V4-M3-TEMPORAL-DELTA-01"
    value["scenes"] = SCENE_CONTRACTS[value["task_id"]]
    value["data"]["expected_frames_by_scene"] = FRAME_CONTRACTS[value["task_id"]]
    result = validate_config(value, tmp_path)
    assert result["scene_count"] == 18
    command, _, iterations = build_train_command(
        value, "scene-0919", "formal", tmp_path / "run"
    )
    assert iterations == 30000
    assert "data.scene_idx=704" in command
    assert "+data.pixel_source.excluded_remainders=[2,4]" in command
    assert "render.render_test=false" in command


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["training"].update(seed=1), "seed/iteration"),
        (lambda value: value["training"]["modes"].update(formal=1000), "seed/iteration"),
        (lambda value: value["data"]["frame_partition"].update(heldout_remainder=3), "frame partition"),
        (lambda value: value["data"].update(test_image_stride=10), "stride split"),
        (lambda value: value["scenes"].pop("scene-0994"), "冻结场景"),
    ],
)
def test_training_config_fails_closed(tmp_path: Path, mutation, message: str) -> None:
    value = copy.deepcopy(config(tmp_path))
    mutation(value)
    with pytest.raises(StreetGSTrainingError, match=message):
        validate_config(value, tmp_path)


def test_direct_script_entry_can_resolve_project_imports() -> None:
    project = Path(__file__).resolve().parents[1]
    process = subprocess.run(
        [sys.executable, str(project / "scripts/run_worldsim_v4_streetgs_scene.py"), "--help"],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert "--scene" in process.stdout
