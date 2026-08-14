from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/preprocess_worldsim_v5_m1_development.py"
SPEC = importlib.util.spec_from_file_location("worldsim_v5_m1_preprocess", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write(path: Path, payload: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _processed_fixture(root: Path, frames: int = 2, cameras: int = 2) -> None:
    for frame in range(frames):
        for camera in range(cameras):
            _write(root / "images" / f"{frame:03d}_{camera}.jpg")
            _write(root / "extrinsics" / f"{frame:03d}_{camera}.txt")
            for category in ("all", "human", "vehicle"):
                _write(root / "dynamic_masks" / category / f"{frame:03d}_{camera}.png")
        _write(root / "lidar" / f"{frame:03d}.bin")
        _write(root / "lidar_pose" / f"{frame:03d}.txt")
    for camera in range(cameras):
        _write(root / "intrinsics" / f"{camera}.txt")
    _write(root / "instances/instances_info.json", b"{}")
    _write(root / "instances/frame_instances.json", b"{}")


def test_processor_output_root_and_frame_contract() -> None:
    target = Path("/tmp/worldsim_v5/drivestudio_processed")
    assert MODULE.processor_output_root(target) == Path(
        "/tmp/worldsim_v5/drivestudio_processed_10Hz/trainval"
    )
    assert MODULE.expected_frames(40) == 196
    with pytest.raises(MODULE.PreprocessError):
        MODULE.expected_frames(1)


def test_validate_processed_scene_checks_all_modalities(tmp_path: Path) -> None:
    scene = tmp_path / "382"
    _processed_fixture(scene)
    result = MODULE.validate_processed_scene(scene, frame_count=2, camera_count=2)
    assert result["counts"]["images"] == 4
    assert result["counts"]["dynamic_masks_vehicle"] == 4
    (scene / "lidar/001.bin").unlink()
    with pytest.raises(MODULE.PreprocessError, match="lidar"):
        MODULE.validate_processed_scene(scene, frame_count=2, camera_count=2)


def test_load_inputs_rejects_identity_drift(tmp_path: Path) -> None:
    config = {
        "task_id": "WS-V5-M1-STRUCTURED-OWNERSHIP-01",
        "status": "running",
        "fresh_cohort_binding": {
            "development_scenes": [{"scene": "scene-0471", "scene_index": 382}]
        },
    }
    batch = {
        "complete": True,
        "quality_read": False,
        "scenes": [{"scene_name": "scene-0471", "scene_index": 999}],
    }
    config_path = tmp_path / "config.yaml"
    batch_path = tmp_path / "batch.json"
    config_path.write_text(
        "task_id: WS-V5-M1-STRUCTURED-OWNERSHIP-01\n"
        "status: running\n"
        "fresh_cohort_binding:\n"
        "  development_scenes:\n"
        "    - {scene: scene-0471, scene_index: 382}\n",
        encoding="utf-8",
    )
    batch_path.write_text(json.dumps(batch), encoding="utf-8")
    with pytest.raises(MODULE.PreprocessError, match="identity"):
        MODULE.load_inputs(config_path, batch_path)
