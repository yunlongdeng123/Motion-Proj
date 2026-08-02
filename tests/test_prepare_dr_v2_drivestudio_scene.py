from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_dr_v2_drivestudio_scene.py"
SPEC = importlib.util.spec_from_file_location("prepare_dr_v2_drivestudio_scene", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload))


def fixture_metadata(root: Path) -> None:
    write_json(
        root / "scene.json",
        [{"name": "scene-0230", "token": "scene", "first_sample_token": "s0", "last_sample_token": "s1"}],
    )
    write_json(
        root / "sample.json",
        [
            {"token": "s0", "scene_token": "scene", "next": "s1"},
            {"token": "s1", "scene_token": "scene", "next": ""},
            {"token": "outside", "scene_token": "other", "next": ""},
        ],
    )
    sensors = []
    calibrations = []
    sample_data = []
    for index, channel in enumerate(MODULE.SENSORS):
        sensors.append({"token": f"sensor-{index}", "channel": channel})
        calibrations.append({"token": f"cal-{index}", "sensor_token": f"sensor-{index}"})
        sample_data.append(
            {
                "token": f"data-{index}",
                "sample_token": "s0" if index % 2 == 0 else "s1",
                "calibrated_sensor_token": f"cal-{index}",
                "filename": f"samples/{channel}/file-{index}",
                "timestamp": index,
                "is_key_frame": True,
            }
        )
    sample_data.append(
        {
            "token": "outside-data",
            "sample_token": "outside",
            "calibrated_sensor_token": "cal-0",
            "filename": "samples/CAM_FRONT/outside.jpg",
            "timestamp": 999,
            "is_key_frame": True,
        }
    )
    write_json(root / "sensor.json", sensors)
    write_json(root / "calibrated_sensor.json", calibrations)
    write_json(root / "sample_data.json", sample_data)


def test_collect_required_is_scene_exact(tmp_path: Path) -> None:
    fixture_metadata(tmp_path)
    result = MODULE.collect_required(tmp_path, "scene-0230")
    assert result["sample_count"] == 2
    assert len(result["sample_data"]) == len(MODULE.SENSORS)
    assert all(result["sensor_counts"][channel] == 1 for channel in MODULE.SENSORS)
    assert "outside-data" not in {row["token"] for row in result["sample_data"]}


def test_sample_chain_cycle_fails_closed(tmp_path: Path) -> None:
    fixture_metadata(tmp_path)
    samples = json.loads((tmp_path / "sample.json").read_text())
    samples[1]["token"] = "s2"
    write_json(tmp_path / "sample.json", samples)
    try:
        MODULE.scene_sample_tokens(tmp_path, "scene-0230")
    except KeyError:
        pass
    else:
        raise AssertionError("broken sample chain must fail")
