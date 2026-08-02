import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "record_dr_v2_m3_asset_reuse.py"
SPEC = importlib.util.spec_from_file_location("record_dr_v2_m3_asset_reuse", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_counts_and_success_stage_validation(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    suffixes = {
        "images": ".jpg",
        "lidar": ".bin",
        "lidar_pose": ".txt",
        "extrinsics": ".txt",
        "sky_masks": ".png",
    }
    expected = {}
    for index, (directory, suffix) in enumerate(suffixes.items(), start=1):
        path = scene / directory
        path.mkdir(parents=True)
        for item in range(index):
            (path / f"{item}{suffix}").write_bytes(b"x")
        expected[directory] = index
    counts = MODULE.count_assets(scene)
    MODULE.validate_counts(counts, expected)
    with pytest.raises(RuntimeError, match="counts changed"):
        MODULE.validate_counts(counts, {**expected, "images": 99})

    stage = tmp_path / "stage.json"
    stage.write_text('{"status":"done","return_code":0}\n', encoding="utf-8")
    assert MODULE.load_success(stage)["status"] == "done"


def test_atomic_json_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "stage.json"
    MODULE.atomic_json(path, {"status": "done"})
    assert json.loads(path.read_text())["status"] == "done"
    with pytest.raises(FileExistsError, match="refuse to overwrite"):
        MODULE.atomic_json(path, {"status": "done"})
