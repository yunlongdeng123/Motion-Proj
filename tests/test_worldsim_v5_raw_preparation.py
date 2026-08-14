from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/prepare_worldsim_v5_drivestudio_raw.py"
SPEC = importlib.util.spec_from_file_location("worldsim_v5_raw_preparation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _config(tmp_path: Path, count: int = 8) -> Path:
    rows = "\n".join(
        f"    - {{scene: scene-{index:04d}, scene_index: {index}}}"
        for index in range(count)
    )
    path = tmp_path / "m1.yaml"
    path.write_text(
        "fresh_cohort_binding:\n"
        f"  development_scenes:\n{rows}\n"
        "  validation_quality_read: false\n"
        "  test_quality_read: false\n",
        encoding="utf-8",
    )
    return path


def test_load_development_scenes_requires_frozen_eight(tmp_path: Path) -> None:
    rows = MODULE.load_development_scenes(_config(tmp_path))
    assert len(rows) == 8
    assert rows[3] == {"scene_name": "scene-0003", "scene_index": 3}
    with pytest.raises(RuntimeError, match="必须为 8 scenes"):
        MODULE.load_development_scenes(_config(tmp_path, count=7))


def test_seed_index_filters_required_and_normalizes_shards(tmp_path: Path) -> None:
    seed = tmp_path / "seed.json"
    seed.write_text(
        json.dumps(
            {
                "samples/CAM_FRONT/a.jpg": "04",
                "samples/CAM_FRONT/ignored.jpg": "v1.0-trainval05_blobs.tgz",
            }
        ),
        encoding="utf-8",
    )
    index = tmp_path / "index.json"
    mapping = MODULE.seed_index(
        index, {"samples/CAM_FRONT/a.jpg"}, [seed]
    )
    assert mapping == {
        "samples/CAM_FRONT/a.jpg": "v1.0-trainval04_blobs.tgz"
    }
    assert json.loads(index.read_text(encoding="utf-8")) == mapping


def test_seed_index_rejects_conflicting_evidence(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"a": "01"}), encoding="utf-8")
    second.write_text(json.dumps({"a": "02"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="member shard 冲突"):
        MODULE.seed_index(tmp_path / "index.json", {"a"}, [first, second])
