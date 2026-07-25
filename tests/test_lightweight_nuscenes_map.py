import importlib.metadata
import importlib.util
import json

import numpy as np

from motion_proj.resim.lightweight_nuscenes_map import (
    LightweightNuScenesMap,
    _discretize_lane,
)


def _reference_module():
    source = importlib.metadata.distribution("nuscenes-devkit").locate_file(
        "nuscenes/map_expansion/arcline_path_utils.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_nuscenes_arcline_reference",
        source,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_discretization_matches_nuscenes_reference():
    lane = [
        {
            "start_pose": [1.0, 2.0, 0.2],
            "end_pose": [0.0, 0.0, 0.0],
            "shape": "LSR",
            "radius": 12.0,
            "segment_length": [4.0, 8.0, 3.0],
        }
    ]
    expected = _reference_module().discretize_lane(lane, 0.5)
    actual = _discretize_lane(lane, 0.5)
    assert np.allclose(actual, expected, atol=1e-12)


def test_lightweight_map_exposes_lane_connectivity_and_centerlines(tmp_path):
    expansion = tmp_path / "maps" / "expansion"
    expansion.mkdir(parents=True)
    lane_path = {
        "start_pose": [0.0, 0.0, 0.0],
        "end_pose": [5.0, 0.0, 0.0],
        "shape": "LSR",
        "radius": 10.0,
        "segment_length": [0.0, 5.0, 0.0],
    }
    payload = {
        "version": "1.3",
        "lane": [{"token": "lane-a"}],
        "lane_connector": [{"token": "connector-b"}],
        "arcline_path_3": {
            "lane-a": [lane_path],
            "connector-b": [lane_path],
        },
        "connectivity": {
            "lane-a": {"incoming": [], "outgoing": ["connector-b"]},
            "connector-b": {"incoming": ["lane-a"], "outgoing": []},
        },
    }
    (expansion / "boston-seaport.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    nmap = LightweightNuScenesMap(tmp_path, "boston-seaport")

    assert [row["token"] for row in nmap.lane] == ["lane-a"]
    assert nmap.connectivity["lane-a"]["outgoing"] == ["connector-b"]
    assert set(nmap.discretize_lanes(["lane-a"], 0.5)) == {"lane-a"}
