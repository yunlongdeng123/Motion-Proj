from __future__ import annotations

from pathlib import Path

from scripts.run_worldsim_v5_m3_clip_inventory import (
    enumerate_windows,
    load_config,
    select_scene_clip,
)


PROJECT = Path(__file__).resolve().parents[1]


def _annotations(indices: list[int], lidar: int = 3, visibility: int = 4) -> list[dict]:
    return [
        {
            "sample_token": f"sample-{index}",
            "num_lidar_pts": lidar,
            "visibility_token": str(visibility),
        }
        for index in indices
    ]


def test_enumerate_windows_requires_seven_consecutive_keyframes() -> None:
    sample_index = {f"sample-{index}": index for index in range(9)}
    sample_timestamp = {
        f"sample-{index}": 1_000_000 + index * 500_000 for index in range(9)
    }
    windows = enumerate_windows(
        _annotations(list(range(9))),
        sample_index=sample_index,
        sample_timestamp=sample_timestamp,
        keyframe_count=7,
        maximum_gap_seconds=0.51,
        minimum_lidar_points=1,
        minimum_visibility=2,
    )
    assert len(windows) == 3
    broken = enumerate_windows(
        _annotations([0, 1, 2, 4, 5, 6, 7]),
        sample_index=sample_index,
        sample_timestamp=sample_timestamp,
        keyframe_count=7,
        maximum_gap_seconds=0.51,
        minimum_lidar_points=1,
        minimum_visibility=2,
    )
    assert broken == []


def test_selection_uses_frozen_metadata_rank_not_quality() -> None:
    priority = {"vehicle.car": 0, "vehicle.truck": 1}
    candidates = [
        {
            "category_name": "vehicle.truck",
            "minimum_lidar_points": 20,
            "total_lidar_points": 200,
            "instance_token": "a",
            "start_sample_index": 0,
        },
        {
            "category_name": "vehicle.car",
            "minimum_lidar_points": 2,
            "total_lidar_points": 20,
            "instance_token": "b",
            "start_sample_index": 1,
        },
        {
            "category_name": "vehicle.car",
            "minimum_lidar_points": 2,
            "total_lidar_points": 25,
            "instance_token": "c",
            "start_sample_index": 2,
        },
    ]
    selected = select_scene_clip(candidates, category_priority=priority)
    assert selected is not None
    assert selected["instance_token"] == "c"


def test_config_keeps_all_quality_and_blob_reads_locked() -> None:
    config = load_config(PROJECT / "configs/worldsim_v5/m3_development_clip_inventory_v1.yaml")
    assert config["clip_policy"]["keyframe_count"] == 7
    assert config["scope"]["image_read"] is False
    assert config["scope"]["lidar_blob_read"] is False
    assert config["scope"]["development_quality_read"] is False
    assert config["next_gate"]["render_quality_unlocked"] is False
