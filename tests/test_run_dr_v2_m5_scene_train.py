from pathlib import Path

from scripts.run_dr_v2_m5_scene_train import processed_ready, validate_processed


def test_processed_ready_requires_exact_counts() -> None:
    counts = {
        "images": 1176,
        "lidar": 196,
        "lidar_pose": 196,
        "extrinsics": 1176,
        "sky_masks": 0,
        "dynamic_all": 1176,
        "dynamic_human": 1176,
        "dynamic_vehicle": 1176,
        "instances_info": 1,
    }
    assert processed_ready(counts)
    counts["images"] = 1175
    assert not processed_ready(counts)


def test_validate_processed_empty_tree(tmp_path: Path) -> None:
    assert validate_processed(tmp_path) == {
        "images": 0,
        "lidar": 0,
        "lidar_pose": 0,
        "extrinsics": 0,
        "sky_masks": 0,
        "dynamic_all": 0,
        "dynamic_human": 0,
        "dynamic_vehicle": 0,
        "instances_info": 0,
    }
