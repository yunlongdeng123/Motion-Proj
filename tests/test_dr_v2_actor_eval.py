import math

import numpy as np
import pytest

from motion_proj.dynamic_editing_v2.actor_projection import (
    clip_polygon_to_image,
    lwh_to_wlh,
    project_box,
    transform_matrix,
    wlh_to_lwh,
)
from motion_proj.dynamic_editing_v2.actor_selection import select_cohort
from motion_proj.dynamic_editing_v2.frame_mapping import nearest_camera_frame
from motion_proj.dynamic_editing_v2.nuscenes_actor_eval import (
    build_actor_candidates,
    stream_filter_rows,
    validate_selected_chain,
)
from motion_proj.dynamic_editing_v2.schema import (
    INTERPOLATED_PROVENANCE,
    RAW_PROVENANCE,
    validate_actor_record,
)


def test_instance_token_chain_traversal_is_chronological_and_closed():
    rows = [
        {"token": "b", "sample_token": "s2", "prev": "a", "next": ""},
        {"token": "a", "sample_token": "s1", "prev": "", "next": "b"},
    ]
    ordered = validate_selected_chain(rows, {"s1": 10, "s2": 20})
    assert [row["token"] for row in ordered] == ["a", "b"]
    rows[0]["prev"] = "wrong"
    with pytest.raises(ValueError, match="链断裂"):
        validate_selected_chain(rows, {"s1": 10, "s2": 20})


def test_stream_filter_rows_supports_non_unique_foreign_key(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text(
        '[{"token":"a","sample_token":"s1"},'
        '{"token":"b","sample_token":"s1"},'
        '{"token":"c","sample_token":"s2"}]',
        encoding="utf-8",
    )
    rows = stream_filter_rows(path, {"s1"}, "sample_token")
    assert [row["token"] for row in rows] == ["a", "b"]


def test_timestamp_mapping_checks_token_and_render_index():
    rows = [
        {
            "camera": "CAM_FRONT",
            "timestamp": 100,
            "frame_idx": 14,
            "image_id": 12,
            "sample_data_token": "sd",
            "filename": "image.jpg",
        }
    ]
    sample_data = {
        "sd": {
            "timestamp": 100,
            "sample_token": "sample",
            "calibrated_sensor_token": "cal",
            "ego_pose_token": "ego",
        }
    }
    result = nearest_camera_frame(105, "sample", "CAM_FRONT", rows, sample_data, 10)
    assert result["timestamp_delta_us"] == 5
    assert result["sample_token_match"] is True
    assert result["adgs_processed_frame"] == 4
    assert result["adgs_render_split"] == "test"
    assert result["adgs_render_index"] == 0
    with pytest.raises(ValueError, match="timestamp"):
        nearest_camera_frame(1000, "sample", "CAM_FRONT", rows, sample_data, 10)


def test_timestamp_mapping_prefers_exact_token_over_closer_sweep():
    rows = [
        {"camera": "CAM_FRONT", "timestamp": 99, "frame_idx": 13, "image_id": 9, "sample_data_token": "sweep", "filename": "sweep.jpg"},
        {"camera": "CAM_FRONT", "timestamp": 120, "frame_idx": 14, "image_id": 12, "sample_data_token": "exact", "filename": "exact.jpg"},
    ]
    sample_data = {
        "sweep": {"timestamp": 99, "sample_token": "other", "calibrated_sensor_token": "cal", "ego_pose_token": "ego"},
        "exact": {"timestamp": 120, "sample_token": "sample", "calibrated_sensor_token": "cal", "ego_pose_token": "ego"},
    }
    result = nearest_camera_frame(100, "sample", "CAM_FRONT", rows, sample_data, 50)
    assert result["sample_data_token"] == "exact"
    assert result["mapping_basis"] == "timestamp_plus_exact_sample_token"


def test_wlh_lwh_round_trip():
    assert lwh_to_wlh(wlh_to_lwh([2.0, 4.0, 1.5])) == [2.0, 4.0, 1.5]


def test_global_ego_camera_projection_and_transform_round_trip():
    identity = [1.0, 0.0, 0.0, 0.0]
    annotation = {
        "translation_global": [0.0, 0.0, 10.0],
        "size_wlh": [2.0, 4.0, 2.0],
        "rotation_quaternion": identity,
    }
    calibrated = {
        "translation": [0.0, 0.0, 0.0],
        "rotation": identity,
        "camera_intrinsic": [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
    }
    ego = {"translation": [0.0, 0.0, 0.0], "rotation": identity}
    result = project_box(annotation, calibrated, ego, 100, 100)
    assert result["valid"] is True
    assert np.allclose(result["center_pixel"], [50.0, 50.0])
    matrix = transform_matrix([1.0, 2.0, 3.0], identity)
    point = np.asarray([4.0, 5.0, 6.0, 1.0])
    assert np.allclose(np.linalg.inv(matrix) @ (matrix @ point), point)


def test_behind_camera_projection_has_complete_zero_area_schema():
    identity = [1.0, 0.0, 0.0, 0.0]
    annotation = {
        "translation_global": [0.0, 0.0, -10.0],
        "size_wlh": [2.0, 4.0, 2.0],
        "rotation_quaternion": identity,
    }
    calibrated = {
        "translation": [0.0, 0.0, 0.0],
        "rotation": identity,
        "camera_intrinsic": [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
    }
    result = project_box(
        annotation,
        calibrated,
        {"translation": [0.0, 0.0, 0.0], "rotation": identity},
        100,
        100,
    )
    assert result["valid"] is False
    assert result["visible_area_px"] == 0.0
    assert result["polygon_after_clip"] == []
    assert result["center_inside_image"] is False


def test_image_boundary_clipping():
    clipped = clip_polygon_to_image([[-10, -10], [110, -10], [110, 110], [-10, 110]], 100, 100)
    assert len(clipped) == 4
    assert all(0 <= x <= 99 and 0 <= y <= 99 for x, y in clipped)


def raw_annotation():
    return {
        "sample_token": "sample",
        "timestamp_us": 1,
        "translation_global": [0.0, 0.0, 10.0],
        "size_wlh": [2.0, 4.0, 2.0],
        "rotation_quaternion": [1.0, 0.0, 0.0, 0.0],
        "visibility_token": "4",
        "num_lidar_pts": 1,
        "num_radar_pts": 0,
        "provenance": RAW_PROVENANCE,
    }


def test_raw_and_interpolated_provenance_are_physically_separate():
    actor = {
        "scene_id": "scene",
        "instance_token": "token",
        "category_name": "vehicle.car",
        "raw_annotations": [raw_annotation()],
        "interpolated_visualization": [
            {"sample_token": None, "provenance": INTERPOLATED_PROVENANCE}
        ],
    }
    validate_actor_record(actor)
    actor["interpolated_visualization"][0]["sample_token"] = "sample"
    with pytest.raises(ValueError, match="sample_token"):
        validate_actor_record(actor)


def actor(token, score_area):
    annotations = [
        {**raw_annotation(), "sample_token": f"s{i}", "num_lidar_pts": 2}
        for i in range(4)
    ]
    observations = [
        {"projection": {"valid": True, "center_inside_image": True, "visible_area_px": score_area}}
        for _ in range(4)
    ]
    return {
        "scene_id": "scene",
        "instance_token": token,
        "category_name": "vehicle.car",
        "raw_annotations": annotations,
        "camera_observations": observations,
        "interpolated_visualization": [],
    }


def selection_config():
    return {
        "category_prefix": "vehicle.",
        "thresholds": {
            "min_raw_annotations": 4,
            "min_point_supported_annotations": 2,
            "min_center_inside_observations": 4,
            "min_median_visible_area_px": 500,
        },
        "score_weights": {
            "raw_annotation_count": 1,
            "valid_camera_observation_count": 1,
            "log1p_lidar_radar_point_sum": 1,
            "log1p_median_visible_area_px": 1,
        },
    }


def test_actor_selection_is_deterministic_and_no_visible_camera_fails_closed():
    cohort = select_cohort([actor("b", 1000), actor("a", 1000), actor("z", 500)], selection_config())
    assert cohort["selected"][0] == {"role": "high-support", "instance_token": "a"}
    assert cohort["selected"][1] == {"role": "boundary-support", "instance_token": "z"}
    invisible = actor("x", 1000)
    for observation in invisible["camera_observations"]:
        observation["projection"]["valid"] = False
        observation["projection"]["center_inside_image"] = False
    result = select_cohort([invisible], selection_config())
    assert result["selected"] == []
    assert "center_inside_observation_count" in result["actors"][0]["support_summary"]["failure_reasons"]


def test_missing_annotation_wrong_scene_and_nan_fail_closed():
    with pytest.raises(ValueError, match="缺 sample_annotation"):
        build_actor_candidates("scene", "scene-token", [], [], [], {}, {}, {}, {}, {}, 50_000)
    rows = [{"token": "a", "sample_token": "outside", "prev": "", "next": "", "instance_token": "i"}]
    with pytest.raises(ValueError, match="越出"):
        build_actor_candidates("scene", "scene-token", [], [], rows, {}, {}, {}, {}, {}, 50_000)
    bad = raw_annotation()
    bad["translation_global"][0] = math.nan
    actor_record = {
        "scene_id": "scene",
        "instance_token": "token",
        "category_name": "vehicle.car",
        "raw_annotations": [bad],
    }
    with pytest.raises(ValueError, match="translation"):
        validate_actor_record(actor_record)
