import math
from types import SimpleNamespace

import numpy as np

from motion_proj.resim.event_interaction import temporal_relation
from motion_proj.resim.event_kinematics import (
    motion_features,
    project_to_polyline,
)


class FakeLaneIndex:
    def __init__(self, lines: dict[str, list[tuple[float, float]]], connectivity: dict):
        self.nmap = SimpleNamespace(connectivity=connectivity)
        self.centerlines = {}
        self.arc_lengths = {}
        for token, points in lines.items():
            xy = np.asarray(points, dtype=float)
            deltas = np.diff(xy, axis=0)
            headings = np.asarray(
                [math.atan2(delta[1], delta[0]) for delta in deltas]
                + [math.atan2(deltas[-1, 1], deltas[-1, 0])]
            )
            self.centerlines[token] = np.column_stack([xy, headings])
            distances = np.linalg.norm(deltas, axis=1)
            self.arc_lengths[token] = np.concatenate([[0.0], np.cumsum(distances)])


KINEMATICS = {
    "annotation_keyframe_stride": 5,
    "dense_frame_period_s": 0.1,
    "keyframes_each_side": 3,
    "branch_lookback_m": 8.0,
    "min_median_speed_mps": 0.1,
    "max_net_course_change_deg": 30.0,
    "max_net_yaw_change_deg": 30.0,
    "max_course_deviation_deg": 35.0,
    "max_lane_join_heading_error_deg": 30.0,
    "min_lateral_displacement_m": 0.8,
    "min_lateral_span_m": 1.0,
    "min_peak_lateral_speed_mps": 0.2,
    "max_peak_lateral_speed_mps": 6.0,
    "min_lateral_direction_consistency": 0.6,
    "min_longitudinal_progress_m": 1.0,
    "max_source_distance_m": 2.5,
    "max_target_distance_m": 2.5,
    "min_lane_preference_margin_m": 0.4,
    "merge_min_target_incoming_lanes": 2,
    "min_merge_branch_alignment_disadvantage_deg": 3.0,
    "min_merge_branch_convergence_angle_deg": 3.0,
    "max_acceleration_mps2": 20.0,
    "max_yaw_rate_deg_s": 90.0,
}


def row(frame: int, x: float, y: float, yaw: float = 0.0) -> dict:
    return {
        "frame_index": frame,
        "xy": [x, y],
        "yaw": yaw,
        "dimensions_lwh": [4.0, 2.0, 1.5],
    }


def test_project_to_polyline_is_continuous_and_signed() -> None:
    line = np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    result = project_to_polyline([4.25, 2.0], line, np.asarray([0.0, 10.0]))
    assert result["s_m"] == 4.25
    assert result["distance_m"] == 2.0
    assert result["signed_lateral_m"] == 2.0


def test_mainline_route_continuation_is_not_merge() -> None:
    lane_index = FakeLaneIndex(
        {
            "source": [(0, 0), (10, 0)],
            "alternate": [(0, 5), (8, 3), (10, 0)],
            "target": [(10, 0), (30, 0)],
        },
        {
            "source": {"incoming": [], "outgoing": ["target"]},
            "alternate": {"incoming": [], "outgoing": ["target"]},
            "target": {"incoming": ["source", "alternate"], "outgoing": []},
        },
    )
    rows = [
        row(0, 2, 0),
        row(5, 5, 0),
        row(10, 8, 0),
        row(15, 12, 0),
        row(20, 16, 0),
        row(25, 20, 0),
    ]
    result = motion_features(
        rows,
        {"token": "source", "start_frame": 0, "end_frame": 10},
        {"token": "target", "start_frame": 15, "end_frame": 25},
        lane_index,
        {"type": "merge"},
        KINEMATICS,
    )
    assert result["join_geometry"]["source_is_best_aligned_incoming"]
    assert not result["checks"]["branch_merge_evidence"]
    assert not result["physical_motion_pass"]


def test_parallel_lane_change_uses_road_relative_preference_flip() -> None:
    lane_index = FakeLaneIndex(
        {
            "source": [(0, 0), (40, 0)],
            "target": [(0, 3), (40, 3)],
        },
        {
            "source": {"incoming": [], "outgoing": []},
            "target": {"incoming": [], "outgoing": []},
        },
    )
    rows = [
        row(0, 2, 0.1),
        row(5, 6, 0.3),
        row(10, 10, 0.8),
        row(15, 14, 2.2),
        row(20, 18, 2.7),
        row(25, 22, 2.9),
    ]
    result = motion_features(
        rows,
        {"token": "source", "start_frame": 0, "end_frame": 10},
        {"token": "target", "start_frame": 15, "end_frame": 25},
        lane_index,
        {"type": "lane_change"},
        KINEMATICS,
    )
    assert result["checks"]["lane_preference_flip"]
    assert result["checks"]["parallel_lane_change_evidence"]
    assert result["physical_motion_pass"]


def test_normal_turn_fails_course_gate() -> None:
    lane_index = FakeLaneIndex(
        {
            "source": [(0, 0), (10, 0)],
            "target": [(10, 0), (10, 20)],
        },
        {
            "source": {"incoming": [], "outgoing": []},
            "target": {"incoming": [], "outgoing": []},
        },
    )
    rows = [
        row(0, 2, 0, 0),
        row(5, 5, 0, 0),
        row(10, 8, 0, 0),
        row(15, 10, 2, math.pi / 2),
        row(20, 10, 5, math.pi / 2),
        row(25, 10, 8, math.pi / 2),
    ]
    result = motion_features(
        rows,
        {"token": "source", "start_frame": 0, "end_frame": 10},
        {"token": "target", "start_frame": 15, "end_frame": 25},
        lane_index,
        {"type": "lane_change"},
        KINEMATICS,
    )
    assert not result["checks"]["not_normal_turn"]
    assert not result["physical_motion_pass"]


INTERACTION = {
    "annotation_keyframe_stride": 5,
    "graph_hops": 2,
    "max_edge_heading_error_deg": 20.0,
    "max_edge_endpoint_gap_m": 3.0,
    "max_actor_heading_error_deg": 25.0,
    "temporal_keyframe_count": 3,
    "min_identity_support_keyframes": 2,
    "min_identity_support_fraction": 2 / 3,
    "min_bumper_gap_m": 0.5,
    "max_bumper_gap_m": 60.0,
}


def matched(frame: int, x: float, actor_id: int) -> dict:
    return {
        **row(frame, x, 0.0),
        "actor_id": actor_id,
        "lane_token": "target",
        "match_status": "MATCHED",
    }


def test_temporal_relation_requires_persistent_front_and_rear() -> None:
    lane_index = FakeLaneIndex(
        {"target": [(0, 0), (100, 0)]},
        {"target": {"incoming": [], "outgoing": []}},
    )
    matches = {
        0: {frame: matched(frame, 50 + frame / 5, 0) for frame in (0, 5, 10, 15, 20)},
        1: {frame: matched(frame, 70 + frame / 5, 1) for frame in (0, 5, 10, 15, 20)},
        2: {frame: matched(frame, 30 + frame / 5, 2) for frame in (0, 5, 10, 15, 20)},
    }
    times = {frame: frame / 10 for frame in (0, 5, 10, 15, 20)}
    result = temporal_relation(0, 10, "target", matches, lane_index, times, INTERACTION)
    assert result["status"] == "PASS"
    assert result["identity_support_keyframes"] == 3
    assert result["front_bumper_gap_m"] == 16.0
    assert result["rear_bumper_gap_m"] == 16.0


def test_temporal_relation_rejects_one_frame_identity() -> None:
    lane_index = FakeLaneIndex(
        {"target": [(0, 0), (100, 0)]},
        {"target": {"incoming": [], "outgoing": []}},
    )
    matches = {
        0: {frame: matched(frame, 50, 0) for frame in (5, 10, 15)},
        1: {10: matched(10, 70, 1)},
        2: {10: matched(10, 30, 2)},
        3: {5: matched(5, 70, 3), 15: matched(15, 70, 3)},
        4: {5: matched(5, 30, 4), 15: matched(15, 30, 4)},
    }
    times = {frame: frame / 10 for frame in (0, 5, 10, 15, 20)}
    result = temporal_relation(0, 10, "target", matches, lane_index, times, INTERACTION)
    assert result["status"] == "FAIL"
    assert result["identity_support_keyframes"] == 1
