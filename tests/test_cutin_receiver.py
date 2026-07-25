from types import SimpleNamespace

import numpy as np

from motion_proj.resim.cutin_receiver import receiver_centric_cutin


def _line(x0, x1, y):
    xs = np.linspace(x0, x1, int(x1 - x0) + 1)
    return np.stack([xs, np.full_like(xs, y), np.zeros_like(xs)], axis=1)


class _LaneIndex:
    def __init__(self, centerlines, connectivity):
        self.centerlines = centerlines
        self.arc_lengths = {
            token: np.concatenate(
                ([0.0], np.cumsum(np.linalg.norm(np.diff(line[:, :2], axis=0), axis=1)))
            )
            for token, line in centerlines.items()
        }
        self.nmap = SimpleNamespace(connectivity=connectivity)


def _row(actor_id, frame, x, y, token):
    return {
        "actor_id": actor_id,
        "frame_index": frame,
        "xy": [x, y],
        "yaw": 0.0,
        "dimensions_lwh": [4.0, 1.8, 1.5],
        "lane_token": token,
        "match_status": "MATCHED",
    }


def _config():
    return {
        "annotation_keyframe_stride": 5,
        "dense_frame_period_s": 0.1,
        "subject_pre_keyframes": 3,
        "subject_post_keyframes": 3,
        "graph_hops": 2,
        "max_edge_heading_error_deg": 25.0,
        "max_edge_endpoint_gap_m": 4.0,
        "lane_half_width_m": 1.75,
        "pre_box_outside_clearance_m": 0.1,
        "post_box_inside_tolerance_m": 0.35,
        "min_pre_center_outside_keyframes": 2,
        "min_post_inside_keyframes": 2,
        "min_pre_center_lateral_m": 2.4,
        "max_post_center_lateral_m": 0.8,
        "min_lateral_convergence_m": 1.5,
        "min_lateral_convergence_consistency": 0.6,
        "min_pre_side_consistency": 1.0,
        "max_pre_heading_error_deg": 30.0,
        "max_post_heading_error_deg": 25.0,
        "min_settle_duration_s": 1.0,
        "min_median_speed_mps": 0.5,
        "max_receiver_heading_error_deg": 25.0,
        "max_receiver_centerline_distance_m": 1.5,
        "min_receiver_pre_keyframes": 2,
        "min_receiver_post_keyframes": 2,
        "min_receiver_bumper_gap_m": 0.5,
        "max_receiver_bumper_gap_m": 40.0,
    }


def test_parallel_lane_change_requires_body_entry_and_persistent_receiver():
    centerlines = {
        "target_in": _line(-30, 0, 0),
        "target": _line(0, 30, 0),
        "target_out": _line(30, 60, 0),
        "source": _line(-30, 30, 3.5),
    }
    connectivity = {
        "target_in": {"incoming": [], "outgoing": ["target"]},
        "target": {"incoming": ["target_in"], "outgoing": ["target_out"]},
        "target_out": {"incoming": ["target"], "outgoing": []},
        "source": {"incoming": [], "outgoing": []},
    }
    lane_index = _LaneIndex(centerlines, connectivity)
    frames = [0, 5, 10, 15, 20, 25]
    subject_y = [3.5, 3.4, 3.2, 0.8, 0.2, 0.0]
    subject = [
        _row(1, frame, -5 + index * 4, subject_y[index], "source" if index < 3 else "target")
        for index, frame in enumerate(frames)
    ]
    receiver = [
        _row(
            2,
            frame,
            -15 + index * 4,
            0.0,
            "target_in" if -15 + index * 4 < 0 else "target",
        )
        for index, frame in enumerate(frames)
    ]
    tracks = {1: subject, 2: receiver}
    matches = {
        actor_id: {row["frame_index"]: row for row in rows}
        for actor_id, rows in tracks.items()
    }
    result = receiver_centric_cutin(
        1,
        {"token": "source", "start_frame": 0, "end_frame": 10},
        {"token": "target", "start_frame": 15, "end_frame": 25},
        {"type": "lane_change"},
        tracks,
        matches,
        lane_index,
        {frame: frame / 10.0 for frame in frames},
        _config(),
    )
    assert result["event_pass"]
    assert result["receiver_actor_id"] == 2
    assert result["pre_box_outside_count"] == 3
    assert result["post_box_inside_count"] == 3

    for row in subject[:3]:
        row["yaw"] = np.deg2rad(45.0)
    high_angle = receiver_centric_cutin(
        1,
        {"token": "source", "start_frame": 0, "end_frame": 10},
        {"token": "target", "start_frame": 15, "end_frame": 25},
        {"type": "lane_change"},
        tracks,
        matches,
        lane_index,
        {frame: frame / 10.0 for frame in frames},
        _config(),
    )
    assert not high_angle["event_pass"]
    assert not high_angle["subject_checks"]["pre_heading_alignment"]


def test_merge_does_not_reuse_subject_source_vehicle_as_receiver():
    centerlines = {
        "source": _line(-30, 0, 0),
        "receiver_branch": _line(-30, 0, 3.5),
        "target": _line(0, 30, 0),
        "target_out": _line(30, 60, 0),
    }
    connectivity = {
        "source": {"incoming": [], "outgoing": ["target"]},
        "receiver_branch": {"incoming": [], "outgoing": ["target"]},
        "target": {
            "incoming": ["source", "receiver_branch"],
            "outgoing": ["target_out"],
        },
        "target_out": {"incoming": ["target"], "outgoing": []},
    }
    lane_index = _LaneIndex(centerlines, connectivity)
    frames = [0, 5, 10, 15, 20, 25]
    subject = [
        _row(1, frame, -10 + index * 4, 0.0, "source" if index < 3 else "target")
        for index, frame in enumerate(frames)
    ]
    same_stream_rear = [
        _row(2, frame, -20 + index * 4, 0.0, "source" if index < 5 else "target")
        for index, frame in enumerate(frames)
    ]
    tracks = {1: subject, 2: same_stream_rear}
    matches = {
        actor_id: {row["frame_index"]: row for row in rows}
        for actor_id, rows in tracks.items()
    }
    result = receiver_centric_cutin(
        1,
        {"token": "source", "start_frame": 0, "end_frame": 10},
        {"token": "target", "start_frame": 15, "end_frame": 25},
        {"type": "merge"},
        tracks,
        matches,
        lane_index,
        {frame: frame / 10.0 for frame in frames},
        _config(),
    )
    assert not result["event_pass"]
    assert result["receiver_actor_id"] is None
    assert result["reason"] in {
        "subject_body_did_not_cross_target_lane",
        "no_stable_independent_target_lane_receiver",
    }


def test_physical_entry_window_may_lag_map_token_boundary():
    centerlines = {
        "target": _line(-50, 100, 0),
        "source": _line(-50, 100, 3.5),
    }
    connectivity = {
        "target": {"incoming": [], "outgoing": []},
        "source": {"incoming": [], "outgoing": []},
    }
    lane_index = _LaneIndex(centerlines, connectivity)
    frames = list(range(0, 65, 5))
    subject_y = [
        3.5,
        3.5,
        3.5,
        3.4,
        3.2,
        2.5,
        2.0,
        2.0,
        1.5,
        0.9,
        0.3,
        0.1,
        0.0,
    ]
    subject = [
        _row(
            1,
            frame,
            index * 4,
            subject_y[index],
            "source" if frame <= 30 else "target",
        )
        for index, frame in enumerate(frames)
    ]
    receiver = [
        _row(2, frame, index * 4 - 12, 0.0, "target")
        for index, frame in enumerate(frames)
    ]
    tracks = {1: subject, 2: receiver}
    matches = {
        actor_id: {row["frame_index"]: row for row in rows}
        for actor_id, rows in tracks.items()
    }
    config = {
        **_config(),
        "subject_pre_search_keyframes": 9,
        "subject_post_search_keyframes": 9,
        "max_entry_transition_duration_s": 4.0,
        "timestamp_tolerance_s": 0.02,
    }

    result = receiver_centric_cutin(
        1,
        {"token": "source", "start_frame": 0, "end_frame": 30},
        {"token": "target", "start_frame": 35, "end_frame": 60},
        {"type": "lane_change"},
        tracks,
        matches,
        lane_index,
        {frame: frame / 10.0 for frame in frames},
        config,
    )

    assert result["event_pass"]
    assert result["post_window_frames"][0] >= 45
    assert result["entry_transition_duration_s"] <= 4.0
