from types import SimpleNamespace

import numpy as np

from motion_proj.resim.cutin_receiver import evaluate_receiver_across_corridors_v2


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
    }


def _config():
    return {
        "strict": {
            "lane_half_width_m": 1.75,
            "subject": {"raw_frame_stride": 5},
            "receiver": {
                "class_prefix": "vehicle.",
                "min_raw_pre_support": 2,
                "min_raw_post_support": 2,
                "min_total_non_null_support": 5,
                "max_missing_required_frames": 1,
                "min_local_displacement_m": 1.0,
                "min_median_longitudinal_speed_mps": 1.0,
                "max_heading_error_deg": 15.0,
                "max_centerline_distance_m": 1.5,
                "min_bumper_gap_m": 0.5,
                "max_bumper_gap_m": 40.0,
            },
            "corridor": {
                "graph_hops": 2,
                "max_edge_heading_error_deg": 15.0,
                "max_edge_endpoint_gap_m": 4.0,
            },
        }
    }


def _standard_inputs():
    centerlines = {
        "source": _line(-30, 40, 3.5),
        "target_in": _line(-30, 0, 0.0),
        "target": _line(0, 30, 0.0),
        "target_out": _line(30, 60, 0.0),
    }
    connectivity = {
        "source": {"incoming": [], "outgoing": []},
        "target_in": {"incoming": [], "outgoing": ["target"]},
        "target": {"incoming": ["target_in"], "outgoing": ["target_out"]},
        "target_out": {"incoming": ["target"], "outgoing": []},
    }
    frames = [0, 5, 10, 15, 20, 25]
    subject = [
        _row(1, frame, -10 + index * 4, 3.0 if index < 3 else 0.2, "source")
        for index, frame in enumerate(frames)
    ]
    receiver = [
        _row(
            2,
            frame,
            -22 + index * 4,
            0.0,
            "target_in" if -22 + index * 4 < 0 else "target",
        )
        for index, frame in enumerate(frames)
    ]
    matches = {
        1: {row["frame_index"]: row for row in subject},
        2: {row["frame_index"]: row for row in receiver},
    }
    frame_times = {frame: frame / 10.0 for frame in frames}
    return _LaneIndex(centerlines, connectivity), subject[:3], subject[3:], matches, frame_times


def _evaluate(matches=None):
    lane_index, pre, post, default_matches, frame_times = _standard_inputs()
    return evaluate_receiver_across_corridors_v2(
        actor_id=1,
        pre_rows=pre,
        post_rows=post,
        matches_by_actor=matches or default_matches,
        lane_index=lane_index,
        source_token="source",
        target_token="target",
        frame_times_s=frame_times,
        config=_config(),
    )


def test_single_missing_receiver_frame_allowed_with_unique_identity():
    lane_index, pre, post, matches, frame_times = _standard_inputs()
    del matches[2][10]
    result = evaluate_receiver_across_corridors_v2(
        actor_id=1,
        pre_rows=pre,
        post_rows=post,
        matches_by_actor=matches,
        lane_index=lane_index,
        source_token="source",
        target_token="target",
        frame_times_s=frame_times,
        config=_config(),
    )
    assert result["status"] == "PASS"
    assert result["receiver"]["selected_actor_id"] == 2
    assert result["receiver"]["missing_frames"] == [10]


def test_receiver_identity_switch_rejected_even_when_support_count_passes():
    lane_index, pre, post, matches, frame_times = _standard_inputs()
    last = matches[2].pop(25)
    matches[38] = {25: {**last, "actor_id": 38}}
    result = evaluate_receiver_across_corridors_v2(
        actor_id=1,
        pre_rows=pre,
        post_rows=post,
        matches_by_actor=matches,
        lane_index=lane_index,
        source_token="source",
        target_token="target",
        frame_times_s=frame_times,
        config=_config(),
    )
    assert result["status"] == "FAIL"
    assert result["reasons"][0] == "RECEIVER_IDENTITY_SWITCH"


def test_stationary_receiver_rejected_and_intermediate_actor_breaks_path_clear():
    lane_index, pre, post, matches, frame_times = _standard_inputs()
    for row in matches[2].values():
        row["xy"] = [-22.0, 0.0]
    stationary = evaluate_receiver_across_corridors_v2(
        actor_id=1,
        pre_rows=pre,
        post_rows=post,
        matches_by_actor=matches,
        lane_index=lane_index,
        source_token="source",
        target_token="target",
        frame_times_s=frame_times,
        config=_config(),
    )
    assert stationary["status"] == "FAIL"
    assert "RECEIVER_WRONG_DIRECTION" in stationary["reasons"]

    lane_index, pre, post, matches, frame_times = _standard_inputs()
    matches[3] = {15: _row(3, 15, -4.0, 0.0, "target")}
    blocked = evaluate_receiver_across_corridors_v2(
        actor_id=1,
        pre_rows=pre,
        post_rows=post,
        matches_by_actor=matches,
        lane_index=lane_index,
        source_token="source",
        target_token="target",
        frame_times_s=frame_times,
        config=_config(),
    )
    assert blocked["status"] == "FAIL"
    assert "PATH_NOT_CLEAR" in blocked["reasons"]


def test_multiple_corridors_different_receivers_abstain():
    centerlines = {
        "source": _line(-30, 40, 3.5),
        "target": _line(0, 40, 0.0),
        "in_a": _line(-30, 0, 0.0),
        "in_b": _line(-30, 0, 0.0),
    }
    connectivity = {
        "source": {"incoming": [], "outgoing": []},
        "target": {"incoming": ["in_a", "in_b"], "outgoing": []},
        "in_a": {"incoming": [], "outgoing": ["target"]},
        "in_b": {"incoming": [], "outgoing": ["target"]},
    }
    lane_index = _LaneIndex(centerlines, connectivity)
    frames = [0, 5, 10, 15, 20, 25]
    subject = [_row(1, frame, -10 + index * 4, 0.2, "target") for index, frame in enumerate(frames)]
    first = [_row(2, frame, -22 + index * 4, 0.0, "in_a") for index, frame in enumerate(frames)]
    second = [_row(3, frame, -22 + index * 4, 0.0, "in_b") for index, frame in enumerate(frames)]
    matches = {
        1: {row["frame_index"]: row for row in subject},
        2: {row["frame_index"]: row for row in first},
        3: {row["frame_index"]: row for row in second},
    }
    result = evaluate_receiver_across_corridors_v2(
        actor_id=1,
        pre_rows=subject[:3],
        post_rows=subject[3:],
        matches_by_actor=matches,
        lane_index=lane_index,
        source_token="source",
        target_token="target",
        frame_times_s={frame: frame / 10.0 for frame in frames},
        config=_config(),
    )
    assert result["status"] == "ABSTAIN"
    assert result["reasons"] == ["AMBIGUOUS_RECEIVER_CORRIDOR"]
