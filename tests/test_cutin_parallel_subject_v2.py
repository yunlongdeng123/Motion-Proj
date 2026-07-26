from types import SimpleNamespace

import numpy as np

from motion_proj.resim.cutin_receiver import evaluate_parallel_subject_v2


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


def _row(frame, x, y, *, yaw=0.0):
    return {
        "actor_id": 1,
        "frame_index": frame,
        "xy": [x, y],
        "yaw": yaw,
        "dimensions_lwh": [4.0, 1.8, 1.5],
        "observation_source": "raw_2hz",
    }


def _config():
    return {
        "strict": {
            "lane_half_width_m": 1.75,
            "subject": {
                "raw_frame_stride": 5,
                "raw_pre_keyframes": 3,
                "raw_post_keyframes": 3,
                "min_pre_center_outside_keyframes": 2,
                "min_post_box_inside_keyframes": 2,
                "min_pre_center_lateral_m": 1.85,
                "max_post_center_lateral_m": 0.80,
                "min_lateral_convergence_m": 1.40,
                "min_lateral_convergence_consistency": 0.80,
                "min_pre_side_consistency": 0.80,
                "min_source_target_shift_m": 2.0,
                "max_source_target_shift_m": 6.0,
                "min_parallel_overlap_m": 8.0,
                "max_source_target_heading_error_deg": 10.0,
                "max_pre_heading_error_deg": 15.0,
                "max_post_heading_error_deg": 10.0,
                "max_accumulated_yaw_change_deg": 15.0,
                "min_median_speed_mps": 1.0,
                "min_settle_duration_s": 1.0,
                "timestamp_tolerance_s": 0.02,
            },
        }
    }


def _standard_inputs():
    lane_index = _LaneIndex(
        {"source": _line(-30, 40, 3.5), "target": _line(-30, 40, 0.0)},
        {"source": {"incoming": [], "outgoing": []}, "target": {"incoming": [], "outgoing": []}},
    )
    pre = [_row(0, -10, 3.5), _row(5, -6, 3.3), _row(10, -2, 2.5)]
    post = [_row(15, 2, 0.6), _row(20, 6, 0.2), _row(25, 10, 0.0)]
    frame_times = {row["frame_index"]: row["frame_index"] / 10.0 for row in [*pre, *post]}
    return lane_index, pre, post, frame_times


def test_raw_pre_outside_and_post_inside_pass_standard_cutin():
    lane_index, pre, post, frame_times = _standard_inputs()
    result = evaluate_parallel_subject_v2(
        actor_id=1,
        source_token="source",
        target_token="target",
        pre_rows=pre,
        post_rows=post,
        lane_index=lane_index,
        frame_times_s=frame_times,
        config=_config(),
    )
    assert result["status"] == "PASS"
    assert all(result["checks"].values())


def test_route_continuation_not_parallel_lane_change():
    lane_index, pre, post, frame_times = _standard_inputs()
    lane_index.nmap.connectivity["source"]["outgoing"] = ["target"]
    result = evaluate_parallel_subject_v2(
        actor_id=1,
        source_token="source",
        target_token="target",
        pre_rows=pre,
        post_rows=post,
        lane_index=lane_index,
        frame_times_s=frame_times,
        config=_config(),
    )
    assert result["status"] == "FAIL"
    assert result["reasons"][0] == "SOURCE_TARGET_NOT_PARALLEL"


def test_post_heading_instability_rejects_map_jitter():
    lane_index, pre, post, frame_times = _standard_inputs()
    for row in post:
        row["yaw"] = np.deg2rad(20.0)
    result = evaluate_parallel_subject_v2(
        actor_id=1,
        source_token="source",
        target_token="target",
        pre_rows=pre,
        post_rows=post,
        lane_index=lane_index,
        frame_times_s=frame_times,
        config=_config(),
    )
    assert result["status"] == "FAIL"
    assert "POST_HEADING_UNSTABLE" in result["reasons"]


def test_interpolation_cannot_supply_hard_support():
    lane_index, pre, post, frame_times = _standard_inputs()
    pre[-1]["frame_index"] = 11
    pre[-1]["observation_source"] = "interpolated_10hz"
    result = evaluate_parallel_subject_v2(
        actor_id=1,
        source_token="source",
        target_token="target",
        pre_rows=pre,
        post_rows=post,
        lane_index=lane_index,
        frame_times_s=frame_times,
        config=_config(),
    )
    assert result["status"] == "ABSTAIN"
    assert result["reasons"] == ["INTERPOLATION_ONLY"]


def test_boundary_raw_entry_is_abstain_never_relaxed_to_pass():
    lane_index = _LaneIndex(
        {"source": _line(-30, 40, 1.9), "target": _line(-30, 40, 0.0)},
        {"source": {"incoming": [], "outgoing": []}, "target": {"incoming": [], "outgoing": []}},
    )
    pre = [_row(0, -10, 1.6), _row(5, -6, 1.5), _row(10, -2, 1.4)]
    post = [_row(15, 2, 0.6), _row(20, 6, 0.2), _row(25, 10, 0.0)]
    frame_times = {row["frame_index"]: row["frame_index"] / 10.0 for row in [*pre, *post]}
    result = evaluate_parallel_subject_v2(
        actor_id=1,
        source_token="source",
        target_token="target",
        pre_rows=pre,
        post_rows=post,
        lane_index=lane_index,
        frame_times_s=frame_times,
        config=_config(),
    )
    assert result["status"] == "ABSTAIN"
    assert result["reasons"] == ["BOUNDARY_RAW_ENTRY_EVIDENCE"]
    assert result["geometry"]["boundary_raw_entry_evidence"] is True
