from types import SimpleNamespace

import numpy as np

from motion_proj.resim.cutin_receiver import receiver_centric_cutin_v2


def _line(x0, x1, y):
    xs = np.linspace(x0, x1, int(x1 - x0) + 1)
    return np.stack([xs, np.full_like(xs, y), np.zeros_like(xs)], axis=1)


class _LaneIndex:
    def __init__(self):
        self.centerlines = {
            "source": _line(-30, 40, 3.5),
            "target_in": _line(-30, 0, 0.0),
            "target": _line(0, 30, 0.0),
            "target_out": _line(30, 60, 0.0),
        }
        self.arc_lengths = {
            token: np.concatenate(
                ([0.0], np.cumsum(np.linalg.norm(np.diff(line[:, :2], axis=0), axis=1)))
            )
            for token, line in self.centerlines.items()
        }
        self.nmap = SimpleNamespace(
            connectivity={
                "source": {"incoming": [], "outgoing": []},
                "target_in": {"incoming": [], "outgoing": ["target"]},
                "target": {"incoming": ["target_in"], "outgoing": ["target_out"]},
                "target_out": {"incoming": ["target"], "outgoing": []},
            }
        )


def _row(actor_id, frame, x, y, token):
    return {
        "actor_id": actor_id,
        "frame_index": frame,
        "xy": [x, y],
        "yaw": 0.0,
        "dimensions_lwh": [4.0, 1.8, 1.5],
        "lane_token": token,
        "observation_source": "raw_2hz",
    }


def _config():
    return {
        "strict": {
            "hard_evidence_source": "raw_2hz_annotations",
            "lane_half_width_m": 1.75,
            "lane_width_source": "configured_nominal_fallback",
            "subject": {
                "raw_frame_stride": 5,
                "raw_pre_keyframes": 3,
                "raw_post_keyframes": 3,
                "raw_pre_search_keyframes": 3,
                "raw_post_search_keyframes": 3,
                "max_entry_transition_duration_s": 4.0,
                "min_pre_center_outside_keyframes": 2,
                "min_post_box_inside_keyframes": 2,
                "min_pre_center_lateral_m": 1.85,
                "max_post_center_lateral_m": 0.8,
                "min_lateral_convergence_m": 1.4,
                "min_lateral_convergence_consistency": 0.8,
                "min_pre_side_consistency": 0.8,
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


def _inputs():
    frames = [0, 5, 10, 15, 20, 25]
    subject = [
        _row(1, frame, -10 + index * 4, [3.5, 3.3, 2.5, 0.6, 0.2, 0.0][index], "source")
        for index, frame in enumerate(frames)
    ]
    receiver = [
        _row(2, frame, -22 + index * 4, 0.0, "target_in")
        for index, frame in enumerate(frames)
    ]
    tracks = {1: subject, 2: receiver}
    matches = {actor: {row["frame_index"]: row for row in rows} for actor, rows in tracks.items()}
    return tracks, matches, {frame: frame / 10.0 for frame in frames}


def test_final_verifier_passes_only_raw_parallel_cutin():
    tracks, matches, times = _inputs()
    result = receiver_centric_cutin_v2(
        1,
        {"token": "source", "start_frame": 0, "end_frame": 10},
        {"token": "target", "start_frame": 15, "end_frame": 25},
        {"type": "lane_change"},
        tracks,
        matches,
        _LaneIndex(),
        times,
        _config(),
    )
    assert result["status"] == "PASS"
    assert result["machine_positive"] is True
    assert result["hard_evidence_source"] == "raw_2hz_annotations"
    assert result["receiver"]["selected_actor_id"] == 2


def test_final_verifier_never_promotes_branch_merge():
    tracks, matches, times = _inputs()
    result = receiver_centric_cutin_v2(
        1,
        {"token": "source", "start_frame": 0, "end_frame": 10},
        {"token": "target", "start_frame": 15, "end_frame": 25},
        {"type": "merge"},
        tracks,
        matches,
        _LaneIndex(),
        times,
        _config(),
    )
    assert result["status"] == "ABSTAIN"
    assert result["machine_positive"] is False
    assert result["primary_reason"] == "UNSUPPORTED_BRANCH_MERGE_MODE"
