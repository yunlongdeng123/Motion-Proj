import math

from resim.event_first_n1_event import (
    _angle_error,
    _gate_decision,
    _stable_runs,
)


def test_angle_error_wraps_at_pi():
    assert math.degrees(_angle_error(math.radians(179), math.radians(-179))) < 3


def test_stable_runs_split_unknown_and_short_segments():
    rows = [
        {"frame_index": 0, "lane_token": "a"},
        {"frame_index": 1, "lane_token": "a"},
        {"frame_index": 2, "lane_token": None},
        {"frame_index": 3, "lane_token": "b"},
        {"frame_index": 4, "lane_token": "b"},
        {"frame_index": 5, "lane_token": "b"},
    ]

    assert _stable_runs(rows, 2) == [
        {"token": "a", "start_frame": 0, "end_frame": 1, "frame_count": 2},
        {"token": "b", "start_frame": 3, "end_frame": 5, "frame_count": 3},
    ]


def test_gate_fails_closed_when_pairs_are_missing():
    summary = {
        "eligible_actor_count_by_scene": {"003": 3, "004": 3},
        "positive_event_count": 2,
        "negative_event_count": 2,
        "same_actor_pair_count": 0,
        "positive_scene_count": 2,
    }
    gates = {
        "min_eligible_actors_per_scene": 2,
        "min_positive_events": 2,
        "min_negative_events": 2,
        "min_same_actor_pairs": 2,
        "min_positive_scenes": 2,
        "unknown_is_positive": False,
        "noninteractive_transition_is_positive": False,
    }

    passed, checks = _gate_decision(summary, gates)

    assert not passed
    assert not checks["same_actor_pairs"]


def test_gate_passes_only_when_all_preregistered_checks_pass():
    summary = {
        "eligible_actor_count_by_scene": {"003": 3, "004": 3},
        "positive_event_count": 2,
        "negative_event_count": 2,
        "same_actor_pair_count": 2,
        "positive_scene_count": 2,
    }
    gates = {
        "min_eligible_actors_per_scene": 2,
        "min_positive_events": 2,
        "min_negative_events": 2,
        "min_same_actor_pairs": 2,
        "min_positive_scenes": 2,
        "unknown_is_positive": False,
        "noninteractive_transition_is_positive": False,
    }

    passed, checks = _gate_decision(summary, gates)

    assert passed
    assert all(checks.values())
