from __future__ import annotations

from types import SimpleNamespace

from motion_proj.preference.selective_order import decide_selective_relation


SETTINGS = {
    "maximum_camera_distance_px": 1.0,
    "maximum_active_fraction_drop": 0.05,
    "maximum_survival_drop": 0.05,
    "maximum_motion_drop_fraction": 0.10,
    "maximum_motion_increase_fraction": 1.0,
    "activity_equivalence_absolute": 0.05,
    "motion_equivalence_fraction": 0.10,
    "high_confidence_margin": 1.0,
}


def _support() -> SimpleNamespace:
    return SimpleNamespace(valid=True, candidate_a="a", candidate_b="b", start_frame=0, end_frame=4)


def _activity(speed: float = 1.0) -> dict:
    return {
        "active_fraction": 1.0,
        "survival": 1.0,
        "residual_speed": speed,
        "net_residual_displacement": speed * 2.0,
    }


def _intervals(lower: float, upper: float) -> dict:
    return {
        name: {"lower": lower, "upper": upper, "holm_significant": True}
        for name in ("punc", "acceleration", "curvature", "coherence")
    }


def test_strict_tie_and_low_motion_abstention_are_distinct() -> None:
    evidence = SimpleNamespace(
        valid=True, reason=None, camera_distance_px=0.1,
        activity_a=_activity(), activity_b=_activity(),
    )
    strict = decide_selective_relation(
        condition_id="c", edge_id="e", support=_support(), evidence=evidence,
        intervals=_intervals(3.0, 4.0),
        ropes={name: 1.0 for name in _intervals(0, 0)},
        strict_threshold=2.0, quality={"comparable": True}, settings=SETTINGS,
    )
    assert strict["relation"] == "strict"
    assert strict["winner_candidate_id"] == "a"

    tie_intervals = {
        name: {"lower": -0.5, "upper": 0.5, "holm_significant": False}
        for name in _intervals(0, 0)
    }
    tie = decide_selective_relation(
        condition_id="c", edge_id="e", support=_support(), evidence=evidence,
        intervals=tie_intervals,
        ropes={name: 1.0 for name in tie_intervals},
        strict_threshold=2.0, quality={"comparable": True}, settings=SETTINGS,
    )
    assert tie["relation"] == "tie"

    collapsed = SimpleNamespace(
        valid=True, reason=None, camera_distance_px=0.1,
        activity_a=_activity(speed=0.5), activity_b=_activity(speed=1.0),
    )
    rejected = decide_selective_relation(
        condition_id="c", edge_id="e", support=_support(), evidence=collapsed,
        intervals=_intervals(3.0, 4.0),
        ropes={name: 1.0 for name in _intervals(0, 0)},
        strict_threshold=2.0, quality={"comparable": True}, settings=SETTINGS,
    )
    assert rejected["relation"] == "incomparable"
    assert rejected["reason"] == "activity_noninferiority"
