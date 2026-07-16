from __future__ import annotations

from types import SimpleNamespace

from motion_proj.preference.selective_order import decide_selective_relation


def test_camera_mismatch_is_incomparable_before_motion_score() -> None:
    support = SimpleNamespace(
        valid=True, candidate_a="a", candidate_b="b", start_frame=0, end_frame=4
    )
    evidence = SimpleNamespace(
        valid=True,
        reason=None,
        camera_distance_px=5.0,
        activity_a={"active_fraction": 1.0, "survival": 1.0, "residual_speed": 1.0, "net_residual_displacement": 2.0},
        activity_b={"active_fraction": 1.0, "survival": 1.0, "residual_speed": 1.0, "net_residual_displacement": 2.0},
    )
    result = decide_selective_relation(
        condition_id="condition",
        edge_id="edge",
        support=support,
        evidence=evidence,
        intervals={},
        ropes={name: 1.0 for name in ("punc", "acceleration", "curvature", "coherence")},
        strict_threshold=2.0,
        quality={"comparable": True},
        settings={"maximum_camera_distance_px": 1.0},
    )
    assert result["relation"] == "incomparable"
    assert result["reason"] == "camera_mismatch"
