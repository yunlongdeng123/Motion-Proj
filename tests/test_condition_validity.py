from __future__ import annotations

from motion_proj.diagnostics.condition_validity import summarize_condition_validity


def _case(index: int) -> dict:
    correction = {
        "finite": True,
        "first_frame_frozen": True,
        "first_frame_mask_zero": True,
    }
    residual = {"mean": float(index + 1)}
    return {
        "case_id": f"case-{index}",
        "modes": {
            "gt_ego_debug": {
                "uses_future_gt_ego": True,
                "residual": residual,
                "correction": correction,
            },
            "identity_ego": {
                "uses_future_gt_ego": False,
                "residual": residual,
                "correction": correction,
            },
            "estimated_background_motion": {
                "uses_future_gt_ego": False,
                "residual": residual,
                "correction": correction,
            },
        },
    }


def test_condition_summary_promotes_only_after_review_threshold() -> None:
    cases = [_case(index) for index in range(12)]
    reviews = [
        {
            "case_id": f"case-{index}",
            "gt_ego_valid": "no",
            "self_estimated_valid": "yes" if index < 9 else "no",
        }
        for index in range(12)
    ]

    summary = summarize_condition_validity(
        cases,
        reviews,
        required_reviews=12,
        minimum_reasonable_rate=0.70,
    )

    assert summary["automated_checks_passed"] is True
    assert summary["reviews"]["reasonable_rate"] == 0.75
    assert summary["static_branch_decision"] == "promote"


def test_condition_summary_stays_pending_before_required_reviews() -> None:
    cases = [_case(index) for index in range(12)]
    reviews = [
        {"case_id": "case-0", "gt_ego_valid": "no", "self_estimated_valid": "yes"}
    ]

    summary = summarize_condition_validity(
        cases,
        reviews,
        required_reviews=12,
        minimum_reasonable_rate=0.70,
    )

    assert summary["status"] == "awaiting_reviews"
    assert summary["static_branch_decision"] == "pending_review"


def test_generated_track_summary_requires_its_own_review_field() -> None:
    cases = [_case(index) for index in range(8)]
    for case in cases:
        case["modes"]["estimated_background_motion"].update({
            "uses_future_gt_track": False,
            "track_diagnostics": {"valid_track_count": 3, "median_track_length": 5.0},
        })
    reviews = [
        {"case_id": f"case-{index}", "point_track_valid": "yes" if index < 6 else "no"}
        for index in range(8)
    ]

    summary = summarize_condition_validity(
        cases,
        reviews,
        required_reviews=8,
        minimum_reasonable_rate=0.70,
        review_target="generated_tracks",
    )

    assert summary["automated_checks_passed"] is True
    assert summary["reviews"]["field"] == "point_track_valid"
    assert summary["reviews"]["reasonable_rate"] == 0.75
    assert summary["static_branch_decision"] == "not_assessed"
    assert summary["generated_track_decision"] == "promote"
