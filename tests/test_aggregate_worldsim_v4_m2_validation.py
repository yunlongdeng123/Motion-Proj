from __future__ import annotations

import pytest

from scripts.aggregate_worldsim_v4_m2_validation import (
    M2ValidationAggregationError,
    collect_validation_requests,
    select_frozen_baseline,
)


ARMS = ["ABSTAIN", "OBSERVED", "TELEA", "ROADPATCH", "GENERATED", "RISK_ROUTER"]


def _matched() -> list[dict]:
    return [
        {
            "arm": arm,
            "status": "atomic_noop" if arm == "ABSTAIN" else "abstain",
            "metrics": {"atomic_noop": True, "edit_error": 0.7}
            if arm == "ABSTAIN"
            else {},
        }
        for arm in ARMS
    ]


def test_collect_validation_requests_retains_measured_asset_abstain() -> None:
    summaries = [
        {
            "scene": "scene-a",
            "status": "done",
            "request_count": 1,
            "blocked_request_count": 1,
            "total_request_count": 2,
            "requests": [
                {
                    "request_id": "ready",
                    "matched_arms": _matched(),
                    "candidates": [{"arm": "TELEA"}],
                }
            ],
            "blocked_requests": [
                {
                    "request_id": "blocked",
                    "status": "abstain",
                    "reason": "ABSTAIN_NO_ROLE_MATCHED_ERASE_PACKAGE",
                    "matched_arms": _matched(),
                    "candidates": [],
                }
            ],
        }
    ]
    rows = collect_validation_requests(summaries, ARMS)
    assert [row["request_id"] for row in rows] == ["ready", "blocked"]


def test_collect_validation_requests_rejects_dropped_blocked_denominator() -> None:
    summaries = [
        {
            "scene": "scene-a",
            "status": "done",
            "request_count": 1,
            "blocked_request_count": 1,
            "total_request_count": 2,
            "requests": [
                {
                    "request_id": "ready",
                    "matched_arms": _matched(),
                    "candidates": [],
                }
            ],
            "blocked_requests": [],
        }
    ]
    with pytest.raises(M2ValidationAggregationError, match="blocked accounting"):
        collect_validation_requests(summaries, ARMS)


def test_validation_uses_development_frozen_baseline() -> None:
    table = [
        {"arm": "OBSERVED", "quality": {"edit_error": 0.1}},
        {"arm": "TELEA", "quality": {"edit_error": 0.9}},
        {"arm": "RISK_ROUTER", "quality": {"edit_error": 0.2}},
    ]
    selected = select_frozen_baseline(
        table, {"best_matched_non_router": "TELEA"}
    )
    assert selected["arm"] == "TELEA"
