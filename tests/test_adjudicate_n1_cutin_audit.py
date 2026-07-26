from scripts.adjudicate_n1_cutin_audit import (
    _component_counts,
    _require_expected_counts,
    _unique_terminal,
)


def test_unique_terminal_requires_awaiting_human_review(tmp_path):
    (tmp_path / "AWAITING_HUMAN_REVIEW").write_text("ready\n", encoding="utf-8")
    assert _unique_terminal(tmp_path) == "AWAITING_HUMAN_REVIEW"


def test_component_counts_are_field_specific():
    rows = [
        {
            "subject_maneuver_verdict": "INVALID",
            "receiver_corridor_verdict": "VALID",
            "receiver_relation_verdict": "VALID",
            "temporal_persistence_verdict": "INVALID",
        },
        {
            "subject_maneuver_verdict": "INVALID",
            "receiver_corridor_verdict": "VALID",
            "receiver_relation_verdict": "INVALID",
            "temporal_persistence_verdict": "VALID",
        },
    ]
    counts = _component_counts(rows)
    assert counts["subject_maneuver_verdict"] == {"INVALID": 2}
    assert counts["receiver_corridor_verdict"] == {"VALID": 2}
    assert counts["receiver_relation_verdict"] == {"INVALID": 1, "VALID": 1}
    assert counts["temporal_persistence_verdict"] == {"INVALID": 1, "VALID": 1}


def test_expected_counts_rejects_a_changed_human_result():
    config = {
        "expected_review_record_count": 18,
        "expected_overall_counts": {"TRUE_POSITIVE": 3, "FALSE_POSITIVE": 15},
    }
    validation = {
        "reviewed_item_count": 18,
        "overall_counts": {"TRUE_POSITIVE": 3, "FALSE_POSITIVE": 15},
    }
    _require_expected_counts(config, validation)
    validation["overall_counts"] = {"TRUE_POSITIVE": 4, "FALSE_POSITIVE": 14}
    try:
        _require_expected_counts(config, validation)
    except RuntimeError as exc:
        assert "总体计数" in str(exc)
    else:
        raise AssertionError("冻结计数变化必须拒绝")
