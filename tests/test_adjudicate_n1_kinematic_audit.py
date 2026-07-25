from scripts.adjudicate_n1_kinematic_audit import (
    _component_counts,
    _unique_terminal,
)


def test_unique_terminal_requires_awaiting_human_review(tmp_path):
    (tmp_path / "AWAITING_HUMAN_REVIEW").write_text("ready\n", encoding="utf-8")
    assert _unique_terminal(tmp_path) == "AWAITING_HUMAN_REVIEW"


def test_component_counts_are_field_specific():
    rows = [
        {
            "subject_maneuver_verdict": "INVALID",
            "target_corridor_verdict": "VALID",
            "front_relation_verdict": "VALID",
            "rear_relation_verdict": "INVALID",
        },
        {
            "subject_maneuver_verdict": "INVALID",
            "target_corridor_verdict": "VALID",
            "front_relation_verdict": "INVALID",
            "rear_relation_verdict": "VALID",
        },
    ]
    counts = _component_counts(rows)
    assert counts["subject_maneuver_verdict"] == {"INVALID": 2}
    assert counts["target_corridor_verdict"] == {"VALID": 2}
    assert counts["front_relation_verdict"] == {"INVALID": 1, "VALID": 1}
    assert counts["rear_relation_verdict"] == {"INVALID": 1, "VALID": 1}
