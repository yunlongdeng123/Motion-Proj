from __future__ import annotations

from scripts.closeout_worldsim_v521 import class_stats


def test_closeout_class_stats_uses_view_cases_only() -> None:
    rows = [
        {"base": "adgs", "entity_kind": "view", "scene": "s1", "failure_class": ["B-ACTOR", "B-MIXED"]},
        {"base": "adgs", "entity_kind": "view", "scene": "s2", "failure_class": ["B-BOUNDARY"]},
        {"base": "adgs", "entity_kind": "temporal_window", "scene": "s1", "failure_class": ["B-UNRESOLVED"]},
        {"base": "streetgs", "entity_kind": "view", "scene": "s1", "failure_class": ["B-ACTOR"]},
    ]
    result = class_stats(rows, "adgs")
    assert result["cases"] == 2
    assert result["class_counts"] == {"B-ACTOR": 1, "B-BOUNDARY": 1}
    assert result["scenes"] == ["s1", "s2"]
