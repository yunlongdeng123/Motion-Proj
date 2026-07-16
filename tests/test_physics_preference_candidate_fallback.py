from __future__ import annotations

import json

import pytest

from motion_proj.diagnostics.physics_preference_candidate_fallback import (
    CandidateFallbackError,
    _review_summary,
    _select_new_conditions,
)
from motion_proj.diagnostics import physics_preference_candidate_fallback as fallback_module


def test_fallback_selects_exactly_offset_120_new_scenes(monkeypatch) -> None:
    rows = [
        {"scene_token": f"scene-{index:03d}", "clip_id": f"clip-{index:03d}"}
        for index in range(130)
    ]
    monkeypatch.setattr(
        fallback_module,
        "select_profile_conditions",
        lambda split, *, partition, condition_count, required_start_index: rows[:condition_count],
    )
    used = {f"scene-{index:03d}" for index in range(120)}
    selected = _select_new_conditions(
        {}, partition="preference_train", offset=120, count=8,
        required_start_index=0, used_scene_tokens=used,
    )
    assert [row["scene_token"] for row in selected] == [f"scene-{index:03d}" for index in range(120, 128)]

    with pytest.raises(CandidateFallbackError, match="重叠"):
        _select_new_conditions(
            {}, partition="preference_train", offset=120, count=8,
            required_start_index=0, used_scene_tokens=used | {"scene-123"},
        )


def test_structure_review_requires_seven_same_scene_and_no_bad_case(tmp_path) -> None:
    cases = [{"case_id": f"case-{index}"} for index in range(8)]
    (tmp_path / "review_cases.json").write_text(json.dumps(cases), encoding="utf-8")
    rows = []
    for index in range(8):
        rows.append({
            "case_id": f"case-{index}",
            "verdict": "same_scene" if index < 7 else "uncertain",
            "failure_reasons": [],
            "reviewer": "human",
            "notes": "",
        })
    (tmp_path / "reviews.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    cfg = {
        "required_cases": 8, "minimum_same_scene": 7,
        "maximum_bad_cases": 0, "maximum_uncertain": 1,
    }
    assert _review_summary(tmp_path, cfg)["pass"] is True

    rows[-1]["verdict"] = "different_composition"
    rows[-1]["failure_reasons"] = ["layout_change"]
    (tmp_path / "reviews.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert _review_summary(tmp_path, cfg)["status"] == "rejected"
