from __future__ import annotations

import pytest

from motion_proj.worldsim_v4.engineering_metrics import derive_engineering_record, summarize_engineering


def test_engineering_formulas_are_derived_from_raw_counts() -> None:
    row = derive_engineering_record(
        {
            "status": "done",
            "timings_seconds": {"prepare": 10, "train": 20, "render": 5},
            "requested_edits": 4,
            "quality_accepted_edits": 3,
            "source_clips": 2,
            "valid_edited_clips": 3,
            "actual_runtime_seconds": 42,
            "ideal_single_pass_seconds": 35,
            "rerun_seconds": 7,
            "full_rerun_seconds": 35,
            "gpu_hours": 1.5,
        }
    )
    assert row["total_seconds"] == 35
    assert row["valid_edit_yield"] == pytest.approx(0.75)
    assert row["counterfactual_expansion_ratio"] == pytest.approx(1.5)
    assert row["retry_amplification"] == pytest.approx(1.2)
    assert row["resume_efficiency"] == pytest.approx(0.8)
    assert row["gpu_hours_per_accepted_clip"] == pytest.approx(0.5)


def test_summary_keeps_all_terminal_states_in_success_denominator() -> None:
    base = {"timings_seconds": {"render": 1}, "requested_edits": 1, "source_clips": 1, "gpu_hours": 1}
    rows = [
        {**base, "status": "done", "quality_accepted_edits": 1, "valid_edited_clips": 1, "frame_times_seconds": [0.02, 0.04]},
        {**base, "status": "blocked", "quality_accepted_edits": 0, "valid_edited_clips": 0},
    ]
    result = summarize_engineering(rows)
    assert result["pipeline_success_rate"] == pytest.approx(0.5)
    assert result["valid_edit_yield"] == pytest.approx(0.5)
    assert result["frame_time_p50_seconds"] == pytest.approx(0.03)
    assert result["fps"] == pytest.approx(2 / 0.06)


def test_unknown_phase_fails_closed() -> None:
    with pytest.raises(ValueError, match="未知 timing phase"):
        derive_engineering_record({"status": "done", "timings_seconds": {"mystery": 1}})
