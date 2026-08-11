from __future__ import annotations

import pytest

from motion_proj.worldsim_v4.statistics import paired_scene_test, summarize_scenes


def test_scene_summary_retains_failed_blocked_abstain_denominator() -> None:
    rows = [
        {"scene": "a", "status": "done", "psnr": 20.0},
        {"scene": "b", "status": "done", "psnr": 22.0},
        {"scene": "c", "status": "failed", "psnr": None},
        {"scene": "d", "status": "blocked", "psnr": None},
        {"scene": "e", "status": "abstain", "psnr": None},
    ]
    result = summarize_scenes(rows, "psnr", bootstrap_samples=200, seed=7)
    assert result["attempted"] == 5
    assert result["valid"] == 2
    assert result["mean"] == pytest.approx(21.0)
    assert result["states"] == {"done": 2, "failed": 1, "blocked": 1, "abstain": 1}


def test_paired_test_uses_scene_intersection_and_orients_lower_metric() -> None:
    candidate = {"a": 0.1, "b": 0.2, "c": None}
    baseline = {"a": 0.2, "b": 0.3, "d": 0.4}
    result = paired_scene_test(candidate, baseline, higher_is_better=False, bootstrap_samples=200, seed=3)
    assert result["attempted_union"] == 4
    assert result["paired"] == 2
    assert result["mean_delta"] == pytest.approx(0.1)
    assert result["bootstrap_ci95"] == pytest.approx([0.1, 0.1])


def test_illegal_status_is_rejected() -> None:
    with pytest.raises(ValueError, match="非法 scene 状态"):
        summarize_scenes([{"scene": "a", "status": "running", "psnr": 1.0}], "psnr")
