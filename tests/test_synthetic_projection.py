import pytest

from motion_proj.eval.synthetic_projection import CORRUPTIONS, run_cases, summarize


@pytest.fixture
def settings():
    return {
        "num_cases": 10,
        "num_frames": 8,
        "height": 32,
        "width": 48,
        "smooth_lambda": 5.0,
        "prior_weight": 0.1,
        "reliability_probability": 0.78,
        "energy_tolerance": 1e-6,
        "minimum_improvement_rate": 0.70,
        "minimum_eligible_fraction": 0.70,
    }


def test_synthetic_projection_is_deterministic(settings):
    first = run_cases(settings, seed=20260711)
    second = run_cases(settings, seed=20260711)
    assert first == second
    assert {row["corruption"] for row in first} == set(CORRUPTIONS)


def test_synthetic_projection_meets_small_acceptance(settings):
    summary = summarize(run_cases(settings, seed=20260711), settings)
    assert summary["finite_and_mask_valid_rate"] == 1.0
    assert summary["minimum_eligible_fraction"] >= 0.70
    assert summary["improvement_rate"] >= 0.70
    assert summary["acceptance"]["accepted"]
