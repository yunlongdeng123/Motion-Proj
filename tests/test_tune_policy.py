import pytest

from motion_proj.tune.policy import SearchBudget, objective_score, prune_reason


def test_search_budget_hard_limit():
    budget = SearchBudget(started_at=100.0, hard_limit_hours=1.0)
    assert budget.may_start_trial(3699.0)
    assert not budget.may_start_trial(3700.0)


def test_pruning_and_objective():
    good = {"static_drift": 1.0, "track_acceleration": 2.0, "lpips": 0.2,
            "projection_eligible_fraction": 0.8,
            "normalized_static_drift_improvement": 0.4,
            "normalized_track_acceleration_improvement": 0.2}
    assert prune_reason(good, base_lpips=0.2) is None
    assert objective_score(good) == pytest.approx(0.3)
    bad = dict(good, projection_eligible_fraction=0.69)
    assert prune_reason(bad, base_lpips=0.2) == "eligible_fraction_below_70pct"
