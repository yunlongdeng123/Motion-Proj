import pytest

from motion_proj.tune.policy import SearchBudget, objective_score, prune_reason
from motion_proj.tune.optuna_runner import TrialRecord, promotion_candidates


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


def test_promotion_uses_score_then_lpips_without_relaxing_thresholds():
    records = [
        TrialRecord(100, "a", None, {}, {"lpips": 0.20}, None, 0.5),
        TrialRecord(100, "b", None, {}, {"lpips": 0.19}, None, 0.5),
        TrialRecord(100, "c", None, {}, {"lpips": 0.10}, "eligible_fraction_below_70pct", None),
    ]
    assert [record.run_id for record in promotion_candidates(records, 2)] == ["b", "a"]
