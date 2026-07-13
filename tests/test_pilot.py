import torch

from motion_proj.train.pilot import capacity_decision, select_pair_indices


def test_select_pair_indices_is_stable_and_disjoint():
    first = select_pair_indices(122, pair_count=8, train_pair_count=4, seed=20260713)
    second = select_pair_indices(122, pair_count=8, train_pair_count=4, seed=20260713)
    assert first == second
    assert len(set(first["all"])) == 8
    assert set(first["train"]).isdisjoint(first["held_out"])


def test_capacity_decision_requires_every_preregistered_gate():
    metrics = {
        "initial_target_error": 1.0,
        "final_target_error": 0.1,
        "outside_teacher_drift_ratio": 0.01,
        "frame0_teacher_drift": 0.0,
        "gradient_finite": True,
        "gradient_nonzero": True,
        "target_roundtrip_max_error": 0.0,
        "correction_direction_cosine": 0.2,
    }
    accepted = capacity_decision(metrics, required_error_reduction=0.8, max_outside_teacher_drift_ratio=0.02)
    assert accepted["passed"]
    metrics["correction_direction_cosine"] = -0.01
    assert not capacity_decision(metrics, required_error_reduction=0.8, max_outside_teacher_drift_ratio=0.02)["passed"]
