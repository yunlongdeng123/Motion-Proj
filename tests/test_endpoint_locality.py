import pytest
import torch

from motion_proj.diagnostics.endpoint_locality import endpoint_objective, locality_decision


def test_global_preserve_lambda_is_not_normalized_out():
    correction = torch.tensor(2.0)
    preserve = torch.tensor(3.0)
    assert float(endpoint_objective(correction, preserve, 0.0)) == 2.0
    assert float(endpoint_objective(correction, preserve, 4.0)) == 14.0
    with pytest.raises(ValueError, match="non-negative"):
        endpoint_objective(correction, preserve, -1.0)


def test_locality_decision_separates_optimization_and_locality_failure():
    base = {
        "lambda_preserve": 1.0,
        "step": 200,
        "outside_teacher_drift_ratio": 0.03,
        "frame0_teacher_drift": 0.1,
    }
    optimized = locality_decision(
        [base | {"correction_fraction": 0.1}],
        max_correction_fraction=0.2,
        max_outside_teacher_drift_ratio=0.02,
        max_frame0_teacher_drift=1.0e-6,
    )
    assert optimized["classification"] == "locality_failure"

    not_optimized = locality_decision(
        [base | {"correction_fraction": 0.3}],
        max_correction_fraction=0.2,
        max_outside_teacher_drift_ratio=0.02,
        max_frame0_teacher_drift=1.0e-6,
    )
    assert not_optimized["classification"] == "optimization_failure_at_preregistered_budget"

    feasible = locality_decision(
        [base | {
            "correction_fraction": 0.1,
            "outside_teacher_drift_ratio": 0.01,
            "frame0_teacher_drift": 0.0,
        }],
        max_correction_fraction=0.2,
        max_outside_teacher_drift_ratio=0.02,
        max_frame0_teacher_drift=1.0e-6,
    )
    assert feasible["classification"] == "feasible_region_found"
    assert feasible["passed"]
