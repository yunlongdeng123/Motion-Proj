from __future__ import annotations

import numpy as np

from motion_proj.worldsim_v4.evidence_metrics import (
    brier_score,
    expected_calibration_error,
    probability_mass_metrics,
)


def test_perfect_probability_metrics_are_zero() -> None:
    target = np.asarray([0.0, 1.0, 1.0, 0.0])
    assert brier_score(target, target) == 0.0
    assert expected_calibration_error(target, target, bins=5) == 0.0
    assert probability_mass_metrics(target, target) == {
        "false_positive_semantic_mass": 0.0,
        "false_negative_semantic_mass": 0.0,
    }
