from __future__ import annotations

import numpy as np

from motion_proj.worldsim_v4.evidence_calibration import (
    BetaCalibrator,
    RawCalibrator,
    TemperatureCalibrator,
    calibrator_from_dict,
    fit_beta_calibration,
    fit_temperature,
)
from motion_proj.worldsim_v4.evidence_metrics import brier_score


def test_raw_calibration_is_identity() -> None:
    probability = np.asarray([0.1, 0.5, 0.9])
    np.testing.assert_allclose(RawCalibrator().transform(probability), probability)


def test_temperature_fit_improves_overconfident_probabilities() -> None:
    probability = np.asarray([0.01, 0.01, 0.99, 0.99] * 30)
    target = np.asarray([0.0, 1.0, 0.0, 1.0] * 30)
    calibrated = fit_temperature(probability, target)
    assert calibrated.temperature > 1.0
    assert brier_score(calibrated.transform(probability), target) < brier_score(probability, target)


def test_beta_fit_handles_asymmetric_miscalibration() -> None:
    probability = np.linspace(0.02, 0.98, 300)
    target_probability = 1.0 / (1.0 + np.exp(-(-0.7 + 0.8 * np.log(probability) - 1.4 * np.log1p(-probability))))
    target = (np.arange(probability.size) / probability.size < target_probability).astype(float)
    calibrated = fit_beta_calibration(probability, target, l2_regularization=1e-5)
    assert brier_score(calibrated.transform(probability), target) < brier_score(probability, target)


def test_frozen_calibrator_round_trip_does_not_refit() -> None:
    probability = np.asarray([0.1, 0.5, 0.9])
    calibrators = [
        RawCalibrator(),
        TemperatureCalibrator(2.5),
        BetaCalibrator(0.8, 1.2, -0.3),
    ]
    for calibrator in calibrators:
        restored = calibrator_from_dict(calibrator.to_dict())
        np.testing.assert_allclose(
            restored.transform(probability), calibrator.transform(probability)
        )


def test_frozen_calibrator_rejects_unregistered_parameters() -> None:
    with np.testing.assert_raises(ValueError):
        calibrator_from_dict({"name": "raw", "temperature": 1.0})
