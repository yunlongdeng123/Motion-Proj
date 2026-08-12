"""Raw, temperature and beta calibration for Gaussian evidence probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np
from scipy.optimize import minimize


_EPSILON = 1e-6


def _probabilities(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite one-dimensional array")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return np.clip(array, _EPSILON, 1.0 - _EPSILON)


def _logit(probability: np.ndarray) -> np.ndarray:
    return np.log(probability) - np.log1p(-probability)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    output = np.empty_like(value, dtype=np.float64)
    positive = value >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponent = np.exp(value[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def _training_data(
    probability: np.ndarray, target: np.ndarray, sample_weight: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = _probabilities(probability, name="probability")
    y = np.asarray(target, dtype=np.float64)
    if y.ndim != 1 or not np.isfinite(y).all() or np.any((y < 0.0) | (y > 1.0)):
        raise ValueError("target must be a finite one-dimensional array in [0, 1]")
    if p.shape != y.shape or p.size == 0:
        raise ValueError("probability/target must have the same non-empty shape")
    if sample_weight is None:
        weight = np.ones_like(p)
    else:
        weight = np.asarray(sample_weight, dtype=np.float64)
        if weight.shape != p.shape or not np.isfinite(weight).all() or np.any(weight < 0.0):
            raise ValueError("sample_weight must match data and be non-negative")
    if weight.sum() <= 0.0:
        raise ValueError("sample_weight must have positive mass")
    return p, y, weight / weight.sum()


class ProbabilityCalibrator(Protocol):
    name: str

    def transform(self, probability: np.ndarray) -> np.ndarray: ...

    def to_dict(self) -> dict[str, float | str]: ...


@dataclass(frozen=True)
class RawCalibrator:
    name: str = "raw"

    def transform(self, probability: np.ndarray) -> np.ndarray:
        return _probabilities(probability, name="probability")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name}


@dataclass(frozen=True)
class TemperatureCalibrator:
    temperature: float
    name: str = "temperature"

    def __post_init__(self) -> None:
        if not np.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")

    def transform(self, probability: np.ndarray) -> np.ndarray:
        p = _probabilities(probability, name="probability")
        return _sigmoid(_logit(p) / self.temperature)

    def to_dict(self) -> dict[str, float | str]:
        return {"name": self.name, "temperature": float(self.temperature)}


@dataclass(frozen=True)
class BetaCalibrator:
    positive_log_coefficient: float
    negative_log_coefficient: float
    intercept: float
    name: str = "beta"

    def __post_init__(self) -> None:
        values = (
            self.positive_log_coefficient,
            self.negative_log_coefficient,
            self.intercept,
        )
        if not np.isfinite(values).all():
            raise ValueError("beta calibration coefficients must be finite")

    def transform(self, probability: np.ndarray) -> np.ndarray:
        p = _probabilities(probability, name="probability")
        calibrated_logit = (
            self.positive_log_coefficient * np.log(p)
            - self.negative_log_coefficient * np.log1p(-p)
            + self.intercept
        )
        return _sigmoid(calibrated_logit)

    def to_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "positive_log_coefficient": float(self.positive_log_coefficient),
            "negative_log_coefficient": float(self.negative_log_coefficient),
            "intercept": float(self.intercept),
        }


def calibrator_from_dict(payload: Mapping[str, Any]) -> ProbabilityCalibrator:
    """只从 development 冻结参数恢复 calibrator，不重新拟合。"""

    name = payload.get("name")
    if name == "raw":
        if set(payload) != {"name"}:
            raise ValueError("raw calibrator payload has unexpected parameters")
        return RawCalibrator()
    if name == "temperature":
        if set(payload) != {"name", "temperature"}:
            raise ValueError("temperature calibrator payload is incomplete")
        return TemperatureCalibrator(float(payload["temperature"]))
    if name == "beta":
        expected = {
            "name",
            "positive_log_coefficient",
            "negative_log_coefficient",
            "intercept",
        }
        if set(payload) != expected:
            raise ValueError("beta calibrator payload is incomplete")
        return BetaCalibrator(
            positive_log_coefficient=float(payload["positive_log_coefficient"]),
            negative_log_coefficient=float(payload["negative_log_coefficient"]),
            intercept=float(payload["intercept"]),
        )
    raise ValueError(f"unsupported calibrator payload: {name}")


def _weighted_log_loss(
    prediction: np.ndarray, target: np.ndarray, weight: np.ndarray
) -> float:
    p = np.clip(prediction, _EPSILON, 1.0 - _EPSILON)
    return float(-(weight * (target * np.log(p) + (1.0 - target) * np.log1p(-p))).sum())


def fit_temperature(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> TemperatureCalibrator:
    p, y, weight = _training_data(probability, target, sample_weight)
    logits = _logit(p)

    def objective(log_temperature: np.ndarray) -> float:
        return _weighted_log_loss(
            _sigmoid(logits / np.exp(log_temperature[0])), y, weight
        )

    result = minimize(objective, np.zeros(1), method="L-BFGS-B", bounds=[(-6.0, 6.0)])
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"temperature calibration failed: {result.message}")
    return TemperatureCalibrator(float(np.exp(result.x[0])))


def fit_beta_calibration(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    l2_regularization: float = 1e-6,
) -> BetaCalibrator:
    p, y, weight = _training_data(probability, target, sample_weight)
    regularization = float(l2_regularization)
    if not np.isfinite(regularization) or regularization < 0.0:
        raise ValueError("l2_regularization must be finite and non-negative")
    features = np.stack([np.log(p), -np.log1p(-p), np.ones_like(p)], axis=1)

    def objective(parameters: np.ndarray) -> float:
        predicted = _sigmoid(features @ parameters)
        penalty = regularization * float(parameters[:2] @ parameters[:2])
        return _weighted_log_loss(predicted, y, weight) + penalty

    result = minimize(
        objective,
        np.asarray([1.0, 1.0, 0.0]),
        method="L-BFGS-B",
        bounds=[(-20.0, 20.0), (-20.0, 20.0), (-20.0, 20.0)],
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"beta calibration failed: {result.message}")
    return BetaCalibrator(*[float(value) for value in result.x])
