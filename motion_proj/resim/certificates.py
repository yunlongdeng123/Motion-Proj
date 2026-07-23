"""V7.1 三态 certificate 与可测性校准。"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np

from .safety_geometry import OrientedBox, swept_obb_collision


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ComponentResult:
    name: str
    verdict: Verdict
    metrics: dict
    reason: str


def aggregate_components(
    components: Iterable[ComponentResult], *, required: Iterable[str] | None = None
) -> Verdict:
    values = list(components)
    selected = values if required is None else [
        value for value in values if value.name in set(required)
    ]
    if not selected:
        return Verdict.UNKNOWN
    if any(value.verdict is Verdict.FAIL for value in selected):
        return Verdict.FAIL
    if any(value.verdict is Verdict.UNKNOWN for value in selected):
        return Verdict.UNKNOWN
    return Verdict.PASS


def kinematic_certificate(
    centers_xyz: np.ndarray,
    times_s: np.ndarray,
    *,
    max_speed_mps: float,
    max_acceleration_mps2: float,
    max_step_m: float,
) -> ComponentResult:
    centers = np.asarray(centers_xyz, dtype=float)
    times = np.asarray(times_s, dtype=float)
    if len(centers) < 2 or len(times) != len(centers) or np.any(np.diff(times) <= 0):
        return ComponentResult("kinematic", Verdict.UNKNOWN, {}, "insufficient_or_invalid_time")
    step = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    dt = np.diff(times)
    speed = step / dt
    acceleration = (
        np.abs(np.diff(speed)) / ((dt[1:] + dt[:-1]) / 2.0)
        if len(speed) >= 2 else np.asarray([0.0])
    )
    metrics = {
        "max_step_m": float(step.max(initial=0.0)),
        "max_speed_mps": float(speed.max(initial=0.0)),
        "max_acceleration_mps2": float(acceleration.max(initial=0.0)),
    }
    failed = (
        metrics["max_step_m"] > max_step_m
        or metrics["max_speed_mps"] > max_speed_mps
        or metrics["max_acceleration_mps2"] > max_acceleration_mps2
    )
    return ComponentResult(
        "kinematic", Verdict.FAIL if failed else Verdict.PASS, metrics,
        "threshold_exceeded" if failed else "within_threshold",
    )


def dynamic_collision_certificate(
    own: list[OrientedBox],
    others: list[list[OrientedBox]],
    *,
    clearance_m: float = 0.0,
) -> ComponentResult:
    if len(own) < 2:
        return ComponentResult("dynamic_safety_geometry", Verdict.UNKNOWN, {}, "insufficient_frames")
    checked = 0
    minimum = float("inf")
    for other in others:
        for index in range(min(len(own), len(other)) - 1):
            result = swept_obb_collision(
                own[index], own[index + 1], other[index], other[index + 1],
                clearance_m=clearance_m,
            )
            checked += 1
            minimum = min(minimum, result["minimum_signed_separation_m"])
            if result["collision"]:
                return ComponentResult(
                    "dynamic_safety_geometry", Verdict.FAIL,
                    {"checked_intervals": checked, "minimum_signed_separation_m": minimum},
                    "continuous_obb_collision",
                )
    if checked == 0:
        return ComponentResult("dynamic_safety_geometry", Verdict.UNKNOWN, {}, "no_comparable_actor")
    return ComponentResult(
        "dynamic_safety_geometry", Verdict.PASS,
        {"checked_intervals": checked, "minimum_signed_separation_m": minimum},
        "no_continuous_obb_collision",
    )


def unavailable_component(name: str, reason: str) -> ComponentResult:
    return ComponentResult(name, Verdict.UNKNOWN, {}, reason)
