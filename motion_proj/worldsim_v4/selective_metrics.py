"""M2 selective editing 的 coverage/error 指标。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class SelectiveSample:
    sample_id: str
    uncertainty: float
    error: float
    accepted: bool

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must be non-empty")
        if not math.isfinite(self.uncertainty) or not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("uncertainty must lie in [0, 1]")
        if not math.isfinite(self.error) or self.error < 0.0:
            raise ValueError("error must be finite and non-negative")


def selective_group_metrics(samples: Iterable[SelectiveSample]) -> dict[str, Any]:
    rows = list(samples)
    if not rows:
        raise ValueError("selective metrics require at least one sample")
    if len({row.sample_id for row in rows}) != len(rows):
        raise ValueError("selective sample IDs must be unique")
    accepted = [row.error for row in rows if row.accepted]
    abstained = [row.error for row in rows if not row.accepted]
    accepted_mean = float(math.fsum(accepted) / len(accepted)) if accepted else None
    abstained_mean = (
        float(math.fsum(abstained) / len(abstained)) if abstained else None
    )
    separation = (
        abstained_mean - accepted_mean
        if accepted_mean is not None and abstained_mean is not None
        else None
    )
    return {
        "requested_count": len(rows),
        "accepted_count": len(accepted),
        "abstain_count": len(abstained),
        "coverage": len(accepted) / len(rows),
        "accepted_mean_error": accepted_mean,
        "abstain_mean_error": abstained_mean,
        "abstain_minus_accepted_error": separation,
        "meaningful_abstention_gate": separation is not None and separation > 0.0,
    }


def selective_risk_curve(
    samples: Iterable[SelectiveSample],
    *,
    requested_coverages: Sequence[float],
) -> list[dict[str, float | int]]:
    rows = list(samples)
    if not rows:
        raise ValueError("selective curve requires at least one sample")
    if not requested_coverages:
        raise ValueError("requested coverages must be non-empty")
    ordered = sorted(rows, key=lambda row: (row.uncertainty, row.sample_id))
    output = []
    for requested in requested_coverages:
        value = float(requested)
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError("requested coverage must lie in (0, 1]")
        retained = max(1, min(len(ordered), math.ceil(value * len(ordered))))
        selected = ordered[:retained]
        output.append(
            {
                "requested_coverage": value,
                "retained_count": retained,
                "coverage": retained / len(ordered),
                "mean_error": float(
                    math.fsum(row.error for row in selected) / retained
                ),
                "maximum_retained_uncertainty": selected[-1].uncertainty,
            }
        )
    return output
