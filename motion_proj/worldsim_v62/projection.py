"""Closed-form hard-evidence projection for sparse tri-state queries."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

FREE_INDEX = 0
OCCUPIED_INDEX = 1
UNKNOWN_INDEX = 2


class ProjectionOutput(NamedTuple):
    probabilities: Tensor
    constrained: Tensor
    contradiction: Tensor


def _mask(value: Tensor | None, reference: Tensor) -> Tensor:
    if value is None:
        return torch.zeros(reference.shape[:-1], dtype=torch.bool, device=reference.device)
    result = value.to(device=reference.device, dtype=torch.bool)
    if result.shape != reference.shape[:-1]:
        raise ValueError(f"evidence mask shape {tuple(result.shape)} != query shape {tuple(reference.shape[:-1])}")
    return result


def _state(reference: Tensor, index: int) -> Tensor:
    values = torch.zeros_like(reference)
    values[..., index] = 1
    return values


def project_feasible_tristate(
    logits: Tensor,
    *,
    observed_free: Tensor | None = None,
    observed_occupied: Tensor | None = None,
    contradiction: Tensor | None = None,
    outside_lifecycle: Tensor | None = None,
) -> ProjectionOutput:
    """Project logits to FREE/OCCUPIED/UNKNOWN under query-wise hard evidence.

    Precedence is contradiction, observed FREE/OCCUPIED, lifecycle, then soft
    prior. Simultaneous FREE and OCCUPIED evidence is a contradiction. Hard
    rows become exact one-hot states; unconstrained rows retain the softmax and
    therefore their gradients.
    """

    if logits.ndim < 2 or logits.shape[-1] != 3:
        raise ValueError(f"expected logits [..., 3], got {tuple(logits.shape)}")

    free_mask = _mask(observed_free, logits)
    occupied_mask = _mask(observed_occupied, logits)
    explicit_contradiction = _mask(contradiction, logits)
    lifecycle_mask = _mask(outside_lifecycle, logits)
    effective_contradiction = explicit_contradiction | (free_mask & occupied_mask)

    free_state = _state(logits, FREE_INDEX)
    occupied_state = _state(logits, OCCUPIED_INDEX)
    unknown_state = _state(logits, UNKNOWN_INDEX)

    probabilities = torch.softmax(logits, dim=-1)
    probabilities = torch.where(lifecycle_mask[..., None], unknown_state, probabilities)
    probabilities = torch.where((free_mask & ~effective_contradiction)[..., None], free_state, probabilities)
    probabilities = torch.where(
        (occupied_mask & ~effective_contradiction)[..., None], occupied_state, probabilities
    )
    probabilities = torch.where(effective_contradiction[..., None], unknown_state, probabilities)

    constrained = free_mask | occupied_mask | explicit_contradiction | lifecycle_mask
    return ProjectionOutput(probabilities, constrained, effective_contradiction)
