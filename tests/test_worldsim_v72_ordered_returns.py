from __future__ import annotations

import torch

from motion_proj.worldsim_v72.eas_vggt.ordered_returns import (
    ordered_return_distribution,
    ordered_return_nll,
    split_blocking_probability,
)


def _logit(probability: torch.Tensor) -> torch.Tensor:
    return torch.logit(probability, eps=1.0e-6)


def test_distribution_sorts_geometry_and_normalizes_with_no_return() -> None:
    depth = torch.tensor([[5.0, 2.0, 8.0]])
    blocking = _logit(torch.tensor([[0.3, 0.7, 0.2]]))
    detection = _logit(torch.tensor([[0.8, 0.6, 0.9]]))
    result = ordered_return_distribution(depth, blocking, detection, torch.ones_like(depth, dtype=torch.bool))
    assert result.sorted_original_index.tolist() == [[1, 0, 2]]
    assert torch.allclose(result.event_probability.sum(1) + result.no_return_probability, torch.ones(1))


def test_undetected_opaque_front_surface_blocks_background_and_yields_no_return() -> None:
    depth = torch.tensor([[2.0, 8.0]])
    blocking = torch.tensor([[30.0, 30.0]])
    detection = torch.tensor([[-30.0, 30.0]])
    result = ordered_return_distribution(depth, blocking, detection, torch.ones_like(depth, dtype=torch.bool))
    assert result.event_probability[0, 1] < 1.0e-10
    assert torch.allclose(result.no_return_probability, torch.ones(1), atol=1.0e-6)


def test_conserved_primitive_split_preserves_total_detection_mass() -> None:
    total_blocking = torch.tensor(0.64)
    detection = torch.tensor(0.75)
    single = ordered_return_distribution(
        torch.tensor([[3.0]]),
        _logit(total_blocking.reshape(1, 1)),
        _logit(detection.reshape(1, 1)),
        torch.ones((1, 1), dtype=torch.bool),
    )
    child = split_blocking_probability(total_blocking, 4)
    split = ordered_return_distribution(
        torch.full((1, 4), 3.0),
        _logit(child.repeat(1, 4)),
        _logit(detection.repeat(1, 4)),
        torch.ones((1, 4), dtype=torch.bool),
    )
    assert torch.allclose(single.event_probability.sum(), split.event_probability.sum(), atol=1.0e-6)
    assert torch.allclose(single.no_return_probability, split.no_return_probability, atol=1.0e-6)


def test_nll_has_finite_gradients_for_return_and_no_return() -> None:
    blocking = torch.zeros((2, 2), requires_grad=True)
    detection = torch.zeros((2, 2), requires_grad=True)
    result = ordered_return_distribution(
        torch.tensor([[1.0, 2.0], [2.0, 3.0]]),
        blocking,
        detection,
        torch.ones((2, 2), dtype=torch.bool),
    )
    loss = ordered_return_nll(result, torch.tensor([0, -1]))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(blocking.grad).all()
    assert torch.isfinite(detection.grad).all()
