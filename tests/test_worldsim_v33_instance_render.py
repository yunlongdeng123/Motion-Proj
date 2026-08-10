import numpy as np
import pytest
import torch

from motion_proj.worldsim_v33.instance_renderer import (
    aggregate_metrics,
    binary_mask_metrics,
    composite_alpha_by_ray,
    instance_mask_losses,
)


def test_sparse_alpha_compositor_matches_closed_form_and_backpropagates() -> None:
    logits = torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))
    result = composite_alpha_by_ray(
        ray_ids=torch.tensor([0, 0, 1]),
        gaussian_logits=logits,
        projected_kernel=torch.tensor([1.0, 0.5, 1.0]),
        ray_count=2,
    )
    torch.testing.assert_close(result, torch.tensor([0.625, 0.5]))
    result.sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0


def test_balanced_mask_loss_has_finite_gradient() -> None:
    prediction = torch.tensor([[0.2, 0.8], [0.1, 0.4]], requires_grad=True)
    target = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    opacity = torch.sigmoid(torch.tensor([0.2, -0.3], requires_grad=True))
    losses = instance_mask_losses(
        prediction=prediction,
        target=target,
        candidate_opacity=opacity,
        prior_opacity=torch.tensor([0.5, 0.5]),
        background_candidate=torch.tensor([False, True]),
        weights={"mask_bce": 1, "dice": 1, "sparse": 0.01, "prior": 0.05},
    )
    losses["total"].backward()
    assert torch.isfinite(prediction.grad).all()


def test_mask_metrics_reward_exact_boundary() -> None:
    target = np.zeros((16, 16), dtype=bool)
    target[4:12, 5:11] = True
    exact = binary_mask_metrics(target, target, boundary_tolerance_pixels=1.0)
    shifted = binary_mask_metrics(
        np.roll(target, 3, axis=1), target, boundary_tolerance_pixels=1.0
    )
    assert exact["iou"] == pytest.approx(1.0)
    assert exact["boundary_f1"] == pytest.approx(1.0)
    assert exact["normalized_boundary_distance"] == pytest.approx(0.0)
    assert shifted["boundary_f1"] < exact["boundary_f1"]
    assert shifted["normalized_boundary_distance"] > 0


def test_aggregate_metrics_rejects_empty_rows() -> None:
    with pytest.raises(ValueError, match="没有"):
        aggregate_metrics([])
