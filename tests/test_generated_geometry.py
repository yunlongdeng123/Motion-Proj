from __future__ import annotations

import torch

from motion_proj.auditor.generated_geometry import (
    estimate_generated_geometry,
    fit_affine_background_flow,
    render_pairwise_background_correction,
)


def _affine_flow(height: int, width: int) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, height),
        torch.linspace(-1, 1, width),
        indexing="ij",
    )
    u = 2.0 + 0.4 * x - 0.2 * y
    v = -1.0 + 0.1 * x + 0.3 * y
    return torch.stack([u, v], dim=-1)


def test_robust_affine_background_fit_rejects_local_outlier() -> None:
    expected = _affine_flow(40, 64)
    observed = expected.clone()
    observed[10:18, 20:32] += torch.tensor([24.0, -18.0])
    confidence = torch.ones(1, 40, 64)

    estimate = fit_affine_background_flow(observed.unsqueeze(0), confidence, min_points=64)

    background = torch.ones(40, 64, dtype=torch.bool)
    background[10:18, 20:32] = False
    error = (estimate.flow[0][background] - expected[background]).abs().max()
    assert float(error) < 0.15
    assert estimate.uses_future_gt_ego is False
    assert estimate.diagnostics["fit_valid_fraction"] == 1.0


def test_formal_generated_modes_do_not_require_future_gt_pose() -> None:
    flow = torch.zeros(2, 8, 12, 2)
    confidence = torch.ones(2, 8, 12)
    depth = torch.ones(3, 8, 12)
    sample = {"intrinsics": torch.eye(3), "cam2ego": torch.eye(4)}

    identity = estimate_generated_geometry("identity_ego", flow, confidence, depth, sample)
    estimated = estimate_generated_geometry(
        "estimated_background_motion",
        flow,
        confidence,
        depth,
        sample,
        fit_options={"min_points": 16},
    )

    assert identity.uses_future_gt_ego is False
    assert estimated.uses_future_gt_ego is False
    assert torch.isfinite(estimated.flow).all()


def test_controlled_geometry_fails_without_actual_control_condition() -> None:
    flow = torch.zeros(1, 4, 6, 2)
    confidence = torch.ones(1, 4, 6)
    depth = torch.ones(2, 4, 6)
    sample = {"intrinsics": torch.eye(3), "cam2ego": torch.eye(4)}

    try:
        estimate_generated_geometry("controlled_ego", flow, confidence, depth, sample)
    except ValueError as exc:
        assert "control_ego2global" in str(exc)
    else:
        raise AssertionError("缺少 control condition 时必须 fail closed")


def test_background_correction_freezes_first_frame_and_is_finite() -> None:
    frames = torch.rand(3, 3, 16, 24) * 2 - 1
    expected_flow = torch.zeros(2, 16, 24, 2)
    confidence = torch.ones(2, 16, 24)

    target, mask = render_pairwise_background_correction(frames, expected_flow, confidence)

    assert torch.equal(target[0], frames[0])
    assert torch.count_nonzero(mask[0]) == 0
    assert torch.isfinite(target).all()
    assert torch.isfinite(mask).all()
    assert tuple(mask.shape) == (3, 1, 16, 24)
