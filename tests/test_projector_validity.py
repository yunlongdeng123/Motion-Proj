from __future__ import annotations

import torch

from motion_proj.auditor.state import Track
from motion_proj.diagnostics.projector_validity import (
    CANDIDATES,
    build_candidate_tracks,
    constrained_smooth_points,
    synthetic_calibration,
)


def _settings() -> dict:
    return {
        "candidates": list(CANDIDATES),
        "current_smooth_lambda": 5.0,
        "constrained": {
            "lambda_acceleration": 0.08,
            "lambda_jerk": 0.02,
            "robust_delta_px": 1.0,
            "robust_iterations": 3,
            "confidence_floor": 0.5,
            "uncertainty_floor_px": 0.25,
            "uncertainty_confidence_scale_px": 1.5,
            "snr_threshold": 1.0,
        },
        "synthetic": {"frame_count": 8},
        "thresholds": {"minimum_noisy_improvement_px": 1.0e-4},
    }


def _track(points: torch.Tensor, present: torch.Tensor | None = None) -> Track:
    present = torch.ones(points.shape[0], dtype=torch.bool) if present is None else present
    xyxy = torch.stack([
        points[:, 0] - 2, points[:, 1] - 2,
        points[:, 0] + 2, points[:, 1] + 2,
    ], dim=-1)
    xyxy[~present] = float("nan")
    return Track(
        "generated_dynamic_residual_000", "generated_point/dynamic_residual",
        xyxy, torch.ones(points.shape[0]), present,
    )


def test_constrained_smoother_preserves_frame0_visibility_and_endpoints():
    points = torch.tensor([[0.0, 0.0], [1.0, 0.1], [5.0, -3.0], [3.0, 0.3], [4.0, 0.4]])
    visible = torch.tensor([True, True, True, True, False])
    projected = constrained_smooth_points(
        points, visible, torch.ones(5), torch.tensor([True, True, True, True, True]),
        lambda_acceleration=0.08, lambda_jerk=0.02, robust_delta_px=1.0, robust_iterations=3,
    )
    assert torch.equal(projected[0], points[0])
    assert torch.equal(projected[3], points[3])
    assert torch.isnan(projected[4]).all()
    assert not torch.equal(projected[2], points[2])


def test_candidate_constraints_leave_background_identity_and_uncertainty_gate():
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [5.0, -3.0], [3.0, 0.0], [4.0, 0.0]])
    dynamic = _track(points)
    background = _track(points)
    background.instance_token = "generated_background_001"
    background.category = "generated_point/background"
    confidence = torch.full((2, 5), 0.95)
    candidates = build_candidate_tracks([dynamic, background], confidence, (96, 96), _settings())
    assert set(candidates) == set(CANDIDATES)
    for name in ("P-ID", "P-CON", "P-UNC"):
        assert torch.equal(candidates[name].tracks[0].center[0], dynamic.center[0])
        assert torch.equal(candidates[name].tracks[0].present, dynamic.present)
    assert torch.equal(candidates["P-CON"].tracks[1].center, background.center)
    assert torch.equal(candidates["P-UNC"].tracks[1].center, background.center)
    # P-UNC can only be a subset of the constrained displacement.
    con_delta = torch.linalg.vector_norm(candidates["P-CON"].tracks[0].center - dynamic.center, dim=-1)
    unc_delta = torch.linalg.vector_norm(candidates["P-UNC"].tracks[0].center - dynamic.center, dim=-1)
    assert torch.all(unc_delta <= con_delta + 1.0e-7)


def test_synthetic_calibration_covers_required_motion_and_corruption_cases():
    rows = synthetic_calibration(_settings())
    assert {row["candidate"] for row in rows} == set(CANDIDATES)
    assert {row["motion"] for row in rows} >= {
        "constant_velocity", "constant_acceleration", "brake", "smooth_turn", "lane_change",
    }
    assert {row["corruption"] for row in rows} >= {
        "clean", "tracker_jitter", "single_frame_outlier", "occlusion_recovery",
    }
