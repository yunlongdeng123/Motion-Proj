from __future__ import annotations

import numpy as np
import pytest

from motion_proj.worldsim_v5.geometry_repair import (
    GeometryRepairError,
    depth_error_summary,
    fit_inverse_depth_surface,
    geometry_reference_confidence,
    geometry_risk_mapping,
    staged_geometry_metrics,
)


def _plane_depth(height: int = 48, width: int = 64) -> tuple[np.ndarray, np.ndarray]:
    intrinsics = np.asarray([[80.0, 0.0, 31.5], [0.0, 80.0, 23.5], [0.0, 0.0, 1.0]])
    rows, columns = np.indices((height, width), dtype=np.float64)
    x = (columns - intrinsics[0, 2]) / intrinsics[0, 0]
    y = (rows - intrinsics[1, 2]) / intrinsics[1, 1]
    depth = 1.0 / (0.1 + 0.012 * x - 0.018 * y)
    return depth.astype(np.float32), intrinsics


def test_g0_recovers_plane_with_outliers() -> None:
    depth, intrinsics = _plane_depth()
    target = np.zeros_like(depth, dtype=bool)
    target[16:34, 22:44] = True
    support = ~target
    corrupted = depth.copy()
    corrupted[::7, ::9] += 25.0
    result = fit_inverse_depth_surface(
        depth=corrupted,
        support_mask=support,
        target_mask=target,
        intrinsics=intrinsics,
        model="G0_ROBUST_PLANE",
        minimum_support_points=100,
    )
    assert result.support_count == int(support.sum())
    assert depth_error_summary(result.depth, depth, target)["mae_m"] < 0.02
    assert result.surface_fit_residual_p90_m < 0.03


def test_quadratic_surface_is_available_but_not_implicitly_selected() -> None:
    depth, intrinsics = _plane_depth()
    target = np.zeros_like(depth, dtype=bool)
    target[15:30, 20:40] = True
    result = fit_inverse_depth_surface(
        depth=depth,
        support_mask=~target,
        target_mask=target,
        intrinsics=intrinsics,
        model="G3_ROBUST_QUADRATIC",
        minimum_support_points=100,
    )
    assert result.model == "G3_ROBUST_QUADRATIC"
    assert depth_error_summary(result.depth, depth, target)["mae_m"] < 1e-4


def test_insufficient_support_fails_closed() -> None:
    depth, intrinsics = _plane_depth()
    target = np.zeros_like(depth, dtype=bool)
    target[10:20, 10:20] = True
    support = np.zeros_like(target)
    support[0, :8] = True
    with pytest.raises(GeometryRepairError, match="ABSTAIN_INSUFFICIENT"):
        fit_inverse_depth_surface(
            depth=depth,
            support_mask=support,
            target_mask=target,
            intrinsics=intrinsics,
            minimum_support_points=32,
        )


def test_staged_metrics_keep_pipeline_errors_separate() -> None:
    reference = np.full((4, 5), 10.0, dtype=np.float32)
    mask = np.ones_like(reference, dtype=bool)
    result = staged_geometry_metrics(
        raw_surface_depth=reference + 0.1,
        pre_gaussianization_depth=reference + 0.2,
        post_gaussianization_render_depth=reference + 0.7,
        reference_depth=reference,
        evaluation_mask=mask,
    )
    assert result["raw_geometry_error"]["mae_m"] == pytest.approx(0.1, abs=1e-5)
    assert result["pre_gaussianization_geometry_error"]["mae_m"] == pytest.approx(
        0.2, abs=1e-5
    )
    assert result["post_gaussianization_render_error"]["mae_m"] == pytest.approx(
        0.7, abs=1e-5
    )
    assert result["representation_gap"]["mae_m"] == pytest.approx(0.5, abs=1e-5)
    assert result["gaussianization_delta_mae_m"] == pytest.approx(0.5, abs=1e-5)


def test_reference_confidence_combines_coverage_and_agreement() -> None:
    high = geometry_reference_confidence(
        observed_reference_pixels=80,
        target_pixels=100,
        lidar_agreement_mae_m=0.05,
        agreement_scale_m=0.5,
    )
    low = geometry_reference_confidence(
        observed_reference_pixels=20,
        target_pixels=100,
        lidar_agreement_mae_m=1.0,
        agreement_scale_m=0.5,
    )
    assert 0.0 <= low < high <= 1.0


def test_nonclipped_risk_mappings_preserve_bad_tail_order() -> None:
    errors = np.asarray([0.6, 2.0, 5.5])
    clipped = geometry_risk_mapping(errors, mapping="R0_CLIP", scale_m=0.5)
    logged = geometry_risk_mapping(errors, mapping="R1_LOG", scale_m=0.5)
    huber = geometry_risk_mapping(errors, mapping="R2_PSEUDO_HUBER", scale_m=0.5)
    assert np.unique(clipped).size == 1
    assert np.all(np.diff(logged) > 0.0)
    assert np.all(np.diff(huber) > 0.0)
