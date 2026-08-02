import numpy as np
import pytest

from motion_proj.dynamic_editing_v2.pilot_metrics import (
    binary_dilate,
    counterfactual_effect_mask,
    depth_order_diagnostic,
    masked_mean_abs,
    masked_psnr,
)


def test_counterfactual_mask_and_dilation_are_deterministic() -> None:
    original = np.zeros((7, 7, 3), dtype=np.uint8)
    removed = original.copy()
    original[3, 3] = 10
    mask = counterfactual_effect_mask(
        original, removed, threshold_uint8=2, dilation_radius=1
    )
    assert int(mask.sum()) == 9
    assert np.array_equal(mask, binary_dilate(original[..., 0] > 2, radius=1))


def test_masked_metrics_exclude_target_region() -> None:
    reference = np.zeros((2, 2, 3), dtype=np.uint8)
    candidate = reference.copy()
    candidate[0, 0] = 255
    target = np.zeros((2, 2), dtype=bool)
    target[0, 0] = True
    assert masked_mean_abs(reference, candidate, ~target) == 0
    assert masked_psnr(reference, candidate, ~target) == float("inf")
    assert masked_mean_abs(reference, candidate, target) == 1.0


def test_depth_order_diagnostic_reports_front_and_violation() -> None:
    edited = np.array([[4.0, 8.0], [1.0, np.nan]], dtype=np.float32)
    background = np.array([[5.0, 7.0], [1.0, 2.0]], dtype=np.float32)
    result = depth_order_diagnostic(
        edited, background, np.ones((2, 2), dtype=bool), tolerance_m=0.05
    )
    assert result["valid_pixel_count"] == 3
    assert result["actor_in_front_rate"] == pytest.approx(1 / 3)
    assert result["depth_ordering_violation_rate"] == pytest.approx(1 / 3)


def test_invalid_mask_shape_fails_closed() -> None:
    with pytest.raises(ValueError, match="二维"):
        binary_dilate(np.zeros((2, 2, 1)), radius=1)
