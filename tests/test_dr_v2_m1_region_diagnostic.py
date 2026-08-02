import numpy as np
import pytest

from scripts import run_dr_v2_m1_region_diagnostic as diagnostic


def test_extract_three_view_uses_upstream_left_center_right_layout():
    composite = np.zeros((294, 1574, 3), dtype=np.uint8)
    composite[:, :518] = 11
    composite[:, 528:1046] = 22
    composite[:, 1056:1574] = 33
    assert np.all(diagnostic.extract_three_view(composite, 0) == 22)
    assert np.all(diagnostic.extract_three_view(composite, 1) == 11)
    assert np.all(diagnostic.extract_three_view(composite, 2) == 33)


def test_region_metrics_is_exact_and_fails_closed_on_empty_region():
    gt = np.zeros((4, 4, 3), dtype=np.uint8)
    pred = gt.copy()
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    metrics = diagnostic.region_metrics(pred, gt, mask)
    assert metrics["pixel_count"] == 4
    assert metrics["rgb_mae_0_1"] == 0.0
    assert metrics["psnr"] > 100
    with pytest.raises(RuntimeError, match="为空"):
        diagnostic.region_metrics(pred, gt, np.zeros_like(mask))


def test_boundary_band_contains_both_sides_of_object_edge():
    mask = np.zeros((15, 15), dtype=bool)
    mask[5:10, 5:10] = True
    boundary = diagnostic.boundary_band(mask, radius=1)
    assert boundary[4, 7]
    assert boundary[5, 7]
    assert not boundary[7, 7]
