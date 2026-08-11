from __future__ import annotations

import numpy as np
import pytest

from motion_proj.worldsim_v4.region_masks import (
    RegionMaskError,
    RegionMaskProtocol,
    build_baseline_region_masks,
)


def test_baseline_regions_exclude_dynamic_and_egocar_from_static() -> None:
    dynamic = np.zeros((11, 13), dtype=bool)
    dynamic[4:7, 5:8] = True
    egocar = np.zeros_like(dynamic)
    egocar[9:, :] = True
    regions = build_baseline_region_masks(
        dynamic, egocar, protocol=RegionMaskProtocol(boundary_radius_pixels=2)
    )
    assert not np.any(regions["static"] & dynamic)
    assert not np.any(regions["static"] & egocar)
    assert np.array_equal(regions["actor"], dynamic)
    assert regions["boundary"].sum() > dynamic.sum()
    assert regions["edit_roi"].sum() == 0


def test_explicit_edit_roi_is_preserved() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    roi = np.zeros_like(mask)
    roi[2:4, 3:5] = True
    regions = build_baseline_region_masks(mask, mask, edit_roi=roi)
    assert np.array_equal(regions["edit_roi"], roi)


def test_region_contract_rejects_shape_or_radius_drift() -> None:
    with pytest.raises(RegionMaskError, match="shape"):
        build_baseline_region_masks(np.zeros((3, 3)), np.zeros((4, 3)))
    with pytest.raises(RegionMaskError, match=">= 1"):
        build_baseline_region_masks(
            np.zeros((3, 3)),
            np.zeros((3, 3)),
            protocol=RegionMaskProtocol(boundary_radius_pixels=0),
        )
