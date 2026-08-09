import numpy as np
import pytest

from motion_proj.worldsim_v3.actor_metrics import (
    binary_dilate,
    binary_erode,
    boundary_band,
    complement_of_mask_union,
    counterfactual_effect_mask,
    psnr_from_sums,
    region_error_sums,
)


def test_binary_morphology_and_boundary_band() -> None:
    mask = np.zeros((9, 9), dtype=bool)
    mask[3:6, 3:6] = True
    assert binary_dilate(mask, 1).sum() == 25
    assert binary_erode(mask, 1).sum() == 1
    band = boundary_band(mask, 1)
    assert band.sum() == 24
    assert not band[4, 4]


def test_effect_mask_is_thresholded_and_dilated() -> None:
    original = np.zeros((7, 7, 3), dtype=np.uint8)
    deleted = original.copy()
    deleted[3, 3, 0] = 2
    mask = counterfactual_effect_mask(original, deleted)
    assert mask.sum() == 25
    assert mask[3, 3]


def test_effect_mask_rejects_non_uint8() -> None:
    image = np.zeros((2, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="uint8"):
        counterfactual_effect_mask(image, image)


def test_region_sums_and_psnr() -> None:
    target = np.zeros((2, 2, 3), dtype=np.float32)
    prediction = target.copy()
    prediction[0, 0] = 0.5
    mask = np.array([[True, False], [False, False]])
    result = region_error_sums(prediction, target, mask)
    assert result == {
        "pixel_count": 1,
        "value_count": 3,
        "squared_error_sum": 0.75,
        "absolute_error_sum": 1.5,
    }
    assert psnr_from_sums(result["squared_error_sum"], result["value_count"]) == pytest.approx(
        6.020599913279624
    )


def test_non_target_is_complement_of_selected_actor_union() -> None:
    first = np.zeros((3, 4), dtype=bool)
    second = np.zeros((3, 4), dtype=bool)
    first[0, 0] = True
    second[1, 1:3] = True
    non_target = complement_of_mask_union([first, second])
    assert non_target.sum() == 9
    assert not non_target[0, 0]
    assert not non_target[1, 1]
    assert non_target[2, 3]


def test_non_target_union_rejects_shape_drift() -> None:
    with pytest.raises(ValueError, match="share a shape"):
        complement_of_mask_union(
            [np.zeros((2, 2), dtype=bool), np.zeros((3, 2), dtype=bool)]
        )
