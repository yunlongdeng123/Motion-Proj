import numpy as np

from scripts.build_worldsim_v32_sam_masks import quality_gate


QUALITY = {
    "minimum_positive_pixels": 4,
    "minimum_prompt_bbox_iou": 0.02,
    "maximum_mask_to_prompt_area_ratio": 6.0,
    "maximum_centroid_to_prompt_diagonal": 1.0,
    "minimum_temporal_iou": 0.02,
    "maximum_temporal_centroid_jump_fraction": 0.12,
    "maximum_temporal_area_ratio": 5.0,
}


def test_quality_gate_accepts_prompt_aligned_mask() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:12, 6:14] = True
    accepted, reasons, metrics = quality_gate(
        binary=mask,
        projected_box=[5.0, 4.0, 15.0, 13.0],
        previous=None,
        quality=QUALITY,
    )
    assert accepted
    assert reasons == []
    assert metrics["prompt_bbox_iou"] is not None


def test_quality_gate_fails_closed_on_prompt_drift() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[15:19, 15:19] = True
    accepted, reasons, _ = quality_gate(
        binary=mask,
        projected_box=[1.0, 1.0, 5.0, 5.0],
        previous=None,
        quality=QUALITY,
    )
    assert not accepted
    assert "prompt_bbox_iou" in reasons


def test_quality_gate_fails_closed_on_temporal_area_jump() -> None:
    previous = np.zeros((30, 30), dtype=bool)
    previous[10:13, 10:13] = True
    current = np.zeros((30, 30), dtype=bool)
    current[5:25, 5:25] = True
    accepted, reasons, _ = quality_gate(
        binary=current,
        projected_box=None,
        previous=previous,
        quality=QUALITY,
    )
    assert not accepted
    assert "temporal_area_ratio" in reasons
