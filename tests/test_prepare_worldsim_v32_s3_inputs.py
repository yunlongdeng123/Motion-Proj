from __future__ import annotations

import numpy as np
from PIL import Image
import pytest

from scripts.prepare_worldsim_v32_s3_inputs import (
    choose_requested_views,
    choose_views,
    dilate_binary,
    square_crop,
)


def row(
    frame: int,
    prompt_frame: int,
    camera: int,
    box: list[float],
    positive: int,
    prompt_iou: float | None,
    temporal_iou: float | None,
) -> dict:
    return {
        "frame": frame,
        "prompt_frame": prompt_frame,
        "camera_id": camera,
        "camera_name": f"CAM_{camera}",
        "projected_box_xyxy": box,
        "width": 800,
        "height": 450,
        "positive_pixels": positive,
        "quality_metrics": {
            "prompt_bbox_iou": prompt_iou,
            "temporal_iou": temporal_iou,
        },
    }


def test_choose_views_rejects_large_clipped_drift() -> None:
    rows = [
        row(21, 21, 2, [0.0, 160.0, 527.0, 449.5], 63_000, 0.84, None),
        row(0, 0, 0, [511.0, 221.0, 633.0, 301.0], 3_400, 0.52, None),
        row(9, 0, 0, [539.0, 210.0, 749.0, 335.0], 9_500, 0.58, 0.86),
    ]
    selected = choose_views(rows)
    assert [(item["frame"], item["camera_id"]) for item in selected] == [
        (0, 0),
        (9, 0),
    ]


def test_choose_requested_views_preserves_frozen_order() -> None:
    rows = [
        row(51, 51, 1, [100.0, 100.0, 500.0, 300.0], 5_000, 0.6, None),
        row(91, 91, 1, [150.0, 120.0, 650.0, 400.0], 15_000, 0.7, None),
    ]
    selected = choose_requested_views(rows, ["91:CAM_1", "51:1"])
    assert [(item["frame"], item["camera_id"]) for item in selected] == [
        (91, 1),
        (51, 1),
    ]


def test_choose_requested_views_rejects_propagated_frame() -> None:
    rows = [
        row(52, 51, 1, [100.0, 100.0, 500.0, 300.0], 5_000, 0.6, 0.9),
        row(91, 91, 1, [150.0, 120.0, 650.0, 400.0], 15_000, 0.7, None),
    ]
    with pytest.raises(RuntimeError, match="不是直接 prompt"):
        choose_requested_views(rows, ["91:1", "52:1"])


def test_square_crop_keeps_binary_mask_and_white_padding() -> None:
    image = Image.new("RGB", (8, 6), color=(10, 20, 30))
    mask = np.zeros((6, 8), dtype=bool)
    mask[1:5, 0:2] = True
    cropped_image, cropped_mask, crop = square_crop(
        image, mask, padding_fraction=0.5, output_size=16
    )
    assert cropped_image.size == (16, 16)
    assert cropped_mask.size == (16, 16)
    assert crop[0] < 0
    values = set(np.unique(np.asarray(cropped_mask)).tolist())
    assert values <= {0, 255}
    assert (np.asarray(cropped_mask) > 0).any()
    assert (np.asarray(cropped_image) == 255).all(axis=2).any()


def test_dilate_binary_expands_exact_radius() -> None:
    mask = np.zeros((7, 7), dtype=bool)
    mask[3, 3] = True
    expanded = dilate_binary(mask, radius=2)
    assert int(expanded.sum()) == 25
    assert expanded[1:6, 1:6].all()
