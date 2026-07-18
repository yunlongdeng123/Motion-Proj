import pytest
import torch

from motion_proj.data.motion_feature_records import (
    local_correlation_window,
    sample_temporal_features,
)


def test_temporal_bilinear_sampling_uses_image_pixel_coordinates():
    features = torch.zeros((2, 2, 3, 4))
    for t in range(2):
        for y in range(3):
            for x in range(4):
                features[t, 0, y, x] = 100 * t + 10 * y + x
                features[t, 1, y, x] = 1.0
    sampled = sample_temporal_features(
        features,
        torch.tensor([0, 1]),
        torch.tensor([[2.0, 1.0], [3.0, 2.0]]),
        image_hw=(3, 4),
    )
    assert sampled[:, 0].tolist() == pytest.approx([12.0, 123.0])


def test_local_cost_window_peaks_at_same_feature_cell():
    features = torch.zeros((2, 9, 3, 3))
    for index in range(9):
        y, x = divmod(index, 3)
        features[:, index, y, x] = 1.0
    correlation = local_correlation_window(
        features,
        torch.tensor([0]),
        torch.tensor([[1.0, 1.0]]),
        torch.tensor([[1.0, 1.0]]),
        image_hw=(3, 3),
        radius_cells=1,
    )
    assert correlation.shape == (1, 9)
    assert int(correlation.argmax(dim=1)) == 4
    assert float(correlation[0, 4]) == pytest.approx(1.0)

