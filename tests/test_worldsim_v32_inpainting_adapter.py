from __future__ import annotations

import numpy as np
import pytest
import torch

from motion_proj.worldsim_v3.gaussian_ancestry import GaussianAncestryLedger, InitSource
from motion_proj.worldsim_v32.inpainting_adapter import (
    CompletionPoints,
    append_generated_background,
    completion_points_from_view,
    merge_completion_points,
)


def test_completion_points_and_voxel_merge_prefer_confidence() -> None:
    rgb = np.full((2, 2, 3), 128, dtype=np.uint8)
    depth = np.full((2, 2), 2.0, dtype=np.float32)
    group = completion_points_from_view(
        rgb=rgb,
        depth=depth,
        mask=np.ones((2, 2), dtype=bool),
        observed_cross_view=np.array([[True, False], [False, False]]),
        intrinsics=np.eye(3),
        camera_to_world=np.eye(4),
        stride=1,
        scale_multiplier=1.0,
        minimum_scale_m=0.01,
        maximum_scale_m=0.2,
    )
    assert group.means.shape == (4, 3)
    assert sorted(group.confidence.tolist()) == pytest.approx([0.5, 0.5, 0.5, 0.9])
    duplicated = CompletionPoints(
        means=group.means[:1].copy(),
        rgb=np.ones((1, 3), dtype=np.float32),
        scales=group.scales[:1].copy(),
        confidence=np.array([0.4], dtype=np.float32),
        observed_cross_view=np.array([False]),
        source_pixels_xy=np.array([[9, 9]]),
    )
    merged = merge_completion_points([group, duplicated], voxel_size_m=0.001)
    assert merged.means.shape[0] == 4
    assert merged.confidence.max() == pytest.approx(0.9)


def test_append_generated_background_preserves_old_rows_and_aligns_ancestry() -> None:
    class FakeBackground:
        @property
        def num_points(self):
            return self._means.shape[0]

    background = FakeBackground()
    background._means = torch.nn.Parameter(torch.tensor([[0.0, 0.0, 0.0]]))
    background._features_dc = torch.nn.Parameter(torch.zeros((1, 3)))
    background._features_rest = torch.nn.Parameter(torch.zeros((1, 0, 3)))
    background._opacities = torch.nn.Parameter(torch.zeros((1, 1)))
    background._scales = torch.nn.Parameter(torch.zeros((1, 3)))
    background._quats = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    background.sh_degree = 0
    background._a2_ancestry = GaussianAncestryLedger.initialize(
        means=background._means.detach(),
        actor_ids=-1,
        init_sources=int(InitSource.LIDAR),
    )
    points = CompletionPoints(
        means=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        rgb=np.array([[0.25, 0.5, 0.75]], dtype=np.float32),
        scales=np.array([[0.1, 0.1, 0.1]], dtype=np.float32),
        confidence=np.array([0.5], dtype=np.float32),
        observed_cross_view=np.array([False]),
        source_pixels_xy=np.array([[2, 3]], dtype=np.int64),
    )
    result = append_generated_background(background, points, opacity=0.8, birth_step=30000)
    assert result["old_background_count"] == 1
    assert result["generated_count"] == 1
    assert background._means.shape == (2, 3)
    assert len(background._a2_ancestry) == 2
    assert background._a2_ancestry.init_source.tolist() == [1, 0]
    torch.testing.assert_close(background._means[0], torch.zeros(3))


def test_empty_append_fails_closed() -> None:
    points = CompletionPoints(
        means=np.empty((0, 3), dtype=np.float32),
        rgb=np.empty((0, 3), dtype=np.float32),
        scales=np.empty((0, 3), dtype=np.float32),
        confidence=np.empty((0,), dtype=np.float32),
        observed_cross_view=np.empty((0,), dtype=bool),
        source_pixels_xy=np.empty((0, 2), dtype=np.int64),
    )
    with pytest.raises(ValueError, match="空"):
        append_generated_background(object(), points, opacity=0.8, birth_step=1)
