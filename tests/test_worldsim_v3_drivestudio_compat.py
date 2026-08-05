from __future__ import annotations

import pytest
import torch

from motion_proj.worldsim_v3 import drivestudio_compat
from motion_proj.worldsim_v3.drivestudio_compat import (
    concatenate_paired_lidar_chunks,
)


def test_filters_empty_pairs_and_preserves_order(monkeypatch) -> None:
    point_chunks = [
        torch.empty((0, 3)),
        torch.tensor([[1.0, 2.0, 3.0]]),
        torch.empty((0, 3)),
        torch.tensor([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
    ]
    color_chunks = [
        torch.empty((0, 3)),
        torch.tensor([[0.1, 0.2, 0.3]]),
        torch.empty((0, 3)),
        torch.tensor([[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
    ]
    original_cat = torch.cat
    observed_row_counts: list[list[int]] = []

    def traced_cat(chunks, *args, **kwargs):
        observed_row_counts.append([chunk.shape[0] for chunk in chunks])
        return original_cat(chunks, *args, **kwargs)

    monkeypatch.setattr(drivestudio_compat.torch, "cat", traced_cat)
    points, colors = concatenate_paired_lidar_chunks(
        point_chunks, color_chunks
    )

    assert observed_row_counts == [[1, 2], [1, 2]]
    assert torch.equal(
        points,
        torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        ),
    )
    assert torch.allclose(
        colors,
        torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        ),
    )


def test_all_empty_returns_prototype_views_without_cat(monkeypatch) -> None:
    points_proto = torch.empty((0, 3), dtype=torch.float64)
    colors_proto = torch.empty((0, 4), dtype=torch.float16)

    def fail_cat(*args, **kwargs):
        raise AssertionError("torch.cat must not run for an all-empty instance")

    monkeypatch.setattr(drivestudio_compat.torch, "cat", fail_cat)
    points, colors = concatenate_paired_lidar_chunks(
        [points_proto, points_proto.clone()],
        [colors_proto, colors_proto.clone()],
    )

    assert points.shape == (0, 3)
    assert colors.shape == (0, 4)
    assert points.dtype == torch.float64
    assert colors.dtype == torch.float16


def test_rejects_unpaired_or_row_mismatched_chunks() -> None:
    with pytest.raises(ValueError, match="chunk counts differ"):
        concatenate_paired_lidar_chunks(
            [torch.empty((0, 3))],
            [],
        )
    with pytest.raises(ValueError, match="at least one"):
        concatenate_paired_lidar_chunks([], [])
    with pytest.raises(ValueError, match="rows differ"):
        concatenate_paired_lidar_chunks(
            [torch.empty((1, 3))],
            [torch.empty((2, 3))],
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_scene0255_mixed_empty_cuda_contract() -> None:
    shapes = (
        [(0, 3)] * 152
        + [(1, 3)] * 3
        + [(2, 3)] * 3
        + [(4, 3)] * 3
        + [(5, 3), (7, 3), (8, 3)]
        + [(9, 3)] * 2
    )
    point_chunks = [torch.arange(rows * 3, device="cuda").reshape(rows, 3) for rows, _ in shapes]
    color_chunks = [chunk + 1000 for chunk in point_chunks]

    points, colors = concatenate_paired_lidar_chunks(
        point_chunks, color_chunks
    )
    torch.cuda.synchronize()

    assert points.shape == (59, 3)
    assert colors.shape == (59, 3)
    assert points.numel() == 177
    assert torch.equal(colors, points + 1000)
