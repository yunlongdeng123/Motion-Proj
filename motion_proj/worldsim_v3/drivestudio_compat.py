"""Narrow compatibility helpers for the frozen DriveStudio baseline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor


def dense_instance_frame_count(
    frame_instances: Mapping[str, Any],
    instances_info: Mapping[str, Any],
    minimum_frame_count: int,
) -> int:
    """Return a safe dense timeline length for sparse instance annotations.

    NuScenes preprocessing may omit ``frame_instances`` keys for frames with
    no visible instances.  The upstream loader indexes its dense arrays by the
    original frame numbers, so ``len(frame_instances)`` is not a valid array
    length for such scenes.  Include the configured camera timeline and every
    frame index referenced by either instance JSON file.
    """

    if minimum_frame_count < 0:
        raise ValueError("minimum_frame_count must be non-negative")

    frame_indices: list[int] = []
    try:
        frame_indices.extend(int(frame_idx) for frame_idx in frame_instances)
        for instance in instances_info.values():
            frame_indices.extend(
                int(frame_idx)
                for frame_idx in instance["frame_annotations"]["frame_idx"]
            )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid instance timeline annotations") from error

    if any(frame_idx < 0 for frame_idx in frame_indices):
        raise ValueError("instance frame indices must be non-negative")
    return max([minimum_frame_count, *(frame_idx + 1 for frame_idx in frame_indices)])


def concatenate_paired_lidar_chunks(
    point_chunks: Sequence[Tensor],
    color_chunks: Sequence[Tensor],
) -> tuple[Tensor, Tensor]:
    """Concatenate paired LiDAR rows after discarding empty frame chunks.

    PyTorch 2.1.2+cu118 can raise ``CUDA error: invalid configuration
    argument`` when ``torch.cat`` receives the scene-0255 pattern of many
    empty CUDA tensors mixed with a few small non-empty tensors. Filtering
    empty *pairs* is value preserving and avoids separating points from their
    colors. The all-empty case returns a zero-row view with the prototype
    tensor's dtype, device, and trailing shape.
    """

    if len(point_chunks) != len(color_chunks):
        raise ValueError(
            "point/color chunk counts differ: "
            f"{len(point_chunks)} != {len(color_chunks)}"
        )
    if not point_chunks:
        raise ValueError("at least one point/color chunk pair is required")

    nonempty_points: list[Tensor] = []
    nonempty_colors: list[Tensor] = []
    for index, (points, colors) in enumerate(zip(point_chunks, color_chunks)):
        if points.ndim == 0 or colors.ndim == 0:
            raise ValueError(f"chunk {index} must have a row dimension")
        if points.shape[0] != colors.shape[0]:
            raise ValueError(
                f"chunk {index} point/color rows differ: "
                f"{points.shape[0]} != {colors.shape[0]}"
            )
        if points.shape[0] == 0:
            continue
        nonempty_points.append(points)
        nonempty_colors.append(colors)

    if not nonempty_points:
        return point_chunks[0][:0], color_chunks[0][:0]

    return (
        torch.cat(nonempty_points, dim=0),
        torch.cat(nonempty_colors, dim=0),
    )
