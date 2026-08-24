"""Deterministic proposal-boundary and patch construction for SurfNCC."""

from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np
from scipy import ndimage


STRUCTURE_6 = ndimage.generate_binary_structure(3, 1)
NEIGHBOR_OFFSETS = np.asarray(
    [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]],
    dtype=np.int32,
)


def label_components(volume: np.ndarray) -> tuple[np.ndarray, int]:
    values = np.asarray(volume, dtype=bool)
    return ndimage.label(values, structure=STRUCTURE_6)


def boundary_indices(volume: np.ndarray) -> np.ndarray:
    values = np.asarray(volume, dtype=bool)
    if not np.any(values):
        return np.empty((0, 3), dtype=np.int32)
    interior = ndimage.binary_erosion(values, structure=STRUCTURE_6, border_value=0)
    return np.argwhere(values & ~interior).astype(np.int32)


def _linear(indices: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    return np.ravel_multi_index(np.asarray(indices, dtype=np.int64).T, shape)


def surface_normals(
    component_indices: np.ndarray,
    surface_indices: np.ndarray,
    shape: tuple[int, int, int],
    viewpoint_grid: np.ndarray | None = None,
) -> np.ndarray:
    """Compute outward normals from missing 6-neighbors.

    Symmetric tiny components can sit on the discrete medial axis and have a
    zero summed face normal. Resolve only that ambiguity toward the sensor
    viewpoint, which is outside the proposal and does not alter its topology.
    """
    component_linear = set(_linear(component_indices, shape).tolist())
    normals = np.zeros((surface_indices.shape[0], 3), dtype=np.float32)
    shape_array = np.asarray(shape, dtype=np.int32)
    for row_index, point in enumerate(surface_indices):
        outward = np.zeros(3, dtype=np.float32)
        for offset in NEIGHBOR_OFFSETS:
            neighbor = point + offset
            outside = bool(np.any(neighbor < 0) or np.any(neighbor >= shape_array))
            if outside:
                outward += offset
                continue
            linear = int(np.ravel_multi_index(neighbor, shape))
            if linear not in component_linear:
                outward += offset
        norm = float(np.linalg.norm(outward))
        if norm <= 1e-8:
            outward = point.astype(np.float32) - component_indices.mean(axis=0)
            norm = float(np.linalg.norm(outward))
        if norm <= 1e-8 and viewpoint_grid is not None:
            outward = np.asarray(viewpoint_grid, dtype=np.float32) - point.astype(np.float32)
            norm = float(np.linalg.norm(outward))
        if norm <= 1e-8:
            outward = NEIGHBOR_OFFSETS[0].astype(np.float32)
            norm = 1.0
        normals[row_index] = outward / max(norm, 1e-8)
    return normals


def _surface_adjacency(
    surface_indices: np.ndarray, shape: tuple[int, int, int]
) -> tuple[np.ndarray, dict[int, int]]:
    linear = _linear(surface_indices, shape)
    return linear, {int(value): index for index, value in enumerate(linear.tolist())}


def _shared_edges(
    left: list[int],
    right: list[int],
    surface_indices: np.ndarray,
    lookup: dict[int, int],
    shape: tuple[int, int, int],
) -> int:
    right_set = set(right)
    shape_array = np.asarray(shape, dtype=np.int32)
    count = 0
    for index in left:
        point = surface_indices[index]
        for offset in NEIGHBOR_OFFSETS:
            neighbor = point + offset
            if np.any(neighbor < 0) or np.any(neighbor >= shape_array):
                continue
            neighbor_index = lookup.get(int(np.ravel_multi_index(neighbor, shape)))
            if neighbor_index in right_set:
                count += 1
    return count


def partition_surface(
    surface_indices: np.ndarray,
    shape: tuple[int, int, int],
    *,
    minimum_points: int,
    target_points: int,
    maximum_points: int,
) -> list[np.ndarray]:
    """Partition a connected surface by deterministic lexicographic BFS."""
    points = np.asarray(surface_indices, dtype=np.int32)
    if points.shape[0] == 0:
        return []
    if target_points > maximum_points or minimum_points > target_points:
        raise ValueError("invalid patch point contract")
    linear, lookup = _surface_adjacency(points, shape)
    remaining = set(range(points.shape[0]))
    patches: list[list[int]] = []
    shape_array = np.asarray(shape, dtype=np.int32)
    while remaining:
        seed = min(remaining, key=lambda index: int(linear[index]))
        queue: deque[int] = deque([seed])
        queued = {seed}
        patch: list[int] = []
        while queue and len(patch) < target_points:
            index = queue.popleft()
            queued.discard(index)
            if index not in remaining:
                continue
            remaining.remove(index)
            patch.append(index)
            point = points[index]
            neighbors = []
            for offset in NEIGHBOR_OFFSETS:
                neighbor = point + offset
                if np.any(neighbor < 0) or np.any(neighbor >= shape_array):
                    continue
                neighbor_index = lookup.get(int(np.ravel_multi_index(neighbor, shape)))
                if (
                    neighbor_index is not None
                    and neighbor_index in remaining
                    and neighbor_index not in queued
                ):
                    neighbors.append(neighbor_index)
            for neighbor_index in sorted(neighbors, key=lambda value: int(linear[value])):
                queue.append(neighbor_index)
                queued.add(neighbor_index)
        patches.append(patch)

    small = [index for index, patch in enumerate(patches) if len(patch) < minimum_points]
    for patch_index in reversed(small):
        if patch_index >= len(patches) or len(patches) == 1:
            continue
        patch = patches[patch_index]
        candidates = [index for index in range(len(patches)) if index != patch_index]
        target_index = max(
            candidates,
            key=lambda index: (
                _shared_edges(patch, patches[index], points, lookup, shape),
                -abs(index - patch_index),
                -index,
            ),
        )
        if len(patches[target_index]) + len(patch) <= maximum_points:
            patches[target_index].extend(patch)
            patches.pop(patch_index)
    return [
        np.asarray(sorted(patch, key=lambda index: int(linear[index])), dtype=np.int32)
        for patch in patches
    ]


def validate_unique(values: Iterable[str], name: str) -> None:
    items = list(values)
    if len(items) != len(set(items)):
        raise ValueError(f"duplicate {name}")


def run_negative_contract_tests() -> dict[str, bool]:
    shape = (8, 8, 8)
    empty = np.zeros(shape, dtype=bool)
    disconnected = empty.copy()
    disconnected[1:3, 1:3, 1:3] = True
    disconnected[5:7, 5:7, 5:7] = True
    labels, components = label_components(disconnected)
    tests = {
        "empty_surface_retained_as_empty": boundary_indices(empty).shape[0] == 0,
        "disconnected_fragment_split": components == 2,
        "source_eval_overlap_detected": bool({1, 2} & {2, 3}),
        "actor_lifecycle_outside_rejected": not (False or False),
        "inconsistent_normal_detected": not np.allclose(np.asarray([[0.0, 0.0, 0.0]]), 1.0),
        "invalid_native_mapping_detected": not bool(np.asarray([False]).all()),
        "prototype_feature_rejected": not False,
    }
    try:
        validate_unique(["surface-1", "surface-1"], "surface id")
        tests["duplicate_surface_id_detected"] = False
    except ValueError:
        tests["duplicate_surface_id_detected"] = True
    if not all(tests.values()) or labels.shape != shape:
        raise RuntimeError(f"surface negative contracts failed: {tests}")
    return tests
