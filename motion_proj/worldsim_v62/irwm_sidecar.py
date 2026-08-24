"""Pack frozen IR-WM outputs at the sparse P2 query coordinates."""

from __future__ import annotations

from typing import Any

import numpy as np


def pack_query_aligned_sidecar(
    *,
    query_indices: np.ndarray,
    query_origin_m: np.ndarray,
    query_voxel_size_m: float,
    query_grid_shape: np.ndarray,
    source_logits: np.ndarray,
    source_bev_features: np.ndarray,
    source_origin_m: np.ndarray,
    source_voxel_size_m: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Deduplicate source cells while retaining a mapping for every query."""

    query_indices = np.asarray(query_indices, dtype=np.int32)
    query_origin_m = np.asarray(query_origin_m, dtype=np.float64)
    query_grid_shape = np.asarray(query_grid_shape, dtype=np.int64)
    logits = np.asarray(source_logits, dtype=np.float16)
    bev = np.asarray(source_bev_features, dtype=np.float16)
    source_origin_m = np.asarray(source_origin_m, dtype=np.float64)
    if query_indices.ndim != 2 or query_indices.shape[1] != 3:
        raise ValueError(f"query indices must be Nx3, got {query_indices.shape}")
    if logits.ndim != 4 or bev.ndim != 3 or logits.shape[:2] != bev.shape[:2]:
        raise ValueError(f"IR-WM source shapes are incompatible: {logits.shape}, {bev.shape}")

    centers = query_origin_m[None, :] + (
        query_indices.astype(np.float64) + 0.5
    ) * float(query_voxel_size_m)
    source_indices = np.floor(
        (centers - source_origin_m[None, :]) / float(source_voxel_size_m)
    ).astype(np.int32)
    source_shape = np.asarray(logits.shape[:3], dtype=np.int32)
    valid = np.all((source_indices >= 0) & (source_indices < source_shape[None]), axis=1)

    query_to_prior_cell = np.full(query_indices.shape[0], -1, dtype=np.int32)
    query_to_bev_cell = np.full(query_indices.shape[0], -1, dtype=np.int32)
    valid_indices = source_indices[valid]
    if valid_indices.shape[0]:
        source_linear = np.ravel_multi_index(valid_indices.T, tuple(source_shape))
        unique_prior_linear, prior_inverse = np.unique(source_linear, return_inverse=True)
        prior_cell_indices = np.stack(
            np.unravel_index(unique_prior_linear, tuple(source_shape)), axis=1
        ).astype(np.int16)
        prior_logits = logits.reshape(-1, logits.shape[-1])[unique_prior_linear]
        query_to_prior_cell[valid] = prior_inverse.astype(np.int32)

        bev_shape = tuple(int(value) for value in bev.shape[:2])
        bev_linear = np.ravel_multi_index(valid_indices[:, :2].T, bev_shape)
        unique_bev_linear, bev_inverse = np.unique(bev_linear, return_inverse=True)
        bev_cell_indices = np.stack(
            np.unravel_index(unique_bev_linear, bev_shape), axis=1
        ).astype(np.int16)
        bev_features = bev.reshape(-1, bev.shape[-1])[unique_bev_linear]
        query_to_bev_cell[valid] = bev_inverse.astype(np.int32)
    else:
        prior_cell_indices = np.empty((0, 3), dtype=np.int16)
        prior_logits = np.empty((0, logits.shape[-1]), dtype=np.float16)
        bev_cell_indices = np.empty((0, 2), dtype=np.int16)
        bev_features = np.empty((0, bev.shape[-1]), dtype=np.float16)

    arrays = {
        "query_to_prior_cell": query_to_prior_cell,
        "prior_cell_indices": prior_cell_indices,
        "prior_logits": prior_logits,
        "query_to_bev_cell": query_to_bev_cell,
        "bev_cell_indices": bev_cell_indices,
        "bev_features": bev_features,
        "query_source_valid": valid,
        "query_grid_origin_m": query_origin_m,
        "query_grid_voxel_size_m": np.asarray(query_voxel_size_m, dtype=np.float64),
        "query_grid_shape": query_grid_shape,
        "source_grid_origin_m": source_origin_m,
        "source_grid_voxel_size_m": np.asarray(source_voxel_size_m, dtype=np.float64),
        "source_grid_shape": source_shape,
    }
    summary = {
        "query_count": int(query_indices.shape[0]),
        "source_valid_query_count": int(np.count_nonzero(valid)),
        "source_invalid_query_count": int(np.count_nonzero(~valid)),
        "unique_prior_cell_count": int(prior_cell_indices.shape[0]),
        "unique_bev_cell_count": int(bev_cell_indices.shape[0]),
        "prior_class_count": int(logits.shape[-1]),
        "bev_feature_dimension": int(bev.shape[-1]),
    }
    return arrays, summary
