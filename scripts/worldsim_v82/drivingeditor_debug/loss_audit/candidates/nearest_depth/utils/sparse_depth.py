"""Permutation-invariant nearest visible sparse depth; NumPy only.

This is an explicit geometric projection choice. The paper does not specify
how multiple LiDAR points in a raster pixel are reduced.
"""
import numpy as np


def nearest_positive_depth(uv, depth, height, width):
    """Return1xHxW nearest finite positive depths; invalid/empty pixels are0.

    uv may contain finite nonnegative float coordinates or integer pixels.
    Coordinate quantization matches camera_utils' positive-coordinate truncation.
    np.minimum.at resolves repeated indices without dependence on point order.
    """
    uv = np.asarray(uv)
    depth = np.asarray(depth)
    if uv.ndim != 2 or uv.shape[1] != 2 or depth.shape != (len(uv),):
        raise ValueError("Expected Nx2 uv and N depth")
    if height <= 0 or width <= 0:
        raise ValueError("Raster dimensions must be positive")
    valid = (np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 0)
             & (uv[:, 0] >= 0) & (uv[:, 0] < width)
             & (uv[:, 1] >= 0) & (uv[:, 1] < height))
    pixels = uv[valid].astype(np.int64)
    indices = pixels[:, 1] * width + pixels[:, 0]
    nearest = np.full(height * width, np.inf, dtype=np.float64)
    np.minimum.at(nearest, indices, depth[valid])
    nearest[~np.isfinite(nearest)] = 0.0
    return nearest.reshape(1, height, width)
