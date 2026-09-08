"""Preserve per-return origins with VDBFusion's one-origin integration API."""
import numpy as np


def integrate_per_origin(volume, points, origins):
    points = np.asarray(points, np.float64)
    origins = np.asarray(origins, np.float64)
    if not len(points):
        return {'integrated_points': 0, 'origin_groups': 0}
    unique, inverse = np.unique(origins, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind='stable')
    grouped = inverse[order]
    edges = np.r_[0, np.flatnonzero(np.diff(grouped)) + 1, len(order)]
    # Partition once: no O(points * origins) repeated boolean masks, no rounding,
    # frame-mean origin, ray cap, or loss of the acquisition-time geometry.
    for left, right in zip(edges[:-1], edges[1:]):
        take = order[left:right]
        volume.integrate(np.ascontiguousarray(points[take]),
                         np.ascontiguousarray(unique[grouped[left]]))
    return {'integrated_points': len(points), 'origin_groups': len(unique)}
