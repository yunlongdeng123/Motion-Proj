"""Unmodified inherited CPU PCA/spacing helper for actual PVD point patches.

Extracted from full_pvd.py, full-source SHA256:
dc442b68c2428f731705bf48a67b22d7ed95f232324cbac8e3cf13d1794b5728
The mathematical function body and its original docstring are retained verbatim.
Scene/default assets, model classes, Torch and CLI are omitted.
"""
import numpy as np
from scipy.spatial import cKDTree


def pca_point_patches(points):
    """Exact CPU PCA/spacing algorithm of prepare_pvd_surface.py, no filtering."""
    _, first = np.unique(np.floor(points/.005).astype(np.int64), axis=0, return_index=True)
    original_index = np.sort(first); p = points[original_index]
    tree = cKDTree(p); k = min(16, len(p)); distance, index = tree.query(p, k=k)
    if k < 4:
        raise ValueError('Too few distinct actual PVD points for point patches')
    spacing = np.median(distance[:, 1:min(4, k)], axis=1)
    radius = np.minimum(.25, np.maximum(4*spacing, .03))
    weight = (distance <= radius[:, None]).astype(float)
    q = p[index]-p[:, None]
    mean = (q*weight[:, :, None]).sum(1)/np.maximum(weight.sum(1), 1)[:, None]
    q = q-mean[:, None]
    covariance = np.einsum('nki,nkj,nk->nij', q, q, weight)/np.maximum(weight.sum(1), 1)[:, None, None]
    eigenvalue, eigenvector = np.linalg.eigh(covariance); normal = eigenvector[:, :, 0]
    flip = (normal*p).sum(1) < 0; normal[flip] *= -1
    valid = (weight.sum(1) >= 4) & (eigenvalue[:, 1] > 1e-7) & (spacing > 1e-6)
    confidence = np.clip((eigenvalue[:, 1]-eigenvalue[:, 0])/np.maximum(eigenvalue[:, 2], 1e-10), 0, 1)
    sigma = np.minimum(.60*spacing, .060)
    sigma = np.minimum(sigma, .45*radius)
    return dict(points=p, normal=normal, sigma=sigma, normal_valid=valid,
                normal_confidence=confidence, knn_spacing=spacing,
                normal_radius=radius, original_index=original_index)
