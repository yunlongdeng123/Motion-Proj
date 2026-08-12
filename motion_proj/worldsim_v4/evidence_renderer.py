"""Frozen-geometry renderer for Gaussian evidence probabilities."""

from __future__ import annotations

from typing import Any

import numpy as np


def probability_to_logit(probability: np.ndarray, *, epsilon: float = 1e-6) -> np.ndarray:
    value = np.asarray(probability, dtype=np.float64)
    if not np.isfinite(value).all() or np.any((value < 0.0) | (value > 1.0)):
        raise ValueError("probability must be finite and lie in [0, 1]")
    clipped = np.clip(value, epsilon, 1.0 - epsilon)
    return (np.log(clipped) - np.log1p(-clipped)).astype(np.float32)


def rasterize_evidence_mask(
    *,
    means: Any,
    quats: Any,
    scales: Any,
    probability: Any,
    viewmats: Any,
    intrinsics: Any,
    width: int,
    height: int,
    near_plane: float,
    far_plane: float,
    packed: bool,
    radius_clip: float,
    antialiased: bool,
):
    """Rasterize evidence only; reject any trainable base geometry tensor."""

    import torch
    from gsplat import rasterization

    for name, value in (("means", means), ("quats", quats), ("scales", scales)):
        if value.requires_grad or value.grad_fn is not None:
            raise RuntimeError(f"base {name} is not immutable")
    ownership = torch.as_tensor(
        probability, dtype=means.dtype, device=means.device
    )
    if ownership.ndim != 1 or ownership.shape[0] != means.shape[0]:
        raise ValueError("probability/Gaussian shapes differ")
    if not torch.isfinite(ownership).all() or bool(
        ((ownership < 0.0) | (ownership > 1.0)).any()
    ):
        raise ValueError("probability must lie in [0, 1]")
    colors = torch.ones((means.shape[0], 3), dtype=means.dtype, device=means.device)
    _, alpha, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=ownership,
        colors=colors,
        viewmats=viewmats,
        Ks=intrinsics,
        width=int(width),
        height=int(height),
        near_plane=float(near_plane),
        far_plane=float(far_plane),
        packed=bool(packed),
        absgrad=False,
        sparse_grad=False,
        radius_clip=float(radius_clip),
        render_mode="RGB",
        rasterize_mode="antialiased" if antialiased else "classic",
    )
    return alpha[0, ..., 0], info
