"""在 immutable base opacity 上渲染 V5 Gaussian ownership sidecar。"""

from __future__ import annotations

from typing import Any

import numpy as np


def compose_ownership_opacity(
    base_opacity: np.ndarray, probability: np.ndarray
) -> np.ndarray:
    """返回 ``base alpha × ownership probability``，不改写任一输入。"""

    opacity = np.asarray(base_opacity, dtype=np.float64)
    ownership = np.asarray(probability, dtype=np.float64)
    if opacity.ndim == 2 and opacity.shape[1] == 1:
        opacity = opacity[:, 0]
    if opacity.ndim != 1 or ownership.shape != opacity.shape:
        raise ValueError("base_opacity/probability 必须是一一对齐的一维数组")
    for name, value in (("base_opacity", opacity), ("probability", ownership)):
        if not np.isfinite(value).all() or np.any((value < 0.0) | (value > 1.0)):
            raise ValueError(f"{name} 必须为有限 [0,1]")
    return (opacity * ownership).astype(np.float32)


def rasterize_ownership_probability(
    *,
    means: Any,
    quats: Any,
    scales: Any,
    base_opacities: Any,
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
    """用真实 base alpha 合成 2D ownership probability；base 必须无梯度。"""

    import torch
    from gsplat import rasterization

    for name, value in (
        ("means", means),
        ("quats", quats),
        ("scales", scales),
        ("base_opacities", base_opacities),
    ):
        if value.requires_grad or value.grad_fn is not None:
            raise RuntimeError(f"base {name} 不是 immutable tensor")
    ownership = torch.as_tensor(
        probability, dtype=means.dtype, device=means.device
    ).reshape(-1)
    opacity = base_opacities.reshape(-1)
    if ownership.shape != opacity.shape or ownership.shape[0] != means.shape[0]:
        raise ValueError("ownership/base opacity/Gaussian shape 不一致")
    if not torch.isfinite(ownership).all() or bool(
        ((ownership < 0.0) | (ownership > 1.0)).any()
    ):
        raise ValueError("ownership probability 必须为有限 [0,1]")
    if not torch.isfinite(opacity).all() or bool(
        ((opacity < 0.0) | (opacity > 1.0)).any()
    ):
        raise ValueError("base opacity 必须为有限 [0,1]")
    semantic_opacity = opacity * ownership
    colors = torch.ones((means.shape[0], 3), dtype=means.dtype, device=means.device)
    _, alpha, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=semantic_opacity,
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
