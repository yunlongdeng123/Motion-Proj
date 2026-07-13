"""标准去噪目标 L_real（方案第 2 节）。"""
from __future__ import annotations

import torch

from .tube import edm_weight


def real_loss(
    backbone,
    x0: torch.Tensor,           # [B,T,C,h,w] GT 片段的 latent
    cond,
    use_edm_weight: bool = True,
    sigma: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> dict:
    """在 GT 片段上的单步去噪损失；可注入共享的 sigma/noise。"""
    device = x0.device
    b = x0.shape[0]
    sigma = backbone.sample_training_sigma(b, device) if sigma is None else sigma.to(device)
    noise = torch.randn_like(x0) if noise is None else noise.to(device=device, dtype=x0.dtype)
    if sigma.shape != (b,):
        raise ValueError(f"sigma 必须为 [B]，实际为 {tuple(sigma.shape)}")
    if noise.shape != x0.shape:
        raise ValueError("noise 必须与 x0 形状一致")
    z = backbone.add_noise(x0, sigma, noise)
    x0_hat = backbone.predict_x0(z, sigma, cond)

    se = (x0_hat - x0) ** 2
    per_sample = se.flatten(1).mean(1)
    if use_edm_weight:
        per_sample = per_sample * edm_weight(sigma)
    return {"loss": per_sample.mean(), "z": z, "x0_hat": x0_hat, "sigma": sigma}
