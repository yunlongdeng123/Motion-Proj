"""标准去噪目标 L_real（方案第 2 节）。"""
from __future__ import annotations

import torch

from .tube import edm_weight


def real_loss(
    backbone,
    x0: torch.Tensor,           # [B,T,C,h,w] GT 片段的 latent
    cond,
    use_edm_weight: bool = True,
) -> dict:
    """在 GT 片段上的单步去噪损失，采用对数正态的 EDM sigma 采样。"""
    device = x0.device
    b = x0.shape[0]
    sigma = backbone.sample_training_sigma(b, device)
    noise = torch.randn_like(x0)
    z = backbone.add_noise(x0, sigma, noise)
    x0_hat = backbone.predict_x0(z, sigma, cond)

    se = (x0_hat - x0) ** 2
    per_sample = se.flatten(1).mean(1)
    if use_edm_weight:
        per_sample = per_sample * edm_weight(sigma)
    return {"loss": per_sample.mean(), "z": z, "x0_hat": x0_hat, "sigma": sigma}
