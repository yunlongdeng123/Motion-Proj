"""投影诱导的去噪分数匹配（方案第 7 节）。

在低噪声“管道”内，我们用投影目标 ``x_dagger`` 替换去噪目标 ``y``，并在可靠性
掩码 ``M_y`` 与界限门控下训练去噪器重建它。同时实现了 x0 形式（EDM/SVD 的默认
形式）和 eps 形式（VP 参数化），依据方案中的等价关系
``grad log q_t = -(1/sigma) eps_dagger``。
"""
from __future__ import annotations

import torch

from .tube import bound_gate, edm_weight, sample_tube_sigma


def _broadcast_mask(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """``[B,T,1,h,w]`` -> 可广播到 ``[B,T,C,h,w]``。"""
    if mask.dim() == like.dim() and mask.shape[2] == 1:
        return mask
    if mask.dim() == like.dim() - 1:  # [B,T,h,w]
        mask = mask.unsqueeze(2)
    return mask


def projection_loss(
    backbone,
    y: torch.Tensor,            # [B,T,C,h,w] latent
    x_dagger: torch.Tensor,     # [B,T,C,h,w] latent
    mask: torch.Tensor,         # [B,T,1,h,w]，取值 [0,1]（latent 分辨率）
    cond,
    cfg_tube,
    use_edm_weight: bool = False,
    use_tube: bool = True,
) -> dict:
    """返回 ``{loss, z, x0_hat, sigma, gate_frac}``（其中的辅助项会被锚定损失复用）。"""
    device = y.device
    b = y.shape[0]
    quantile_range = tuple(cfg_tube.sigma_quantile_range) if use_tube else (0.0, 1.0)
    sigma = sample_tube_sigma(backbone, b, device, quantile_range)
    noise = torch.randn_like(y)
    z = backbone.add_noise(y, sigma, noise)
    x0_hat = backbone.predict_x0(z, sigma, cond)

    target = x_dagger.detach()                              # 对投影结果做梯度截断
    m = _broadcast_mask(mask, y).to(x0_hat.dtype)

    if backbone.parameterization == "eps":
        pred = backbone.eps_from_x0(z, sigma, x0_hat)
        tgt = backbone.eps_from_x0(z, sigma, target)
    else:  # x0 形式
        pred, tgt = x0_hat, target

    se = (pred - tgt) ** 2 * m
    m_full = m.expand_as(se)
    per_sample = se.flatten(1).sum(1) / m_full.flatten(1).sum(1).clamp_min(1.0)

    gate = (bound_gate(sigma, y, x_dagger, float(cfg_tube.bound_B)).float()
            if use_tube else torch.ones(b, device=device))
    if use_edm_weight:
        per_sample = per_sample * edm_weight(sigma)
    loss = (per_sample * gate).sum() / gate.sum().clamp_min(1.0)

    return {
        "loss": loss,
        "z": z,
        "x0_hat": x0_hat,
        "sigma": sigma,
        "gate_frac": float(gate.mean()),
    }
