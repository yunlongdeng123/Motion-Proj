"""低噪声“管道”时间步采样与投影界限门控。

投影损失仅在低噪声 sigma 上施加，此时带噪样本仍处于 ``y`` 的语义邻域内
（方案第 8.3 节），并且仅当目标修正的幅度不会爆炸时才施加：

    (1 / sigma) * || y - x_dagger ||  <=  B
"""
from __future__ import annotations

import torch


def sample_tube_sigma(
    backbone,
    batch_size: int,
    device,
    quantile_range: tuple[float, float] = (0.0, 0.35),
    schedule_len: int = 50,
) -> torch.Tensor:
    """从调度表的低噪声子集中为每个样本采样一个 sigma。"""
    sigmas = backbone.sample_sigmas(schedule_len, device)         # 升序
    n = sigmas.numel()
    lo = int(quantile_range[0] * n)
    hi = max(lo + 1, int(quantile_range[1] * n))
    pool = sigmas[lo:hi]
    idx = torch.randint(0, pool.numel(), (batch_size,), device=device)
    return pool[idx]


def bound_gate(
    sigma: torch.Tensor,        # [B]
    y: torch.Tensor,            # [B,T,C,h,w]
    x_dagger: torch.Tensor,     # [B,T,C,h,w]
    B: float,
) -> torch.Tensor:
    """逐样本的布尔门控 ``(1/sigma)*||y - x_dagger|| <= B`` -> ``[B]``。"""
    diff = (y - x_dagger).flatten(1).norm(dim=1)                  # [B]
    rms = diff / (y[0].numel() ** 0.5)                            # 尺度归一化
    return (rms / sigma.clamp_min(1e-8)) <= B


def edm_weight(sigma: torch.Tensor, sigma_data: float = 0.5) -> torch.Tensor:
    """EDM 的逐 sigma 损失权重 ``(sigma^2+sigma_data^2)/(sigma*sigma_data)^2``。"""
    return (sigma**2 + sigma_data**2) / (sigma * sigma_data).clamp_min(1e-8) ** 2
