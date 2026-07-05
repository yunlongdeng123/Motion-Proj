"""锚定损失，使微调后的模型与冻结的基座模型保持接近（方案第 7 节）。

L_anchor = || f_theta(z,t,c) - f_theta0(z,t,c) ||^2，在学生模型已经使用的*同一个*
带噪 latent 上计算（因此不额外加噪）。实现于 x0 空间，在不计 EDM 缩放的意义下与
eps 空间的形式等价。
"""
from __future__ import annotations

import torch


def anchor_loss(backbone, z: torch.Tensor, sigma: torch.Tensor, cond, x0_hat_student: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        x0_anchor = backbone.anchor_predict_x0(z, sigma, cond)
    return ((x0_hat_student - x0_anchor.detach()) ** 2).flatten(1).mean(1).mean()
