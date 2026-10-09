"""洞区掩码约定：1 是待生成区域，0 是已知可见区域。"""

from __future__ import annotations

import math

import torch


def make_border_outpaint_mask(
    height: int,
    width: int,
    left_fraction: float = 0.33,
    right_fraction: float = 0.33,
    *,
    top_fraction: float = 0.0,
    bottom_fraction: float = 0.0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """返回 ``[1,1,H,W]`` 的扩图洞区掩码，默认左右各遮 33%。

    与公开训练 mask 一致，以 ``round`` 确定边界；角落取并集。
    上下边默认不遮，供显式的四边扩图对照使用。
    """
    if height <= 0 or width <= 0:
        raise ValueError("height 和 width 必须为正整数")
    fractions = (left_fraction, right_fraction, top_fraction, bottom_fraction)
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in fractions):
        raise ValueError("各边 fraction 必须在 [0, 1] 内")

    mask = torch.zeros((1, 1, height, width), dtype=torch.float32, device=device)
    left = round(width * left_fraction)
    right = width - round(width * (1 - right_fraction))
    top = round(height * top_fraction)
    bottom = height - round(height * (1 - bottom_fraction))
    if left:
        mask[..., :left] = 1
    if right:
        mask[..., width - right :] = 1
    if top:
        mask[..., :top, :] = 1
    if bottom:
        mask[..., height - bottom :, :] = 1
    return mask


def apply_hole_mask(
    tensor: torch.Tensor,
    hole_mask: torch.Tensor,
    *,
    fill: float = 0.0,
) -> torch.Tensor:
    """构造已知区域条件，强制覆盖洞区原值，防止真值特征泄露。

    ``tensor`` 为 ``[B,C,H,W]``，``hole_mask`` 为 ``[B,1,H,W]``；
    掩码中任何正值都视为洞，因而重采样后的软边缘也不会透露隐藏值。
    """
    if tensor.ndim != 4 or hole_mask.ndim != 4:
        raise ValueError("tensor 与 hole_mask 都必须为 4 维")
    if hole_mask.shape != (tensor.shape[0], 1, *tensor.shape[-2:]):
        raise ValueError("hole_mask 形状必须是 [B,1,H,W] 并匹配 tensor")
    if tensor.device != hole_mask.device:
        raise ValueError("tensor 与 hole_mask 必须在同一设备")
    return torch.where(hole_mask > 0, torch.as_tensor(fill, dtype=tensor.dtype, device=tensor.device), tensor)
