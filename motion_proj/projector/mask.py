"""可靠性掩码 M_y：标记投影目标在何处是可信的。"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def build_reliability_mask(
    static_mask: torch.Tensor,   # [F,H,W] 可靠静态像素（F = K-1）
    obj_mask: torch.Tensor,      # [K,H,W] 可靠修复的目标区域
    num_frames: int,
) -> torch.Tensor:
    """将 静态 + 目标 的可靠性合并为 ``[K,1,H,W]``，取值 [0,1]。

    帧 k 的静态可靠性在可用时使用帧对 (k-1,k)/(k,k+1)；
    目标可靠性来自合成器（compositor），为逐帧结果。
    """
    h, w = obj_mask.shape[-2:]
    device = obj_mask.device
    static_k = torch.zeros(num_frames, h, w, device=device)
    f = static_mask.shape[0]
    # 帧 0 取帧对 0；帧 k>0 取帧对 k-1
    static_k[0] = static_mask[0]
    for k in range(1, num_frames):
        static_k[k] = static_mask[min(k - 1, f - 1)]
    m = torch.maximum(static_k, obj_mask)
    return m.unsqueeze(1).clamp(0, 1)


def downsample_mask_to_latent(mask: torch.Tensor, scale: int) -> torch.Tensor:
    """``[K,1,H,W]`` -> ``[K,1,H/scale,W/scale]``（区域平均，保持在 [0,1]）。"""
    return F.avg_pool2d(mask, kernel_size=scale, stride=scale).clamp(0, 1)
