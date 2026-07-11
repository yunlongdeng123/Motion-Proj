"""离线 RAFT flow 驱动的 latent 时序一致性损失。"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _sampling_grid(flow: torch.Tensor) -> torch.Tensor:
    batch, height, width, _ = flow.shape
    ys, xs = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=flow.dtype),
        torch.arange(width, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    x = xs.unsqueeze(0) + flow[..., 0]
    y = ys.unsqueeze(0) + flow[..., 1]
    gx = 2.0 * x / max(width - 1, 1) - 1.0
    gy = 2.0 * y / max(height - 1, 1) - 1.0
    return torch.stack((gx, gy), dim=-1).expand(batch, -1, -1, -1)


def flow_warp_charbonnier_loss(
    video: torch.Tensor,
    flow: torch.Tensor,
    confidence: torch.Tensor,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    """用 t->t+1 flow 把下一帧采样回 t，并计算置信度加权 Charbonnier loss。"""
    if video.dim() != 5:
        raise ValueError("video 必须为 [B,T,C,H,W]")
    batch, frames, channels, height, width = video.shape
    expected_flow = (batch, frames - 1, height, width, 2)
    expected_conf = (batch, frames - 1, 1, height, width)
    if flow.shape != expected_flow or confidence.shape != expected_conf:
        raise ValueError(f"flow/confidence shape 不匹配: {flow.shape}, {confidence.shape}")
    flat_flow = flow.reshape(-1, height, width, 2).to(video.dtype)
    # CUDA grid_sample backward 非确定；把下一帧视为固定 warp target，
    # 仍对当前帧预测反传，避免破坏 checkpoint/resume 的逐位一致性。
    next_frames = video[:, 1:].detach().reshape(-1, channels, height, width)
    warped_next = F.grid_sample(
        next_frames,
        _sampling_grid(flat_flow),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).reshape(batch, frames - 1, channels, height, width)
    residual = torch.sqrt((warped_next - video[:, :-1]).square() + epsilon**2)
    weights = confidence.to(residual.dtype).expand_as(residual)
    return (residual * weights).sum() / weights.sum().clamp_min(1.0)
