"""可微的目标到参考帧反向传播，以及只使用可见参考特征的融合。"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .masks import apply_hole_mask


def resize_flow(flow: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """按非正方形尺寸分别缩放像素位移流，保持 target→source 方向。

    输入/输出均为 ``[B,2,H,W]``；通道 0 是 x/列位移，通道 1 是 y/行位移。
    与 ``align_corners=False`` 的图像重采样约定一致。
    """
    if flow.ndim != 4 or flow.shape[1] != 2 or not flow.is_floating_point():
        raise ValueError("flow 必须为浮点 [B,2,H,W]")
    out_h, out_w = size
    if out_h <= 0 or out_w <= 0:
        raise ValueError("输出高宽必须为正")
    in_h, in_w = flow.shape[-2:]
    resized = F.interpolate(flow, size=size, mode="bilinear", align_corners=False)
    scale = flow.new_tensor([out_w / in_w, out_h / in_h]).view(1, 2, 1, 1)
    return resized * scale


def backward_warp(source: torch.Tensor, target_to_source_flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """在目标网格上用 ``grid_sample`` 从源帧取值。

    流是目标坐标到源坐标的像素位移：``source_xy = target_xy + flow_xy``。
    输入 ``source=[B,C,H,W]``、``flow=[B,2,H,W]``，返回采样值及
    ``[B,1,H,W]`` 严格界内有效掩码；越界采样填零。边界有效条件是
    ``0 <= source_x <= W-1`` 与 ``0 <= source_y <= H-1``。
    """
    if source.ndim != 4 or not source.is_floating_point():
        raise ValueError("source 必须为浮点 [B,C,H,W]")
    if target_to_source_flow.ndim != 4 or target_to_source_flow.shape != (source.shape[0], 2, *source.shape[-2:]):
        raise ValueError("flow 必须为 [B,2,H,W] 并匹配 source")
    if not target_to_source_flow.is_floating_point() or source.device != target_to_source_flow.device:
        raise ValueError("flow 必须为同设备浮点张量")

    _, _, height, width = source.shape
    dtype, device = source.dtype, source.device
    y, x = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    flow = target_to_source_flow.to(dtype=dtype)
    source_x = x[None] + flow[:, 0]
    source_y = y[None] + flow[:, 1]
    valid = ((source_x >= 0) & (source_x <= width - 1) & (source_y >= 0) & (source_y <= height - 1))[:, None]
    grid = torch.stack((2 * (source_x + 0.5) / width - 1, 2 * (source_y + 0.5) / height - 1), dim=-1)
    warped = F.grid_sample(source, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return warped * valid.to(dtype), valid


def fuse_reference_latents(
    target_latent: torch.Tensor,
    target_hole_mask: torch.Tensor,
    reference_latents: torch.Tensor,
    target_to_reference_flows: torch.Tensor,
    reference_hole_masks: torch.Tensor,
    reliabilities: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """用可靠参考特征填目标洞区，同时逐元素保留目标已知区域。

    参考张量为 ``[B,R,C,H,W]``，流为 ``[B,R,2,H,W]``，洞区与可靠性
    为 ``[B,R,1,H,W]``。可靠性在目标网格，范围 ``[0,1]``。先在源网格
    擦除每个参考洞区，再反向采样；双线性边缘以可见支持量归一化，避免
    已隐藏的参考值进入结果。返回融合特征和 ``[B,1,H,W]`` 支持置信度。
    已知目标的置信度为 1；洞区置信度为各参考有效权重的平均，范围 [0,1]。
    无有效参考的洞区值为 0。这只是传播契约，不包含论文的参考选择或细化。
    """
    if target_latent.ndim != 4 or not target_latent.is_floating_point():
        raise ValueError("target_latent 必须为浮点 [B,C,H,W]")
    batch, channels, height, width = target_latent.shape
    if target_hole_mask.shape != (batch, 1, height, width):
        raise ValueError("target_hole_mask 必须为 [B,1,H,W]")
    if reference_latents.ndim != 5 or reference_latents.shape[0] != batch or reference_latents.shape[2:] != (channels, height, width):
        raise ValueError("reference_latents 必须为 [B,R,C,H,W]")
    refs = reference_latents.shape[1]
    if refs < 1:
        raise ValueError("至少需要一个参考帧")
    if target_to_reference_flows.shape != (batch, refs, 2, height, width):
        raise ValueError("target_to_reference_flows 必须为 [B,R,2,H,W]")
    if reference_hole_masks.shape != (batch, refs, 1, height, width):
        raise ValueError("reference_hole_masks 必须为 [B,R,1,H,W]")
    if reliabilities is not None and reliabilities.shape != (batch, refs, 1, height, width):
        raise ValueError("reliabilities 必须为 [B,R,1,H,W]")
    inputs = (target_hole_mask, reference_latents, target_to_reference_flows, reference_hole_masks)
    if reliabilities is not None:
        inputs += (reliabilities,)
    if any(item.device != target_latent.device for item in inputs):
        raise ValueError("所有输入必须在同一设备")
    if reference_latents.dtype != target_latent.dtype:
        raise ValueError("参考与目标 latent 的 dtype 必须一致")
    if reliabilities is not None and (not reliabilities.is_floating_point() or not torch.isfinite(reliabilities).all() or (reliabilities < 0).any() or (reliabilities > 1).any()):
        raise ValueError("reliabilities 必须是 [0,1] 内有限浮点数")

    weighted_sum = torch.zeros_like(target_latent)
    weight_sum = torch.zeros((batch, 1, height, width), dtype=target_latent.dtype, device=target_latent.device)
    for index in range(refs):
        visible = (reference_hole_masks[:, index] <= 0).to(target_latent.dtype)
        safe_reference = apply_hole_mask(reference_latents[:, index], reference_hole_masks[:, index])
        warped_value, in_bounds = backward_warp(safe_reference, target_to_reference_flows[:, index])
        warped_visible, _ = backward_warp(visible, target_to_reference_flows[:, index])
        warped_visible = warped_visible.clamp(0, 1)
        reliability = 1 if reliabilities is None else reliabilities[:, index]
        weight = warped_visible * in_bounds.to(target_latent.dtype) * reliability
        # 只用可见样本做双线性插值，0 支持位置不会参与融合。
        visible_value = warped_value / warped_visible.clamp_min(1e-6)
        weighted_sum = weighted_sum + visible_value * weight
        weight_sum = weight_sum + weight

    fused_hole = weighted_sum / weight_sum.clamp_min(1e-6)
    hole = target_hole_mask > 0
    output = torch.where(hole, fused_hole, target_latent)
    confidence = torch.where(hole, weight_sum / refs, torch.ones_like(weight_sum))
    return output, confidence
