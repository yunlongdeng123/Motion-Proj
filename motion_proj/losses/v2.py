"""P2-V2 的 teacher-relative residual-v 损失。

这些函数只表达 V2 的单步目标，不负责 cache 选择、优化器或训练调度；因而既可被
trainer 复用，也可由 gradient audit 在固定 batch/noise 下逐项求导。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_channel_mask(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """规范化为 ``[B,T,1,H,W]`` 的浮点 mask。"""
    if mask.dim() == like.dim() - 1:
        mask = mask.unsqueeze(2)
    if mask.dim() != like.dim() or mask.shape[2] != 1:
        raise ValueError("mask 必须为 [B,T,1,H,W] 或 [B,T,H,W]")
    return mask.to(device=like.device, dtype=like.dtype).clamp(0.0, 1.0)


def _expand_batch(value: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    while value.dim() < like.dim():
        value = value.unsqueeze(-1)
    return value


def _weighted_huber(error: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor | float) -> torch.Tensor:
    """按 mask/置信度归一化的 Huber，并对空 component 返回可求导零。"""
    m = _as_channel_mask(mask, error)
    w = torch.as_tensor(weight, device=error.device, dtype=error.dtype)
    w = _expand_batch(w, m)
    weighted = (m * w).expand_as(error)
    value = F.huber_loss(error, torch.zeros_like(error), reduction="none") * weighted
    return value.sum() / weighted.sum().clamp_min(1.0)


def _masked_rms(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = _as_channel_mask(mask, value).expand_as(value)
    numerator = (value.square() * m).flatten(1).sum(1)
    denominator = m.flatten(1).sum(1).clamp_min(1.0)
    return (numerator / denominator).sqrt()


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius < 0:
        raise ValueError("dilation radius 不得小于 0")
    if radius == 0:
        return mask
    b, t, _, h, w = mask.shape
    return F.max_pool2d(mask.reshape(b * t, 1, h, w), kernel_size=2 * radius + 1,
                        stride=1, padding=radius).reshape(b, t, 1, h, w)


def teacher_relative_v_target(
    backbone,
    z: torch.Tensor,
    sigma: torch.Tensor,
    cond,
    base_latent: torch.Tensor,
    projected_latent: torch.Tensor,
    static_mask: torch.Tensor,
    object_mask: torch.Tensor,
    *,
    eta: float,
    trust_region_B: float,
    eps: float = 1.0e-8,
) -> dict[str, torch.Tensor]:
    """构造带连续 trust-region 的 stop-gradient residual-v target。"""
    if eta <= 0 or trust_region_B <= 0:
        raise ValueError("eta 和 trust_region_B 必须为正数")
    if base_latent.shape != projected_latent.shape or base_latent.shape != z.shape:
        raise ValueError("base/projected/z latent 形状必须一致")
    static = _as_channel_mask(static_mask, z)
    obj = _as_channel_mask(object_mask, z)
    union = (static + obj).clamp_max(1.0)
    delta = (projected_latent - base_latent).detach()
    correction_rms = _masked_rms(delta, union)
    sigma_e = _expand_batch(sigma.to(device=z.device, dtype=z.dtype), z)
    eta_cap = trust_region_B * (sigma.to(device=z.device, dtype=z.dtype) + eps) / (correction_rms + eps)
    eta_eff = torch.minimum(torch.full_like(sigma, float(eta)), eta_cap)
    with torch.no_grad():
        teacher = backbone.anchor_predict_model_output(z, sigma, cond).detach()
        target = teacher - _expand_batch(eta_eff, z) * ((1.0 + sigma_e.square()).sqrt() / sigma_e) * union * delta
    return {
        "teacher": teacher,
        "target": target,
        "union_mask": union,
        "static_mask": static,
        "object_mask": obj,
        "eta_eff": eta_eff,
        "correction_rms": correction_rms.detach(),
        "trust_region_clipping_fraction": (eta_eff < float(eta)).float().mean(),
    }


def correction_v_loss(
    student_v: torch.Tensor,
    target_v: torch.Tensor,
    static_mask: torch.Tensor,
    object_mask: torch.Tensor,
    *,
    static_weight: torch.Tensor | float = 1.0,
    object_weight: torch.Tensor | float = 1.0,
) -> dict[str, torch.Tensor]:
    """分别返回 static/object correction，禁止只留下不可定位的 union loss。"""
    error = student_v - target_v.detach()
    static = _weighted_huber(error, static_mask, static_weight)
    obj = _weighted_huber(error, object_mask, object_weight)
    return {"loss": static + obj, "loss_static": static, "loss_object": obj}


def outside_mask_preserve_v_loss(
    student_v: torch.Tensor,
    teacher_v: torch.Tensor,
    union_mask: torch.Tensor,
    *,
    weight: torch.Tensor | float = 1.0,
    dilation_radius: int = 1,
) -> dict[str, torch.Tensor]:
    """仅在 correction mask 外对齐冻结 Base；边缘带不施加 Base imitation。"""
    union = _as_channel_mask(union_mask, student_v)
    dilated = _dilate(union, dilation_radius).clamp(0.0, 1.0)
    outside = 1.0 - dilated
    loss = _weighted_huber(student_v - teacher_v.detach(), outside, weight)
    return {"loss": loss, "outside_mask": outside, "boundary_mask": (dilated - union).clamp_min(0.0)}
