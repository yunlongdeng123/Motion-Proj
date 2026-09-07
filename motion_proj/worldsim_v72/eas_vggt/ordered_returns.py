"""可组合的场景级表面事件与第一回波/无回波分布。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OrderedReturnDistribution:
    sorted_depth_m: torch.Tensor
    sorted_original_index: torch.Tensor
    event_probability: torch.Tensor
    no_return_probability: torch.Tensor
    clear_probability: torch.Tensor
    undetected_block_probability: torch.Tensor


def ordered_return_distribution(
    depth_m: torch.Tensor,
    blocking_logits: torch.Tensor,
    detection_logits: torch.Tensor,
    event_valid_mask: torch.Tensor,
) -> OrderedReturnDistribution:
    """按几何深度排序后组合 blocking 与 detection，两者不共享语义。"""
    if depth_m.ndim != 2:
        raise ValueError("depth_m 必须为 (R,K)")
    if blocking_logits.shape != depth_m.shape or detection_logits.shape != depth_m.shape:
        raise ValueError("blocking/detection logits 必须与 depth_m 同形")
    if event_valid_mask.shape != depth_m.shape:
        raise ValueError("event_valid_mask 必须与 depth_m 同形")
    valid = event_valid_mask.to(dtype=torch.bool)
    if torch.any(valid & (~torch.isfinite(depth_m) | (depth_m < 0.0))):
        raise ValueError("有效事件必须具有有限非负深度")
    sortable_depth = torch.where(valid, depth_m, torch.full_like(depth_m, torch.inf))
    sorted_depth, order = torch.sort(sortable_depth, dim=1, stable=True)
    sorted_valid = torch.gather(valid, 1, order)
    blocking = torch.sigmoid(torch.gather(blocking_logits, 1, order)) * sorted_valid
    detection = torch.sigmoid(torch.gather(detection_logits, 1, order)) * sorted_valid
    survival = 1.0 - blocking
    prefix = torch.cumprod(
        torch.cat([torch.ones_like(survival[:, :1]), survival[:, :-1]], dim=1),
        dim=1,
    )
    event_probability = prefix * blocking * detection
    undetected = torch.sum(prefix * blocking * (1.0 - detection), dim=1)
    clear = torch.prod(survival, dim=1)
    no_return = clear + undetected
    return OrderedReturnDistribution(
        sorted_depth_m=sorted_depth,
        sorted_original_index=order,
        event_probability=event_probability,
        no_return_probability=no_return,
        clear_probability=clear,
        undetected_block_probability=undetected,
    )


def ordered_return_nll(
    distribution: OrderedReturnDistribution,
    target_sorted_index: torch.Tensor,
    *,
    ignore_index: int = -2,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """target=-1 表示真实 no-return，非负值表示排序后的表面事件。"""
    targets = target_sorted_index.to(dtype=torch.long).reshape(-1)
    if len(targets) != len(distribution.event_probability):
        raise ValueError("target 数量与射线数量不一致")
    probabilities = torch.cat(
        [distribution.event_probability, distribution.no_return_probability[:, None]], dim=1
    )
    no_return_column = probabilities.shape[1] - 1
    selected = torch.where(targets == -1, no_return_column, targets)
    usable = targets != ignore_index
    if torch.any(usable & ((selected < 0) | (selected >= probabilities.shape[1]))):
        raise ValueError("target_sorted_index 越界")
    if not torch.any(usable):
        return probabilities.sum() * 0.0
    chosen = probabilities[torch.arange(len(probabilities), device=probabilities.device), selected.clamp_min(0)]
    return -torch.log(chosen[usable].clamp_min(epsilon)).mean()


def split_blocking_probability(probability: torch.Tensor, parts: int) -> torch.Tensor:
    """在相同表面位置分裂 primitive 时保持总 blocking mass。"""
    if parts <= 0:
        raise ValueError("parts 必须为正")
    probability = probability.clamp(0.0, 1.0 - torch.finfo(probability.dtype).eps)
    return -torch.expm1(torch.log1p(-probability) / float(parts))
