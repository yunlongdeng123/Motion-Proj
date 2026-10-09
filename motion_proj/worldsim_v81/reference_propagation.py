"""论文 Eq. 5–6 双向参考传播的 owned 适配层。

此模块只负责参考图、目标到来源的 pull flow 和条件 latent 汇集；最终仍调用
固定官方 ``LatentPropagation._post_fuse``，不改变细化/融合的参数与结构。
协议名称必须与旧版单向树 ``reference-m4`` 区分，不能据此恢复旧断点。
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass

import torch
from torch.nn import functional as F


PROTOCOL = "paper-bidirectional-m4"


@dataclass(frozen=True)
class ReferencePlan:
    total_frames: int
    refs: tuple[int, ...]
    pairs: tuple[tuple[int, int], ...]
    protocol: str = PROTOCOL


def build_reference_plan(total_frames: int, refs: list[int] | tuple[int, ...]) -> ReferencePlan:
    """为每个非参考帧保留最近过去/未来两条直接边，再加参考链边。

    pair 统一按较早帧→较晚帧排列。RAFT.forward_pairs 同时给出反向 flow，
    因此每条无向边只估计一次双向光流。pair 顺序也是 FCNet 的 K 轴顺序，
    调用方必须原样记录并沿用；它不是作者已公开的训练排序协议。
    """
    refs = tuple(refs)
    if total_frames < 2 or not refs or refs != tuple(sorted(set(refs))):
        raise ValueError("参考帧必须递增、唯一，视频至少两帧")
    if refs[0] != 0 or refs[-1] != total_frames - 1:
        raise ValueError("参考链必须包含首帧和末帧")
    if any(later - earlier > 4 for earlier, later in zip(refs, refs[1:])):
        raise ValueError("paper-bidirectional-m4 的相邻参考间距不能超过4帧")
    pairs = list(zip(refs, refs[1:]))
    refset = set(refs)
    for target in range(total_frames):
        if target in refset:
            continue
        pos = bisect_left(refs, target)
        past, future = refs[pos - 1], refs[pos]
        pairs.extend(((past, target), (target, future)))
    expected = 2 * (total_frames - len(refs)) + len(refs) - 1
    if len(pairs) != expected or len(set(pairs)) != expected:
        raise AssertionError("参考图边数或唯一性错误")
    return ReferencePlan(total_frames, refs, tuple(pairs))


def static_fcnet_pair_masks(frame_masks: torch.Tensor, plan: ReferencePlan) -> torch.Tensor:
    """仅在各帧 mask 完全相同的 P1 中适配官方 FCNet 的 K+1 mask 接口。

    官方 FCNet 取 masks[:, :-1] / masks[:, 1:] 处理 K 条 flow。此扩展只
    修正长度和索引，不证明 FCNet 已适配非相邻 pair 的时序分布。
    """
    if frame_masks.ndim != 5 or frame_masks.shape[1] != plan.total_frames:
        raise ValueError("frame_masks 必须按 plan 的视频帧数排列")
    if not torch.equal(frame_masks, frame_masks[:, :1].expand_as(frame_masks)):
        raise ValueError("动态 mask 需要逐 pair 的 FCNet 适配，不能静态扩展")
    return frame_masks[:, :1].expand(-1, len(plan.pairs) + 1, -1, -1, -1)


def _rescale_flow(flow: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """与固定官方 rescale_flow 一致：双线性缩放后按宽高比例缩放位移。"""
    old_height, old_width = flow.shape[-2:]
    resized = F.interpolate(flow, size=(height, width), mode="bilinear", align_corners=True)
    scale = flow.new_tensor((width / old_width, height / old_height)).view(1, 2, 1, 1)
    return resized * scale


def _compose_flow(next_edge: torch.Tensor, accumulated: torch.Tensor, warp) -> torch.Tensor:
    """与固定官方 Eq. 6 组合相同：F(i→c)+W(F(c→r), F(i→c))。"""
    return accumulated + warp(next_edge, accumulated)


def _fb_consistency_mask(forward: torch.Tensor, backward: torch.Tensor, warp) -> torch.Tensor:
    """固定官方 fb_consistency_mask 的公式，流方向为 target→source / source→target。"""
    backward_warped = warp(backward, forward)
    difference = forward + backward_warped
    magnitude = (forward.square() + backward_warped.square()).sum(dim=1, keepdim=True)
    threshold = 0.01 * magnitude + 0.5
    return (difference.square().sum(dim=1, keepdim=True) < threshold).float()


def propagate_reference_latents(module, cond_lat: torch.Tensor,
                                flow_fw: torch.Tensor, flow_bw: torch.Tensor,
                                frame_masks: torch.Tensor, plan: ReferencePlan):
    """从真实过去/未来来源拉取 latent，并交给原细化与融合模块。

    输入 flow 的第 k 条必须对应 plan.pairs[k]，forward 为 a→b，backward
    为 b→a。为拉取 source latent，grid_sample 必须使用 target→source flow。
    不能调用官方 forward：它硬编码只重采样 T-1 条 flow。
    """
    if plan.protocol != PROTOCOL or plan != build_reference_plan(plan.total_frames, plan.refs):
        raise ValueError("只接受固定 paper-bidirectional-m4 参考图")
    if cond_lat.ndim != 5:
        raise ValueError("cond_lat 必须为 [B,T,C,H,W]")
    batch, total, _, height, width = cond_lat.shape
    if total != plan.total_frames or frame_masks.ndim != 5 or frame_masks.shape[:3] != (batch, total, 1):
        raise ValueError("latent 与 mask 的帧数/批次不一致")
    if flow_fw.ndim != 5 or flow_fw.shape[:3] != (batch, len(plan.pairs), 2) or flow_bw.shape != flow_fw.shape:
        raise ValueError("双向 flow 的 K 轴必须与 plan.pairs 完全对齐")
    if flow_fw.shape[-2:] != frame_masks.shape[-2:]:
        raise ValueError("flow 与原始 frame mask 的空间尺寸不一致")

    resized_masks = F.interpolate(frame_masks.flatten(0, 1).float(),
                                  size=(height, width), mode="nearest")
    masks = (resized_masks.reshape(batch, total, 1, height, width) != 0).float()
    coverage = 1 - masks
    fw = torch.stack([_rescale_flow(flow_fw[:, k], height, width)
                      for k in range(len(plan.pairs))], dim=1)
    bw = torch.stack([_rescale_flow(flow_bw[:, k], height, width)
                      for k in range(len(plan.pairs))], dim=1)
    edge_index = {edge: k for k, edge in enumerate(plan.pairs)}

    def target_to_source(target: int, source: int) -> torch.Tensor:
        if target < source:
            return fw[:, edge_index[(target, source)]]
        return bw[:, edge_index[(source, target)]]

    def gather(target: int, ordered_sources: list[int]):
        latent = cond_lat[:, target]
        covered = coverage[:, target]
        accumulated = cond_lat.new_zeros((batch, 2, height, width))
        reverse_accumulated = cond_lat.new_zeros((batch, 2, height, width))
        previous = target
        for source in ordered_sources:
            edge = target_to_source(previous, source)
            reverse_edge = target_to_source(source, previous)
            if previous == target:
                accumulated, reverse_accumulated = edge, reverse_edge
            else:
                # Eq. 6：i→source；反向一致性则按 source→i 的逆序组合。
                reverse_accumulated = _compose_flow(
                    reverse_accumulated, reverse_edge, module.backward_warp_cached)
                accumulated = _compose_flow(edge, accumulated, module.backward_warp_cached)
            # 在 source 坐标先屏蔽未知 latent，避免双线性插值混入洞区特征；
            # 拉取后的 latent 已含来源覆盖权重，不再重复乘 warped_coverage，
            # 避免在半像素边界把有效证据衰减两次。与公开未mask来源有意不同。
            known_source = cond_lat[:, source] * coverage[:, source]
            warped_latent = module.backward_warp_cached(known_source, accumulated)
            warped_coverage = module.backward_warp_cached(coverage[:, source], accumulated)
            if module.use_fb_check:
                valid = _fb_consistency_mask(accumulated, reverse_accumulated,
                                             module.backward_warp_cached)
                warped_latent = warped_latent * valid
                warped_coverage = warped_coverage * valid
            latent = latent + (1 - covered) * warped_latent
            covered = covered + (1 - covered) * warped_coverage
            previous = source
        return latent, covered, accumulated

    future_latents, past_latents = [], []
    future_coverages, past_coverages = [], []
    future_flows, past_flows = [], []
    for target in range(total):
        f_lat, f_cov, f_flow = gather(target, [r for r in plan.refs if r > target])
        p_lat, p_cov, p_flow = gather(target, [r for r in reversed(plan.refs) if r < target])
        future_latents.append(f_lat)
        past_latents.append(p_lat)
        future_coverages.append(f_cov)
        past_coverages.append(p_cov)
        future_flows.append(f_flow)
        past_flows.append(p_flow)
    return module._post_fuse(
        cond_lat, masks, torch.stack(future_latents, dim=1),
        torch.stack(past_latents, dim=1), torch.stack(future_coverages, dim=1),
        torch.stack(past_coverages, dim=1), torch.stack(future_flows, dim=1),
        torch.stack(past_flows, dim=1))
