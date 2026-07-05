"""由自车运动（ego-motion）诱导的静态光流，以及可靠静态像素掩码。"""
from __future__ import annotations

import torch

from ..utils.geometry import ego_induced_flow


def compute_ego_flow(
    depth: torch.Tensor,      # [K,H,W]
    intrinsics: torch.Tensor, # [3,3]
    cam2ego: torch.Tensor,    # [4,4]
    ego2global: torch.Tensor, # [K,4,4]
) -> torch.Tensor:
    """逐帧对（pair）由自车运动诱导的静态光流 ``[F,H,W,2]``（F = K-1）。"""
    k = depth.shape[0]
    flows = []
    for t in range(k - 1):
        flows.append(
            ego_induced_flow(depth[t], intrinsics, cam2ego, ego2global[t], ego2global[t + 1])
        )
    return torch.stack(flows, 0)


def boxes_to_dynamic_mask(boxes_per_frame: list, h: int, w: int, device) -> torch.Tensor:
    """逐帧对 GT 框区域取并集 -> 动态掩码 ``[K,H,W]``，取值 {0,1}。"""
    k = len(boxes_per_frame)
    mask = torch.zeros(k, h, w, device=device)
    for t, blist in enumerate(boxes_per_frame):
        for b in blist:
            u0, v0, u1, v1 = [int(round(float(x))) for x in b["xyxy"]]
            mask[t, max(0, v0):min(h, v1 + 1), max(0, u0):min(w, u1 + 1)] = 1.0
    return mask


def build_static_mask(
    flow_conf: torch.Tensor,        # [F,H,W]
    dynamic_mask: torch.Tensor,     # [K,H,W]
    conf_thresh: float = 0.5,
) -> torch.Tensor:
    """可靠静态像素：具有高前后向一致性（fb-consistency）且位于目标框之外。

    对帧对 (t, t+1)，使用 *源* 帧 t 处的动态掩码。
    """
    dyn_src = dynamic_mask[:-1]                       # [F,H,W]
    static = (flow_conf >= conf_thresh).float() * (1.0 - dyn_src)
    return static
