"""Motion Auditor：以 no-grad 方式编排 感知 + 几何 -> MotionState。

这里的所有内容都是 ``@torch.no_grad``，并与训练计算图分离
（方案第 8.4 节）。auditor 永远不会看到带噪的扩散状态 x_t。
"""
from __future__ import annotations

from typing import Any

import torch

from ..utils.logging import get_logger
from .boxes_nuscenes import build_tracks
from .depth_anything import DepthEstimator
from .ego_flow import boxes_to_dynamic_mask, build_static_mask, compute_ego_flow
from .flow_raft import RAFTFlow
from .state import MotionState

log = get_logger(__name__)


class MotionAuditor:
    def __init__(self, device: str = "cuda", enable_depth: bool = True, conf_thresh: float = 0.5):
        self.device = device
        self.conf_thresh = conf_thresh
        self.flow = RAFTFlow(device=device)
        self.depth = DepthEstimator(device=device, enable=enable_depth)

    @torch.no_grad()
    def audit(self, sample: dict) -> MotionState:
        """在单个（未成批的）数据集样本 dict 上运行 auditor。"""
        frames = sample["frames"].to(self.device)          # [K,3,H,W]
        boxes = sample["boxes"]                              # list[K] of list[dict]
        K = sample["intrinsics"].to(self.device)
        cam2ego = sample["cam2ego"].to(self.device)
        ego2global = sample["ego2global"].to(self.device)
        k, _, h, w = frames.shape

        # 1) 光流 + 前后向一致性（fb-consistency）置信度
        u_static, flow_conf = self.flow.flow_with_confidence(frames)

        # 2) 深度 + 由自车运动（ego）诱导的静态光流
        depth = self.depth.depth(frames)                    # [K,H,W]
        u_ego = compute_ego_flow(depth, K, cam2ego, ego2global)

        # 3) 可靠静态掩码（高一致性，且位于目标框之外）
        dyn_mask = boxes_to_dynamic_mask(boxes, h, w, self.device)
        static_mask = build_static_mask(flow_conf, dyn_mask, self.conf_thresh)

        # 4) 由 GT 框得到的目标轨迹（tracks）
        tracks = build_tracks(boxes, k)

        state = MotionState(
            u_static=u_static,
            u_ego=u_ego.to(u_static.dtype),
            static_mask=static_mask,
            flow_conf=flow_conf,
            depth=depth,
            tracks=tracks,
            meta={
                "intrinsics": K,
                "cam2ego": cam2ego,
                "ego2global": ego2global,
                "hw": (h, w),
                "sample_id": sample.get("sample_id"),
            },
        )
        return state

    @torch.no_grad()
    def static_drift_score(self, state: MotionState) -> float:
        """在可靠静态像素上，经自车运动补偿后的静态漂移幅值的均值。

        值越大 = 越差（漂移越多）。用于诊断排序。
        """
        res = state.static_residual                          # [F,H,W,2]
        m = state.static_mask                                # [F,H,W]
        mag = res.norm(dim=-1) * m
        denom = m.sum().clamp_min(1.0)
        return float(mag.sum() / denom)
