"""Motion Auditor：以 no-grad 方式编排 感知 + 几何 -> MotionState。

这里的所有内容都是 ``@torch.no_grad``，并与训练计算图分离
（方案第 8.4 节）。auditor 永远不会看到带噪的扩散状态 x_t。
"""
from __future__ import annotations

from typing import Any

import torch

from ..utils.logging import get_logger
from .ego_flow import boxes_to_dynamic_mask, build_static_mask
from .generated_geometry import estimate_generated_geometry
from .providers import (
    DepthProvider,
    EgoMotionProvider,
    FlowProvider,
    LidarCalibratedDepthProvider,
    NuScenesTrackProvider,
    ProvidedEgoMotionProvider,
    RAFTFlowProvider,
    TrackProvider,
)
from .state import MotionState

log = get_logger(__name__)


class MotionAuditor:
    def __init__(
        self,
        device: str = "cuda",
        enable_depth: bool = True,
        conf_thresh: float = 0.5,
        flow_provider: FlowProvider | None = None,
        depth_provider: DepthProvider | None = None,
        track_provider: TrackProvider | None = None,
        ego_motion_provider: EgoMotionProvider | None = None,
        generated_geometry_mode: str | None = None,
        background_fit_options: dict[str, Any] | None = None,
    ):
        self.device = device
        self.conf_thresh = conf_thresh
        self.flow_provider = flow_provider or RAFTFlowProvider(device=device)
        self.depth_provider = depth_provider or LidarCalibratedDepthProvider(device=device, enable=enable_depth)
        self.track_provider = track_provider or NuScenesTrackProvider()
        self.ego_motion_provider = ego_motion_provider or ProvidedEgoMotionProvider()
        self.generated_geometry_mode = generated_geometry_mode
        self.background_fit_options = dict(background_fit_options or {})

    @torch.no_grad()
    def audit(
        self,
        sample: dict,
        *,
        generated_geometry_mode: str | None = None,
        observed_flow: torch.Tensor | None = None,
        flow_confidence: torch.Tensor | None = None,
        depth: torch.Tensor | None = None,
    ) -> MotionState:
        """在单个（未成批的）数据集样本 dict 上运行 auditor。"""
        frames = sample["frames"].to(self.device)          # [K,3,H,W]
        boxes = sample.get("boxes", [[] for _ in range(frames.shape[0])])
        K = sample["intrinsics"].to(self.device)
        cam2ego = sample["cam2ego"].to(self.device)
        k, _, h, w = frames.shape

        # 1) 光流 + 前后向一致性（fb-consistency）置信度
        if (observed_flow is None) != (flow_confidence is None):
            raise ValueError("observed_flow 与 flow_confidence 必须同时提供")
        if observed_flow is None:
            u_static, flow_conf = self.flow_provider.estimate(frames)
        else:
            u_static = observed_flow.to(self.device)
            flow_conf = flow_confidence.to(self.device)

        # 2) 深度 + 由自车运动（ego）诱导的静态光流
        if depth is None:
            depth = self.depth_provider.estimate(frames, sample)  # [K,H,W]
        else:
            depth = depth.to(self.device)
        mode = generated_geometry_mode or self.generated_geometry_mode
        geometry_diagnostics = None
        uses_future_gt_ego = False
        if mode is not None:
            estimate = estimate_generated_geometry(
                mode,
                u_static,
                flow_conf,
                depth,
                sample,
                fit_options=self.background_fit_options,
            )
            u_ego = estimate.flow
            geometry_confidence = estimate.confidence
            ego_valid = geometry_confidence > 0
            geometry_diagnostics = estimate.diagnostics
            uses_future_gt_ego = estimate.uses_future_gt_ego
        else:
            flow_with_validity = getattr(self.ego_motion_provider, "flow_with_validity", None)
            if callable(flow_with_validity):
                u_ego, ego_valid = flow_with_validity(depth, sample)
            else:
                u_ego = self.ego_motion_provider.flow(depth, sample)
                ego_valid = torch.ones(u_ego.shape[:-1], device=u_ego.device, dtype=torch.bool)
            geometry_confidence = ego_valid.to(flow_conf.dtype)

        # 3) 可靠静态掩码（高一致性，且位于目标框之外）
        dyn_mask = boxes_to_dynamic_mask(boxes, h, w, self.device)
        static_mask = build_static_mask(flow_conf, dyn_mask, self.conf_thresh)
        static_mask = static_mask * geometry_confidence.to(static_mask.dtype)

        # 4) 由 GT 框得到的目标轨迹（tracks）
        tracks = self.track_provider.estimate(frames, sample)

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
                "hw": (h, w),
                "sample_id": sample.get("sample_id"),
                "ego_valid_fraction": float(ego_valid.float().mean()),
                "depth_diagnostics": getattr(self.depth_provider, "last_diagnostics", None),
                "generated_geometry_mode": mode,
                "geometry_diagnostics": geometry_diagnostics,
                "uses_future_gt_ego": uses_future_gt_ego,
                "uses_future_gt_track": False if mode is not None else None,
            },
        )
        pose_key = "control_ego2global" if mode == "controlled_ego" else "ego2global"
        if pose_key in sample and (mode is None or mode in {"gt_ego_debug", "controlled_ego"}):
            state.meta["ego2global"] = sample[pose_key].to(self.device)
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
