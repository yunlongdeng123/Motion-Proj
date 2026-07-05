"""Dynamics Projector（动力学投影器）：P(y) = Gamma(a_y, s_dagger)。

给定干净层级（clean-level）的帧 ``y`` 以及 auditor 的 ``MotionState``，产出
经动力学修复的目标 ``x_dagger`` 和可靠性掩码 ``M_y``。整个投影器都是
no-grad 的，并与训练分离（方案第 6/8.4 节）。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..auditor.state import MotionState
from ..utils.logging import get_logger
from . import energies as E
from .mask import build_reliability_mask
from .smoothing import smooth_tracks
from .support import classify_support
from .warper import composite_objects, render_static

log = get_logger(__name__)


@dataclass
class ProjectionResult:
    y: torch.Tensor            # [K,3,H,W] 输入帧
    x_dagger: torch.Tensor     # [K,3,H,W] 修复后的目标
    mask: torch.Tensor         # [K,1,H,W] 可靠性，取值 [0,1]
    metadata: dict


class DynamicsProjector:
    def __init__(self, smooth_lambda: float = 5.0, anchor: int = 0):
        self.smooth_lambda = smooth_lambda
        self.anchor = anchor

    @torch.no_grad()
    def project(self, frames: torch.Tensor, state: MotionState) -> ProjectionResult:
        device = frames.device
        k, _, h, w = frames.shape
        meta = state.meta

        # 1) 静态层：由自车光流得到的无漂移背景
        static_bg = render_static(
            frames,
            state.depth.to(device),
            meta["intrinsics"].to(device),
            meta["cam2ego"].to(device),
            meta["ego2global"].to(device),
            anchor=self.anchor,
        )

        # 2) 动态层：平滑轨迹、按支持性过滤、合成
        smoothed = smooth_tracks(state.tracks, lam=self.smooth_lambda)
        support = classify_support(state.tracks, (h, w))
        x_dagger, obj_mask = composite_objects(
            static_bg, frames, state.tracks, smoothed, support
        )

        # 3) 可靠性掩码
        mask = build_reliability_mask(state.static_mask.to(device), obj_mask, k)

        # 4) 能量报告（诊断）：比较投影前后
        try:
            terms = E.e_dyn(state, smoothed, support)
            energy_report = {kk: float(vv) for kk, vv in terms.items()}
            energy_report["obj_before"] = float(E.e_obj(state.tracks))
            energy_report["obj_after"] = float(E.e_obj(smoothed))
            energy_report["prior_before"] = float(E.e_prior(state.tracks))
            energy_report["prior_after"] = float(E.e_prior(smoothed))
            energy_report["static_drift"] = float(E.e_static(state))
        except Exception as e:  # pragma: no cover
            log.warning("energy report failed: %s", e)
            energy_report = {}

        metadata = {
            "sample_id": meta.get("sample_id"),
            "num_tracks": len(state.tracks),
            "num_unsupported": int(
                sum(int((~v).sum()) for v in support.values())
            ),
            "energies": energy_report,
            "hw": (h, w),
        }
        return ProjectionResult(y=frames, x_dagger=x_dagger, mask=mask, metadata=metadata)
