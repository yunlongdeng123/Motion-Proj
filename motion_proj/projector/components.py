"""DynamicsProjector 的 background/object/support/reliability 组件。"""
from __future__ import annotations

from typing import Protocol

import torch

from ..auditor.state import MotionState, Track
from ..auditor.generated_geometry import render_pairwise_background_correction
from .mask import build_reliability_mask
from .smoothing import smooth_tracks
from .support import classify_support
from .warper import composite_objects, render_static


class BackgroundProjector(Protocol):
    def project(self, frames: torch.Tensor, state: MotionState, anchor: int) -> torch.Tensor: ...


class ObjectProjector(Protocol):
    def project(self, background: torch.Tensor, frames: torch.Tensor, original: list[Track],
                projected: list[Track], support: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]: ...


class SupportProvider(Protocol):
    def classify(self, tracks: list[Track], hw: tuple[int, int]) -> dict[str, torch.Tensor]: ...


class ReliabilityProvider(Protocol):
    def build(self, state: MotionState, object_mask: torch.Tensor, num_frames: int) -> torch.Tensor: ...


class EgoWarpBackground:
    def project(self, frames: torch.Tensor, state: MotionState, anchor: int) -> torch.Tensor:
        meta = state.meta
        # generated rollout 没有可用的 future ego pose。此处只使用审计器已经从生成帧
        # 估计出的背景 flow，随后由局部 point-track overlay 恢复候选动态区域。
        if meta.get("generated_geometry_mode") is not None and "ego2global" not in meta:
            target, _ = render_pairwise_background_correction(
                frames, state.u_ego.to(frames.device),
                meta.get("geometry_confidence", state.flow_conf).to(frames.device),
            )
            return target
        return render_static(
            frames, state.depth.to(frames.device), meta["intrinsics"].to(frames.device),
            meta["cam2ego"].to(frames.device), meta["ego2global"].to(frames.device), anchor=anchor,
        )


class SmoothedObjectProjector:
    def __init__(self, smooth_lambda: float = 5.0):
        self.smooth_lambda = float(smooth_lambda)

    def smooth(self, tracks: list[Track]) -> list[Track]:
        return smooth_tracks(tracks, lam=self.smooth_lambda)

    def project(self, background, frames, original, projected, support):
        return composite_objects(background, frames, original, projected, support)


class TemporalBorderSupport:
    def classify(self, tracks: list[Track], hw: tuple[int, int]) -> dict[str, torch.Tensor]:
        return classify_support(tracks, hw)


class FlowObjectReliability:
    def build(self, state: MotionState, object_mask: torch.Tensor, num_frames: int) -> torch.Tensor:
        return build_reliability_mask(state.static_mask.to(object_mask.device), object_mask, num_frames)
