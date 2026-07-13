"""由四个可替换组件构成的动力学投影器。"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..auditor.state import MotionState
from ..utils.logging import get_logger
from . import energies as E
from .components import (
    BackgroundProjector,
    EgoWarpBackground,
    FlowObjectReliability,
    ObjectProjector,
    ReliabilityProvider,
    SmoothedObjectProjector,
    SupportProvider,
    TemporalBorderSupport,
)

log = get_logger(__name__)


@dataclass
class ProjectionResult:
    y: torch.Tensor
    x_dagger: torch.Tensor
    mask: torch.Tensor
    energy_before: dict[str, float]
    energy_after: dict[str, float]
    diagnostics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @property
    def target(self) -> torch.Tensor:
        return self.x_dagger

    @property
    def valid_mask(self) -> torch.Tensor:
        return self.mask


class DynamicsProjector:
    def __init__(
        self,
        smooth_lambda: float = 5.0,
        anchor: int = 0,
        background: BackgroundProjector | None = None,
        objects: ObjectProjector | None = None,
        support: SupportProvider | None = None,
        reliability: ReliabilityProvider | None = None,
    ):
        self.anchor = int(anchor)
        self.background = background or EgoWarpBackground()
        self.objects = objects or SmoothedObjectProjector(smooth_lambda)
        self.support = support or TemporalBorderSupport()
        self.reliability = reliability or FlowObjectReliability()

    @torch.no_grad()
    def project(self, frames: torch.Tensor, state: MotionState) -> ProjectionResult:
        k, _, h, w = frames.shape
        background = self.background.project(frames, state, self.anchor)
        if not hasattr(self.objects, "smooth"):
            raise TypeError("ObjectProjector 必须实现 smooth(tracks)，以便统一能量诊断")
        projected_tracks = self.objects.smooth(state.tracks)  # type: ignore[attr-defined]
        support = self.support.classify(state.tracks, (h, w))
        target, object_mask = self.objects.project(background, frames, state.tracks, projected_tracks, support)
        mask = self.reliability.build(state, object_mask, k).clamp(0, 1)

        before = self._energies(state, state.tracks, support)
        after = self._energies(state, projected_tracks, support)
        # 必须用总能量严格下降；空 track 时 obj/prior 的 0<=0 不能虚报成功。
        if "total" in before and "total" in after:
            energy_decreased = float(after["total"]) < float(before["total"]) - 1e-6
        else:
            energy_decreased = False
        diagnostics = {
            "eligible_fraction": float((mask > 0).float().mean()),
            "energy_decreased": energy_decreased,
            "num_tracks": len(state.tracks),
            "num_unsupported": int(sum(int((~value).sum()) for value in support.values())),
            "components": {
                "background": type(self.background).__name__, "object": type(self.objects).__name__,
                "support": type(self.support).__name__, "reliability": type(self.reliability).__name__,
            },
        }
        track_diagnostics = state.meta.get("track_diagnostics")
        if track_diagnostics is not None:
            diagnostics["generated_tracks"] = track_diagnostics
            diagnostics["object_correction_coverage"] = float(object_mask.gt(0).float().mean())
        legacy_energies = {
            **{f"{key}_before": value for key, value in before.items()},
            **{f"{key}_after": value for key, value in after.items()},
            "static_drift": before.get("static", 0.0),
        }
        metadata = {"sample_id": state.meta.get("sample_id"), "hw": (h, w),
                    "energies": legacy_energies, "diagnostics": diagnostics}
        return ProjectionResult(frames, target, mask, before, after, diagnostics, metadata)

    @staticmethod
    def _energies(state, tracks, support) -> dict[str, float]:
        try:
            return {key: float(value) for key, value in E.e_dyn(state, tracks, support).items()}
        except Exception as exc:  # pragma: no cover - 仅用于保留 cache 主流程
            log.warning("energy report failed: %s", exc)
            return {}
