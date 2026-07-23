"""渲染支持 provenance；禁止被安全几何 evaluator 读取。"""
from __future__ import annotations

from dataclasses import dataclass

from .canonical_hash import canonical_sha256


@dataclass(frozen=True)
class RenderSupportFrame:
    checkpoint_sha256: str
    gaussian_primitive_count: int
    source_observations: tuple[str, ...]
    supporting_camera_times: tuple[str, ...]
    reprojection_residual_px: float | None
    uncertainty: float | None

    def __post_init__(self) -> None:
        if self.gaussian_primitive_count < 0:
            raise ValueError("Gaussian 数量不能为负")

    def content_hash(self) -> str:
        return canonical_sha256(
            {
                "schema": "render-support-v1",
                "checkpoint_sha256": self.checkpoint_sha256,
                "gaussian_primitive_count": self.gaussian_primitive_count,
                "source_observations": sorted(self.source_observations),
                "supporting_camera_times": sorted(self.supporting_camera_times),
                "reprojection_residual_px": self.reprojection_residual_px,
                "uncertainty": self.uncertainty,
            }
        )


def safety_input_from_render_support(_: RenderSupportFrame) -> None:
    raise TypeError("RenderSupportFrame 不得转换为 SafetyGeometry")
