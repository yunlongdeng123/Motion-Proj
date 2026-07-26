"""可复现实验、原子 stage 和断点恢复基础设施。

这里保留原有的 package-level API，但按需导入实现。轻量级的运行器（例如
nuScenes cut-in 编排器）只需要 ``runtime.atomic``，不应因为 package 初始化而
加载依赖 PyTorch 的 sampler。
"""
from __future__ import annotations

from typing import TYPE_CHECKING


__all__ = [
    "ExperimentRegistry",
    "JsonlMetrics",
    "RunManifest",
    "ResumableRandomSampler",
    "StageManifest",
]


if TYPE_CHECKING:
    from .experiment import ExperimentRegistry, JsonlMetrics, RunManifest
    from .sampler import ResumableRandomSampler
    from .stage import StageManifest


def __getattr__(name: str):
    if name in {"ExperimentRegistry", "JsonlMetrics", "RunManifest"}:
        from . import experiment

        return getattr(experiment, name)
    if name == "ResumableRandomSampler":
        from .sampler import ResumableRandomSampler

        return ResumableRandomSampler
    if name == "StageManifest":
        from .stage import StageManifest

        return StageManifest
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
