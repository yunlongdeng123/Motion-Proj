"""危险保真的确定性物理表面编译器。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class SurfaceAction(str, Enum):
    """V7 物理表面的四种允许动作。"""

    KEEP = "KEEP"
    PROJECT = "PROJECT"
    COMPLETE = "COMPLETE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ActorState:
    """编译器不得修改的 Actor 身份、轨迹与尺寸。"""

    actor_id: str
    trajectory_xyz_m: tuple[tuple[float, float, float], ...]
    size_lwh_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not self.actor_id:
            raise ValueError("actor_id must be non-empty")
        if not self.trajectory_xyz_m:
            raise ValueError("trajectory_xyz_m must contain at least one state")
        if len(self.size_lwh_m) != 3 or min(self.size_lwh_m) <= 0.0:
            raise ValueError("size_lwh_m must contain three positive values")


@dataclass(frozen=True)
class PhysicalEvidence:
    """单个 Actor-local primitive 的传感器与表面证据。"""

    actor: ActorState
    primitive_id: str
    sensor_hit_count: int
    temporal_support_count: int
    view_direction_count: int
    provenance_supported: bool
    free_space_violation_m: float
    surface_distance_m: float
    hole_radius_m: float = 0.0
    normal_alignment: float = 1.0
    evidence_known: bool = True

    def __post_init__(self) -> None:
        counts = (
            self.sensor_hit_count,
            self.temporal_support_count,
            self.view_direction_count,
        )
        if min(counts) < 0:
            raise ValueError("evidence counts must be non-negative")
        distances = (
            self.free_space_violation_m,
            self.surface_distance_m,
            self.hole_radius_m,
        )
        if min(distances) < 0.0:
            raise ValueError("evidence distances must be non-negative")
        if not -1.0 <= self.normal_alignment <= 1.0:
            raise ValueError("normal_alignment must be in [-1, 1]")


@dataclass(frozen=True)
class CompilerThresholds:
    minimum_sensor_hits: int = 1
    minimum_temporal_support: int = 2
    minimum_completion_views: int = 2
    maximum_project_distance_m: float = 0.25
    minimum_free_space_violation_m: float = 0.02
    minimum_hole_radius_m: float = 0.15
    minimum_normal_alignment: float = 0.25


@dataclass(frozen=True)
class CompiledSurfaceDecision:
    actor: ActorState
    primitive_id: str
    action: SurfaceAction
    reason_codes: tuple[str, ...]
    collision_query_enabled: bool
    projected_distance_m: float = 0.0


class HazardPreservingPhysicalCompiler:
    """只修复局部物理表面，永不删除或改写 Actor 状态。"""

    def __init__(self, thresholds: CompilerThresholds | None = None) -> None:
        self.thresholds = thresholds or CompilerThresholds()

    def compile(self, evidence: PhysicalEvidence) -> CompiledSurfaceDecision:
        thresholds = self.thresholds
        supported = (
            evidence.provenance_supported
            and evidence.sensor_hit_count >= thresholds.minimum_sensor_hits
            and evidence.temporal_support_count >= thresholds.minimum_temporal_support
        )

        if not evidence.evidence_known:
            return self._unknown(evidence, "evidence_not_observed")
        if not evidence.provenance_supported and evidence.sensor_hit_count == 0:
            return self._unknown(evidence, "sensor_and_provenance_missing")

        if evidence.free_space_violation_m >= thresholds.minimum_free_space_violation_m:
            if (
                supported
                and evidence.surface_distance_m <= thresholds.maximum_project_distance_m
                and evidence.normal_alignment >= thresholds.minimum_normal_alignment
            ):
                return CompiledSurfaceDecision(
                    actor=evidence.actor,
                    primitive_id=evidence.primitive_id,
                    action=SurfaceAction.PROJECT,
                    reason_codes=("observed_free_contradiction", "near_supported_surface"),
                    collision_query_enabled=True,
                    projected_distance_m=float(evidence.surface_distance_m),
                )
            return self._unknown(evidence, "unrepairable_free_space_contradiction")

        if evidence.hole_radius_m >= thresholds.minimum_hole_radius_m:
            if (
                supported
                and evidence.view_direction_count >= thresholds.minimum_completion_views
                and evidence.normal_alignment >= thresholds.minimum_normal_alignment
            ):
                return CompiledSurfaceDecision(
                    actor=evidence.actor,
                    primitive_id=evidence.primitive_id,
                    action=SurfaceAction.COMPLETE,
                    reason_codes=("locally_supported_surface_hole",),
                    collision_query_enabled=True,
                )
            return self._unknown(evidence, "insufficient_multiview_completion_support")

        if supported:
            return CompiledSurfaceDecision(
                actor=evidence.actor,
                primitive_id=evidence.primitive_id,
                action=SurfaceAction.KEEP,
                reason_codes=("stable_sensor_supported_surface",),
                collision_query_enabled=True,
            )
        return self._unknown(evidence, "insufficient_physical_surface_support")

    def compile_many(
        self, evidence_rows: Iterable[PhysicalEvidence]
    ) -> list[CompiledSurfaceDecision]:
        return [self.compile(evidence) for evidence in evidence_rows]

    @staticmethod
    def _unknown(
        evidence: PhysicalEvidence, reason: str
    ) -> CompiledSurfaceDecision:
        return CompiledSurfaceDecision(
            actor=evidence.actor,
            primitive_id=evidence.primitive_id,
            action=SurfaceAction.UNKNOWN,
            reason_codes=(reason,),
            collision_query_enabled=False,
        )
