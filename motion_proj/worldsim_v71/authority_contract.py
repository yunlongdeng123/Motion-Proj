"""Typed authority boundary for WorldSim V7.1 Actor fields.

The physical query deliberately has no appearance argument. Image-trained visual
geometry and attributes live in a sibling type and therefore cannot enter collision
energy through this API. Geometry correctness still comes from GT supervision; this
module only makes the deployment ownership boundary executable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PhysicalActorField:
    """Actor-canonical geometry trained by set, plane and first-return GT losses."""

    actor_id: str
    centers_actor_m: torch.Tensor
    isotropic_scales_m: torch.Tensor


@dataclass(frozen=True)
class ActorPose:
    """Read-only rigid transform from Actor-canonical coordinates to world."""

    actor_id: str
    rotation_world_from_actor: torch.Tensor
    translation_world_from_actor_m: torch.Tensor


@dataclass(frozen=True)
class VisualActorLayer:
    """Render-only geometry and attributes; never accepted by physical queries."""

    actor_id: str
    centers_actor_m: torch.Tensor
    log_scales: torch.Tensor
    quaternions_wxyz: torch.Tensor
    spherical_harmonics: torch.Tensor
    opacity_logits: torch.Tensor


def actor_gaussian_energy(
    queries_actor_m: torch.Tensor,
    field: PhysicalActorField,
) -> torch.Tensor:
    """Evaluate the frozen isotropic physical field in Actor coordinates."""

    scales = field.isotropic_scales_m.reshape(1, -1).clamp_min(1.0e-4)
    distances = torch.cdist(
        queries_actor_m,
        field.centers_actor_m,
        p=2.0,
        compute_mode="donot_use_mm_for_euclid_dist",
    )
    return torch.logsumexp(-0.5 * (distances / scales).square(), dim=1)


def world_to_actor_queries(
    queries_world_m: torch.Tensor,
    pose: ActorPose,
) -> torch.Tensor:
    """Inverse rigid query; canonical geometry is never rewritten per frame."""

    centered = queries_world_m - pose.translation_world_from_actor_m.reshape(1, 3)
    return centered @ pose.rotation_world_from_actor


def world_gaussian_energy(
    queries_world_m: torch.Tensor,
    field: PhysicalActorField,
    pose: ActorPose,
) -> torch.Tensor:
    """World-frame physical query whose dependency set is exactly field + pose."""

    if field.actor_id != pose.actor_id:
        raise ValueError("physical field and pose must belong to the same Actor")
    return actor_gaussian_energy(world_to_actor_queries(queries_world_m, pose), field)


def world_from_actor_points(
    points_actor_m: torch.Tensor,
    pose: ActorPose,
) -> torch.Tensor:
    """Forward rigid composition for rendering or visualization."""

    return (
        points_actor_m @ pose.rotation_world_from_actor.transpose(0, 1)
        + pose.translation_world_from_actor_m.reshape(1, 3)
    )
