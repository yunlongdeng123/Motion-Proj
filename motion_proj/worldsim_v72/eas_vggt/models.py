"""E2 的容量可比物理表面与回波证据 heads。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PhysicalEvidenceOutput:
    surface_delta_actor_m: torch.Tensor
    blocking_logit: torch.Tensor
    detection_logit: torch.Tensor
    evidence_fou: torch.Tensor
    evidence_strength: torch.Tensor
    added_evidence_fou: torch.Tensor
    added_evidence_strength: torch.Tensor


class EvidenceConditionedSurfaceAdapter(nn.Module):
    """Keep actor-local geometry independent from appearance, then fuse evidence."""

    def __init__(
        self,
        *,
        base_feature_dim: int,
        visual_feature_dim: int,
        geometric_feature_dim: int = 5,
        hidden_dim: int = 256,
        visual_dim: int = 128,
        maximum_surface_delta_m: float = 0.30,
    ) -> None:
        super().__init__()
        self.maximum_surface_delta_m = float(maximum_surface_delta_m)
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(visual_feature_dim),
            nn.Linear(visual_feature_dim, visual_dim),
            nn.GELU(),
            nn.Linear(visual_dim, visual_dim),
        )
        # Physical ownership excludes visual features and global actor pose.
        physical_dim = base_feature_dim + 3 + 3 + 1
        self.physical_projection = nn.Sequential(
            nn.LayerNorm(physical_dim),
            nn.Linear(physical_dim, hidden_dim),
            nn.GELU(),
        )
        self.geometry_projection = nn.Sequential(
            nn.LayerNorm(geometric_feature_dim),
            nn.Linear(geometric_feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.physical_fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.GELU(),
        )
        self.evidence_trunk = nn.Sequential(
            nn.Linear(hidden_dim + visual_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.surface_head = nn.Linear(hidden_dim, 3)
        self.evidence_head = nn.Linear(hidden_dim, 4)
        self.return_head = nn.Linear(hidden_dim, 2)
        with torch.no_grad():
            self.evidence_head.bias.zero_()
            self.evidence_head.bias[0] = -4.0

    def forward(
        self,
        *,
        base_features: torch.Tensor,
        canonical_xyz: torch.Tensor,
        evidence_fou: torch.Tensor,
        opportunity_count: torch.Tensor,
        visual_features: torch.Tensor,
        visual_observed: torch.Tensor,
        geometric_features: torch.Tensor,
        geometry_observed: torch.Tensor,
    ) -> PhysicalEvidenceOutput:
        if evidence_fou.shape[-1] != 3 or canonical_xyz.shape[-1] != 3:
            raise ValueError("canonical_xyz/evidence_fou 最后一维必须为 3")
        opportunity = torch.log1p(opportunity_count.float()).unsqueeze(-1)
        observed = visual_observed.float().unsqueeze(-1)
        geometry_present = geometry_observed.float().unsqueeze(-1)
        physical = torch.cat(
            [base_features, canonical_xyz, evidence_fou, opportunity], dim=-1
        )
        visual = self.visual_projection(visual_features) * observed
        geometry_hidden = self.geometry_projection(geometric_features) * geometry_present
        physical_hidden = self.physical_fusion(
            torch.cat([self.physical_projection(physical), geometry_hidden, geometry_present], dim=-1)
        )
        evidence_hidden = self.evidence_trunk(torch.cat([physical_hidden, visual, observed], dim=-1))
        delta = torch.tanh(self.surface_head(physical_hidden)) * self.maximum_surface_delta_m
        evidence_update = self.evidence_head(evidence_hidden)
        added_strength = torch.nn.functional.softplus(evidence_update[..., :1])
        added_fou = torch.softmax(evidence_update[..., 1:], dim=-1)
        prior_strength = 1.0 + torch.log1p(opportunity_count.float()).unsqueeze(-1)
        concentration = prior_strength * evidence_fou + added_strength * added_fou
        evidence_strength = concentration.sum(dim=-1, keepdim=True)
        fou = concentration / evidence_strength.clamp_min(1.0e-8)
        return_logits = self.return_head(evidence_hidden)
        return PhysicalEvidenceOutput(
            surface_delta_actor_m=delta,
            blocking_logit=return_logits[..., 0],
            detection_logit=return_logits[..., 1],
            evidence_fou=fou,
            evidence_strength=evidence_strength.squeeze(-1),
            added_evidence_fou=added_fou,
            added_evidence_strength=added_strength.squeeze(-1),
        )


class MatchedScalarSurfaceAdapter(EvidenceConditionedSurfaceAdapter):
    """同一输入与 trunk 的 scalar response 对照。"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        hidden_dim = self.return_head.in_features
        self.scalar_head = nn.Linear(hidden_dim, 1)
        del self.evidence_head
        del self.return_head

    def forward(self, **inputs) -> tuple[torch.Tensor, torch.Tensor]:
        opportunity = torch.log1p(inputs["opportunity_count"].float()).unsqueeze(-1)
        observed = inputs["visual_observed"].float().unsqueeze(-1)
        geometry_present = inputs["geometry_observed"].float().unsqueeze(-1)
        physical = torch.cat(
            [
                inputs["base_features"],
                inputs["canonical_xyz"],
                inputs["evidence_fou"],
                opportunity,
            ],
            dim=-1,
        )
        visual = self.visual_projection(inputs["visual_features"]) * observed
        geometry_hidden = self.geometry_projection(inputs["geometric_features"]) * geometry_present
        physical_hidden = self.physical_fusion(
            torch.cat([self.physical_projection(physical), geometry_hidden, geometry_present], dim=-1)
        )
        evidence_hidden = self.evidence_trunk(torch.cat([physical_hidden, visual, observed], dim=-1))
        delta = torch.tanh(self.surface_head(physical_hidden)) * self.maximum_surface_delta_m
        return delta, self.scalar_head(evidence_hidden).squeeze(-1)


def lift_actor_surface_to_world(
    candidates_actor_m: torch.Tensor,
    surface_delta_actor_m: torch.Tensor,
    world_from_actor: torch.Tensor,
) -> torch.Tensor:
    """Compose actor-local surfaces with an SE(3) pose; this map is equivariant by construction."""

    if candidates_actor_m.shape != surface_delta_actor_m.shape or candidates_actor_m.shape[-1] != 3:
        raise ValueError("candidate/delta shapes must match and end in xyz")
    if world_from_actor.shape[-2:] != (4, 4):
        raise ValueError("world_from_actor must end in (4,4)")
    local = candidates_actor_m + surface_delta_actor_m
    rotation = world_from_actor[..., :3, :3]
    translation = world_from_actor[..., :3, 3]
    return torch.einsum("...ij,...nj->...ni", rotation, local) + translation.unsqueeze(-2)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
