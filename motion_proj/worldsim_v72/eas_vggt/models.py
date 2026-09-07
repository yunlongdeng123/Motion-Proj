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


class EvidenceConditionedSurfaceAdapter(nn.Module):
    """共享 trunk 后分离物理位移、连续证据和分类回波参数。"""

    def __init__(
        self,
        *,
        base_feature_dim: int,
        visual_feature_dim: int,
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
        # base + canonical xyz + F/O/U + log opportunity + visual presence
        physical_dim = base_feature_dim + 3 + 3 + 1 + 1
        self.physical_projection = nn.Sequential(
            nn.LayerNorm(physical_dim),
            nn.Linear(physical_dim, hidden_dim),
            nn.GELU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim + visual_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.surface_head = nn.Linear(hidden_dim, 3)
        self.evidence_head = nn.Linear(hidden_dim, 4)
        self.return_head = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        *,
        base_features: torch.Tensor,
        canonical_xyz: torch.Tensor,
        evidence_fou: torch.Tensor,
        opportunity_count: torch.Tensor,
        visual_features: torch.Tensor,
        visual_observed: torch.Tensor,
    ) -> PhysicalEvidenceOutput:
        if evidence_fou.shape[-1] != 3 or canonical_xyz.shape[-1] != 3:
            raise ValueError("canonical_xyz/evidence_fou 最后一维必须为 3")
        opportunity = torch.log1p(opportunity_count.float()).unsqueeze(-1)
        observed = visual_observed.float().unsqueeze(-1)
        physical = torch.cat(
            [base_features, canonical_xyz, evidence_fou, opportunity, observed], dim=-1
        )
        visual = self.visual_projection(visual_features) * observed
        hidden = self.trunk(torch.cat([self.physical_projection(physical), visual], dim=-1))
        delta = torch.tanh(self.surface_head(hidden)) * self.maximum_surface_delta_m
        raw_evidence = torch.nn.functional.softplus(self.evidence_head(hidden))
        evidence_strength = raw_evidence.sum(dim=-1, keepdim=True)
        fou = raw_evidence[..., 1:] / raw_evidence[..., 1:].sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        return_logits = self.return_head(hidden)
        return PhysicalEvidenceOutput(
            surface_delta_actor_m=delta,
            blocking_logit=return_logits[..., 0],
            detection_logit=return_logits[..., 1],
            evidence_fou=fou,
            evidence_strength=evidence_strength.squeeze(-1),
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
        physical = torch.cat(
            [
                inputs["base_features"],
                inputs["canonical_xyz"],
                inputs["evidence_fou"],
                opportunity,
                observed,
            ],
            dim=-1,
        )
        visual = self.visual_projection(inputs["visual_features"]) * observed
        hidden = self.trunk(torch.cat([self.physical_projection(physical), visual], dim=-1))
        delta = torch.tanh(self.surface_head(hidden)) * self.maximum_surface_delta_m
        return delta, self.scalar_head(hidden).squeeze(-1)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
