"""M1：Actor-local evidential anisotropic surface completion field。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class EvidentialActorSurfaceField(nn.Module):
    """PointNet evidence encoder + implicit coordinate decoder。"""

    def __init__(self, evidence_dim: int, latent_dim: int = 128, hidden_dim: int = 128) -> None:
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(evidence_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5),
        )

    def encode(self, evidence_features: torch.Tensor) -> torch.Tensor:
        encoded = self.point_encoder(evidence_features)
        return encoded.max(dim=0).values

    def forward(
        self,
        evidence_features: torch.Tensor,
        query_normalized_xyz: torch.Tensor,
        actor_size_lwh_m: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decode(
            self.encode(evidence_features), query_normalized_xyz, actor_size_lwh_m
        )

    def decode(
        self,
        latent: torch.Tensor,
        query_normalized_xyz: torch.Tensor,
        actor_size_lwh_m: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = latent.reshape(1, -1).expand(len(query_normalized_xyz), -1)
        size = actor_size_lwh_m.reshape(1, 3).expand(len(query_normalized_xyz), -1)
        output = self.decoder(torch.cat([latent, query_normalized_xyz, size], dim=-1))
        # SCF两个分量是非负的planar/vertical boundary distance。
        return F.softplus(output[:, :2]), output[:, 2:]


def anisotropic_scf_distance(scf: torch.Tensor) -> torch.Tensor:
    if scf.shape[-1] != 2:
        raise ValueError("SCF 必须包含 planar 与 vertical 两个分量")
    return torch.linalg.vector_norm(scf, dim=-1)


def occupied_collision_mask(evidence_logits: torch.Tensor) -> torch.Tensor:
    """类别顺序固定为 FREE/OCCUPIED/UNKNOWN；UNKNOWN 不进入碰撞表面。"""
    if evidence_logits.shape[-1] != 3:
        raise ValueError("evidence logits 最后一维必须为3")
    return evidence_logits.argmax(dim=-1) == 1
