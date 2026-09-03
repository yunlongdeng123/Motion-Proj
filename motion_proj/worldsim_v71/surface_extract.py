"""从 anisotropic SCF 抽取物理表面。"""

from __future__ import annotations

import torch

from motion_proj.worldsim_v71.actor_surface_field import occupied_collision_mask


def extract_zero_crossing_surface(
    query_points: torch.Tensor,
    scf: torch.Tensor,
    evidence_logits: torch.Tensor,
    *,
    planar_band_m: float,
    vertical_band_m: float,
) -> torch.Tensor:
    if len(query_points) != len(scf) or len(query_points) != len(evidence_logits):
        raise ValueError("query/scf/evidence 数量不一致")
    near_surface = (scf[:, 0].abs() <= float(planar_band_m)) & (
        scf[:, 1].abs() <= float(vertical_band_m)
    )
    return query_points[near_surface & occupied_collision_mask(evidence_logits)]


def merge_hard_anchors(anchors: torch.Tensor, extracted: torch.Tensor) -> torch.Tensor:
    return torch.cat([anchors, extracted], dim=0)
