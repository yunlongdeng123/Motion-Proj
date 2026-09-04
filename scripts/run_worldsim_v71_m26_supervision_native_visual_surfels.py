"""Expand frozen physical carriers into surface-bounded visual-only surfels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from pytorch3d.transforms import matrix_to_quaternion

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import run_worldsim_v71_m25_geometry_locked_attribute_optimization as m25


def _surface_frames(
    centers: torch.Tensor, neighbor_count: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    distances = torch.cdist(centers, centers)
    neighbors = distances.topk(
        k=min(neighbor_count + 1, len(centers)), largest=False
    ).indices[:, 1:]
    offsets = centers[neighbors] - centers[:, None]
    covariance = torch.einsum("nki,nkj->nij", offsets, offsets) / offsets.shape[1]
    _, eigenvectors = torch.linalg.eigh(covariance)
    normal = eigenvectors[..., 0]
    tangent_a = eigenvectors[..., 2]
    tangent_b = torch.linalg.cross(normal, tangent_a, dim=-1)
    return tangent_a, tangent_b, normal


def _expand_visual_surfels(
    carrier: Mapping[str, np.ndarray], config: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    surface = config["visual_surface"]
    centers = torch.as_tensor(carrier["centers"], dtype=torch.float32)
    radii = torch.as_tensor(carrier["scales"], dtype=torch.float32)
    tangent_a, tangent_b, normal = _surface_frames(
        centers, int(surface["pca_neighbors"])
    )

    coordinates = torch.as_tensor(surface["grid_coordinates"], dtype=torch.float32)
    grid = torch.cartesian_prod(coordinates, coordinates)
    offsets = (
        grid[None, :, 0, None] * tangent_a[:, None]
        + grid[None, :, 1, None] * tangent_b[:, None]
    ) * radii[:, None, None]
    visual_centers = centers[:, None] + offsets

    tangent_scale = radii * float(surface["tangent_scale_fraction"])
    normal_scale = torch.full_like(radii, float(surface["normal_thickness_m"]))
    scales_xyz = torch.stack([tangent_scale, tangent_scale, normal_scale], dim=-1)
    scales_xyz = scales_xyz[:, None].expand(-1, len(grid), -1)

    rotations = torch.stack([tangent_a, tangent_b, normal], dim=-1)
    quaternions = matrix_to_quaternion(rotations)
    quaternions = quaternions[:, None].expand(-1, len(grid), -1)

    child_count = int(len(grid))
    opacity_logits = torch.as_tensor(carrier["opacity_logits"], dtype=torch.float32)
    parent_alpha = torch.sigmoid(opacity_logits)
    child_alpha = 1.0 - torch.pow(1.0 - parent_alpha, 1.0 / child_count)
    child_alpha = child_alpha.clamp(1.0e-6, 1.0 - 1.0e-6)
    child_logits = torch.logit(child_alpha)

    def repeat_rows(value: np.ndarray) -> np.ndarray:
        return np.repeat(value[:, None], child_count, axis=1).reshape(
            (-1,) + value.shape[1:]
        )

    expanded = {
        "centers": visual_centers.reshape(-1, 3).numpy().astype(np.float32),
        "scales_xyz": scales_xyz.reshape(-1, 3).numpy().astype(np.float32),
        "quaternions": quaternions.reshape(-1, 4).numpy().astype(np.float32),
        "features_dc": repeat_rows(carrier["features_dc"]).astype(np.float32),
        "features_rest": repeat_rows(carrier["features_rest"]).astype(np.float32),
        "opacity_logits": child_logits[:, None]
        .expand(-1, child_count, -1)
        .reshape(-1, 1)
        .numpy()
        .astype(np.float32),
    }
    metadata = {
        "representation": "physical_anchor_bounded_visual_surfel_grid",
        "physical_center_source": "frozen_m8_gt_supervised_surface",
        "visual_center_source": "deterministic_local_pca_tangent_grid",
        "surface_frame_source": "frozen_physical_carrier_only",
        "image_to_visual_geometry_gradient": False,
        "visual_geometry_trainable": False,
        "visual_geometry_in_physical_query": False,
        "physical_carrier_count": int(len(centers)),
        "visual_children_per_carrier": child_count,
        "visual_surfel_count": int(len(expanded["centers"])),
        "grid_coordinates": coordinates.tolist(),
        "pca_neighbors": int(surface["pca_neighbors"]),
        "tangent_scale_fraction": float(surface["tangent_scale_fraction"]),
        "normal_thickness_m": float(surface["normal_thickness_m"]),
        "opacity_mass_initialization": "equal_overlapping_transmittance",
    }
    return expanded, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    summary = m25.run(
        args.config.resolve(),
        args.run_id,
        carrier_transform=_expand_visual_surfels,
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
