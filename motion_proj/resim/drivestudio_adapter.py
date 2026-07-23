"""DriveStudio/gsplat 到 V7.1 typed render 产品的窄适配层。"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def gsplat_first_hit_from_info(
    info: dict,
    *,
    alpha_threshold: float,
    gaussian_mask: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """从 gsplat near→far intersections 计算累计 alpha 首次过阈值深度。"""
    from gsplat.cuda._wrapper import rasterize_to_indices_in_range

    if not 0 < alpha_threshold < 1:
        raise ValueError("alpha_threshold 必须在 (0,1)")
    height, width = int(info["height"]), int(info["width"])
    device = info["means2d"].device
    transmittance = torch.ones((1, height, width), device=device)
    gaussian_ids, pixel_ids, _ = rasterize_to_indices_in_range(
        0,
        2**31 - 1,
        transmittance,
        info["means2d"],
        info["conics"],
        info["opacities"],
        width,
        height,
        int(info["tile_size"]),
        info["isect_offsets"],
        info["flatten_ids"],
    )
    if gaussian_mask is not None:
        keep = gaussian_mask[gaussian_ids]
        gaussian_ids, pixel_ids = gaussian_ids[keep], pixel_ids[keep]
    if len(gaussian_ids) == 0:
        return np.zeros((height, width), np.float32), np.zeros((height, width), bool)
    # wrapper 保留每条 ray 的 near→far 顺序；稳定按 pixel 分组。
    order = torch.argsort(pixel_ids, stable=True)
    gaussian_ids, pixel_ids = gaussian_ids[order], pixel_ids[order]
    x = pixel_ids % width
    y = pixel_ids // width
    coords = torch.stack([x, y], dim=-1).float() + 0.5
    means = info["means2d"][0, gaussian_ids]
    conics = info["conics"][0, gaussian_ids]
    delta = coords - means
    sigma = (
        0.5 * (conics[:, 0] * delta[:, 0] ** 2 + conics[:, 2] * delta[:, 1] ** 2)
        + conics[:, 1] * delta[:, 0] * delta[:, 1]
    )
    alpha = torch.clamp_max(
        info["opacities"][0, gaussian_ids] * torch.exp(-sigma), 0.999
    ).clamp_min(0)
    log_trans = torch.log1p(-alpha)
    cumulative = torch.cumsum(log_trans, dim=0)
    starts = torch.ones_like(pixel_ids, dtype=torch.bool)
    starts[1:] = pixel_ids[1:] != pixel_ids[:-1]
    start_indices = torch.nonzero(starts, as_tuple=False).flatten()
    lengths = torch.diff(
        torch.cat([start_indices, torch.tensor([len(pixel_ids)], device=device)])
    )
    bases = torch.zeros_like(start_indices, dtype=cumulative.dtype)
    nonzero = start_indices > 0
    bases[nonzero] = cumulative[start_indices[nonzero] - 1]
    cumulative_alpha = 1.0 - torch.exp(cumulative - torch.repeat_interleave(bases, lengths))
    reached = cumulative_alpha >= alpha_threshold
    candidate = torch.where(
        reached,
        torch.arange(len(pixel_ids), device=device),
        torch.full((len(pixel_ids),), len(pixel_ids), device=device),
    )
    first = torch.full((height * width,), len(pixel_ids), dtype=torch.long, device=device)
    first.scatter_reduce_(0, pixel_ids.long(), candidate.long(), reduce="amin", include_self=True)
    valid = first < len(pixel_ids)
    depth = torch.zeros((height * width,), dtype=torch.float32, device=device)
    depth[valid] = info["depths"][0, gaussian_ids[first[valid]]].float()
    return (
        depth.reshape(height, width).cpu().numpy(),
        valid.reshape(height, width).cpu().numpy(),
    )


def global_actor_gaussian_mask(
    pts_labels: torch.Tensor,
    rigid_class_label: int,
    rigid_point_ids: torch.Tensor,
    model_index: int,
) -> torch.Tensor:
    rigid_positions = torch.nonzero(
        pts_labels == rigid_class_label, as_tuple=False
    ).flatten()
    if len(rigid_positions) != len(rigid_point_ids):
        raise ValueError("RigidNodes global/local Gaussian 数量不一致")
    result = torch.zeros_like(pts_labels, dtype=torch.bool)
    result[rigid_positions] = rigid_point_ids.reshape(-1) == int(model_index)
    return result
