"""从冻结 gsplat renderer 提取 V5 ownership evidence intersections。"""

from __future__ import annotations

from typing import Any

import numpy as np


def renderer_intersections(
    info: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 Gaussian、pixel、projected pixel、T-before-alpha 与投影深度。"""

    import torch
    from gsplat.cuda._wrapper import rasterize_to_indices_in_range

    height, width = int(info["height"]), int(info["width"])
    means2d = info["means2d"]
    transmittance = torch.ones(
        (int(info["n_cameras"]), height, width),
        device=means2d.device,
        dtype=means2d.dtype,
    )
    gaussian_id, pixel_id, camera_id = rasterize_to_indices_in_range(
        0,
        2**31 - 1,
        transmittance,
        means2d,
        info["conics"],
        info["opacities"],
        width,
        height,
        int(info["tile_size"]),
        info["isect_offsets"],
        info["flatten_ids"],
    )
    if gaussian_id.numel() == 0:
        empty_integer = np.empty(0, dtype=np.int64)
        return (
            empty_integer,
            empty_integer,
            np.empty((0, 2), dtype=np.float32),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float32),
        )
    rays = camera_id * (height * width) + pixel_id
    order = torch.argsort(rays, stable=True)
    gaussian_id = gaussian_id[order]
    pixel_id = pixel_id[order]
    camera_id = camera_id[order]
    rays = rays[order]
    projected = means2d[camera_id, gaussian_id]
    conics = info["conics"][camera_id, gaussian_id]
    opacity = info["opacities"][camera_id, gaussian_id]
    pixel_x = pixel_id.remainder(width).to(projected.dtype) + 0.5
    pixel_y = torch.div(pixel_id, width, rounding_mode="floor").to(
        projected.dtype
    ) + 0.5
    delta_x = projected[:, 0] - pixel_x
    delta_y = projected[:, 1] - pixel_y
    sigma = (
        0.5
        * (conics[:, 0] * delta_x.square() + conics[:, 2] * delta_y.square())
        + conics[:, 1] * delta_x * delta_y
    )
    alpha = torch.minimum(
        opacity * torch.exp(-sigma), opacity.new_tensor(0.999)
    ).double()
    log_survival = torch.log1p(-alpha)
    prefix = torch.cumsum(log_survival, 0)
    starts = torch.ones_like(rays, dtype=torch.bool)
    starts[1:] = rays[1:] != rays[:-1]
    starts_at = torch.nonzero(starts, as_tuple=False).flatten()
    bases = torch.zeros(
        starts_at.numel(), device=means2d.device, dtype=torch.float64
    )
    if starts_at.numel() > 1:
        bases[1:] = prefix[starts_at[1:] - 1]
    segments = torch.cumsum(starts.long(), 0) - 1
    contribution = torch.exp(prefix - log_survival - bases[segments]) * alpha
    depths = info["depths"]
    if depths.ndim == 3:
        depths = depths[..., 0]
    projected_depth = (
        depths[camera_id, gaussian_id]
        if depths.ndim == 2
        else depths[gaussian_id]
    )
    return (
        gaussian_id.cpu().numpy().astype(np.int64),
        pixel_id.cpu().numpy().astype(np.int64),
        projected.detach().cpu().numpy().astype(np.float32),
        contribution.cpu().numpy().astype(np.float64),
        projected_depth.detach().cpu().numpy().astype(np.float32),
    )
