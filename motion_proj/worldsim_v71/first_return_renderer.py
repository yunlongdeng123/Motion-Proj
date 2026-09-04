"""可微首返回渲染与共享 literal beam-tube 评估。"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch


def _ray_geometry(origins: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    vectors = targets - origins
    depths = torch.linalg.vector_norm(vectors, dim=-1).clamp_min(1.0e-6)
    return depths, vectors / depths[:, None]


def differentiable_first_return_depth(
    surface: torch.Tensor,
    origins: torch.Tensor,
    targets: torch.Tensor,
    *,
    sample_count: int = 64,
    surface_sigma_m: float = 0.08,
    density_scale: float = 2.0,
    fallback_margin_m: float = 0.50,
    ray_chunk_size: int = 256,
    point_chunk_size: int = 256,
) -> torch.Tensor:
    """以固定深度采样和 alpha 合成近似首返回，梯度直接回到表面位置。"""
    if surface.ndim != 2 or surface.shape[-1] != 3:
        raise ValueError("surface 必须为 [N,3]")
    if len(surface) == 0:
        target_depth, _ = _ray_geometry(origins, targets)
        return target_depth + float(fallback_margin_m)
    fractions = torch.linspace(
        1.0 / sample_count,
        1.0,
        steps=sample_count,
        dtype=surface.dtype,
        device=surface.device,
    )
    outputs = []
    sigma_sq = max(float(surface_sigma_m) ** 2, 1.0e-8)
    for start in range(0, len(targets), int(ray_chunk_size)):
        chunk_origins = origins[start : start + ray_chunk_size]
        chunk_targets = targets[start : start + ray_chunk_size]
        target_depth, directions = _ray_geometry(chunk_origins, chunk_targets)
        fallback = target_depth + float(fallback_margin_m)
        sample_depth = fallback[:, None] * fractions[None, :]
        samples = chunk_origins[:, None, :] + sample_depth[:, :, None] * directions[:, None, :]
        flat_samples = samples.reshape(-1, 3)
        density = torch.zeros(len(flat_samples), dtype=surface.dtype, device=surface.device)
        for point_start in range(0, len(surface), int(point_chunk_size)):
            points = surface[point_start : point_start + point_chunk_size]
            squared = torch.cdist(flat_samples, points).square()
            density = density + torch.exp(-0.5 * squared / sigma_sq).sum(dim=1)
        density = density.reshape(len(chunk_targets), sample_count)
        alpha = 1.0 - torch.exp(-float(density_scale) * density)
        survival = torch.cumprod(
            torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1.0e-7], dim=1),
            dim=1,
        )
        weights = survival[:, :-1] * alpha
        outputs.append((weights * sample_depth).sum(dim=1) + survival[:, -1] * fallback)
    return torch.cat(outputs, dim=0)


def differentiable_scaled_first_return_depth(
    centers: torch.Tensor,
    scales: torch.Tensor,
    origins: torch.Tensor,
    targets: torch.Tensor,
    *,
    sample_count: int = 64,
    density_scale: float = 2.0,
    fallback_margin_m: float = 0.50,
    ray_chunk_size: int = 256,
    point_chunk_size: int = 256,
) -> torch.Tensor:
    """Alpha-compose first returns while preserving each Gaussian's physical scale."""
    if centers.ndim != 2 or centers.shape[-1] != 3:
        raise ValueError("centers must be [N,3]")
    scales = scales.reshape(-1)
    if len(centers) != len(scales):
        raise ValueError("centers and scales must align")
    if len(centers) == 0:
        target_depth, _ = _ray_geometry(origins, targets)
        return target_depth + float(fallback_margin_m)
    fractions = torch.linspace(
        1.0 / sample_count,
        1.0,
        steps=sample_count,
        dtype=centers.dtype,
        device=centers.device,
    )
    outputs = []
    for start in range(0, len(targets), int(ray_chunk_size)):
        chunk_origins = origins[start : start + ray_chunk_size]
        chunk_targets = targets[start : start + ray_chunk_size]
        target_depth, directions = _ray_geometry(chunk_origins, chunk_targets)
        fallback = target_depth + float(fallback_margin_m)
        sample_depth = fallback[:, None] * fractions[None, :]
        samples = (
            chunk_origins[:, None, :]
            + sample_depth[:, :, None] * directions[:, None, :]
        )
        flat_samples = samples.reshape(-1, 3)
        density = torch.zeros(
            len(flat_samples), dtype=centers.dtype, device=centers.device
        )
        for point_start in range(0, len(centers), int(point_chunk_size)):
            points = centers[point_start : point_start + point_chunk_size]
            sigma = scales[point_start : point_start + point_chunk_size].clamp_min(1.0e-4)
            squared = torch.cdist(flat_samples, points).square()
            density = density + torch.exp(
                -0.5 * squared / sigma.square().reshape(1, -1)
            ).sum(dim=1)
        density = density.reshape(len(chunk_targets), sample_count)
        alpha = 1.0 - torch.exp(-float(density_scale) * density)
        survival = torch.cumprod(
            torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1.0e-7], dim=1),
            dim=1,
        )
        weights = survival[:, :-1] * alpha
        outputs.append(
            (weights * sample_depth).sum(dim=1) + survival[:, -1] * fallback
        )
    return torch.cat(outputs, dim=0)


def differentiable_oriented_first_return_depth(
    centers: torch.Tensor,
    normals: torch.Tensor,
    tangent_scales: torch.Tensor,
    normal_thickness: torch.Tensor,
    origins: torch.Tensor,
    targets: torch.Tensor,
    *,
    sample_count: int = 64,
    density_scale: float = 2.0,
    fallback_margin_m: float = 0.50,
    ray_chunk_size: int = 256,
    point_chunk_size: int = 256,
) -> torch.Tensor:
    """Alpha-compose an anisotropic density whose thin axis is the surface normal."""
    if centers.ndim != 2 or centers.shape[-1] != 3:
        raise ValueError("centers must be [N,3]")
    normals = torch.nn.functional.normalize(normals.reshape(-1, 3), dim=1, eps=1.0e-6)
    tangent_scales = tangent_scales.reshape(-1)
    normal_thickness = normal_thickness.reshape(-1)
    if not (
        len(centers)
        == len(normals)
        == len(tangent_scales)
        == len(normal_thickness)
    ):
        raise ValueError("Oriented Gaussian attributes must align")
    if len(centers) == 0:
        target_depth, _ = _ray_geometry(origins, targets)
        return target_depth + float(fallback_margin_m)
    fractions = torch.linspace(
        1.0 / sample_count,
        1.0,
        steps=sample_count,
        dtype=centers.dtype,
        device=centers.device,
    )
    outputs = []
    for start in range(0, len(targets), int(ray_chunk_size)):
        chunk_origins = origins[start : start + ray_chunk_size]
        chunk_targets = targets[start : start + ray_chunk_size]
        target_depth, directions = _ray_geometry(chunk_origins, chunk_targets)
        fallback = target_depth + float(fallback_margin_m)
        sample_depth = fallback[:, None] * fractions[None, :]
        samples = (
            chunk_origins[:, None, :]
            + sample_depth[:, :, None] * directions[:, None, :]
        ).reshape(-1, 3)
        density = torch.zeros(len(samples), dtype=centers.dtype, device=centers.device)
        for point_start in range(0, len(centers), int(point_chunk_size)):
            points = centers[point_start : point_start + point_chunk_size]
            local_normals = normals[point_start : point_start + point_chunk_size]
            tangent = tangent_scales[point_start : point_start + point_chunk_size].clamp_min(
                1.0e-4
            )
            thickness = normal_thickness[
                point_start : point_start + point_chunk_size
            ].clamp_min(1.0e-4)
            displacement = samples[:, None, :] - points[None, :, :]
            normal_coordinate = torch.sum(
                displacement * local_normals[None, :, :], dim=-1
            )
            tangent_sq = (
                torch.sum(displacement.square(), dim=-1) - normal_coordinate.square()
            ).clamp_min(0.0)
            exponent = -0.5 * (
                tangent_sq / tangent.square().reshape(1, -1)
                + normal_coordinate.square() / thickness.square().reshape(1, -1)
            )
            density = density + torch.exp(exponent).sum(dim=1)
        density = density.reshape(len(chunk_targets), sample_count)
        alpha = 1.0 - torch.exp(-float(density_scale) * density)
        survival = torch.cumprod(
            torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1.0e-7], dim=1),
            dim=1,
        )
        weights = survival[:, :-1] * alpha
        outputs.append(
            (weights * sample_depth).sum(dim=1) + survival[:, -1] * fallback
        )
    return torch.cat(outputs, dim=0)


def literal_spherical_first_return_partition(
    centers: np.ndarray | torch.Tensor,
    scales: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    origins: np.ndarray | torch.Tensor,
    *,
    depth_tolerance_m: float,
    device: torch.device,
    ray_chunk_size: int = 512,
    point_chunk_size: int = 2048,
) -> dict[str, np.ndarray]:
    """Exact minimum positive ray intersection with reported 1-sigma spheres."""
    center_tensor = torch.as_tensor(
        centers, dtype=torch.float32, device=device
    ).reshape(-1, 3)
    scale_tensor = torch.as_tensor(
        scales, dtype=torch.float32, device=device
    ).reshape(-1)
    target_tensor = torch.as_tensor(
        targets, dtype=torch.float32, device=device
    ).reshape(-1, 3)
    origin_tensor = torch.as_tensor(
        origins, dtype=torch.float32, device=device
    ).reshape(-1, 3)
    if len(center_tensor) != len(scale_tensor):
        raise ValueError("centers and scales must align")
    if len(target_tensor) != len(origin_tensor):
        raise ValueError("targets and origins must align")
    first_depths = []
    target_depths = []
    with torch.inference_mode():
        for start in range(0, len(target_tensor), int(ray_chunk_size)):
            chunk_targets = target_tensor[start : start + ray_chunk_size]
            chunk_origins = origin_tensor[start : start + ray_chunk_size]
            target_depth, directions = _ray_geometry(chunk_origins, chunk_targets)
            best_depth = torch.full_like(target_depth, torch.inf)
            for point_start in range(0, len(center_tensor), int(point_chunk_size)):
                points = center_tensor[point_start : point_start + point_chunk_size]
                radius = scale_tensor[point_start : point_start + point_chunk_size]
                vectors = points[None, :, :] - chunk_origins[:, None, :]
                projected = torch.sum(vectors * directions[:, None, :], dim=-1)
                perpendicular_sq = torch.sum(vectors.square(), dim=-1) - projected.square()
                discriminant = radius.square().reshape(1, -1) - perpendicular_sq
                near = projected - torch.sqrt(discriminant.clamp_min(0.0))
                valid = (discriminant >= 0.0) & (near > 0.0)
                local = torch.where(valid, near, torch.full_like(near, torch.inf)).min(dim=1).values
                best_depth = torch.minimum(best_depth, local)
            first_depths.append(best_depth.cpu())
            target_depths.append(target_depth.cpu())
    first_depth = (
        torch.cat(first_depths).numpy()
        if first_depths
        else np.empty(0, dtype=np.float32)
    )
    target_depth = (
        torch.cat(target_depths).numpy()
        if target_depths
        else np.empty(0, dtype=np.float32)
    )
    observable = np.isfinite(first_depth)
    return {
        "first_depth": first_depth.astype(np.float32),
        "target_depth": target_depth.astype(np.float32),
        "early": observable
        & (first_depth < target_depth - float(depth_tolerance_m)),
        "hit": observable
        & (np.abs(first_depth - target_depth) <= float(depth_tolerance_m)),
        "observable": observable,
    }


def literal_oriented_first_return_partition(
    centers: np.ndarray | torch.Tensor,
    normals: np.ndarray | torch.Tensor,
    tangent_scales: np.ndarray | torch.Tensor,
    normal_thickness: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    origins: np.ndarray | torch.Tensor,
    *,
    depth_tolerance_m: float,
    device: torch.device,
    ray_chunk_size: int = 512,
    point_chunk_size: int = 2048,
) -> dict[str, np.ndarray]:
    """Exact first positive ray intersection with oriented oblate ellipsoids."""
    center_tensor = torch.as_tensor(centers, dtype=torch.float32, device=device).reshape(
        -1, 3
    )
    normal_tensor = torch.nn.functional.normalize(
        torch.as_tensor(normals, dtype=torch.float32, device=device).reshape(-1, 3),
        dim=1,
        eps=1.0e-6,
    )
    tangent_tensor = torch.as_tensor(
        tangent_scales, dtype=torch.float32, device=device
    ).reshape(-1)
    thickness_tensor = torch.as_tensor(
        normal_thickness, dtype=torch.float32, device=device
    ).reshape(-1)
    target_tensor = torch.as_tensor(targets, dtype=torch.float32, device=device).reshape(
        -1, 3
    )
    origin_tensor = torch.as_tensor(origins, dtype=torch.float32, device=device).reshape(
        -1, 3
    )
    if not (
        len(center_tensor)
        == len(normal_tensor)
        == len(tangent_tensor)
        == len(thickness_tensor)
    ):
        raise ValueError("Oriented Gaussian attributes must align")
    if len(target_tensor) != len(origin_tensor):
        raise ValueError("targets and origins must align")
    first_depths = []
    target_depths = []
    with torch.inference_mode():
        for start in range(0, len(target_tensor), int(ray_chunk_size)):
            chunk_targets = target_tensor[start : start + ray_chunk_size]
            chunk_origins = origin_tensor[start : start + ray_chunk_size]
            target_depth, directions = _ray_geometry(chunk_origins, chunk_targets)
            best_depth = torch.full_like(target_depth, torch.inf)
            for point_start in range(0, len(center_tensor), int(point_chunk_size)):
                points = center_tensor[point_start : point_start + point_chunk_size]
                local_normals = normal_tensor[point_start : point_start + point_chunk_size]
                tangent = tangent_tensor[
                    point_start : point_start + point_chunk_size
                ].clamp_min(1.0e-4)
                thickness = thickness_tensor[
                    point_start : point_start + point_chunk_size
                ].clamp_min(1.0e-4)
                q = chunk_origins[:, None, :] - points[None, :, :]
                qn = torch.sum(q * local_normals[None, :, :], dim=-1)
                dn = torch.sum(
                    directions[:, None, :] * local_normals[None, :, :], dim=-1
                )
                inverse_tangent_sq = tangent.square().reciprocal().reshape(1, -1)
                inverse_thickness_sq = thickness.square().reciprocal().reshape(1, -1)
                q_dot_d = torch.sum(q * directions[:, None, :], dim=-1)
                q_sq = torch.sum(q.square(), dim=-1)
                coefficient_a = (
                    (1.0 - dn.square()) * inverse_tangent_sq
                    + dn.square() * inverse_thickness_sq
                )
                coefficient_b = (
                    (q_dot_d - qn * dn) * inverse_tangent_sq
                    + qn * dn * inverse_thickness_sq
                )
                coefficient_c = (
                    (q_sq - qn.square()) * inverse_tangent_sq
                    + qn.square() * inverse_thickness_sq
                    - 1.0
                )
                discriminant = coefficient_b.square() - coefficient_a * coefficient_c
                root = torch.sqrt(discriminant.clamp_min(0.0))
                near = (-coefficient_b - root) / coefficient_a.clamp_min(1.0e-8)
                far = (-coefficient_b + root) / coefficient_a.clamp_min(1.0e-8)
                positive = torch.where(near > 0.0, near, far)
                valid = (discriminant >= 0.0) & (positive > 0.0)
                local = torch.where(
                    valid, positive, torch.full_like(positive, torch.inf)
                ).min(dim=1).values
                best_depth = torch.minimum(best_depth, local)
            first_depths.append(best_depth.cpu())
            target_depths.append(target_depth.cpu())
    first_depth = (
        torch.cat(first_depths).numpy()
        if first_depths
        else np.empty(0, dtype=np.float32)
    )
    target_depth = (
        torch.cat(target_depths).numpy()
        if target_depths
        else np.empty(0, dtype=np.float32)
    )
    observable = np.isfinite(first_depth)
    return {
        "first_depth": first_depth.astype(np.float32),
        "target_depth": target_depth.astype(np.float32),
        "early": observable
        & (first_depth < target_depth - float(depth_tolerance_m)),
        "hit": observable
        & (np.abs(first_depth - target_depth) <= float(depth_tolerance_m)),
        "observable": observable,
    }


def differentiable_oriented_first_intersection_depth(
    centers: torch.Tensor,
    normals: torch.Tensor,
    tangent_scales: torch.Tensor,
    normal_thickness: torch.Tensor,
    origins: torch.Tensor,
    targets: torch.Tensor,
    *,
    fallback_margin_m: float = 0.50,
    ray_chunk_size: int = 256,
    point_chunk_size: int = 512,
) -> torch.Tensor:
    """Return the exact ellipsoid entrance depth used by the literal deployment audit."""
    if centers.ndim != 2 or centers.shape[-1] != 3:
        raise ValueError("centers must be [N,3]")
    normals = torch.nn.functional.normalize(normals.reshape(-1, 3), dim=1, eps=1.0e-6)
    tangent_scales = tangent_scales.reshape(-1)
    normal_thickness = normal_thickness.reshape(-1)
    if not (
        len(centers)
        == len(normals)
        == len(tangent_scales)
        == len(normal_thickness)
    ):
        raise ValueError("Oriented Gaussian attributes must align")
    target_depths, _ = _ray_geometry(origins, targets)
    if len(centers) == 0:
        return target_depths + float(fallback_margin_m)
    outputs = []
    for start in range(0, len(targets), int(ray_chunk_size)):
        chunk_origins = origins[start : start + ray_chunk_size]
        chunk_targets = targets[start : start + ray_chunk_size]
        target_depth, directions = _ray_geometry(chunk_origins, chunk_targets)
        best_depth = torch.full_like(target_depth, torch.inf)
        for point_start in range(0, len(centers), int(point_chunk_size)):
            points = centers[point_start : point_start + point_chunk_size]
            local_normals = normals[point_start : point_start + point_chunk_size]
            tangent = tangent_scales[
                point_start : point_start + point_chunk_size
            ].clamp_min(1.0e-4)
            thickness = normal_thickness[
                point_start : point_start + point_chunk_size
            ].clamp_min(1.0e-4)
            q = chunk_origins[:, None, :] - points[None, :, :]
            qn = torch.sum(q * local_normals[None, :, :], dim=-1)
            dn = torch.sum(
                directions[:, None, :] * local_normals[None, :, :], dim=-1
            )
            inverse_tangent_sq = tangent.square().reciprocal().reshape(1, -1)
            inverse_thickness_sq = thickness.square().reciprocal().reshape(1, -1)
            q_dot_d = torch.sum(q * directions[:, None, :], dim=-1)
            q_sq = torch.sum(q.square(), dim=-1)
            coefficient_a = (
                (1.0 - dn.square()) * inverse_tangent_sq
                + dn.square() * inverse_thickness_sq
            )
            coefficient_b = (
                (q_dot_d - qn * dn) * inverse_tangent_sq
                + qn * dn * inverse_thickness_sq
            )
            coefficient_c = (
                (q_sq - qn.square()) * inverse_tangent_sq
                + qn.square() * inverse_thickness_sq
                - 1.0
            )
            discriminant = coefficient_b.square() - coefficient_a * coefficient_c
            root = torch.sqrt(discriminant.clamp_min(0.0) + 1.0e-10)
            denominator = coefficient_a.clamp_min(1.0e-8)
            near = (-coefficient_b - root) / denominator
            far = (-coefficient_b + root) / denominator
            positive = torch.where(near > 0.0, near, far)
            valid = (discriminant >= 0.0) & (positive > 0.0)
            local = torch.where(
                valid, positive, torch.full_like(positive, torch.inf)
            ).min(dim=1).values
            best_depth = torch.minimum(best_depth, local)
        fallback = target_depth + float(fallback_margin_m)
        outputs.append(torch.where(torch.isfinite(best_depth), best_depth, fallback))
    return torch.cat(outputs, dim=0)


def literal_first_return_partition(
    surface: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    origins: np.ndarray | torch.Tensor,
    *,
    lateral_tolerance_m: float,
    depth_tolerance_m: float,
    device: torch.device,
    ray_chunk_size: int = 512,
    point_chunk_size: int = 2048,
) -> dict[str, np.ndarray]:
    """逐射线返回 beam tube 内最小正深度，与 V7 P20/P22/P23 语义一致。"""
    surface_tensor = torch.as_tensor(surface, dtype=torch.float32, device=device).reshape(-1, 3)
    target_tensor = torch.as_tensor(targets, dtype=torch.float32, device=device).reshape(-1, 3)
    origin_tensor = torch.as_tensor(origins, dtype=torch.float32, device=device).reshape(-1, 3)
    if len(target_tensor) != len(origin_tensor):
        raise ValueError("targets 与 origins 数量不一致")
    first_depths = []
    first_indices = []
    target_depths = []
    with torch.inference_mode():
        for start in range(0, len(target_tensor), int(ray_chunk_size)):
            chunk_targets = target_tensor[start : start + ray_chunk_size]
            chunk_origins = origin_tensor[start : start + ray_chunk_size]
            target_depth, directions = _ray_geometry(chunk_origins, chunk_targets)
            best_depth = torch.full_like(target_depth, torch.inf)
            best_index = torch.zeros(len(chunk_targets), dtype=torch.long, device=device)
            for point_start in range(0, len(surface_tensor), int(point_chunk_size)):
                points = surface_tensor[point_start : point_start + point_chunk_size]
                if len(points) == 0:
                    continue
                vectors = points[None, :, :] - chunk_origins[:, None, :]
                depths = torch.sum(vectors * directions[:, None, :], dim=-1)
                lateral = torch.linalg.vector_norm(
                    vectors - depths[:, :, None] * directions[:, None, :], dim=-1
                )
                valid = (depths > 0.0) & (lateral <= float(lateral_tolerance_m))
                local_depth, local_index = torch.min(
                    torch.where(valid, depths, torch.full_like(depths, torch.inf)), dim=1
                )
                replace = local_depth < best_depth
                best_depth = torch.where(replace, local_depth, best_depth)
                best_index = torch.where(replace, local_index + point_start, best_index)
            first_depths.append(best_depth.cpu())
            first_indices.append(best_index.cpu())
            target_depths.append(target_depth.cpu())
    first_depth = torch.cat(first_depths).numpy() if first_depths else np.empty(0, dtype=np.float32)
    first_index = torch.cat(first_indices).numpy() if first_indices else np.empty(0, dtype=np.int64)
    target_depth = torch.cat(target_depths).numpy() if target_depths else np.empty(0, dtype=np.float32)
    observable = np.isfinite(first_depth)
    return {
        "first_indices": first_index.astype(np.int64),
        "first_depth": first_depth.astype(np.float32),
        "target_depth": target_depth.astype(np.float32),
        "early": observable & (first_depth < target_depth - float(depth_tolerance_m)),
        "hit": observable & (np.abs(first_depth - target_depth) <= float(depth_tolerance_m)),
        "observable": observable,
    }


def renderer_kwargs(config: Mapping[str, float | int]) -> dict[str, float | int]:
    return {
        "sample_count": int(config["sample_count"]),
        "surface_sigma_m": float(config["surface_sigma_m"]),
        "density_scale": float(config["density_scale"]),
        "fallback_margin_m": float(config["fallback_margin_m"]),
        "ray_chunk_size": int(config["ray_chunk_size"]),
        "point_chunk_size": int(config["point_chunk_size"]),
    }
