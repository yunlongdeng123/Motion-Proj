"""冻结 3D Gaussian 几何上的 instance-opacity 渲染、loss 与指标。"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def composite_alpha_by_ray(
    *,
    ray_ids: "Any",
    gaussian_logits: "Any",
    projected_kernel: "Any",
    ray_count: int,
):
    """纯 Torch 的可微 alpha 合成，用于 correctness test 与稀疏后端。"""
    import torch

    rays = torch.as_tensor(ray_ids, dtype=torch.long, device=gaussian_logits.device)
    kernel = torch.as_tensor(
        projected_kernel, dtype=gaussian_logits.dtype, device=gaussian_logits.device
    )
    if rays.ndim != 1 or kernel.shape != rays.shape or gaussian_logits.shape != rays.shape:
        raise ValueError("ray/logit/kernel 必须是一维同 shape")
    if ray_count <= 0 or rays.numel() and (
        int(rays.min()) < 0 or int(rays.max()) >= ray_count
    ):
        raise ValueError("ray id 越界")
    alpha = (torch.sigmoid(gaussian_logits) * kernel).clamp(0.0, 0.999)
    log_survival = torch.zeros(
        ray_count, dtype=alpha.dtype, device=alpha.device
    )
    log_survival.index_add_(0, rays, torch.log1p(-alpha))
    return 1.0 - torch.exp(log_survival)


def rasterize_instance_mask(
    *,
    means: "Any",
    quats: "Any",
    scales: "Any",
    opacity_logits: "Any",
    viewmats: "Any",
    intrinsics: "Any",
    width: int,
    height: int,
    near_plane: float,
    far_plane: float,
    packed: bool,
    radius_clip: float,
    antialiased: bool,
):
    """调用 gsplat，但只传冻结几何和独立 instance opacity。"""
    import torch
    from gsplat import rasterization

    for name, value in (("means", means), ("quats", quats), ("scales", scales)):
        if value.requires_grad:
            raise RuntimeError(f"base {name} requires_grad=true，拒绝 instance render")
    if not opacity_logits.requires_grad and opacity_logits.grad_fn is not None:
        raise RuntimeError("instance opacity tensor 的梯度状态不一致")
    colors = torch.ones(
        (means.shape[0], 3), dtype=means.dtype, device=means.device
    )
    _, alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=torch.sigmoid(opacity_logits),
        colors=colors,
        viewmats=viewmats,
        Ks=intrinsics,
        width=int(width),
        height=int(height),
        near_plane=float(near_plane),
        far_plane=float(far_plane),
        packed=bool(packed),
        absgrad=False,
        sparse_grad=False,
        radius_clip=float(radius_clip),
        render_mode="RGB",
        rasterize_mode="antialiased" if antialiased else "classic",
    )
    return alphas[0, ..., 0], info


def instance_mask_losses(
    *,
    prediction: "Any",
    target: "Any",
    candidate_opacity: "Any",
    prior_opacity: "Any",
    background_candidate: "Any",
    weights: Mapping[str, float],
) -> dict[str, "Any"]:
    """目标平衡 BCE + Dice + sparse + prior；共享参数使 temporal 恒等。"""
    import torch
    import torch.nn.functional as functional

    if prediction.shape != target.shape:
        raise ValueError("prediction/target mask shape 不一致")
    target = target.to(dtype=prediction.dtype)
    positive = target.sum()
    negative = target.numel() - positive
    positive_weight = torch.clamp(negative / positive.clamp_min(1.0), 1.0, 50.0)
    pixel_weight = torch.where(target > 0.5, positive_weight, 1.0)
    bce = functional.binary_cross_entropy(
        prediction.clamp(1e-5, 1.0 - 1e-5), target, weight=pixel_weight
    )
    intersection = (prediction * target).sum()
    dice = 1.0 - (2.0 * intersection + 1.0) / (
        prediction.sum() + target.sum() + 1.0
    )
    background = background_candidate.to(dtype=torch.bool)
    sparse = (
        candidate_opacity[background].mean()
        if bool(background.any())
        else candidate_opacity.new_zeros(())
    )
    prior = (candidate_opacity - prior_opacity).square().mean()
    temporal = candidate_opacity.new_zeros(())
    total = (
        float(weights["mask_bce"]) * bce
        + float(weights["dice"]) * dice
        + float(weights["sparse"]) * sparse
        + float(weights["prior"]) * prior
        + float(weights.get("temporal", 0.0)) * temporal
    )
    return {
        "total": total,
        "mask_bce": bce,
        "dice": dice,
        "sparse": sparse,
        "prior": prior,
        "temporal": temporal,
    }


def binary_inner_boundary(mask: np.ndarray) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2:
        raise ValueError("mask 必须是二维")
    padded = np.pad(value, 1, constant_values=False)
    interior = (
        value
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return value & ~interior


def binary_mask_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    boundary_tolerance_pixels: float,
) -> dict[str, float]:
    """计算 IoU、boundary F1/NBD 与 FP/FN semantic mass。"""
    from scipy.ndimage import distance_transform_edt

    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    if pred.shape != truth.shape or pred.ndim != 2:
        raise ValueError("metric masks 必须是同 shape 二维数组")
    intersection = int(np.logical_and(pred, truth).sum())
    union = int(np.logical_or(pred, truth).sum())
    pred_count, truth_count = int(pred.sum()), int(truth.sum())
    iou = intersection / union if union else 1.0
    pred_boundary = binary_inner_boundary(pred)
    truth_boundary = binary_inner_boundary(truth)
    pred_boundary_count = int(pred_boundary.sum())
    truth_boundary_count = int(truth_boundary.sum())
    if pred_boundary_count and truth_boundary_count:
        distance_to_truth = distance_transform_edt(~truth_boundary)
        distance_to_pred = distance_transform_edt(~pred_boundary)
        precision = float(
            (distance_to_truth[pred_boundary] <= boundary_tolerance_pixels).mean()
        )
        recall = float(
            (distance_to_pred[truth_boundary] <= boundary_tolerance_pixels).mean()
        )
        boundary_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        diagonal = float(np.hypot(*pred.shape))
        normalized_boundary_distance = 0.5 * (
            float(distance_to_truth[pred_boundary].mean())
            + float(distance_to_pred[truth_boundary].mean())
        ) / diagonal
    elif pred_boundary_count == truth_boundary_count:
        boundary_f1 = 1.0
        normalized_boundary_distance = 0.0
    else:
        boundary_f1 = 0.0
        normalized_boundary_distance = 1.0
    false_positive = int(np.logical_and(pred, ~truth).sum())
    false_negative = int(np.logical_and(~pred, truth).sum())
    return {
        "iou": float(iou),
        "boundary_f1": float(boundary_f1),
        "normalized_boundary_distance": float(normalized_boundary_distance),
        "false_positive_semantic_mass": false_positive / max(pred_count, 1),
        "false_negative_semantic_mass": false_negative / max(truth_count, 1),
        "predicted_positive_pixels": float(pred_count),
        "target_positive_pixels": float(truth_count),
        "identity_present": float(pred_count > 0 and truth_count > 0),
    }


def aggregate_metrics(rows: list[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("没有可聚合的 metric row")
    names = sorted(set.intersection(*(set(row) for row in rows)))
    return {
        name: float(np.mean([float(row[name]) for row in rows])) for name in names
    }
