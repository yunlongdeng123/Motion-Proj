"""GN11 v3 的监督责任与几何约束。

本模块只定义可单测的张量目标。真实联合前向图不做裁剪；车身联合 RGB
的参数梯度路由在 :mod:`utils.v3_runtime_context` 中显式完成。
"""

import torch
import torch.nn.functional as F


STATIC_BODY_RGB_GROUPS = frozenset({"xyz", "opacity", "scaling", "rotation"})


def _mask_like(mask, reference):
    mask = mask.to(device=reference.device, dtype=reference.dtype)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3 or mask.shape[0] != 1:
        raise ValueError("expected a 1xHxW binary mask")
    if mask.shape[-2:] != reference.shape[-2:]:
        raise ValueError("mask and rendered image dimensions differ")
    return mask.clamp(0.0, 1.0)


def route_depth(joint_depth, dynamic_depth_sum, dynamic_alpha, selected_mask):
    """车身深度由独立动态分支承担，前向联合图保持不变。"""
    return torch.where(
        selected_mask.bool(),
        dynamic_depth_sum / dynamic_alpha.clamp_min(1e-5),
        joint_depth,
    )


def observed_rgb_loss(prediction, target, selected_mask, ssim_fn, dssim=0.2):
    """保留 v2 的掩码外联合 RGB，车身直接像素梯度为零。"""
    outside = 1.0 - _mask_like(selected_mask, prediction)
    prediction_outside = prediction * outside
    target_outside = target * outside
    return ((1.0 - dssim) * F.l1_loss(prediction_outside, target_outside)
            + dssim * (1.0 - ssim_fn(prediction_outside, target_outside)))


def body_joint_rgb_loss(prediction, target, selected_mask, ssim_fn, dssim=0.2):
    """在真实联合渲染上检查车身最终合成，不构造裁剪后的伪前向。

    L1 按有效车身像素归一，避免远车因全图平均而几乎没有监督。DSSIM
    仍在同一个真实联合图的掩码视图上计算。参数梯度由运行时路由到全部
    动态参数以及静态几何/opacity，明确排除静态 SH 颜色和 EnvMap。
    """
    mask = _mask_like(selected_mask, prediction)
    channels = prediction.shape[0]
    denominator = (mask.sum() * channels).clamp_min(1.0)
    l1 = ((prediction - target).abs() * mask).sum() / denominator
    masked_prediction = prediction * mask
    masked_target = target * mask
    dssim_loss = 1.0 - ssim_fn(masked_prediction, masked_target)
    return (1.0 - dssim) * l1 + dssim * dssim_loss


def static_front_occlusion_loss(
    static_depth_sum,
    static_alpha,
    measured_depth,
    selected_mask,
    margin_ratio=0.01,
    temperature_ratio=0.01,
):
    """处罚车身测量表面之前的屏幕空间静态贡献。

    这是基于每像素 opacity 加权静态深度的可微分深度顺序代理，不把
    Gaussian 中心是否落入 3D 框当作充分条件，因此宽尺度、中心在框外但
    投影侵入车身的 splat 也能收到信号。它不是逐 splat 的精确 front-alpha。
    """
    if margin_ratio < 0 or temperature_ratio <= 0:
        raise ValueError("front-occlusion margins must be finite and nonnegative")
    mask = _mask_like(selected_mask, static_alpha)
    measured_depth = measured_depth.to(device=static_alpha.device, dtype=static_alpha.dtype)
    if measured_depth.ndim == 2:
        measured_depth = measured_depth.unsqueeze(0)
    if measured_depth.shape != static_alpha.shape:
        raise ValueError("measured depth and static alpha dimensions differ")

    valid = ((measured_depth > 0) & mask.bool()).to(static_alpha)
    denominator = valid.sum().clamp_min(1.0)
    static_depth = static_depth_sum / static_alpha.clamp_min(1e-5)
    relative_gap = (measured_depth - static_depth) / measured_depth.clamp_min(1e-5)
    soft_front_gap = F.softplus(
        (relative_gap - margin_ratio) / temperature_ratio
    ) * temperature_ratio
    loss = (static_alpha * soft_front_gap * valid).sum() / denominator
    hard_front = (relative_gap > margin_ratio).to(static_alpha) * valid
    metrics = {
        "front_valid_pixels": valid.sum(),
        "front_static_alpha_mean": (static_alpha * valid).sum() / denominator,
        "front_hard_fraction": hard_front.sum() / denominator,
        "front_hard_alpha_mass": (static_alpha * hard_front).sum() / denominator,
    }
    return loss, metrics


def _masked_top_mean(values, mask, fraction):
    if not 0 < fraction <= 1:
        raise ValueError("top fraction must be in (0, 1]")
    active = values[mask.bool()]
    if active.numel() == 0:
        return values.sum() * 0
    k = max(1, min(active.numel(), int(round(active.numel() * fraction))))
    return torch.topk(active, k=k, largest=True, sorted=False).values.mean()


def dynamic_color_leak_losses(
    dynamic_premultiplied_rgb,
    dynamic_alpha,
    selected_mask,
    boundary_radius_px=2,
    top_fraction=0.01,
    top_weight=0.1,
    radiance_ratio_cap=4.0,
    ratio_slack=1.0 / 255.0,
):
    """直接约束边界外动态预乘颜色及低 alpha/高颜色退化。

    输入必须是加入 EnvMap 之前的独立动态预乘 RGB。边界膨胀带不参与
    训练，减少混合像素误伤；均值项与 top-1% 项共同避免小而亮的泄漏被
    全图平均淹没。ratio 约束仅针对当前 [0,1] 低动态范围表示约定。
    """
    if boundary_radius_px < 0:
        raise ValueError("boundary radius must be nonnegative")
    if top_weight < 0 or radiance_ratio_cap <= 0 or ratio_slack < 0:
        raise ValueError("dynamic leak settings must be finite and nonnegative")
    mask = _mask_like(selected_mask, dynamic_alpha)
    kernel = 2 * int(boundary_radius_px) + 1
    dilated = F.max_pool2d(mask.unsqueeze(0), kernel, stride=1,
                           padding=int(boundary_radius_px)).squeeze(0)
    strict_outside = (dilated < 0.5).to(dynamic_alpha)
    denominator = strict_outside.sum().clamp_min(1.0)

    pixel_color = dynamic_premultiplied_rgb.abs().amax(dim=0, keepdim=True)
    color_mean = (pixel_color * strict_outside).sum() / denominator
    color_top = _masked_top_mean(pixel_color, strict_outside, top_fraction)
    color_loss = color_mean + top_weight * color_top

    ratio_excess = F.relu(
        pixel_color - radiance_ratio_cap * dynamic_alpha - ratio_slack
    )
    ratio_mean = (ratio_excess * strict_outside).sum() / denominator
    ratio_top = _masked_top_mean(ratio_excess, strict_outside, top_fraction)
    ratio_loss = ratio_mean + top_weight * ratio_top
    metrics = {
        "dynamic_strict_outside_pixels": strict_outside.sum(),
        "dynamic_color_leak_mean": color_mean,
        "dynamic_color_leak_top": color_top,
        "dynamic_ratio_excess_mean": ratio_mean,
        "dynamic_ratio_excess_top": ratio_top,
    }
    return color_loss, ratio_loss, metrics


def vehicle_volume_opacity(
    xyz,
    opacity,
    centers,
    axes,
    half_sizes,
    world_up,
    ground_offset,
    road_clearance,
    softness,
    chunk_size=131072,
):
    """保留 v2 的保守 3D 中心排斥，作为屏幕深度约束的补充。"""
    if centers.numel() == 0:
        return opacity.sum() * 0
    if softness <= 0 or road_clearance <= 0:
        raise ValueError("positive scale-calibrated margins required")
    total = opacity.sum() * 0
    weight_sum = torch.zeros((), device=xyz.device, dtype=xyz.dtype)
    for start in range(0, len(xyz), chunk_size):
        points = xyz[start:start + chunk_size]
        local = torch.einsum("nbi,bij->nbj", points[:, None] - centers, axes)
        distance = (half_sizes[None] - local.abs()).amin(-1)
        membership = (distance / softness).clamp(0, 1).amax(1)
        height = points @ world_up + ground_offset
        membership = membership * ((height - road_clearance) / softness).clamp(0, 1)
        weights = membership.detach()
        total = total + (opacity[start:start + chunk_size, 0] * weights).sum()
        weight_sum = weight_sum + weights.sum()
    return total / weight_sum.clamp_min(1)
