"""只读监督诊断：真实光栅权重归因，不增加训练损失或裁剪物理前向。"""

import torch
import torch.nn.functional as F


def center_in_boxes(xyz, centers, axes, half_sizes, chunk_size=131072):
    result = torch.zeros(len(xyz), device=xyz.device, dtype=torch.bool)
    for start in range(0, len(xyz), chunk_size):
        local = torch.einsum('nbi,bij->nbj', xyz[start:start+chunk_size, None]-centers, axes)
        result[start:start+chunk_size] = (local.abs() <= half_sizes[None]).all(-1).any(-1)
    return result


def _mean(value, domain):
    return float(value[domain.expand_as(value)].mean()) if domain.any() else None


@torch.no_grad()
def supervision_metrics(package, target, mask, measured_depth, scale, boundary_px=2):
    body = mask.bool()
    measured = measured_depth.to(package['alpha']).reshape_as(body)
    valid = body & torch.isfinite(measured) & (measured > 0)
    maps = dict(lidar_depth_m=measured/scale, body_mask=body, valid_body_lidar=valid)
    metrics = dict(body_pixels=int(body.sum()), body_lidar_pixels=int(valid.sum()))
    for branch, prefix in [('joint', ''), ('static', 'static_'), ('dynamic', 'dynamic_')]:
        alpha = package[prefix+'alpha']
        depth = package[prefix+'depth' if prefix else 'depth']/alpha.clamp_min(1e-5)
        maps[branch+'_depth_m'] = depth/scale
        present = valid & (alpha > 1e-5) & torch.isfinite(depth)
        metrics[branch+'_body_depth_valid_pixels'] = int(present.sum())
        metrics[branch+'_body_depth_mae_m'] = _mean((depth-measured).abs()/scale, present)
    metrics['body_joint_rgb_l1'] = _mean((package['render']-target).abs(), body)
    metrics['body_dynamic_raw_rgb_l1'] = _mean((package['dynamic_premultiplied_rgb']-target).abs(), body)
    metrics['body_dynamic_alpha_mse'] = _mean((package['dynamic_alpha']-mask).square(), body)
    metrics['full_dynamic_alpha_mse'] = float((package['dynamic_alpha']-mask).square().mean())
    kernel = 2*boundary_px+1
    dilated = F.max_pool2d(mask[None].float(), kernel, 1, boundary_px)[0] > .5
    eroded = (1-F.max_pool2d(1-mask[None].float(), kernel, 1, boundary_px)[0]) > .5
    boundary = dilated & ~eroded
    outside = ~dilated
    maps['boundary_band'] = boundary
    metrics['boundary_pixels'] = int(boundary.sum())
    metrics['boundary_joint_rgb_l1'] = _mean((package['render']-target).abs(), boundary)
    metrics['boundary_dynamic_alpha_mse'] = _mean((package['dynamic_alpha']-mask).square(), boundary)
    # 边界车内颜色和车外预乘颜色分开，不要求车外动态RGB拟合背景GT。
    metrics['boundary_inside_dynamic_raw_l1'] = _mean((package['dynamic_premultiplied_rgb']-target).abs(), boundary & body)
    color = package['dynamic_premultiplied_rgb'].abs().amax(0, keepdim=True)
    metrics['boundary_outside_dynamic_color'] = _mean(color, boundary & ~body)
    metrics['strict_outside_pixels'] = int(outside.sum())
    metrics['strict_outside_dynamic_color_mean'] = _mean(color, outside)
    if outside.any():
        values = color[outside]
        metrics['strict_outside_dynamic_color_top1pct'] = float(values.topk(max(1, round(values.numel()*.01))).values.mean())
    else:
        metrics['strict_outside_dynamic_color_top1pct'] = None
    return metrics, maps


@torch.no_grad()
def sampled_front_contributions(package, mask, measured_depth, inside_boxes, road_rows,
                                raster_features, margin_ratio=.01, max_samples=14):
    """把每个样本阈值作为 feature 通道，让原生光栅器计算相同 T_i*alpha_i。

    不删除后方高斯，不先筛前方子集再重算 T。沿用原生排序的相机中心 z_i，
    不是连续椭球的首次表面距离。static 与 joint 的透射率分别报告。
    raster_features(features, joint) 返回相同几何/alpha的 feature 图。
    """
    measured = measured_depth.to(package['alpha']).reshape_as(mask)
    valid = mask.bool() & torch.isfinite(measured) & (measured > 0)
    pixels = torch.nonzero(valid[0], as_tuple=False)
    if not len(pixels):
        return dict(sample_pixels_yx=pixels, sample_measured_depth=measured.new_empty(0)), dict(
            sampled_pixels=0, extra_raster_calls=0, status='NO_VALID_BODY_LIDAR')
    indices = torch.linspace(0, len(pixels)-1, min(max_samples, len(pixels)), device=pixels.device).round().long()
    pixels = pixels[indices]
    static_kwargs = package['static_raster_kwargs']
    z = static_kwargs['features'][:, -2]
    alpha = package['static_alpha'][0, pixels[:, 0], pixels[:, 1]]
    average = package['static_depth'][0, pixels[:, 0], pixels[:, 1]]/alpha.clamp_min(1e-5)
    depth = measured[0, pixels[:, 0], pixels[:, 1]]
    threshold = depth*(1-margin_ratio)
    arrays = dict(sample_pixels_yx=pixels, sample_measured_depth=depth,
                  sample_static_mean_depth=average, sample_static_alpha=alpha,
                  sample_mean_front=(alpha > 1e-5) & (average < threshold))
    records = {key: [] for key in ('static_front_alpha', 'static_front_center_inside_alpha',
               'static_front_road_alpha', 'static_center_inside_alpha', 'joint_front_static_alpha')}
    calls = 0
    for start in range(0, len(pixels), 7):
        subset = pixels[start:start+7]
        front = z[:, None] < threshold[start:start+7][None]
        # 7*3+1+2 = 24，包含原生深度与alpha，不超过 compact MAX_FEATURES。
        flags = torch.cat((front, front & inside_boxes[:, None], front & road_rows[:, None],
                           inside_boxes[:, None]), 1).to(z)
        k = len(subset)
        yy, xx = subset.T
        image = raster_features(flags, False)
        joint_image = raster_features(flags, True)
        for result, expected in [(image, package['static_alpha']), (joint_image, package['alpha'])]:
            if not torch.allclose(result[-1, yy, xx], expected[0, yy, xx], atol=2e-5, rtol=2e-5):
                raise RuntimeError('contribution pass changed native transmittance')
        calls += 2
        records['static_front_alpha'].append(image[torch.arange(k, device=z.device), yy, xx])
        records['static_front_center_inside_alpha'].append(image[torch.arange(k, device=z.device)+k, yy, xx])
        records['static_front_road_alpha'].append(image[torch.arange(k, device=z.device)+2*k, yy, xx])
        records['static_center_inside_alpha'].append(image[3*k, yy, xx])
        records['joint_front_static_alpha'].append(joint_image[torch.arange(k, device=z.device), yy, xx])
    arrays.update({key: torch.cat(value) for key, value in records.items()})
    outside_center = arrays['static_front_alpha']-arrays['static_front_center_inside_alpha']
    metrics = dict(sampled_pixels=len(pixels), extra_raster_calls=calls, status='SAMPLED_NATIVE_FEATURE_COMPOSITING',
                   static_front_alpha_mean=float(arrays['static_front_alpha'].mean()),
                   joint_front_static_alpha_mean=float(arrays['joint_front_static_alpha'].mean()),
                   static_front_center_outside_alpha_mean=float(outside_center.mean()),
                   static_front_road_alpha_mean=float(arrays['static_front_road_alpha'].mean()),
                   mean_depth_missed_front_mass_mean=float((arrays['static_front_alpha']*~arrays['sample_mean_front']).mean()))
    # 连续质量指标不依赖人为可见阈值；1/255仅为额外漏检计数的固定报告口径。
    metrics['mean_depth_missed_pixels_alpha_gt_1over255'] = int(((arrays['static_front_alpha'] > 1/255) & ~arrays['sample_mean_front']).sum())
    return arrays, metrics
