"""固定像素域的道路监督；RGB 不截断，指标另用 V1 的 [0,1] 口径。"""

import torch
import torch.nn.functional as F

VERSION = 'GN11_V41_DENSE_COMPOSITE_ROAD_20261023'
POLICY = dict(joint_weight=1., appearance_weight=1., preserve_weight=1.,
              vehicle_body_weight=1., vehicle_edge_weight=1.,
              vehicle_reduction='mean of nonempty identity means, interior and edge separately',
              outside_reliable_target='frozen native full reconstruction; not road RGB ground truth',
              ssim_mix=.2, ssim_window=11, ssim_sigma=1.5, ssim_min_valid_fraction=.8,
              robust_delta=.05, alpha_epsilon=1e-5,
              normalization='mean_valid_pixels_per_frame_then_mean_sampled_frames',
              coverage='unchanged_V4_center_weighted_target_0.95_weight_0.1',
              prior='unchanged_V4_connected_world_neighbors_weight_0.1')


def masked_mean(values, mask):
    if mask.requires_grad or mask.dtype != torch.bool or mask.shape != values.shape[-2:]:
        raise ValueError('监督域须为固定 HxW bool，不允许可导权重或广播域')
    if not mask.any():
        return values.sum()*0
    return values[..., mask].mean()


def valid_windows(mask, size=11):
    # padding 也是无效；窗口不含车身/未知像素，不能把图像乘 mask 后算 SSIM。
    count = F.conv2d(mask[None, None].float(), mask.new_ones((1, 1, size, size),
                    dtype=torch.float32), padding=size//2)[0, 0]
    interior = torch.zeros_like(mask)
    interior[size//2:-(size//2), size//2:-(size//2)] = True
    return (count >= size*size*POLICY['ssim_min_valid_fraction']) & mask & interior


def ssim_map(prediction, target, mask=None):
    size, sigma = POLICY['ssim_window'], POLICY['ssim_sigma']
    coordinate = torch.arange(size, dtype=prediction.dtype, device=prediction.device)-size//2
    gaussian = torch.exp(-coordinate.square()/(2*sigma**2))
    gaussian /= gaussian.sum()
    kernel = (gaussian[:, None]*gaussian[None, :]).expand(3, 1, size, size).contiguous()
    raw_convolve = lambda x: F.conv2d(x[None], kernel, padding=size//2, groups=3)[0]
    if mask is None:
        convolve = raw_convolve
    else:
        weight = mask.to(prediction).expand_as(prediction)
        normalizer = raw_convolve(weight).clamp_min(1e-8)
        convolve = lambda x: raw_convolve(x*weight)/normalizer
    mu_p, mu_t = convolve(prediction), convolve(target)
    var_p = convolve(prediction.square())-mu_p.square()
    var_t = convolve(target.square())-mu_t.square()
    covariance = convolve(prediction*target)-mu_p*mu_t
    return (((2*mu_p*mu_t+.01**2)*(2*covariance+.03**2)) /
            ((mu_p.square()+mu_t.square()+.01**2)*(var_p+var_t+.03**2))).mean(0)


def identity_mean(error, domains):
    values = [masked_mean(error, mask) for mask in domains if mask.any()]
    return torch.stack(values).mean() if values else error.sum()*0


def dense_objective(joint, native, road, target, visible, protect, windows=None, bodies=None, edges=None):
    if any(x.shape != target.shape for x in (joint, native, road['rgb'])):
        raise ValueError('RGB 必须为同一相机的 3xHxW')
    if native.requires_grad or target.requires_grad:
        raise ValueError('原 V1 参考与真实 RGB 必须停止梯度')
    if (visible & protect).any():
        raise ValueError('可见路面与保护域重叠')
    if windows is None:
        windows = valid_windows(visible)
    if windows.requires_grad or (windows & ~visible).any():
        raise ValueError('非法 SSIM 窗口')
    rgb = masked_mean((joint-target).abs(), visible)
    structural = masked_mean(1-ssim_map(joint, target, visible), windows)
    # 没有有效窗口时只计算 RGB，同时明确计数，不把零值冒充 SSIM 通过。
    mixed = ((1-POLICY['ssim_mix'])*rgb+POLICY['ssim_mix']*structural
             if windows.any() else rgb)
    surface = road['rgb']/road['alpha'].clamp_min(POLICY['alpha_epsilon'])
    appearance = masked_mean(torch.sqrt((surface-target).square()+POLICY['robust_delta']**2)
                             -POLICY['robust_delta'], visible)
    change = (joint-native).abs()
    body = edge = change.sum()*0
    remaining = protect
    if bodies is not None or edges is not None:
        if bodies is None or edges is None or bodies.shape != edges.shape or bodies.shape[1:] != visible.shape:
            raise ValueError('车身内部/边缘域必须成对提供')
        vehicles = bodies.any(0) | edges.any(0)
        if (vehicles & ~protect).any() or (vehicles & visible).any():
            raise ValueError('车身保护域不能得到道路真值')
        remaining = protect & ~vehicles
        body, edge = identity_mean(change, bodies), identity_mean(change, edges)
    preservation = masked_mean(change, remaining)
    total = (POLICY['joint_weight']*mixed + POLICY['appearance_weight']*appearance
             + POLICY['preserve_weight']*preservation
             + POLICY['vehicle_body_weight']*body + POLICY['vehicle_edge_weight']*edge)
    terms = dict(joint_rgb=rgb, joint_dssim=structural, appearance_rgb=appearance,
                 preservation=preservation, vehicle_body_preservation=body, vehicle_edge_preservation=edge)
    with torch.no_grad():
        terms.update(visible_pixels=visible.sum(), ssim_windows=windows.sum(), protect_pixels=protect.sum(),
            independent_alpha=masked_mean(road['alpha'], visible),
            native_visible_mae=masked_mean((native-target).abs(), visible),
            joint_minus_native_luma=masked_mean(((joint-native)*joint.new_tensor(
                [.2126, .7152, .0722])[:, None, None]).sum(0), visible))
    return total, terms
