"""域内均值显式分开；完整 mask 监督输入从不预乘 GT mask。"""
import torch
import torch.nn.functional as F
from .math3d import srgb_to_linear


def masked_mean(value, mask):
    mask = mask.to(value.dtype)
    while mask.ndim < value.ndim:
        mask = mask[..., None]
    mask = mask.expand_as(value)
    return (value*mask).sum()/mask.sum().clamp_min(1)


def ssim_loss(x, y, valid=None):
    # x/y: HWC；mask 保证窗口内没有被 Stage1 排除的阴影像素。
    if min(x.shape[:2]) < 3:
        return x.sum()*0
    a, b = x.permute(2, 0, 1)[None], y.permute(2, 0, 1)[None]
    pool = lambda z: F.avg_pool2d(z, 3, stride=1)
    ma, mb = pool(a), pool(b)
    va, vb = pool(a*a)-ma*ma, pool(b*b)-mb*mb
    co = pool(a*b)-ma*mb
    score = ((2*ma*mb+.01**2)*(2*co+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2)).clamp_min(1e-9)
    value = (1-score.clamp(-1, 1)).squeeze(0).permute(1, 2, 0)/2
    if valid is None:
        return value.mean()
    windows = F.avg_pool2d(valid.float()[None, None], 3, stride=1).squeeze(0).squeeze(0) >= .999
    return masked_mean(value, windows)


def instance_domain_mean(value, mask, known, reduction='mean'):
    """先各实例域内归一化；空域不稀释其他实例，未知标签无梯度。"""
    if value.shape != mask.shape:
        raise ValueError('实例值与域的HWK形状必须一致')
    mass=mask.to(value.dtype).sum((0,1))
    active=known & (mass>0)
    per_actor=(value*mask.to(value.dtype)).sum((0,1))/mass.clamp_min(1)
    if reduction=='sum':return (per_actor*active).sum()
    if reduction!='mean':raise ValueError('invalid instance reduction')
    return (per_actor*active).sum()/active.sum().clamp_min(1)


def instance_body_loss(error,frame):
    value=error.mean(-1)[...,None].expand_as(frame.masks)
    return instance_domain_mean(value,frame.masks,frame.mask_known)


def mask_terms(package, frame, balanced=False):
    raw, visible = package["raw_actor_alpha"], package["visible_labels"]
    target = frame.masks.to(raw.dtype)
    known = frame.mask_known[None, None].expand_as(raw)
    if balanced:
        inside=instance_domain_mean((raw-1).square(),frame.masks,frame.mask_known)
        # 每个已知实例的完整车外域都在；求和避免大量空实例将约束稀释。
        outside=instance_domain_mean(raw.square(),~frame.masks,frame.mask_known,'sum')
    else:
        inside = masked_mean((raw-1).square(), known & frame.masks)
        outside = masked_mean(raw.square(), known & ~frame.masks)
    identity = masked_mean((visible-target).square(), known)
    return inside, outside, identity


def geometry_and_reference_terms(model, package, frame, config, stage):
    geo, normal, residual = model.field.regularizers(stage)
    inside, outside, identity = mask_terms(package, frame,config.instance_balanced_supervision)
    observed = frame.depth > 0
    # 相机 Z，逆深度；不给无 LiDAR 的像素伪深度监督。
    pred = package["depth"].clamp_min(.1).reciprocal()
    gt = frame.depth.clamp_min(.1).reciprocal()
    depth = masked_mean((pred-gt).abs(), observed)
    field = model.field
    weight = field.reference_confidence if config.use_clean_road else field.reference_confidence*0
    appearance = field.appearance_logit.sigmoid()
    # 参考是太阳下无车辆遮挡的辐照颜色，而非 intrinsic albedo。
    from .math3d import quaternion_matrix
    road_n = quaternion_matrix(field.quaternion)[..., 2]
    if stage == "geometry":
        clean = masked_mean((appearance-srgb_to_linear(field.reference_rgb)).abs(), weight)
    else:
        ambient = model.light.ambient_logit.sigmoid()
        irradiance = ambient+(1-ambient)*(road_n*model.light.direction()).sum(-1).clamp_min(0)[:, None]
        unshadowed = field.albedo_logit.sigmoid()*irradiance
        clean = masked_mean((unshadowed-srgb_to_linear(field.reference_rgb)).abs(), weight)
    return dict(depth=depth, mask_inside=inside, mask_outside=outside, visible_instance=identity,
                clean_road=clean, geometry=geo, normal=normal, view_residual=residual)
