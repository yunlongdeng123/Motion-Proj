"""S+V 局部项：保留原 Eq13 前景项，精确复用历史 V 的掩码外项。"""

import torch


def _validated_mask(raw_alpha, dynamic_mask):
    if not isinstance(raw_alpha, torch.Tensor) or not raw_alpha.is_floating_point():
        raise ValueError("raw alpha 必须是浮点 Tensor")
    if raw_alpha.ndim != 3 or raw_alpha.shape[0] != 1 or min(raw_alpha.shape[-2:]) < 1:
        raise ValueError("raw alpha 必须是非空 1xHxW")
    if not isinstance(dynamic_mask, torch.Tensor):
        raise ValueError("动态 mask 必须是 Tensor")
    mask = dynamic_mask.unsqueeze(0) if dynamic_mask.ndim == 2 else dynamic_mask
    if mask.shape != raw_alpha.shape:
        raise ValueError("动态 mask 必须是匹配 alpha 的 HxW 或 1xHxW")
    if not torch.isfinite(raw_alpha).all() or not torch.isfinite(mask).all():
        raise ValueError("alpha 和 mask 必须全部 finite")
    if not ((mask == 0) | (mask == 1)).all():
        raise ValueError("动态 mask 必须是严格二值 0/1")
    return mask.to(device=raw_alpha.device, dtype=raw_alpha.dtype)


def diagnostic_dynamic_background_opacity_loss(raw_alpha, dynamic_mask, eps=1e-4):
    """历史 V：-mean[(1-Md)*a*log1p(-a/(1+eps))]，a=clamp(raw,0,1)。

    分母是全 H*W；输入不能预乘 mask。它是工程诊断项，非已认证作者公式。
    原 foreground Eq13 仍由未改动的 utils.paper_losses.dynamic_opacity_loss 计算。
    """
    foreground = _validated_mask(raw_alpha, dynamic_mask)
    opacity = raw_alpha.clamp(0.0, 1.0)
    return -((1.0 - foreground) * opacity * torch.log1p(-opacity / (1.0 + eps))).mean()


def raw_alpha_domain_stats(raw_alpha, dynamic_mask):
    """记录未乘 mask、未 clamp 的两域均值；空域计数 0、均值 0。"""
    foreground = _validated_mask(raw_alpha, dynamic_mask).detach()
    alpha = raw_alpha.detach()
    result = {}
    for domain, mask in (("inside", foreground), ("outside", 1.0 - foreground)):
        count = mask.sum()
        result["raw_dynamic_alpha_" + domain + "_pixels"] = int(count.item())
        result["raw_dynamic_alpha_" + domain + "_mean"] = (
            (alpha * mask).sum() / count.clamp_min(1.0)).item()
    return result


def validate_sv_config(args):
    expected = {"paper_method": True, "including_dynamic": True,
                "diagnostic_observed_static_rgb_enabled": True,
                "diagnostic_lambda_observed_static_rgb": 1.0,
                "paper_lambda_dynamic_opacity": 0.05, "lambda_dssim": 0.2,
                "iterations": 50000, "position_lr_max_steps": 50000, "seed": 0}
    for key, value in expected.items():
        actual = args.get(key)
        if type(actual) is not type(value) or actual != value:
            raise ValueError("S+V 配置未锁定: " + key + " 必须为 " + str(value))


def reject_unbound_gpu_training():
    """源码准备没有 GPU harness 绑定；旧审批、旧 binding 或环境变量无效。"""
    raise RuntimeError("S+V 仅完成 CPU 源码准备；新的 GPU harness 尚未接入，禁止训练")
