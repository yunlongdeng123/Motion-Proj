"""Hybrid30k 损失组合；仅依赖 Torch，不加载场景、检查点或 CUDA 扩展。

24 版的有效工程约束继续保留；论文明确改写的 sky、LiDAR 和 NPS
由调用方传入原函数结果。完整动态 alpha MSE 替代前景 Eq13，不叠加。
所有 map 都由调用方从本次前向生成，不能传入乘过 Md 的动态 alpha。
"""

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Sequence

import torch
import torch.nn.functional as F


def _mask_like(mask, reference):
    """保持标签内容，拒绝软 mask；不按 RGB 亮度重新构造标签。"""
    if reference.ndim != 3 or min(reference.shape[-2:]) < 1:
        raise ValueError("参考 Tensor 必须是非空 CxHxW")
    if not isinstance(mask, torch.Tensor):
        raise TypeError("动态 mask 必须是 Tensor")
    mask = mask.unsqueeze(0) if mask.ndim == 2 else mask
    if mask.shape != (1, *reference.shape[-2:]):
        raise ValueError("mask 必须为匹配渲染的 HxW 或 1xHxW")
    if not bool(((mask == 0) | (mask == 1)).all().item()):
        raise ValueError("动态 mask 必须是严格二值 0/1")
    return mask.to(device=reference.device, dtype=reference.dtype)


def full_dynamic_opacity_mse(raw_dynamic_alpha, dynamic_mask):
    """mean[(raw alpha_d-Md)^2]，分母 H*W，两域都有直接监督。

    原始 alpha 不 clamp、不预乘 Md、不 detach。系数 100 是历史控制，
    不表示它最优。已经预乘 mask 的 Tensor 无法由数值可靠识别，调用方负责。
    """
    if raw_dynamic_alpha.ndim != 3 or raw_dynamic_alpha.shape[0] != 1:
        raise ValueError("原始动态 alpha 必须是 1xHxW")
    mask = _mask_like(dynamic_mask, raw_dynamic_alpha)
    return F.mse_loss(raw_dynamic_alpha, mask)


def photometric_rgb_loss(image, target, ssim_fn: Callable, lambda_dssim=0.2):
    """保持原 SSIM 实现；L1 和 DSSIM 都按传入完整图像计算。"""
    if image.shape != target.shape or image.ndim != 3:
        raise ValueError("RGB 输入必须是匹配的 CxHxW")
    return ((1.0 - lambda_dssim) * F.l1_loss(image, target)
            + lambda_dssim * (1.0 - ssim_fn(image, target)))


def static_complement_rgb_loss(static_image, gt_image, dynamic_mask,
                               ssim_fn: Callable, lambda_dssim=0.2):
    """24 静态 RGB：两张整图乘 (1-Md)，全图 .8 L1+.2 DSSIM。

    不裁剪、不按静态像素数重归一化，SSIM 仍在乘 mask 后的完整图像上运行。
    """
    complement = 1.0 - _mask_like(dynamic_mask, static_image)
    return photometric_rgb_loss(static_image * complement, gt_image * complement,
                                ssim_fn, lambda_dssim)


def static_duration_loss(static_t_map, eps=1e-8):
    """24 公式是 1/mean(abs(map))，不是 mean(1/abs(map))。

    map 来源：render(exp(_scaling_t).clamp_max(2))/combined_alpha。
    它是寿命/时间宽度，不是 _t 或场景全局时间。仅当均值 < eps 时对分母
    加下界，避免除零；下界区间该项对 map 的梯度为 0，不冻结模型参数。
    """
    if not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError("寿命分母下界必须为正 finite 数")
    return static_t_map.abs().mean().clamp_min(float(eps)).reciprocal()


def static_velocity_loss(static_v_map):
    """24 静态速度图 L1：mean(abs(map))，不做硬置零或参数冻结。"""
    return static_v_map.abs().mean()


def combined_opacity_entropy(alpha):
    """24 活跃项是总 alpha 的 -mean[o log(o)]，并非单独 static alpha。"""
    opacity = alpha.clamp(1e-6, 1.0 - 1e-6)
    return -(opacity * opacity.log()).mean()


def lidar_decay_factor(iteration, lidar_decay):
    """保留 24 调度；decay<=0 时关闭衰减。"""
    if isinstance(iteration, bool) or int(iteration) != iteration or iteration < 0:
        raise ValueError("iteration 必须是非负整数")
    decay = float(lidar_decay)
    if not math.isfinite(decay):
        raise ValueError("lidar_decay 必须 finite")
    return math.exp(-int(iteration) / 8000.0 * decay) if decay > 0 else 1.0


def _scalar(name, state, reference):
    value = state[name]
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise ValueError(name + " 必须是单元素损失 Tensor")
    if value.device != reference.device or value.dtype != reference.dtype:
        raise ValueError(name + " 必须与 global_rgb 使用相同 device/dtype")
    return value.reshape(())


@dataclass
class HybridLossResult:
    total: torch.Tensor
    raw: dict
    weighted: dict
    weights: dict
    metadata: dict

    def log_values(self):
        """日志全部脱离计算图，返回可直接 JSON 序列化的 Python 标量。"""
        values = {"loss_hybrid_total": float(self.total.detach().item())}
        for name, value in self.raw.items():
            values["loss_hybrid_" + name + "_raw"] = float(value.detach().item())
            values["loss_hybrid_" + name + "_weighted"] = float(self.weighted[name].detach().item())
            values["loss_hybrid_" + name + "_weight"] = float(self.weights[name])
        values.update(self.metadata)
        return values


def compose_hybrid_loss(state: Mapping, config: Mapping, *, nps_fn=None):
    """组合前向状态，NPS 只按 arm flag 调用未改写的 paper nps_fn。

    必需标量：global_rgb/dynamic_rgb/sky_rgb/static_rgb/sky/depth/
    inverse_depth_smoothness/velocity_smoothness。
    必需图：dynamic_alpha(raw)/dynamic_mask/static_t_map/static_v_map/alpha。
    iteration 为整数；NPS 开启时需 nps_features/nps_distances。
    static_{t,v}_map 必须按旧 combined alpha 归一化；nps_features 必须
    是当前论文路径按 static alpha 归一化的静态速度图，二者不要混用。
    """
    reference = state["global_rgb"]
    if not isinstance(reference, torch.Tensor) or reference.numel() != 1:
        raise ValueError("global_rgb 必须是单元素损失 Tensor")
    scalar_names = ("global_rgb", "dynamic_rgb", "sky_rgb", "static_rgb", "sky",
                    "depth", "inverse_depth_smoothness", "velocity_smoothness")
    raw = {name: _scalar(name, state, reference) for name in scalar_names}
    time_eps = float(config.get("hybrid_time_reg_eps", 1e-8))
    raw["dynamic_opacity"] = full_dynamic_opacity_mse(state["dynamic_alpha"], state["dynamic_mask"])
    raw["static_duration"] = static_duration_loss(state["static_t_map"], time_eps)
    raw["static_velocity"] = static_velocity_loss(state["static_v_map"])
    raw["opacity_entropy"] = combined_opacity_entropy(state["alpha"])
    nps_enabled = bool(config.get("hybrid_nps_enabled", False))
    if nps_enabled:
        if nps_fn is None:
            raise ValueError("NPS 开启时必须传入原 paper_losses.nps_loss")
        raw["nps"] = nps_fn(state["nps_features"], state["dynamic_mask"],
                            state["nps_distances"])
    else:
        # 关闭 arm 完全不调用 NPS，也不创建距离图或消耗随机数。
        raw["nps"] = reference.sum() * 0.0
    decay = lidar_decay_factor(state["iteration"], config.get("lidar_decay", 1.0))
    weights = {
        "global_rgb": 1.0,
        "dynamic_rgb": float(config.get("hybrid_lambda_dynamic_rgb", 5.0)),
        "sky_rgb": float(config.get("hybrid_lambda_sky_rgb", 5.0)),
        "static_rgb": float(config.get("hybrid_lambda_static_rgb", 1.0)),
        "sky": float(config.get("paper_lambda_sky", 0.05)),
        "depth": float(config.get("paper_lambda_depth", 0.1)) * decay,
        "dynamic_opacity": float(config.get("hybrid_lambda_dynamic_opacity", 100.0)),
        "static_duration": float(config.get("lambda_t_reg", 0.1)),
        "static_velocity": float(config.get("lambda_v_reg", 0.01)),
        "inverse_depth_smoothness": float(config.get("lambda_inv_depth", 0.1)),
        "velocity_smoothness": float(config.get("lambda_v_smooth", 0.1)),
        "opacity_entropy": float(config.get("lambda_opacity_entropy", 0.05)),
        "nps": float(config.get("paper_lambda_nps", 0.01)) if nps_enabled else 0.0,
    }
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights.values()):
        raise ValueError("全部损失权重必须为非负 finite 数")
    weighted = {name: value * weights[name] for name, value in raw.items()}
    total = sum(weighted.values(), reference.sum() * 0.0)
    duration_mean = state["static_t_map"].detach().abs().mean().item()
    result = HybridLossResult(total, raw, weighted, weights, {
        "hybrid_depth_decay": decay,
        "hybrid_nps_enabled": nps_enabled,
        "hybrid_static_duration_mean_abs": float(duration_mean),
        "hybrid_static_duration_guard_active": bool(duration_mean < time_eps),
        "hybrid_time_reg_eps": time_eps,
    })
    if not bool(torch.isfinite(total).detach().item()):
        raise FloatingPointError("Hybrid 总损失非 finite: " + str(result.log_values()))
    return result


def _gradient_metrics(gradient):
    if gradient is None:
        return {"connected": False, "finite": True, "l2": 0.0, "abs_mean": 0.0, "abs_max": 0.0}
    gradient = gradient.detach()
    finite = bool(torch.isfinite(gradient).all().item())
    return {"connected": True, "finite": finite,
            "l2": float(torch.linalg.vector_norm(gradient).item()),
            "abs_mean": float(gradient.abs().mean().item()),
            "abs_max": float(gradient.abs().max().item()) if gradient.numel() else 0.0}


def _gradient_slots(named_inputs, preserve_grad_tensors):
    """autograd.grad 会写 retained 非 leaf 的 .grad，必须显式恢复这些槽。"""
    unique = {}
    for tensor in (*named_inputs.values(), *preserve_grad_tensors):
        if isinstance(tensor, torch.Tensor) and (tensor.is_leaf or tensor.retains_grad):
            unique[id(tensor)] = tensor
    return [(tensor, tensor.grad, None if tensor.grad is None else tensor.grad.detach().clone())
            for tensor in unique.values()]


def _restore_gradient_slots(slots):
    # 恢复原对象身份；若 retained hook 原地累加过，也恢复原数值。
    with torch.no_grad():
        for tensor, original, saved in slots:
            if original is not None:
                original.copy_(saved)
            tensor.grad = original


def _term_gradients(value, inputs):
    if not value.requires_grad or not inputs:
        return tuple(None for _ in inputs)
    return torch.autograd.grad(value, inputs, allow_unused=True,
                               retain_graph=True, create_graph=False)


def diagnose_gradients(result: HybridLossResult, named_inputs: Mapping,
                       *, term_names: Sequence = None, preserve_grad_tensors=()):
    """在训练 backward 前读梯度量级；不采样、不 backward、不累积参数梯度。

    建议输入 static_feature/raw_static_alpha/velocity 等原图或参数。
    调用方须把 named_inputs 之外所有 retain_grad 中间量（renderer 的
    viewspace_points）放进 preserve_grad_tensors。返回值不持有 Tensor/图。
    先读未加权梯度，乘显式标量得到加权梯度，避免额外反传。
    """
    selected = tuple(result.raw) if term_names is None else tuple(term_names)
    active = {name: tensor for name, tensor in named_inputs.items() if tensor.requires_grad}
    slots = _gradient_slots(named_inputs, preserve_grad_tensors)
    values = {}
    try:
        for term in selected:
            gradients = _term_gradients(result.raw[term], tuple(active.values()))
            for name, tensor in named_inputs.items():
                gradient = gradients[tuple(active).index(name)] if name in active else None
                weighted = None if gradient is None else gradient * result.weights[term]
                for kind, metrics in (("raw", _gradient_metrics(gradient)),
                                      ("weighted", _gradient_metrics(weighted))):
                    for metric, value in metrics.items():
                        values["grad_hybrid/" + term + "/" + name + "/" + kind + "_" + metric] = value
        # 同一输入上比较量级，缺少 RGB 连接时不把零分母称作有效比率。
        if "nps" in selected and "static_rgb" in selected:
            for name in named_inputs:
                stem = "grad_hybrid/"
                denominator = values[stem + "static_rgb/" + name + "/weighted_l2"]
                numerator = values[stem + "nps/" + name + "/weighted_l2"]
                values[stem + "nps/" + name + "/ratio_to_static_rgb_valid"] = denominator > 0
                values[stem + "nps/" + name + "/weighted_l2_over_static_rgb"] = (
                    numerator / denominator if denominator > 0 else 0.0)
    finally:
        _restore_gradient_slots(slots)
    return values


def compare_objective_gradients(reference, candidate, named_inputs: Mapping, *,
                                prefix="rewrite", preserve_grad_tensors=()):
    """同一次前向比较旧/新目标，不据公式相似就认定改写影响小。"""
    active = {name: tensor for name, tensor in named_inputs.items() if tensor.requires_grad}
    slots = _gradient_slots(named_inputs, preserve_grad_tensors)
    values = {prefix + "/reference": float(reference.detach().item()),
              prefix + "/candidate": float(candidate.detach().item()),
              prefix + "/scalar_delta": float((candidate - reference).detach().item())}
    try:
        before = _term_gradients(reference, tuple(active.values()))
        after = _term_gradients(candidate, tuple(active.values()))
        for name, tensor in active.items():
            index = tuple(active).index(name)
            old = torch.zeros_like(tensor) if before[index] is None else before[index].detach()
            new = torch.zeros_like(tensor) if after[index] is None else after[index].detach()
            stem = prefix + "/" + name + "/"
            values[stem + "reference_l2"] = float(torch.linalg.vector_norm(old).item())
            values[stem + "candidate_l2"] = float(torch.linalg.vector_norm(new).item())
            values[stem + "delta_l2"] = float(torch.linalg.vector_norm(new - old).item())
            values[stem + "finite"] = bool(torch.isfinite(old).all().item() and torch.isfinite(new).all().item())
    finally:
        _restore_gradient_slots(slots)
    return values
