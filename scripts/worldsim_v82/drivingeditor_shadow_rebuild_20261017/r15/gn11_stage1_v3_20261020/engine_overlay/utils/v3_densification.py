"""GN11 v3 的分支增密统计。

联合与独立分支的屏幕梯度不相加。每个 Gaussian、每个采样视图只记一次，
在分支梯度和联合梯度之间取较强者；点数上限与原生增密调度保持不变。
"""

import torch


def _xy_gradient_norm(proxy, expected_points, label):
    if proxy.shape[0] != expected_points:
        raise ValueError(f"{label} proxy/topology mismatch")
    if proxy.grad is None:
        raise RuntimeError(f"missing {label} raster screen gradient")
    if proxy.grad.shape != proxy.shape:
        raise RuntimeError(f"invalid {label} raster screen gradient shape")
    gradient = torch.linalg.vector_norm(proxy.grad[:, :2], dim=-1, keepdim=True)
    if not torch.isfinite(gradient).all():
        raise FloatingPointError(f"non-finite {label} raster screen gradient")
    return gradient.detach()


def accumulate_max_screen_gradient(
    model,
    branch_proxy,
    branch_visibility,
    branch_radii,
    joint_proxy,
    joint_visibility,
    joint_radii,
):
    """用逐点 max 融合分支/联合需求，避免重复计数和容量翻倍。"""
    point_count = int(model.get_xyz.shape[0])
    branch_gradient = _xy_gradient_norm(branch_proxy, point_count, "branch")
    joint_gradient = _xy_gradient_norm(joint_proxy, point_count, "joint")
    for name, value in (
        ("branch_visibility", branch_visibility),
        ("joint_visibility", joint_visibility),
        ("branch_radii", branch_radii),
        ("joint_radii", joint_radii),
    ):
        if value.shape[0] != point_count:
            raise ValueError(f"{name}/topology mismatch")

    branch_visible = branch_visibility.bool()
    joint_visible = joint_visibility.bool()
    active = branch_visible | joint_visible
    both = branch_visible & joint_visible
    chosen = torch.where(
        both[:, None],
        torch.maximum(branch_gradient, joint_gradient),
        torch.where(branch_visible[:, None], branch_gradient, joint_gradient),
    )
    chosen_radii = torch.maximum(branch_radii, joint_radii)

    model.max_radii2D[active] = torch.maximum(
        model.max_radii2D[active], chosen_radii[active]
    )
    model.xyz_gradient_accum[active] += chosen[active]
    # 同一个训练视图只增加一次分母；绝不把两个渲染统计直接相加。
    model.denom[active] += 1
    if getattr(model, "_t", None) is not None and model._t.requires_grad:
        if model._t.grad is None:
            raise RuntimeError("missing dynamic temporal gradient")
        if not torch.isfinite(model._t.grad).all():
            raise FloatingPointError("non-finite dynamic temporal gradient")
        model.t_gradient_accum[active] += model._t.grad.detach()[active]
    else:
        model.t_gradient_accum.zero_()

    def mean_or_zero(value, mask):
        weights = mask[:, None].to(value)
        return (value * weights).sum() / weights.sum().clamp_min(1.0)

    return {
        "active_points": active.sum(),
        "branch_only_points": (branch_visible & ~joint_visible).sum(),
        "joint_only_points": (joint_visible & ~branch_visible).sum(),
        "both_points": both.sum(),
        "joint_selected_points": (both & (joint_gradient[:, 0] > branch_gradient[:, 0])).sum(),
        "branch_gradient_mean": mean_or_zero(branch_gradient, branch_visible),
        "joint_gradient_mean": mean_or_zero(joint_gradient, joint_visible),
        "selected_gradient_mean": mean_or_zero(chosen, active),
    }
