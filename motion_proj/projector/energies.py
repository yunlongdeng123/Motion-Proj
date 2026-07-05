"""驾驶动力学能量项（方案第 5 节）。

它们定义了投影器所要逼近的流形。在 V1 中，投影本身是以闭式求解的
（自车光流静态渲染 + 轨迹的时间平滑），因此这些能量项主要用于诊断/报告，
以及让流形显式化、可测试。能量越低 = 动力学一致性越好。
"""
from __future__ import annotations

import torch

from ..auditor.state import MotionState, Track


def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def e_static(state: MotionState) -> torch.Tensor:
    """在可靠静态像素上，经自车运动补偿的鲁棒静态漂移。"""
    res = state.static_residual                 # [F,H,W,2]
    m = state.static_mask                        # [F,H,W]
    val = charbonnier(res).sum(-1) * m
    return val.sum() / m.sum().clamp_min(1.0)


def _second_difference(x: torch.Tensor) -> torch.Tensor:
    """对 ``[K,...]`` 沿时间（dim 0）计算 x[k+1]-2x[k]+x[k-1]。"""
    if x.shape[0] < 3:
        return torch.zeros_like(x[:0])
    return x[2:] - 2 * x[1:-1] + x[:-2]


def e_obj(tracks: list[Track]) -> torch.Tensor:
    """目标运动平滑性：惩罚框中心的加速度以及对数尺度的加加速度（jerk）。"""
    total = torch.zeros(())
    count = 0
    for tr in tracks:
        present = tr.present
        if present.sum() < 3:
            continue
        c = tr.center                                  # [K,2]
        logz = torch.log(tr.scale.clamp_min(1e-3))     # [K,2]
        acc = _second_difference(c)
        jerk = _second_difference(logz)
        # 仅统计完全存在的窗口
        w = (present[2:] & present[1:-1] & present[:-2]).float().unsqueeze(-1)
        total = total + (charbonnier(acc) * w).sum() + (charbonnier(jerk) * w).sum()
        count += int(w.sum())
    return total / max(count, 1)


def e_xview(state: MotionState) -> torch.Tensor:
    """跨视角一致性。V1 为单相机 -> 0（多视角的占位实现）。"""
    return torch.zeros((), device=state.depth.device)


def e_sup(support_flags: dict) -> torch.Tensor:
    """对无支持的新目标出现进行惩罚（基于计数）。"""
    n_unsupported = sum(int((~torch.as_tensor(v)).sum()) for v in support_flags.values()) if support_flags else 0
    return torch.tensor(float(n_unsupported))


def e_prior(tracks: list[Track]) -> torch.Tensor:
    """弱先验：抑制过大的逐步速度（对流形起正则化作用）。"""
    total = torch.zeros(())
    count = 0
    for tr in tracks:
        if tr.present.sum() < 2:
            continue
        v = tr.center[1:] - tr.center[:-1]
        w = (tr.present[1:] & tr.present[:-1]).float().unsqueeze(-1)
        total = total + (charbonnier(v) * w).sum()
        count += int(w.sum())
    return total / max(count, 1)


def e_dyn(
    state: MotionState,
    tracks: list[Track],
    support_flags: dict | None = None,
    weights: dict | None = None,
) -> dict:
    """返回各能量项及其加权总和（用于诊断）。"""
    w = {"static": 1.0, "obj": 1.0, "xview": 1.0, "sup": 0.1, "prior": 0.1}
    if weights:
        w.update(weights)
    terms = {
        "static": e_static(state),
        "obj": e_obj(tracks),
        "xview": e_xview(state),
        "sup": e_sup(support_flags or {}),
        "prior": e_prior(tracks),
    }
    total = sum(w[k] * float(terms[k]) for k in terms)
    terms["total"] = torch.tensor(total)
    return terms
