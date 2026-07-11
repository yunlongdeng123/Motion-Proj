"""对目标轨迹进行置信度加权的时间平滑。

我们通过逐通道最小化以下目标，将每条轨迹的框状态（中心 u,v 与对数尺度
log w, log h）投影到 ``E_obj`` 流形上：

    sum_k present_k * (x_k - data_k)^2  +  lambda * sum_k (x_{k+1} - 2x_k + x_{k-1})^2

这是一个线性（近似三对角）的最小二乘求解。缺失帧的数据权重为零，因此平滑器
会由正则项对其进行插值。这是二阶差分先验下 RTS 平滑的闭式类比。
"""
from __future__ import annotations

import torch

from ..auditor.state import Track


def _smooth_channel(data: torch.Tensor, weight: torch.Tensor, lam: float) -> torch.Tensor:
    """对单个 ``[K]`` 通道求解 (W + lam Dᵀ D) x = W data。"""
    k = data.shape[0]
    if k == 1:
        return data.clone()
    W = torch.diag(weight)
    # 二阶差分算子 D：[(K-2), K]
    if k >= 3:
        D = torch.zeros(k - 2, k, dtype=data.dtype, device=data.device)
        for i in range(k - 2):
            D[i, i] = 1.0
            D[i, i + 1] = -2.0
            D[i, i + 2] = 1.0
    else:  # k == 2 -> 一阶差分
        D = torch.zeros(1, k, dtype=data.dtype, device=data.device)
        D[0, 0] = 1.0
        D[0, 1] = -1.0
    A = W + lam * (D.transpose(0, 1) @ D)
    b = W @ data
    # 为完全缺席的通道（所有权重为 0）添加微小的岭项（ridge）
    A = A + 1e-6 * torch.eye(k, dtype=data.dtype, device=data.device)
    x = torch.linalg.solve(A, b)
    return x


def smooth_track(tr: Track, lam: float = 5.0, conf_floor: float = 0.1) -> Track:
    """返回 ``tr`` 的平滑副本（中心 + 对数尺度），所有帧均被填充。"""
    k = tr.xyxy.shape[0]
    present = tr.present.float()
    weight = present.clamp_min(0.0)
    # 用近似线性的填充为缺失帧的数据设定初值（用 0 即可；由正则项负责填充）
    center = tr.center.clone()
    logscale = torch.log(tr.scale.clamp_min(1e-3))
    # 将 NaN（缺席帧）替换为 数据 0 + 权重 0
    center = torch.nan_to_num(center, nan=0.0)
    logscale = torch.nan_to_num(logscale, nan=0.0)

    sc = torch.stack([_smooth_channel(center[:, i], weight, lam) for i in range(2)], dim=-1)
    sl = torch.stack([_smooth_channel(logscale[:, i], weight, lam) for i in range(2)], dim=-1)
    scale = torch.exp(sl)

    u, v = sc[:, 0], sc[:, 1]
    w2, h2 = scale[:, 0] * 0.5, scale[:, 1] * 0.5
    xyxy = torch.stack([u - w2, v - h2, u + w2, v + h2], dim=-1)

    # 只要原始轨迹在相邻帧有任何支持，平滑后的轨迹在该处即视为"存在"
    new_present = tr.present | _dilate_bool(tr.present)
    depth = torch.nan_to_num(tr.depth, nan=float(torch.nanmean(tr.depth)) if present.sum() > 0 else 0.0)
    return Track(
        instance_token=tr.instance_token,
        category=tr.category,
        xyxy=xyxy,
        depth=depth,
        present=new_present,
    )


def _dilate_bool(x: torch.Tensor) -> torch.Tensor:
    """沿时间的 1D 膨胀（步长 1），用于填补单帧空隙以便插值。"""
    out = x.clone()
    out[1:] = out[1:] | x[:-1]
    out[:-1] = out[:-1] | x[1:]
    return out


def smooth_tracks(tracks: list[Track], lam: float = 5.0) -> list[Track]:
    return [smooth_track(tr, lam=lam) for tr in tracks]
