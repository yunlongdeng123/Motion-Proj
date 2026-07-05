"""目标外观的支持性过滤（方案第 5 节，E_sup）。

一个 *新出现* 的外观（目标在帧 k 存在但在 k-1 不存在）在满足以下任一条件时
被视为有支持：
  - 先前支持（prev support）：在近期的前若干帧中出现过（重检测 / 短暂遮挡）；
  - 边界支持（border support）：从图像边界进入（合理的场景入场）；
  - （跨视角 / 遮挡边界支持：未来工作 / 多视角）。

对于无支持的新出现外观，*不会* 用凭空生成的像素来修复：投影器会将其掩码剔除
（并可将其导向 replay mining）。
"""
from __future__ import annotations

import torch

from ..auditor.state import Track


def classify_support(
    tracks: list[Track],
    hw: tuple[int, int],
    border_frac: float = 0.04,
    prev_window: int = 2,
) -> dict[str, torch.Tensor]:
    """返回 ``{instance_token: supported[K] bool}``（可修复处为 True）。"""
    h, w = hw
    bx = border_frac * w
    by = border_frac * h
    flags: dict[str, torch.Tensor] = {}
    for tr in tracks:
        k = tr.present.shape[0]
        supported = torch.ones(k, dtype=torch.bool)
        for t in range(k):
            if not bool(tr.present[t]):
                supported[t] = False
                continue
            is_new = t == 0 or not bool(tr.present[t - 1])
            if not is_new:
                continue  # 持续存在的目标始终视为有支持
            prev_ok = any(bool(tr.present[max(0, t - dt)]) for dt in range(1, prev_window + 1)) and t > 0
            box = tr.xyxy[t]
            border_ok = bool(
                (box[0] <= bx) or (box[1] <= by) or (box[2] >= w - 1 - bx) or (box[3] >= h - 1 - by)
            )
            supported[t] = prev_ok or border_ok or (t == 0)
        flags[tr.instance_token] = supported
    return flags
