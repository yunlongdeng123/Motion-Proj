"""Auditor / projector 诊断（第 2-3 周的成功标准）。

- ``rank_clips``：按静态漂移 / 轨迹加速度对片段排序，让明显的失败样本
  浮到最前面。
- ``corruption_sensitivity``：注入合成的静态漂移，验证 auditor 的漂移分数
  会随之上升（即 auditor 能够 *检测* 出失败）。
- ``projection_quality``：投影后的目标相较于输入降低了物体轨迹能量的片段
  所占比例（第 3 周的“比输入更干净”标准）。
"""
from __future__ import annotations

import torch

from ..auditor import MotionAuditor
from ..auditor.state import MotionState


def rank_clips(dataset, auditor: MotionAuditor, n: int | None = None) -> list[tuple[str, float]]:
    n = n or len(dataset)
    scored = []
    for i in range(min(n, len(dataset))):
        s = dataset[i]
        state = auditor.audit(s)
        scored.append((s["sample_id"], auditor.static_drift_score(state)))
    return sorted(scored, key=lambda kv: -kv[1])


def corruption_sensitivity(state: MotionState, jitter: float = 3.0) -> dict:
    """向观测到的静态光流中加入合成漂移，确认分数会随之上升。"""
    base = _drift(state)
    noisy = MotionState(
        u_static=state.u_static + jitter,
        u_ego=state.u_ego,
        static_mask=state.static_mask,
        flow_conf=state.flow_conf,
        depth=state.depth,
        tracks=state.tracks,
        meta=state.meta,
    )
    corrupted = _drift(noisy)
    return {"clean_drift": base, "corrupted_drift": corrupted, "detected": corrupted > base}


def _drift(state: MotionState) -> float:
    res = state.static_residual.norm(dim=-1) * state.static_mask
    return float(res.sum() / state.static_mask.sum().clamp_min(1.0))


def projection_quality(cache_metadatas: list[dict]) -> dict:
    """在缓存元数据上聚合：E_obj 得到改善的片段所占比例。"""
    improved, total, deltas = 0, 0, []
    for m in cache_metadatas:
        e = m.get("energies", {})
        if "obj_before" in e and "obj_after" in e:
            total += 1
            d = e["obj_before"] - e["obj_after"]
            deltas.append(d)
            improved += int(d > 0)
    frac = improved / max(total, 1)
    mean_delta = sum(deltas) / max(len(deltas), 1)
    return {"clips": total, "frac_improved": frac, "mean_obj_reduction": mean_delta}
