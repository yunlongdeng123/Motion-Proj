"""纯CPU时序身份证据选择；不读文件、不追踪GPU、不按颜色/类别/邻近猜ID。

previous和seed_support必须是调用方显式接受的同帧实拍观测。拒绝的新候选
只回退同时间戳previous；没有可信观测时mask=None、identity_unknown=True，
不能把它转成全零背景。强制seed分数不构成独立证据。入口接线由调用方负责。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Hashable, Mapping
import numpy as np


@dataclass(frozen=True)
class AcceptedObservation:
    mask: np.ndarray
    provenance: Any


@dataclass(frozen=True)
class TemporalCandidate:
    mask: np.ndarray | None
    visibility: float | None
    iou: float | None
    provenance: Any
    absent: bool = False
    weak_evidence: bool = False
    forced_confidence: bool = False
    geometry_support: np.ndarray | None = None


@dataclass(frozen=True)
class TemporalConfig:
    absent_threshold: float = .5
    min_visibility: float = .5
    min_iou: float = .5
    min_geometry_fraction: float = .8

    def __post_init__(self):
        values = (self.absent_threshold, self.min_visibility, self.min_iou, self.min_geometry_fraction)
        if not all(np.isfinite(x) and 0 <= x <= 1 for x in values) or self.absent_threshold > self.min_visibility:
            raise ValueError("Thresholds must be finite in [0,1], absent_threshold<=min_visibility")


@dataclass(frozen=True)
class FrameDecision:
    actor_id: Hashable
    timestamp: float
    status: str                         # 新候选的accepted/rejected/unknown
    reasons: tuple[str, ...]
    mask: np.ndarray | None
    chosen_kind: str                    # seed_support/candidate/previous/unknown
    chosen_provenance: Any
    identity_unknown: bool
    independent_confidence: bool
    connected_seeds: tuple
    blocked_by: tuple                   # (seed时间戳, 首次真实断链时间戳)


def _mask(value, shape=None):
    arr = np.asarray(value)
    if arr.ndim != 2 or arr.dtype.kind != "b" or min(arr.shape) < 1:
        raise ValueError("Masks must be nonzero-shape boolean (H,W) arrays")
    if shape is not None and arr.shape != shape:
        raise ValueError("All evidence must share one explicit image grid")
    return arr


def _score(value):
    return value is not None and np.isfinite(value) and 0 <= value <= 1


def _absent(candidate, cfg):
    return candidate.absent or (candidate.mask is not None and not candidate.mask.any()) or (
        not candidate.forced_confidence and _score(candidate.visibility) and candidate.visibility <= cfg.absent_threshold)


def select_track(actor_id: Hashable, candidates: Mapping[float, TemporalCandidate],
                 seed_support: Mapping[float, AcceptedObservation], *,
                 previous: Mapping[float, AcceptedObservation] | None = None,
                 config: TemporalConfig | None = None) -> dict[float, FrameDecision]:
    """从每个显式可信seed独立前后遍历。只有新显式seed可重建断开的证据链。

    缺测/weak/几何不相容不伪造身份或背景，也不等同于已观测的实际empty/absent。
    几何支持仅用于否决不相容候选；几何相容本身不能提供身份或可见性证明。
    """
    hash(actor_id)                      # ID只作opaque key，绝不数值排序
    cfg, prior = config or TemporalConfig(), previous or {}
    times = sorted(set(candidates) | set(seed_support) | set(prior))
    if any(not np.isfinite(t) for t in times):
        raise ValueError("Timestamps must be finite numeric values")
    arrays = [x.mask for x in [*candidates.values(), *seed_support.values(), *prior.values()] if x.mask is not None]
    shape = _mask(arrays[0]).shape if arrays else None
    for arr in arrays:
        _mask(arr, shape)
    for obs in [*seed_support.values(), *prior.values()]:
        if obs.provenance is None or not _mask(obs.mask, shape).any():
            raise ValueError("Accepted observations need actual nonempty masks and explicit provenance")
    for candidate in candidates.values():
        if candidate.geometry_support is not None:
            _mask(candidate.geometry_support, shape)
    connected, blocked = {t: [] for t in times}, {t: [] for t in times}
    for seed in sorted(seed_support):
        index = times.index(seed)
        for step in (-1, 1):
            barrier = None
            for j in range(index, -1 if step < 0 else len(times), step):
                t = times[j]
                candidate = candidates.get(t)
                if t != seed and barrier is None and candidate is not None and _absent(candidate, cfg):
                    barrier = t
                if barrier is None:
                    if seed not in connected[t]:
                        connected[t].append(seed)
                else:
                    blocked[t].append((seed, barrier))
    result = {}
    for t in times:
        candidate, reasons = candidates.get(t), []
        if t in seed_support:
            chosen, kind, status = seed_support[t], "seed_support", "accepted"
            reasons = ["explicit_seed_support", "seed_scores_are_not_independent_evidence"]
        else:
            if candidate is None or candidate.mask is None:
                reasons.append("candidate_unobserved")
            if candidate is not None:
                if _absent(candidate, cfg):
                    reasons.append("actual_empty_or_absent")
                if not connected[t]:
                    reasons.append("disconnected_after_actual_absence" if blocked[t] else "no_explicit_seed_support")
                if candidate.forced_confidence:
                    reasons.append("forced_confidence_is_not_independent_evidence")
                if candidate.provenance is None:
                    reasons.append("candidate_provenance_missing")
                if candidate.weak_evidence:
                    reasons.append("external_review_or_weak_evidence")
                if not _score(candidate.visibility) or candidate.visibility < cfg.min_visibility:
                    reasons.append("visibility_unconfirmed")
                if not _score(candidate.iou) or candidate.iou < cfg.min_iou:
                    reasons.append("iou_unconfirmed")
                if candidate.mask is not None and candidate.mask.any() and candidate.geometry_support is not None:
                    if candidate.geometry_support[candidate.mask].mean() < cfg.min_geometry_fraction:
                        reasons.append("geometry_incompatible")
            unobserved = candidate is None or (candidate.mask is None and not _absent(candidate, cfg))
            status = "accepted" if not reasons else ("unknown" if unobserved else "rejected")
            chosen = AcceptedObservation(candidate.mask, candidate.provenance) if status == "accepted" else prior.get(t)
            kind = "candidate" if status == "accepted" else ("previous" if chosen else "unknown")
            if chosen and kind == "previous":
                reasons.append("fallback_same_timestamp_accepted_observation")
            elif chosen is None:
                reasons.append("identity_unknown_not_background")
        result[t] = FrameDecision(actor_id, t, status, tuple(reasons),
            None if chosen is None else chosen.mask.copy(), kind,
            None if chosen is None else chosen.provenance, chosen is None,
            kind == "candidate", tuple(connected[t]), tuple(blocked[t]))
    return result


@dataclass(frozen=True)
class OverlapResult:
    owned_masks: dict
    unknown_mask: np.ndarray
    foreground_mask: np.ndarray
    reasons: dict


def resolve_actual_overlap(masks: Mapping[Hashable, np.ndarray | None], depths: Mapping, *,
                           unknown_mask=None, shape=None, depth_tolerance=1e-6) -> OverlapResult:
    """仅两个实际轮廓重叠时比较同一深度约定的数据；无mask的近OBB不遮挡。

    不推断部件归属，不按邻近合并ID。深度缺测/非正/平局保留unknown；调用方
    显式给入的unknown实际前景也保留unknown，不变成背景或自动归到邻近actor。
    """
    if not np.isfinite(depth_tolerance) or depth_tolerance < 0:
        raise ValueError("depth_tolerance must be finite and nonnegative")
    active = {actor: _mask(mask) for actor, mask in masks.items() if mask is not None}
    if shape is None:
        shape = next(iter(active.values())).shape if active else (_mask(unknown_mask).shape if unknown_mask is not None else None)
    if shape is None:
        raise ValueError("Supply shape when no actual masks are available")
    shape = _mask(np.zeros(shape, dtype=bool)).shape
    unknown = np.zeros(shape, bool) if unknown_mask is None else _mask(unknown_mask, shape).copy()
    for mask in active.values():
        _mask(mask, shape)
    if not active:
        return OverlapResult({}, unknown, unknown.copy(), {"actual_overlap_pixels": 0})
    ids = list(active)                  # ID重命名不改变结果；平局不靠ID大小决定
    stack = np.stack(list(active.values()))
    count = stack.sum(0)
    values = np.full(stack.shape, np.inf)
    invalid = np.zeros(shape, bool)
    for index, actor in enumerate(ids):
        value = np.asarray(depths.get(actor, np.nan), dtype=float)
        if value.ndim != 0 and value.shape != shape:
            raise ValueError("Depth must be scalar or match the mask grid")
        good = np.isfinite(value) & (value > 0)
        values[index] = np.where(stack[index] & good, value, np.inf)
        invalid |= stack[index] & ~good
    best = values.min(0)
    winners = stack & np.isclose(values, best, atol=depth_tolerance, rtol=0) & np.isfinite(best)
    overlap = count > 1
    ambiguous = overlap & (invalid | (winners.sum(0) != 1))
    unknown |= ambiguous
    owned = {actor: ((stack[index] & ~overlap) | (winners[index] & overlap & ~ambiguous)) & ~unknown
             for index, actor in enumerate(ids)}
    return OverlapResult(owned, unknown, (count > 0) | unknown,
        {"actual_overlap_pixels": int(overlap.sum()), "unresolved_overlap_pixels": int(ambiguous.sum())})
