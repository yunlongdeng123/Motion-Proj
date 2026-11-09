"""逐帧 CPU 数组决策；不运行网络、不改 RGB/loss、不接收或删除持久 3D 状态。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence
import numpy as np
try:
    from .v44_inputs import (IdentityRegistry, make_label_masks, candidate_observation,
                             preserve_supported_observations)
    from .v44_visibility import Evidence
    from .v44_temporal import AcceptedObservation, FrameDecision, resolve_actual_overlap
except ImportError:  # 独立暂存目录也可运行 CPU 测试。
    from v44_inputs import (IdentityRegistry, make_label_masks, candidate_observation,
                            preserve_supported_observations)
    from v44_visibility import Evidence
    from v44_temporal import AcceptedObservation, FrameDecision, resolve_actual_overlap


@dataclass(frozen=True)
class FrameMaskEvidence:
    candidate: Optional[np.ndarray] = None
    candidate_provenance: Any = None
    candidate_accepted: bool = False       # 来自独立同帧观测/QA，不是强制 seed 分数
    weak_candidate: bool = False
    positive_xy: Sequence = ()
    accepted_same_frame: Optional[AcceptedObservation] = None
    accepted_timestamp_s: Optional[float] = None
    retention_support: Optional[np.ndarray] = None  # 调用方证据图，不内置语义/场景例外
    visibility_state: Optional[np.ndarray] = None  # v44_visibility.Evidence 输出
    measured_depth: Optional[np.ndarray] = None   # 实际表面深度，不接受标量 OBB 距离
    measured_depth_valid: Optional[np.ndarray] = None
    depth_provenance: Any = None
    temporal_decision: Optional[FrameDecision] = None
    candidate_vehicle_support: Optional[np.ndarray] = None  # 独立确认“是车辆”，不等于身份确认


def _mask(value, shape, name):
    a = np.asarray(value)
    if a.shape != shape or a.dtype.kind != 'b':
        raise ValueError(name + ' 须为同一像素网格上的 bool H,W')
    return a.copy()


def _labels(resolved, registry):
    owner = np.zeros(resolved.foreground_mask.shape, np.int64)
    for actor, mask in resolved.owned_masks.items():
        owner[mask] = actor
    owner[resolved.unknown_mask] = -1
    return make_label_masks(resolved.foreground_mask, owner, registry)


def prepare_frame_masks(evidence: Mapping[int, FrameMaskEvidence], registry: IdentityRegistry, *,
                        timestamp_s: float, shape, frame_id=None, vehicle_foreground=None,
                        certified_nonvehicle_occluder: Optional[AcceptedObservation] = None,
                        depth_tolerance=1e-6):
    """返回 (LabelMasks, trace)，仅 training_ready=True 时可作完整 raw-alpha GT。

    候选身份只由显式独立接受/Temporal FrameDecision/同帧已接受观测授权。
    无正点仅记未决；独立确认是车辆但身份未知时才设 foreground、owner=-1。
    未确认是否车辆的 SAM 区域另记 semantic_unresolved_regions，阻止训练；
    不能把它解释为已知背景或当作 ignore mask 改写 MSE 域/权重。静态实物遮挡只剪
    当前可见 mask；持久 actor/隐藏几何完全不在本接口的修改域内。实际深度
    仅在实际轮廓交叠像素竞争；缺测/平局保持 unknown，绝不用近 OBB 占位。
    """
    shape = tuple(shape)
    if len(shape) != 2 or any(type(x) is not int or x < 1 for x in shape) or not np.isfinite(timestamp_s):
        raise ValueError('显式 H,W 和实际秒 timestamp 无效')
    zero = np.zeros(shape, bool)
    vehicle = zero if vehicle_foreground is None else _mask(vehicle_foreground, shape, 'vehicle_foreground')
    static = zero.copy()
    if certified_nonvehicle_occluder is not None:
        if certified_nonvehicle_occluder.provenance is None:
            raise ValueError('非车辆实物遮挡须有独立认证 provenance')
        static = _mask(certified_nonvehicle_occluder.mask, shape, 'certified nonvehicle occluder')
    known, depth, prior, support = {}, {}, {}, {}
    uncertain = vehicle & ~static; semantic_pending = zero.copy(); pending_by_actor = {}; actor_trace = {}
    for actor, item in evidence.items():
        if type(actor) is not int or actor not in registry.actor_to_track:
            raise ValueError('actor ID 必须使用显式登记表，不能按 alias 重新排序')
        if type(item.candidate_accepted) is not bool or type(item.weak_candidate) is not bool:
            raise ValueError('接受/弱证据标志须为显式 bool')
        candidate, candidate_meta = None, dict(positive_count=0, positive_on_mask=0, support_status='unobserved')
        if item.candidate is not None:
            candidate = _mask(item.candidate, shape, 'candidate')
            candidate, candidate_meta = candidate_observation(candidate, item.positive_xy)
        candidate_vehicle = (zero if item.candidate_vehicle_support is None
                             else _mask(item.candidate_vehicle_support, shape, 'independent candidate vehicle support'))
        confirmed_vehicle = vehicle | candidate_vehicle
        pending_by_actor[actor] = zero.copy()
        if candidate is not None:
            uncertain |= candidate & confirmed_vehicle & ~static
            pending_by_actor[actor] = candidate & ~confirmed_vehicle & ~static
            semantic_pending |= pending_by_actor[actor]
        accepted = item.accepted_same_frame
        if accepted is not None:
            if (item.accepted_timestamp_s is None or item.accepted_timestamp_s != timestamp_s
                    or accepted.provenance is None):
                raise ValueError('旧观测须明确同一实际 timestamp 和接受来源，不能跨帧回填')
            accepted_mask = _mask(accepted.mask, shape, 'accepted same-frame observation')
        else:
            accepted_mask = None
        if item.retention_support is not None and accepted_mask is None:
            raise ValueError('保留支持图必须绑定同帧已接受观测，不能凭支持图造轮廓')
        state = np.zeros(shape, np.uint8) if item.visibility_state is None else np.asarray(item.visibility_state)
        if state.shape != shape or state.dtype.kind not in 'iu' or not np.isin(state, [int(s) for s in Evidence]).all():
            raise ValueError('visibility_state 不是同网格的 Evidence 状态图')
        veto = state == Evidence.MEASURED_FOREGROUND
        if (item.measured_depth is None) != (item.measured_depth_valid is None):
            raise ValueError('实际深度必须同时提供逐像素可靠性图')
        measured = np.full(shape, np.nan)
        if item.measured_depth is not None:
            z = np.asarray(item.measured_depth, dtype=float)
            valid = _mask(item.measured_depth_valid, shape, 'measured_depth_valid')
            if z.shape != shape or (valid.any() and item.depth_provenance is None):
                raise ValueError('仅接受有来源的实际 H,W 表面深度，不能用标量框距离')
            measured[valid & np.isfinite(z) & (z > 0)] = z[valid & np.isfinite(z) & (z > 0)]
        depth[actor] = measured
        chosen, kind, provenance = None, 'unknown', None
        decision = item.temporal_decision
        if decision is not None:
            if decision.actor_id != actor or decision.timestamp != timestamp_s:
                raise ValueError('Temporal 决策与当前 actor/time 不一致')
            if decision.mask is not None and not decision.identity_unknown:
                if decision.chosen_provenance is None or decision.chosen_kind not in ('seed_support', 'candidate', 'previous'):
                    raise ValueError('已接受 Temporal mask 缺少实际来源')
                if not (item.weak_candidate and decision.chosen_kind == 'candidate'):
                    chosen = _mask(decision.mask, shape, 'temporal chosen mask')
                    kind, provenance = 'temporal_' + decision.chosen_kind, decision.chosen_provenance
        elif item.candidate_accepted and not item.weak_candidate:
            if candidate is None or item.candidate_provenance is None:
                raise ValueError('接受候选须提供实际 mask 与独立来源')
            chosen, kind, provenance = candidate, 'accepted_candidate', item.candidate_provenance
        if chosen is None and accepted_mask is not None:
            chosen, kind, provenance = accepted_mask, 'accepted_same_frame_fallback', accepted.provenance
        if chosen is not None:
            known[actor] = chosen & ~static & ~veto
            uncertain |= chosen & veto & ~static
        if accepted_mask is not None and item.retention_support is not None:
            support[actor] = _mask(item.retention_support, shape, 'external retention support')
            prior[actor] = accepted_mask & support[actor] & ~static & ~veto
        actor_trace[actor] = dict(chosen_kind=kind, chosen_provenance=provenance,
            candidate_point_support=candidate_meta, candidate_pixels=0 if candidate is None else int(candidate.sum()),
            independently_confirmed_candidate_vehicle_pixels=0 if candidate is None else int((candidate & confirmed_vehicle & ~static).sum()),
            current_visible_pixels=0 if chosen is None else int((chosen & ~static & ~veto).sum()),
            measured_foreground_pixels=int(veto.sum()), depth_provenance=item.depth_provenance,
            temporal_reasons=() if decision is None else decision.reasons)
    resolved = resolve_actual_overlap(known, depth, shape=shape, depth_tolerance=depth_tolerance)
    labels = _labels(resolved, registry)
    restored_trace = dict(actual_overlap_pixels=0, unresolved_overlap_pixels=0)
    if prior:
        free = labels.owner == 0
        preserve_masks = {actor: mask & free for actor, mask in prior.items()}
        retained = resolve_actual_overlap(preserve_masks, depth, shape=shape, depth_tolerance=depth_tolerance)
        prior_labels = _labels(retained, registry)
        labels = preserve_supported_observations(labels, prior_labels, prior_labels.foreground, registry)
        restored_trace = retained.reasons
        for actor in prior:
            actor_trace[actor]['supported_old_pixels_restored'] = int(retained.owned_masks[actor].sum())
    fill_unknown = uncertain & (labels.owner == 0) & ~static
    owner = labels.owner.copy(); owner[fill_unknown] = -1
    foreground = labels.foreground | fill_unknown
    owner[static] = 0; foreground[static] = False
    labels = make_label_masks(foreground, owner, registry)
    semantic_pending &= ~labels.foreground & ~static
    training_ready = not bool(semantic_pending.any())
    for actor in evidence:
        actor_trace[actor]['semantic_unresolved_pixels'] = int((pending_by_actor[actor] & semantic_pending).sum())
    trace = dict(frame_id=frame_id, timestamp_s=float(timestamp_s), actors=actor_trace,
        overlap=resolved.reasons, supported_preservation_overlap=restored_trace,
        certified_static_occluder_pixels=int(static.sum()),
        certified_static_provenance=None if certified_nonvehicle_occluder is None else certified_nonvehicle_occluder.provenance,
        binary_dynamic_pixels=int(labels.foreground.sum()), unknown_identity_pixels=int(labels.unknown_identity.sum()),
        semantic_unresolved_regions=semantic_pending.copy(), semantic_unresolved_pixels=int(semantic_pending.sum()),
        training_ready=training_ready, binary_dynamic_mask_is_training_gt=training_ready,
        training_blockers=() if training_ready else ('candidate_vehicle_semantics_unresolved',),
        background_semantics='final' if training_ready else 'unresolved_regions_are_not_known_background',
        persistent_actor_ids=tuple(evidence), persistent_3d_state_modified=False,
        loss_domain='full_image_raw_alpha_mse', loss_weights_modified=False,
        unknown_vehicle_remains_foreground=True)
    return labels, trace
