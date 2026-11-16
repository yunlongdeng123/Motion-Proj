"""场景无关的 CPU 输入契约；不导入 renderer，不推断身份或改写 canonical 坐标。"""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Sequence
import numpy as np


def _strings(values, name):
    if isinstance(values, (str, bytes)):
        raise ValueError(name + ' 须为字符串序列，不能传单个字符串')
    values = tuple(values)
    if not values or any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values):
        raise ValueError(name + ' 必须为非空、无重复的原始字符串集合')
    return values


def _binary(value, name):
    a = np.asarray(value)
    if a.ndim < 2 or not a.size or not np.isin(a, [0, 1, 255]).all():
        raise ValueError(name + ' 必须为非空二值像素图')
    return a.astype(bool)


@dataclass(frozen=True)
class IdentityRegistry:
    """ID 必须由调用方显式登记；字符串改名/排序不能重新分配 ID。"""
    track_to_actor: Mapping[str, int]

    def __post_init__(self):
        mapping = dict(self.track_to_actor)
        if mapping:
            _strings(mapping, 'track_ids')
        if any(type(v) is not int or v <= 0 or v > np.iinfo(np.int64).max for v in mapping.values()):
            raise ValueError('actor ID 必须是 int64 范围内正整数')
        if len(set(mapping.values())) != len(mapping):
            raise ValueError('track/actor 必须为双射；部件关系另行登记')
        object.__setattr__(self, 'track_to_actor', MappingProxyType(mapping))
        object.__setattr__(self, 'actor_to_track', MappingProxyType({v: k for k, v in mapping.items()}))

    def manifest(self):
        return dict(schema_version=1, track_to_actor=dict(self.track_to_actor),
                    actor_to_track={str(k): v for k, v in self.actor_to_track.items()})

    @classmethod
    def from_manifest(cls, value):
        registry = cls(value['track_to_actor'])
        if (type(value.get('schema_version')) is not int or value['schema_version'] != 1
                or value.get('actor_to_track') != registry.manifest()['actor_to_track']):
            raise ValueError('登记表版本或双向映射不一致')
        return registry


@dataclass(frozen=True)
class SceneInputs:
    """时间单位秒，camera_to_world 为 SE3；默认全部帧，无内置留出。"""
    images: np.ndarray
    camera_to_world: np.ndarray
    timestamps_s: np.ndarray
    frame_ids: Sequence[str]
    raw_track_ids: Sequence[str]
    registry: IdentityRegistry

    def __post_init__(self):
        ids = _strings(self.frame_ids, 'frame_ids'); n = len(ids)
        images = np.asarray(self.images); cameras = np.asarray(self.camera_to_world, dtype=float)
        times = np.asarray(self.timestamps_s, dtype=float)
        tracks = tuple(self.raw_track_ids)
        if tracks:
            _strings(tracks, 'raw_track_ids')
        if set(tracks) != set(self.registry.track_to_actor):
            raise ValueError('原始 track 集合必须与显式登记表一致')
        if images.ndim != 4 or images.shape[0] != n or images.shape[-1] != 3 or min(images.shape[1:3]) <= 0:
            raise ValueError('images 必须为与 frame_ids 对齐的 N,H,W,3')
        if not np.isfinite(images).all() or cameras.shape != (n, 4, 4) or times.shape != (n,):
            raise ValueError('图像/相机/时间长度或数值不合法')
        if not np.isfinite(cameras).all() or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
            raise ValueError('相机须有限，实际 timestamps_s 须严格递增')
        rotation = cameras[:, :3, :3]
        if (not np.allclose(cameras[:, 3], [0, 0, 0, 1], atol=1e-6, rtol=0)
                or not np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=1e-5, rtol=0)
                or not np.allclose(np.linalg.det(rotation), 1, atol=1e-5, rtol=0)):
            raise ValueError('camera_to_world 必须为刚体 SE3，不能含尺度/反射')
        for name, value in [('images', images), ('camera_to_world', cameras), ('timestamps_s', times)]:
            object.__setattr__(self, name, value)
        object.__setattr__(self, 'frame_ids', ids); object.__setattr__(self, 'raw_track_ids', tracks)

    def indices(self, frame_ids: Optional[Sequence[str]] = None):
        if frame_ids is None:
            return tuple(range(len(self.frame_ids)))
        requested = _strings(frame_ids, 'selected frame_ids')
        lookup = {f: i for i, f in enumerate(self.frame_ids)}
        if not set(requested) <= set(lookup):
            raise ValueError('请求了不存在的帧')
        indices = tuple(lookup[f] for f in requested)
        if any(b <= a for a, b in zip(indices, indices[1:])):
            raise ValueError('显式选帧必须保持真实时间顺序')
        return indices


@dataclass(frozen=True)
class LabelMasks:
    owner: np.ndarray
    foreground: np.ndarray

    def __post_init__(self):
        fg = _binary(self.foreground, 'foreground'); owner = np.asarray(self.owner)
        if owner.shape != fg.shape or owner.dtype.kind != 'i' or (owner < -1).any():
            raise ValueError('owner 须为有符号整数且同形：-1未知、0背景、正数已知')
        if not np.array_equal(owner != 0, fg):
            raise ValueError('前景必须具有已知 owner 或 -1，不能隐式变背景')
        object.__setattr__(self, 'owner', owner.astype(np.int64, copy=True))
        object.__setattr__(self, 'foreground', fg.copy())

    @property
    def binary_dynamic_mask(self):
        return self.foreground.copy()

    @property
    def unknown_identity(self):
        return self.owner == -1

    def export(self):
        return dict(owner=self.owner.copy(), binary_dynamic_mask=np.uint8(self.foreground) * 255,
                    unknown_identity=np.uint8(self.unknown_identity) * 255)


def make_label_masks(foreground, owner=None, registry=None):
    fg = _binary(foreground, 'foreground')
    if owner is None:
        own = np.where(fg, -1, 0).astype(np.int64)
    else:
        own = np.asarray(owner)
        if own.dtype.kind != 'i' or own.shape != fg.shape or ((own != 0) & ~fg).any():
            raise ValueError('owner/foreground 形状、整数类型或背景归属冲突')
        own = own.astype(np.int64, copy=True); own[fg & (own == 0)] = -1
    known = set(np.unique(own[own > 0]).tolist())
    if known and (registry is None or not known <= set(registry.actor_to_track)):
        raise ValueError('已知 owner 未在显式登记表中注册')
    return LabelMasks(own, fg)


def candidate_observation(mask, positive_xy=()):
    """候选没有同帧正点时标未决并保留，不把缺证据当作空轮廓。"""
    candidate = _binary(mask, 'candidate')
    if candidate.ndim != 2:
        raise ValueError('candidate 须为 H,W')
    points = np.asarray(positive_xy, dtype=float)
    if points.size == 0:
        points = np.empty((0, 2))
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('positive_xy 须为有限的 N,2 像素坐标')
    count = 0
    for x, y in np.rint(points).astype(np.int64):
        count += int(0 <= y < candidate.shape[0] and 0 <= x < candidate.shape[1] and candidate[y, x])
    return candidate.copy(), dict(positive_count=len(points), positive_on_mask=count,
                                  support_status='measured' if count else 'unresolved', inferred=False)


def preserve_supported_observations(current, prior, support, registry):
    """支持图由调用方提供；只补背景空位，不覆写未知/已知 owner。"""
    support = _binary(support, 'external support')
    if support.shape != current.owner.shape or prior.owner.shape != current.owner.shape:
        raise ValueError('观测/支持图尺寸不一致')
    take = (current.owner == 0) & prior.foreground & support
    owner = current.owner.copy(); owner[take] = prior.owner[take]
    return make_label_masks(current.foreground | take, owner, registry)


def validate_component_groups(groups, registry):
    """仅接受显式证据；保留部件原姿态，不按近邻合并或拼局部坐标。"""
    result = []; membership = set()
    for group in groups:
        parent = group['parent_track']; members = _strings(group['member_tracks'], 'member_tracks')
        evidence = group.get('evidence')
        provided = (bool(evidence.strip()) if isinstance(evidence, str)
                    else isinstance(evidence, Mapping) and any(bool(v) for v in evidence.values()))
        if (parent not in members or len(members) < 2 or not set(members) <= set(registry.track_to_actor)
                or set(members) & membership or not provided):
            raise ValueError('部件组须注册、互不重叠且包含明确证据')
        membership.update(members)
        result.append(dict(parent_track=parent, member_tracks=members, evidence=evidence,
                           pose_policy='retain_member_transforms'))
    return tuple(result)


def extrapolate_short_gap(timestamps_s, positions, query_timestamps_s, max_gap_s=.3, fit_points=4):
    """仅拟合位置/已解缠向量；不线性外推旋转矩阵，结果绝不是 GT。"""
    t = np.asarray(timestamps_s, dtype=float); p = np.asarray(positions, dtype=float)
    q = np.atleast_1d(np.asarray(query_timestamps_s, dtype=float))
    if (t.ndim != 1 or len(t) < 2 or p.ndim != 2 or p.shape[0] != len(t) or not p.shape[1]
            or q.ndim != 1 or not q.size or not all(np.isfinite(a).all() for a in (t, p, q))
            or (np.diff(t) <= 0).any() or not np.isfinite(max_gap_s) or max_gap_s < 0
            or type(fit_points) is not int or fit_points < 2):
        raise ValueError('外推需有效实际秒时间、至少两观测与有限位置向量')
    values = []; metadata = []; k = min(fit_points, len(t))
    for query in q:
        if t[0] <= query <= t[-1]:
            raise ValueError('只外推观测区间外短缺口；不能把已有观测标为 inferred')
        window = slice(0, k) if query < t[0] else slice(len(t) - k, len(t))
        boundary = t[0] if query < t[0] else t[-1]; horizon = abs(query - boundary)
        if horizon > max_gap_s + 1e-9:
            raise ValueError('缺口超过 max_gap_s，拒绝外推')
        design = np.column_stack([t[window] - boundary, np.ones(k)])
        coef = np.linalg.lstsq(design, p[window], rcond=None)[0]
        residual = float(np.sqrt(np.mean(np.sum((design @ coef - p[window]) ** 2, axis=1))))
        values.append(np.array([query - boundary, 1.]) @ coef)
        metadata.append(dict(source='inferred_linear_extrapolation', is_ground_truth=False,
                             query_timestamp_s=float(query), horizon_s=float(horizon), max_gap_s=float(max_gap_s),
                             support_timestamps_s=t[window].tolist(), fit_residual_rms=residual))
    return np.asarray(values), tuple(metadata)
