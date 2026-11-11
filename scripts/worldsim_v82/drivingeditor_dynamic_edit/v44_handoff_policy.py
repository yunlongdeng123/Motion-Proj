"""V44 BUILD 阶段的候选覆盖与保持域；纯 NumPy，无场景或 QUERY 分支。

target_mask 必须由显式 instance 标签及 manifest 删除集合解析，不从 road
chart、颜色、帧名或候选得分反推。unknown_target 只供覆盖审计，不是背景真值，
也不产生额外 photo 监督或 opacity=0 惩罚。
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import numpy as np


FIT_SUPPORT_THRESHOLD = 0.05


def _array(value, name, ndim, shape=None):
    if np.ma.isMaskedArray(value):
        raise ValueError(name + ' must not contain implicit masked values')
    result = np.asarray(value)
    if result.ndim != ndim or (shape is not None and result.shape != shape):
        raise ValueError(name + ' must have shape ' + (str(shape) if shape is not None else f'{ndim}D'))
    if ndim == 2 and any(size <= 0 for size in result.shape):
        raise ValueError(name + ' must have positive image height and width')
    return result


def _binary_image(value, name, shape=None):
    result = _array(value, name, 2, shape)
    if (result.dtype.kind not in 'biuf' or not np.isfinite(result).all()
            or not ((result == 0) | (result == 1)).all()):
        raise ValueError(name + ' must be boolean or finite binary numeric values')
    return result.astype(bool, copy=True)


def _weight_image(value, name, shape):
    result = _array(value, name, 2, shape)
    if (result.dtype.kind not in 'biuf' or not np.isfinite(result).all()
            or (result < 0).any() or (result > 1).any()):
        raise ValueError(name + ' must be finite real weights in [0, 1]')
    return result.copy()


def _instance_labels(value):
    labels = _array(value, 'instance_labels', 2)
    if (labels.dtype.kind not in 'iu' or (labels < 0).any()
            or (labels > np.iinfo(np.int64).max).any()):
        raise ValueError('instance_labels must be nonnegative integer IDs within int64')
    return labels


def target_mask_from_labels(instance_labels, target_label_ids):
    """用显式标签 ID 求多目标并集；目标不可见或删除集合为空均返回空掩码。

    调用方负责把 manifest track/actor ID 映射成此标签图的 instance ID。
    此函数不假设 actor ID 与标签值相同，不把缺席目标扩张成 chart 区域。
    标签 0 为背景；ID 必须是正整数，不接受浮点截断或布尔 ID。
    """
    labels = _instance_labels(instance_labels)
    if isinstance(target_label_ids, (str, bytes)):
        raise ValueError('target_label_ids must be an explicit collection of positive integers')
    try:
        requested = tuple(target_label_ids)
    except TypeError as error:
        raise ValueError('target_label_ids must be an explicit collection of positive integers') from error
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
           or value <= 0 or value > np.iinfo(np.int64).max for value in requested):
        raise ValueError('target_label_ids must contain positive integers within int64')
    if len(set(requested)) != len(requested):
        raise ValueError('target_label_ids must contain distinct IDs')
    return np.isin(labels, np.asarray(requested, dtype=np.int64))


def rebind_other_vehicles(view_other, labels, delete_label_ids):
    """按当前完整删除集合重建 other，避免旧单目标视图保护第二删除目标。

    labels 只接 manifest 已接受的 instance 标签图；调用方负责该数据绑定。
    非目标正标签补入保护域，无标签的旧 other 仍保留；显式目标退出旧 other。
    此操作只改目标归属，不修改 photo 权重或 road chart。
    """
    labels = _instance_labels(labels)
    target = target_mask_from_labels(labels, delete_label_ids)
    other = _binary_image(view_other, 'view_other', labels.shape)
    return (other | (labels > 0)) & ~target


def handoff_domains(view, target_mask):
    """返回一个 BUILD 视角的保持域、候选覆盖域和审计掩码，输入保持不变。

    view 需提供同形 HW 的 fit/keep 权重与 inside/other/marking 二值图。
    额外字段（如 reference_linear）忽略。输入权重和分辨率保持原合同。
    保护域 P=other|marking；仅移除不与 P 冲突的目标保持约束，并将 P 的 keep
    权重明确设为 1。fit 权重保持原样，不额外创造 photometric 参考。

    返回字段：fit（原 photo 权重副本）、keep、left、protected、
    candidate_support、unknown_target、target_protected_conflict。
    fit>.05 只表示现有 handoff photo 支持；unknown_target 是缺少该支持的
    非保护目标区域。函数不补全 reference_linear，不为 unknown 创建损失。
    """
    if not isinstance(view, Mapping) and not (hasattr(view, 'files') and hasattr(view, '__getitem__')):
        raise ValueError('view must be a field mapping or an opened NumPy NPZ archive')
    required = ('fit', 'keep', 'inside', 'other', 'marking')
    if any(name not in view for name in required):
        raise ValueError('view requires fit, keep, inside, other and marking')
    target = _binary_image(target_mask, 'target_mask')
    shape = target.shape
    fit = _weight_image(view['fit'], 'fit', shape)
    keep = _weight_image(view['keep'], 'keep', shape)
    inside = _binary_image(view['inside'], 'inside', shape)
    other = _binary_image(view['other'], 'other', shape)
    marking = _binary_image(view['marking'], 'marking', shape)

    protected = other | marking
    removable_target = target & ~protected
    keep[removable_target] = 0
    keep[protected] = 1
    left = ~inside & ~other & ~removable_target
    fit_support = fit > FIT_SUPPORT_THRESHOLD
    return dict(
        fit=fit,
        keep=keep,
        left=left,
        protected=protected,
        candidate_support=(fit_support | target) & ~protected,
        unknown_target=removable_target & ~fit_support,
        target_protected_conflict=target & protected,
    )


def _score_vector(value, name):
    result = _array(value, name, 1)
    if (result.dtype.kind not in 'iuf' or not np.isfinite(result).all()
            or (result < 0).any()):
        raise ValueError(name + ' must be finite nonnegative real scores')
    return result.astype(np.float64, copy=False) if result.dtype.kind in 'iu' else result


def eligible_static_contributors(max_candidate_score, max_guard_score, fixed_owner):
    """对全部 BUILD 视角已取 max 的得分应用原有固定阈值，返回 N 维布尔图。

    调用方在 BUILD 时只构造一次候选表；不按 QUERY 帧重建，也不改变 owner。
    空静态模型允许空向量；布尔或浮点 owner 均拒绝，避免隐式身份转换。
    """
    score = _score_vector(max_candidate_score, 'max_candidate_score')
    guard = _score_vector(max_guard_score, 'max_guard_score')
    owner = _array(fixed_owner, 'fixed_owner', 1, score.shape)
    if guard.shape != score.shape:
        raise ValueError('max_guard_score must have the same shape as max_candidate_score')
    if owner.dtype.kind not in 'iu' or (owner > np.iinfo(np.int64).max).any():
        raise ValueError('fixed_owner must be integer IDs within int64')
    # 不把 float32 输入提升为 float64，保持原有阈值边界的数值语义。
    with np.errstate(over='ignore'):
        return (score > 1e-4) & ~(guard > 20 * score) & (owner <= 0)
