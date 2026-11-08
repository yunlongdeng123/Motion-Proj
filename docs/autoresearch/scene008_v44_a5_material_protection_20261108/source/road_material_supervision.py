"""道路材质的显式拟合/保护监督；不加载或修改场景模型。

旧资格只测拟合区贡献，无法约束同一父点在保护区的颜色。本接口保留
C目标的原拟合域，并独立注册原观测道路、非道路保护域；资格取两种
监督的可见依赖并集。重叠域显式记录，由约束求解器优先保护，不能把
保护像素从统计分母移除，也不能把这份BUILD监督称为独立测试。
"""
from dataclasses import dataclass
from typing import Dict

import numpy as np


def _mask(value, shape, name):
    result = np.asarray(value).squeeze().astype(bool)
    if result.shape != shape:
        raise ValueError(name + ': invalid mask shape')
    return result.copy()


def _rgb(value, shape, name):
    value = np.asarray(value)
    if value.shape == shape + (3,):
        value = value.transpose(2, 0, 1)
    if value.shape != (3,) + shape:
        raise ValueError(name + ': invalid RGB shape')
    result = value.astype(np.float32)
    if np.issubdtype(value.dtype, np.integer):
        if value.min() < 0 or value.max() > 255:
            raise ValueError(name + ': integer RGB outside byte range')
        result /= 255.0
    if not np.isfinite(result).all() or result.min() < 0 or result.max() > 1:
        raise ValueError(name + ': expected finite observation RGB in [0,1]')
    return result.copy()


@dataclass(frozen=True)
class RoadMaterialSupervision:
    fit_rgb: np.ndarray
    fit_mask: np.ndarray
    protection_rgb: np.ndarray
    protection_domains: Dict[str, np.ndarray]
    source_protect_mask: np.ndarray

    @property
    def protection_mask(self):
        return np.logical_or.reduce(list(self.protection_domains.values()))

    def summary(self):
        union = self.protection_mask
        return dict(
            schema='road-material-fit-protect-supervision-v1',
            fit_pixels=int(self.fit_mask.sum()),
            protection_pixels=int(union.sum()),
            protection_domains={name: int(mask.sum()) for name, mask in self.protection_domains.items()},
            fit_and_protection_overlap=int((self.fit_mask & union).sum()),
            overlap_policy='Protection constraints take priority; neither domain is erased.',
            input_role='BUILD supervision, not independent held-out evaluation',
            optimization_scope='Shared world road-parent RGB only',
        )


def build_supervision(observation, c_target):
    """由固定观测/合法C目标注册监督，不读取候选RGB、alpha或深度选mask。"""
    shape = np.asarray(observation['road']).squeeze().shape
    if len(shape) != 2:
        raise ValueError('expected two-dimensional image masks')
    road = _mask(observation['road'], shape, 'original_road')
    road_edit = _mask(observation['road_edit'], shape, 'road_edit')
    protect = _mask(observation['protect'], shape, 'observation_protect')
    visible = np.asarray(observation['visible'])
    if visible.shape != (2,) + shape:
        raise ValueError('expected fixed a6/a37 visibility masks')
    a6 = _mask(visible[0], shape, 'a6')
    a37 = _mask(visible[1], shape, 'a37')
    fit = _mask(c_target['fit_mask'], shape, 'C_fit')
    source_protect = _mask(c_target['protect_mask'], shape, 'C_source_protect')
    if np.any(fit & source_protect):
        raise ValueError('original C fit/protect roles overlap')
    nonroad = protect & ~a6 & ~a37 & ~road & ~road_edit
    return RoadMaterialSupervision(
        fit_rgb=_rgb(c_target['edgeblend_rgb'], shape, 'C_fit_target'),
        fit_mask=fit,
        protection_rgb=_rgb(observation['rgb'], shape, 'observation_RGB'),
        protection_domains=dict(protected_nonroad=nonroad, original_road=road),
        source_protect_mask=source_protect,
    )


def qualify_parent_rgb(fit_mass, protection_mass):
    """正拟合贡献或正保护贡献即可获得材质监督；无尾部比例阈值。"""
    fit = np.asarray(fit_mass, dtype=np.float64)
    protection = np.asarray(protection_mass, dtype=np.float64)
    if (fit.ndim != 1 or fit.shape != protection.shape
            or not np.isfinite(fit).all() or not np.isfinite(protection).all()
            or np.any(fit < 0) or np.any(protection < 0)):
        raise ValueError('expected finite nonnegative per-parent alpha*T mass')
    return np.flatnonzero((fit > 0) | (protection > 0))
