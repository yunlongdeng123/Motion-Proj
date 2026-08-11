"""WorldSim V4 baseline 图像评测的冻结区域掩码。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy import ndimage


class RegionMaskError(ValueError):
    """区域输入不满足 V4 mask 合同。"""


@dataclass(frozen=True)
class RegionMaskProtocol:
    boundary_radius_pixels: int = 3


def _mask(value: np.ndarray, label: str) -> np.ndarray:
    mask = np.asarray(value)
    if mask.ndim != 2:
        raise RegionMaskError(f"{label} 必须是 HxW")
    return mask.astype(bool, copy=False)


def build_baseline_region_masks(
    dynamic_mask: np.ndarray,
    egocar_mask: np.ndarray,
    *,
    edit_roi: np.ndarray | None = None,
    protocol: RegionMaskProtocol = RegionMaskProtocol(),
) -> Mapping[str, np.ndarray]:
    """从 DriveStudio dynamic/egocar mask 构造 baseline 的四个显式区域。

    ``global`` 由 evaluator 自身创建。baseline 没有编辑目标时 ``edit_roi``
    固定为空，而不是把任意动态对象冒充 edit ROI。
    """
    dynamic = _mask(dynamic_mask, "dynamic_mask")
    egocar = _mask(egocar_mask, "egocar_mask")
    if dynamic.shape != egocar.shape:
        raise RegionMaskError("dynamic_mask 与 egocar_mask shape 不一致")
    radius = int(protocol.boundary_radius_pixels)
    if radius < 1:
        raise RegionMaskError("boundary_radius_pixels 必须 >= 1")
    structure = ndimage.generate_binary_structure(2, 1)
    dilated = ndimage.binary_dilation(dynamic, structure=structure, iterations=radius)
    eroded = ndimage.binary_erosion(dynamic, structure=structure, iterations=radius)
    boundary = np.logical_xor(dilated, eroded)
    roi = np.zeros(dynamic.shape, dtype=bool) if edit_roi is None else _mask(edit_roi, "edit_roi")
    if roi.shape != dynamic.shape:
        raise RegionMaskError("edit_roi shape 不一致")
    return {
        "static": ~(dynamic | egocar),
        "actor": dynamic,
        "boundary": boundary,
        "edit_roi": roi,
    }
