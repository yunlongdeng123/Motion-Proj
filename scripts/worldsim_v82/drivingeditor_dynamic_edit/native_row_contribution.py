"""原生同一深度排序中的逐行、逐区域可见 alphaT 标记接口。"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RowAttribution:
    region_names: tuple
    alpha_mass: np.ndarray
    owners: np.ndarray
    static_rows: int
    dynamic_rows: int
    region_pixels: dict
    conservation_relative_error: float


def _detached_rows(scene, key, delete_actor_ids):
    import torch
    deleted = scene._query_ids(delete_actor_ids)
    with torch.no_grad():
        rows, owners, camera, _ = scene._rows(key, (), (), False)
        rows = {k: v.detach() if torch.is_tensor(v) else v for k, v in rows.items()}
        rows = dict(rows, mask=rows['mask'] & ~scene._contains(owners, deleted))
    return rows, owners, camera


def attribute_rows(scene, key, regions, *, delete_actor_ids=()):
    """每个固定原生行对各 bool(H,W) 区域的真实 sum(alpha_i*T_i)。

    最多22个区域使用24-feature ABI中的独立标记通道；保留当前参数、
    逐行mask和完整光栅排序，仅对新建标记求梯度。
    """
    rows, owners, camera = _detached_rows(scene, key, delete_actor_ids)
    return attribute_prepared_rows(scene, rows, owners, camera, regions)


def attribute_prepared_rows(scene, rows, owners, camera, regions):
    """对调用者准备好的固定行求标记贡献；不重取行、不改变scene或删除mask。

    调用者先完成授权的mask/opacity/scale状态替换。这里detach所有输入行，
    梯度只流向新建的feature标记，保持传入的真实光栅排序与可见性。
    """
    import torch
    if not isinstance(regions, dict) or not regions or len(regions) > 22:
        raise ValueError('1..22 named regions are required by the 24-feature ABI')
    if any(not isinstance(name, str) or not name for name in regions):
        raise ValueError('region names must be nonempty strings')
    rows = {k: v.detach() if torch.is_tensor(v) else v for k, v in rows.items()}
    if rows['features'].shape[0] != len(owners) or rows['mask'].shape != owners.shape or rows['mask'].dtype != torch.bool:
        raise ValueError('Prepared rows must retain the fixed owner topology and boolean row mask')
    shape = (camera.image_height, camera.image_width)
    masks = []
    for name, value in regions.items():
        a = np.asarray(value)
        if a.dtype != np.bool_ or a.shape != shape:
            raise ValueError(f'{name} must be bool{shape}')
        masks.append(torch.as_tensor(a.copy(), device=owners.device))
    features = rows['features'].new_zeros((len(owners), 24))
    features[:, :2] = rows['features'][:, :2]
    features[:, 2:2+len(masks)] = 1
    features.requires_grad_(True)
    with torch.enable_grad():
        pack = scene._bindings().raster(camera, scene.pipe, 0,
            dict(rows, features=features), scene._background)
        if pack['extra'].shape[0] != 22:
            raise ValueError('raster must preserve the 24-feature ABI')
        objective = sum(pack['extra'][j][mask].sum() for j, mask in enumerate(masks))
        mass = torch.autograd.grad(objective, features)[0][:, 2:2+len(masks)]
    a = mass.detach().cpu().numpy().copy()
    alpha = pack['alpha'].detach().reshape(shape)
    expected = np.asarray([float(alpha[m].sum()) for m in masks])
    actual = a.sum(axis=0, dtype=np.float64)
    error = float(np.max(np.abs(actual-expected)/np.maximum(expected, 1)))
    if not np.isfinite(a).all() or (a < -1e-6).any() or error > 5e-4:
        raise RuntimeError(f'invalid feature-gradient alphaT attribution: relative error {error}')
    a = np.maximum(a, 0)
    return RowAttribution(tuple(regions), a, owners.detach().cpu().numpy().copy(),
        len(scene.static.get_xyz), len(scene.dynamic.get_xyz) if scene._include_dynamic else 0,
        {name: int(mask.sum()) for name, mask in zip(regions, masks)}, error)
