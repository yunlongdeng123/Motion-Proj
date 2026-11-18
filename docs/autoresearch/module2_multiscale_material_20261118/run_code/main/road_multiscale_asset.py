"""可移植的只读共享道路材质入口；权威field重查询，road.npz为派生GS缓存。

该组件不加载场景或相机，不修改默认provider。调用者将query_colors()返回的
固定active-world-row顺序RGB交给原有Gaussian光栅器。几何变化必须重建采样计划。
"""
from pathlib import Path
from dataclasses import dataclass
import hashlib,json
import numpy as np
import torch
from .road_multiscale_material import WorldMaterialField


def _digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


@dataclass
class SharedRoadMaterialAsset:
    model: WorldMaterialField
    active_world_rows: np.ndarray
    eligible: np.ndarray
    derived_colors: torch.Tensor
    metadata: dict

    @torch.no_grad()
    def query_colors(self,*,verify_derived=True,tolerance=1e-6):
        """每次从共享网格重新查询；未许可颜色保持原只读缓存。"""
        ids=torch.as_tensor(np.flatnonzero(self.eligible),device=self.derived_colors.device)
        colors=self.derived_colors.clone()
        values,mass=self.model.query(indices=ids)
        if not bool(torch.isfinite(values).all()) or bool((mass<=1e-12).any()):
            raise ValueError('合法材质查询存在空源核或非finite颜色')
        colors[ids]=values
        if verify_derived and float((colors-self.derived_colors).abs().max())>tolerance:
            raise ValueError('权威共享场与派生Gaussian缓存不同；请重新导出资产')
        return colors


def load_shared_road_material(asset_directory,*,device='cpu'):
    """加载自包含field/资格/派生road；不依赖实验目录或任一相机。

    本入口验证已保存field、row scope及完整road body的绑定。物理query计划的
    source-geometry重构审计由导出流程完成并保存；这里不冒称重新验证外部原测量。
    """
    root=Path(asset_directory);receipt=json.loads((root/'field_receipt.json').read_text())
    for filename,key in [('world_field.pt','field_sha256'),('road.npz','derived_road_sha256'),('field_scope.npz','field_scope_sha256')]:
        if _digest(root/filename)!=receipt[key]:raise ValueError('共享资产绑定损坏：'+filename)
    with np.load(root/'road.npz',allow_pickle=False) as z:
        active=np.flatnonzero(z['active']);rgb=z['reference_srgb'][active].copy()
    with np.load(root/'field_scope.npz',allow_pickle=False) as z:
        ids=z['active_world_rows'].copy();eligible=z['eligible'].copy()
    if not np.array_equal(active,ids) or eligible.dtype!=np.bool_ or eligible.shape!=(len(active),):
        raise ValueError('world-row身份/资格与road body不一致')
    with np.load(root/'provenance.npz',allow_pickle=False) as z:
        prior=z['capacity_source_prior_reference_srgb'][active]
        original_eligible=z['capacity_color_eligible'][active]
    if not np.array_equal(eligible,original_eligible):raise ValueError('field资格与已审世界写回资格不同')
    if not np.array_equal(rgb[~eligible],prior[~eligible]):raise ValueError('未许可行的只读颜色发生变化')
    model,metadata=WorldMaterialField.load(root/'world_field.pt',device=device)
    if len(model.center_uv)!=len(active):raise ValueError('field采样计划与world-row数量不一致')
    model.requires_grad_(False);model.eval()
    return SharedRoadMaterialAsset(model,active,eligible,
        torch.as_tensor(rgb,device=device,dtype=torch.float32),metadata)
