"""公共标记接口的导入、形状和只读协议；真实CUDA等价沿用既有归因审计。"""
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from .native_row_contribution import RowAttribution, attribute_rows, attribute_prepared_rows


class Scene:
    def __init__(self):
        self.owners=torch.tensor([-1,10])
        self.features=torch.zeros((2,24))
        self.mask=torch.ones(2,dtype=torch.bool)
        self.static=SimpleNamespace(get_xyz=torch.zeros((2,3)))
        self.dynamic=None;self._include_dynamic=False;self.pipe=None;self._background=torch.zeros(3)

    def _query_ids(self,ids):return tuple(ids)
    def _contains(self,owners,ids):return torch.isin(owners,owners.new_tensor(ids)) if ids else torch.zeros_like(owners,dtype=torch.bool)
    def _rows(self,*args):return dict(features=self.features,mask=self.mask),self.owners,SimpleNamespace(image_height=2,image_width=2),None
    def _bindings(self):
        def raster(camera,pipe,degree,rows,background):
            weights=torch.tensor([[[.1,.2],[.3,.4]],[[.2,.1],[.1,.2]]])*rows['mask'][:,None,None]
            extra=torch.einsum('nf,nhw->fhw',rows['features'][:,2:],weights)
            return dict(extra=extra,alpha=weights.sum(0)[None])
        return SimpleNamespace(raster=raster)


def test_marker_shape_and_fixed_owner_mask_are_read_only():
    scene=Scene()
    region=np.ones((2,2),bool)
    result=attribute_rows(scene,134,dict(K=region),delete_actor_ids=(10,))
    assert isinstance(result,RowAttribution)
    assert result.alpha_mass.shape==(2,1)
    assert np.allclose(result.alpha_mass[:,0],[1.,0.])
    assert result.region_pixels==dict(K=4)
    assert result.conservation_relative_error<1e-6
    assert torch.equal(scene.mask,torch.ones(2,dtype=torch.bool))
    assert not scene.features.requires_grad


def test_marker_rejects_non_boolean_or_wrong_shape_region():
    with pytest.raises(ValueError,match='bool'):
        attribute_rows(Scene(),134,dict(K=np.ones((2,2),np.float32)))
    with pytest.raises(ValueError,match='bool'):
        attribute_rows(Scene(),134,dict(K=np.ones((3,2),bool)))


def test_prepared_rows_use_supplied_state_and_detach_source_gradients():
    scene=Scene();rows,owners,camera,_=scene._rows(134)
    source_features=torch.zeros((2,24),requires_grad=True)
    supplied=dict(rows,features=source_features,mask=torch.tensor([True,False]))
    result=attribute_prepared_rows(scene,supplied,owners,camera,dict(K=np.ones((2,2),bool)))
    assert np.allclose(result.alpha_mass[:,0],[1.,0.])
    assert source_features.grad is None
    assert scene.mask.all()
