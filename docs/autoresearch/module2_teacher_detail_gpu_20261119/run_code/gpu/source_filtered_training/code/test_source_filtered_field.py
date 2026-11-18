"""固定 source filter 的 CPU 前向、伴随、持久化与 A200 metadata rebase。"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import copy
import json
import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from world_texture_runtime import WorldTextureRuntime
from field_training_state import TrainingState, max_tree_difference, file_sha
from source_filtered_field import (SourceFootprintWorldTexture, SourceFilteredTrainingState,
                                   SIGMA_UV, rebase_zero_H_common_A)

torch.set_num_threads(1)
SPEC = dict(low_lr=.01, high_lr=.003, betas=[.9,.999], eps=1e-8, weight_decay=0.,
    regularizers=dict(low_adjacent_delta_squared=.01, effective_high_energy=.1,
        effective_high_lowfrequency_leakage=.1, effective_high_weighted_DC=.1))


def model(cls=SourceFootprintWorldTexture, gate=1.):
    return cls(np.full((16,18,3),.4), np.ones((16,18)), np.full((16,18),gate),
        np.ones((16,18)), np.ones((16,18)), low_mask=np.ones((4,4)))


def test_filter_float64_matches_scipy_axes_boundary_and_constant():
    m=model().double(); x=np.random.default_rng(101).normal(size=(16,18,3))
    expected=gaussian_filter(x,sigma=(SIGMA_UV[1],SIGMA_UV[0],0),mode='nearest',truncate=3)
    actual=m.filter_high(torch.tensor(x,dtype=torch.float64)).detach().numpy()
    assert np.max(abs(actual-expected))<2e-15
    assert m.filter_settings()['radius_uv']==[2,1]
    assert torch.max(abs(m.filter_high(torch.ones_like(torch.tensor(x)))-1))<2e-15


def test_filter_adjoint_matches_scipy_dense_boundary_operator():
    m=model().double(); h,w=5,7; n=h*w
    matrix=np.column_stack([gaussian_filter(np.eye(n)[:,i].reshape(h,w),
        sigma=(SIGMA_UV[1],SIGMA_UV[0]),mode='nearest',truncate=3).ravel() for i in range(n)])
    rng=np.random.default_rng(101);x=torch.tensor(rng.normal(size=(h,w,3)),dtype=torch.float64,requires_grad=True)
    weights=rng.normal(size=(h,w,3));(m.filter_high(x)*torch.tensor(weights)).sum().backward()
    expected=np.stack([matrix.T@weights[...,c].ravel()for c in range(3)],-1).reshape(h,w,3)
    assert np.max(abs(x.grad.numpy()-expected))<2e-15


def test_fields_float32_match_zero_update_scipy_and_have_real_H_gradient():
    m=model()
    with torch.no_grad():m.high.copy_(torch.sin(torch.arange(m.high.numel()).reshape_as(m.high)*.1))
    low,old=WorldTextureRuntime.fields(m);newlow,new=m.fields()
    expected=gaussian_filter(old.detach().numpy(),sigma=(SIGMA_UV[1],SIGMA_UV[0],0),mode='nearest',truncate=3)
    assert torch.equal(low,newlow)
    assert np.max(abs(new.detach().numpy()-expected))<5e-8
    new[7,8,0].backward();assert m.high.grad.abs().max()>0
    assert set(dict(m.named_parameters()))=={'low','high'}


def test_specialized_save_reload_and_old_loader_refuses(tmp_path):
    m=model()
    with torch.no_grad():m.low.add_(.1);m.high.add_(torch.randn_like(m.high)*.1)
    path=tmp_path/'field.pt';m.save(path,{'arm':'M12'})
    restored,meta=SourceFootprintWorldTexture.load(path,device='cpu')
    assert meta['arm']=='M12'
    for x,y in zip(m.fields(),restored.fields()):assert torch.equal(x,y)
    with pytest.raises(KeyError):WorldTextureRuntime.load(path,device='cpu')
    changed=torch.load(path,weights_only=False);changed['source_filter']['sigma_uv'][0]=1.
    torch.save(changed,tmp_path/'wrong.pt')
    with pytest.raises(ValueError):SourceFootprintWorldTexture.load(tmp_path/'wrong.pt',device='cpu')


class ToyReader:
    kind='pixel';renderer_mode='synthetic_CPU_unit_fixture'
    def require_execution(self,role):assert role=='formal'
    def assert_frozen(self):pass
    def manifest(self):return {}
    def render(self,model,raw,joint=False):
        uv=torch.tensor([[[4.2,5.1]],[[8.3,6.7]],[[10.8,10.2]],[[13.7,12.1]]],dtype=torch.float64)
        values,_=model.sample(uv,torch.ones(1))
        return dict(road_rgb=values.T.reshape(3,1,4))


def test_A200_rebase_only_identity_and_fork_keeps_teacher_gate(tmp_path):
    source=tmp_path/'input.json';source.write_text('{"fixture":true}')
    identity=dict(reader='pixel',source_bindings={str(source):file_sha(source)},seed=101,
                  scope_rows=4,active_world_rows_kind='global_world_row')
    targets={raw:dict(fixed_alpha=torch.ones((1,4)),**{k:torch.ones((1,4),dtype=torch.bool)for k in ('A','C','S')},
        **{'target_'+k:torch.full((3,1,4),.6)for k in ('A','C','S')})for raw in (86,110)}
    old=TrainingState(model(WorldTextureRuntime,0.),ToyReader(),targets,identity,SPEC,role='formal',arm='common_pixel')
    # 微型合成CPU状态，不是场景训练；真实A200由root另做相同核验。
    for _ in range(200):old.step()
    common=old.snapshot();newid=copy.deepcopy(identity);newid['material_model']='SourceFootprintWorldTexture.v1'
    new=SourceFilteredTrainingState(model(),ToyReader(),targets,newid,SPEC,role='formal')
    before=copy.deepcopy(common)
    rebased=rebase_zero_H_common_A(common,new,tmp_path/'rebase.json')
    assert max_tree_difference(common,before)==0
    for key in common:
        if key!='identity':assert max_tree_difference(common[key],rebased[key])==0
    new.fork_common_A(rebased)
    assert torch.all(new.model.H_gate==1)
    assert torch.equal(new.model.low,common['params']['low'])
    assert new.optimizer.state_dict()['state'].get(1)is None
    assert json.loads((tmp_path/'rebase.json').read_text())['passed']
    bad=copy.deepcopy(common);bad['params']['high'][0,0,0]=.01
    with pytest.raises(ValueError):rebase_zero_H_common_A(bad,new)
    bad=copy.deepcopy(common);bad['role']='smoke'
    with pytest.raises(ValueError):rebase_zero_H_common_A(bad,new)
