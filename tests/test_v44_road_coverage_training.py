from dataclasses import replace
import copy
import numpy as np
import pytest
import torch
from drivingeditor_dynamic_edit.road_coverage_training import (
    CoverageContract, CoverageTrainer, opacity_scope, coverage_domains,
    alpha_coverage_metrics, _tensor_difference, tangent_scale_scope)


def test_opacity_scope_keeps_old_geometry_and_protected_fallback_frozen():
    n=8
    road=dict(active=np.array([1,1,1,1,1,1,0,1],bool),minimum_model_depth=np.array([0,0,0,0,0,0,0,.2]))
    p=dict(requested=np.array([1,1,1,1,1,0,1,1],bool),legal_write=np.ones(n,bool),
        surface_supported=np.array([1,1,0,1,1,1,1,1],bool),unknown=np.array([0,0,0,1,0,0,0,0],bool),
        guarded_world_rows=np.array([0,1,0,0,0,0,0,0],bool),trusted_geometry=np.ones(n,bool))
    mask,receipt=opacity_scope(road,p)
    assert np.flatnonzero(mask).tolist()==[0,4]
    assert receipt['trusted_geometry_opacity_allowed']
    assert receipt['trained_parameters']==['road_opacity_logits']


def test_observed_surface_coverage_request_is_independent_of_generated_colour_W():
    road=dict(active=np.ones(2,bool))
    p=dict(requested=np.array([True,False]),coverage_requested=np.ones(2,bool),legal_write=np.ones(2,bool),
        surface_supported=np.ones(2,bool),unknown=np.zeros(2,bool),guarded_world_rows=np.zeros(2,bool))
    mask,receipt=opacity_scope(road,p)
    assert mask.tolist()==[True,True]
    assert receipt['coverage_request_field']=='coverage_requested'
    assert p['requested'].tolist()==[True,False]


def test_evaluation_denominator_retains_unsupported_unknown_holes():
    view=dict(index=np.array([[0,1,2,3]]),target=np.ones((1,4),bool),inside=np.ones((1,4),bool),
        retained_reveal_guard=np.zeros((1,4),bool),retained_box_guard=np.zeros((1,4),bool))
    p=dict(requested=np.ones(4,bool),legal_write=np.ones(4,bool),surface_supported=np.array([1,0,1,1],bool),
        unknown=np.array([0,0,1,0],bool),guarded_world_rows=np.array([0,0,0,1],bool))
    d=coverage_domains(view,np.zeros((1,4),np.uint8),p)
    assert d['evaluation'].sum()==4 and d['fit'].sum()==1
    m=alpha_coverage_metrics(np.array([[.6,0,0,.3]]),d,maximum_alpha=np.array([[.95,0,0,.3]]))
    assert m['geometry_unsupported_pixels']==1 and m['material_unknown_pixels']==1
    assert m['gap_pixels']==2 and m['opacity_uncontrollable_gap_pixels']==2
    assert m['opacity_can_reach_supported_pixels']==1


def toy_trainer(binding=None):
    base=torch.tensor([.3,.4,.5])
    # 各road footprint在3个像素的真实over合成；第三像素无可训练覆盖。
    kernel=torch.tensor([[1.,0.,0.],[0.,1.,0.],[.2,.2,1.]])
    def render(key,opacity):return 1-torch.prod(1-opacity[:,None]*kernel,dim=0)
    baseline=render(0,base).detach();fit=torch.tensor([1,1,0],dtype=torch.bool)
    d={k:dict(fit=fit,retained=torch.tensor([0,0,1],dtype=torch.bool),outside=torch.zeros(3,dtype=torch.bool)) for k in [86,134]}
    return CoverageTrainer(base,[0,1],[86,134],render,d,{86:baseline,134:baseline},binding or dict(input='fixed'))


def test_coverage_update_changes_only_selected_opacity_and_is_finite():
    t=toy_trainer();original=t.base_opacity.clone();first=t.update();second=t.update()
    assert np.isfinite(first['loss']) and first['nonzero_gradient_parameters']==2
    assert second['loss']<first['loss']
    assert torch.equal(t.base_opacity,original)
    assert t.opacity()[2]==original[2] and torch.all(t.opacity()[:2]>original[:2])


def test_full_state_restore_replays_next_update_exactly():
    t=toy_trainer();t.update();t.update();checkpoint=t.state()
    t.update();expected=t.state()
    replay=toy_trainer();receipt=replay.restore(checkpoint)
    assert receipt['full_state_restored']
    replay.update();actual=replay.state()
    assert not _tensor_difference(expected,actual)
    assert actual['optimizer']['state'][0]['exp_avg'].abs().sum()>0


def test_restore_rejects_changed_source_contract_or_scope():
    t=toy_trainer();z=t.state()
    with pytest.raises(ValueError,match='绑定'):toy_trainer(dict(input='different')).restore(z)
    changed=copy.deepcopy(z);changed['ids'][0]=2
    with pytest.raises(ValueError,match='索引'):toy_trainer().restore(changed)
    changed=copy.deepcopy(z);changed['step']=-1
    with pytest.raises(ValueError,match='步数'):toy_trainer().restore(changed)
    changed=copy.deepcopy(z);changed['accounting_role']='smoke'
    with pytest.raises(ValueError,match='记账'):toy_trainer().restore(changed)
    t.step=600
    with pytest.raises(RuntimeError,match='预算'):t.update()
    with pytest.raises(ValueError):replace(CoverageContract(),formal_steps=601)
    with pytest.raises(ValueError):replace(CoverageContract(),tangent_scale_max_factor=1.16)


def test_tangent_bounds_use_actual_world_distance_and_protect_trusted_geometry():
    yy,xx=np.mgrid[:7,:7];keys=np.c_[xx.ravel(),yy.ravel()];n=len(keys)
    # 网格刻意带世界XY伸缩，不能用假定cell_m作为实际距离。
    world=np.c_[keys[:,0]*.021,keys[:,1]*.019,np.zeros(n)]
    road=dict(active=np.ones(n,bool),gaussian_sigma_m=np.tile([.003,.003,.0001],(n,1)))
    p=dict(world_keys=keys,world_xyz=world,requested=np.ones(n,bool),legal_write=np.ones(n,bool),
        surface_supported=np.ones(n,bool),unknown=np.zeros(n,bool),trusted_geometry=np.zeros(n,bool),guarded_world_rows=np.zeros(n,bool))
    center=24;p['trusted_geometry'][center]=True
    eligible,upper,distance,receipt=tangent_scale_scope(road,p,np.ones(n,bool))
    assert not eligible[center] and upper[center]==1
    assert not eligible[0]
    assert np.all(3*np.max(road['gaussian_sigma_m'][eligible,:2],axis=1)*upper[eligible]<distance[eligible])
    assert upper.max()<=1.15 and receipt['normal_sigma_frozen']


def test_bounded_scale_training_preserves_normal_thickness_and_restore():
    base=torch.tensor([.4,.5]);scales=torch.tensor([[.01,.01,.001],[.01,.01,.001]])
    baseline=torch.tensor([.4,.5]);mask=torch.tensor([True,False])
    def render(key,opacity,sigma):return opacity*(sigma[:,0]/.01)
    d={86:dict(fit=mask,retained=~mask,outside=torch.zeros(2,dtype=torch.bool))}
    contract=replace(CoverageContract(),train_tangent_scales=True)
    t=CoverageTrainer(base,[0],[86],render,d,{86:baseline},dict(source='fixed'),contract,
        base_scales=scales,active_scale_ids=[0],scale_upper=[1.01])
    for _ in range(4):t.update()
    assert 1<=float(t.scale_factor[0])<=1.01+1e-6
    assert torch.equal(t.scales()[:,2],scales[:,2]) and torch.equal(t.scales()[1],scales[1])
    state=t.state();t.update();expected=t.state()
    t.restore(state);t.update();assert not _tensor_difference(expected,t.state())
