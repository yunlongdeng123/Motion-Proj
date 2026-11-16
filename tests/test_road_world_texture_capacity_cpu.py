"""纯CPU测试，通过文件路径导入，避免执行旧包的renderer导入链。"""
from pathlib import Path
import importlib.util
import os
import sys
import numpy as np
import pytest
import torch

MODULE=Path(os.environ['ROAD_CAPACITY_MODULE_PATH']) if 'ROAD_CAPACITY_MODULE_PATH' in os.environ else Path(__file__).resolve().parents[1]/'scripts/worldsim_v82/drivingeditor_dynamic_edit/road_world_texture_capacity.py'
spec=importlib.util.spec_from_file_location('road_world_texture_capacity_cpu',MODULE)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

def plan():
    y,x=np.mgrid[0:9,0:9]
    world=np.c_[x.ravel()*.02,y.ravel()*.02,np.zeros(x.size)]
    base=np.full((len(world),3),.3,np.float32)
    return m.make_color_basis_plan(world,base,np.ones(len(world),bool),scope_source='synthetic_grid_fixture',source_role='synthetic_unit_test',spacing_m=.04)


def test_explicit_cpu_environment_and_common_initialization():
    m.require_cpu_only_environment()
    p=plan();i=m.RoadColorParameterization(p,'independent_ci');w=m.RoadColorParameterization(p,'shared_world')
    assert all(parameter.device.type=='cpu' for model in (i,w) for parameter in model.parameters())
    assert torch.equal(i(),w()) and torch.equal(i(),torch.from_numpy(p.base_srgb))
    assert len(w.delta_values)<len(i.delta_values)
    assert np.allclose(p.weights.sum(1),1,atol=1e-15)


def test_shared_world_query_uses_each_own_position_and_couples_rows():
    p=plan();w=m.RoadColorParameterization(p,'shared_world')
    origin=p.origin_xy
    with torch.no_grad():
        grid=np.flatnonzero(p.mutable_texels);u=grid%p.shape[1];v=grid//p.shape[1]
        w.delta_values.copy_(torch.tensor(np.c_[u*.01,v*.02,(u+v)*.01],dtype=torch.float32))
    own,inside=w.query_world_delta(p.world_xyz[:,:2])
    assert inside.all() and torch.equal(own,w.row_delta())
    position=np.array([[.03,.07],[.030000001,.07]])
    values,inside=w.query_world_delta(position)
    assert inside.all() and float((values[0]-values[1]).abs().max().detach())<1e-8
    assert float((w.row_delta()[0]-w.row_delta()[-1]).abs().max().detach())>.01
    outside,valid=w.query_world_delta(np.array([[-1.,-1.]]))
    assert not valid[0] and torch.all(outside==0)


def test_shared_backward_is_bilinear_transpose_without_detach():
    p=plan();w=m.RoadColorParameterization(p,'shared_world')
    coefficients=torch.arange(len(p.world_xyz)*3,dtype=torch.float32).reshape(-1,3)/100
    loss=(w.affine_colors()*coefficients).sum()
    gradient=torch.autograd.grad(loss,w.delta_values)[0]
    expected=np.zeros((len(p.mutable_texels),3),np.float64)
    for corner in range(4):np.add.at(expected,p.corner_ids[:,corner],p.weights[:,corner,None]*coefficients.numpy())
    expected=expected[np.flatnonzero(p.mutable_texels)].astype(np.float32)
    assert np.allclose(gradient.numpy(),expected,atol=1e-6,rtol=1e-6)
    assert gradient.abs().max()>0


def test_protected_rows_use_same_c0_override_without_extra_texel_freeze():
    p=plan();protected=np.zeros(len(p.world_xyz),bool);protected[40]=True
    p=m.make_color_basis_plan(p.world_xyz,p.base_srgb,p.requested_eligible,protected,scope_source=p.scope_source,source_role=p.source_role,spacing_m=.04)
    assert np.array_equal(p.eligible,p.requested_eligible&~protected)
    for mode in ('shared_world','independent_ci'):
        model=m.RoadColorParameterization(p,mode)
        with torch.no_grad():model.delta_values.fill_(.2)
        assert torch.equal(model()[40],torch.from_numpy(p.base_srgb[40]))
        assert torch.all(model.row_delta()[~torch.from_numpy(p.eligible)]==0)
    w=m.RoadColorParameterization(p,'shared_world')
    with torch.no_grad():w.delta_values.fill_(.2)
    # raw连续场与有效shader的guard override必须分开，不能误称同一输出。
    assert torch.all(w.query_world_delta(p.world_xyz[protected,:2])[0]>.1)


@pytest.mark.parametrize('mode',['shared_world','independent_ci'])
def test_cpu_export_reload_and_road_bake_only_color(tmp_path,mode):
    p=plan();model=m.RoadColorParameterization(p,mode)
    with torch.no_grad():model.delta_values.copy_(torch.linspace(-.03,.04,model.delta_values.numel()).reshape_as(model.delta_values))
    m.save_parameterization(model,tmp_path/'params',metadata={'synthetic_test':True})
    restored,metadata=m.load_parameterization(tmp_path/'params')
    assert torch.equal(model(),restored()) and metadata['GPU_executed'] is False
    source=tmp_path/'source.npz';out=tmp_path/'road.npz'
    original=dict(reference_srgb=p.base_srgb,model_xyz=p.world_xyz,road_opacity=np.full(len(p.world_xyz),.99,np.float32),
        gaussian_sigma_m=np.full((len(p.world_xyz),3),.01,np.float32),active=np.ones(len(p.world_xyz),bool),model_from_world=np.eye(4))
    np.savez(source,**original);m.bake_road_colors(source,out,restored)
    with np.load(out) as road:
        for key,value in original.items():
            if key!='reference_srgb':assert np.array_equal(road[key],value)
        assert np.array_equal(road['reference_srgb'],model().detach().numpy())
    with pytest.raises(FileExistsError):m.bake_road_colors(source,out,restored)


def test_conditional_world_tv_penalizes_delta_without_smoothing_prior():
    p=plan();i=m.RoadColorParameterization(p,'independent_ci')
    edges=np.c_[np.arange(8),np.arange(1,9)]
    assert m.world_delta_tv(i,edges,np.full(8,.02))==0
    with torch.no_grad():i.delta_values[::2]=.1
    loss=m.world_delta_tv(i,edges,np.full(8,.02))
    assert loss>0 and torch.autograd.grad(loss,i.delta_values)[0].abs().max()>0


def test_basis_statistics_distinguish_structural_and_numeric_rank():
    result=m.basis_statistics(plan())
    assert result['shared_texels_with_multiple_rows_fraction']==1
    assert result['shared_RGB_scalars']<result['independent_RGB_scalars']
    assert result['structural_rank_upper_bound']<=result['shared_RGB_scalars']//3
    assert result['global_numeric_rank'].startswith('not computed')


def test_scope_provenance_required_and_preview_cannot_become_training_packet():
    p=plan()
    with pytest.raises(TypeError):m.make_color_basis_plan(p.world_xyz,p.base_srgb,p.eligible)
    preview=m.make_color_basis_plan(p.world_xyz,p.base_srgb,p.eligible,scope_source='old_scope.npz',source_role='legacy_static_preview_only')
    model=m.RoadColorParameterization(preview,'shared_world')
    with pytest.raises(ValueError,match='静态预览'):
        m.color_only_native_packet({'means3D':torch.zeros(len(p.world_xyz),3)},model)


def test_fixed_real_teacher_domains_keep_rgb_gradient_and_detach_alpha():
    p=plan();model=m.RoadColorParameterization(p,'shared_world')
    alpha=torch.full((1,9,9),.95,requires_grad=True)
    # 仅CPU线性示意读出，不是原生renderer验收或真实场景训练。
    prediction=model().reshape(9,9,3).permute(2,0,1)*alpha.detach()
    mask=torch.ones(9,9,dtype=torch.bool);q=torch.ones(9,9);target=torch.full((3,9,9),.6)
    loss=m.fixed_coverage_teacher_l1(prediction,alpha,target,mask,q)
    grad=torch.autograd.grad(loss,(model.delta_values,alpha),allow_unused=True)
    assert grad[0].abs().max()>0 and grad[1] is None
    with pytest.raises(ValueError,match='A/C/S'):
        m.separated_real_teacher_loss(prediction,alpha,{},dict(A=1,C=.5))
