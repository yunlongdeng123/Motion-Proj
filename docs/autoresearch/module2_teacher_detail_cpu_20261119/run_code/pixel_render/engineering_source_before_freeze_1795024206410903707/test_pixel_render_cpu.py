"""CPU单元夹具；0native、0CUDA、0优化器更新。理想纹理从解析世界定义产生。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,copy
from pathlib import Path
import numpy as np
import pytest
import torch
from pixel_world_material import (SharedWorldTexture,ContributionBasis,cpu_front_to_back,
    shade_pixel_material,pixel_box_samples,look_at_camera)
from native_basis_adapter import export_native_basis,NativeExecutionBlocked

torch.set_num_threads(1)


def texture(gate=1.):
    h=w=32;base=np.full((h,w,3),.4);valid=np.ones((h,w));permission=np.full((h,w),gate)
    return SharedWorldTexture(base,valid,permission,latent_mask=np.ones((h,w)),reg_support=np.ones((h,w)),high_bound=.34)


def test_camera_change_preserves_world_point_and_pixel_jacobian():
    X=np.array([.13,1.5,0.]);colors=[]
    for camera in[look_at_camera([0,-2,1.6],[0,3,0]),look_at_camera([.4,-1.7,1.4],[.2,3.5,0])]:
        M=np.linalg.inv(camera.world_from_camera);p=M[:3,:3]@X+M[:3,3];uv=camera.K[:2,:2]@(p[:2]/p[2])+camera.K[:2,2]-.5
        hit,valid,J=camera.plane_map(pixels=uv)
        assert valid and np.max(abs(hit-X))<1e-12
        eps=1e-4;numeric=np.stack([(camera.plane_map(pixels=uv+np.eye(2)[i]*eps)[0][:2]-camera.plane_map(pixels=uv-np.eye(2)[i]*eps)[0][:2])/(2*eps)for i in(0,1)],-1)
        assert np.max(abs(numeric-J))<1e-8
        colors.append(.3+.1*np.sin(hit[:2]))
    assert np.max(abs(colors[0]-colors[1]))<1e-12


def alias_metrics():
    u=np.linspace(30,70,129);centers=np.c_[u,np.full(len(u),10.)];J=np.broadcast_to(np.diag([6.,1.]),(len(u),2,2))
    samples,weights,meta=pixel_box_samples(centers,J,max_axis_samples=16)
    point=.4+.25*np.cos(2*np.pi*u/4)
    target=.4+.25*np.cos(2*np.pi*u/4)*np.sinc(6/4)
    pixel=(.4+.25*np.cos(2*np.pi*samples[...,0]/4))@weights
    return dict(point_RMSE=float(np.sqrt(np.mean((point-target)**2))),pixel_RMSE=float(np.sqrt(np.mean((pixel-target)**2))),
        sample_axis=meta['axis_samples'],sampling_qualified=meta['sampling_qualified'],target='analytic box integral of world cosine, not GS output')


def test_pixel_filter_suppresses_alias_and_flags_stress():
    m=alias_metrics();assert m['pixel_RMSE']<.03*m['point_RMSE']and m['sampling_qualified']
    _,_,stress=pixel_box_samples([[10,10]],[[[40,0],[0,1]]],max_axis_samples=16)
    assert not stress['sampling_qualified']and not stress['Gaussian_Sigma_applied']


def test_teacher_permission_not_prior_label_controls_H_gradient():
    observed=texture(0);teacher=texture(1)
    assert torch.equal(observed.low_mask,teacher.low_mask)and torch.equal(observed.latent_mask,teacher.latent_mask)and torch.equal(observed.reg_support,teacher.reg_support)
    uv=np.array([[[20.2,20.3]]]);pred,_=teacher.sample(uv);loss=(pred[0,0]-.65).square();loss.backward()
    gradient=float(teacher.high.grad[20,20,0]);assert abs(gradient)>1e-4
    pred0,_=observed.sample(uv);pred0.sum().backward();assert float(observed.high.grad.abs().max())==0
    eps=1e-5;values=[]
    with torch.no_grad():
        for sign in(1,-1):
            teacher.high[20,20,0]=sign*eps;y,_=teacher.sample(uv);values.append(float((y[0,0]-.65).square()))
        teacher.high[20,20,0]=0
    assert abs((values[0]-values[1])/(2*eps)-gradient)<1e-8
    # 此case显式设定original_source=3/unknown=True/known=False，但最终teacher-base已合法。
    # core不消费这些prior标签；K/conflict许可为0时上面的observed=0路径同样给H梯度0。


def fixture_bases():
    K=torch.zeros((4,5),dtype=torch.bool);K[0,0]=True
    a=torch.full((3,4,5),.4,dtype=torch.float64);a[0]=.65;a[1]=.5;a[2,0,0]=0
    c=torch.tensor([[.1,.2,.3],[.3,.3,.3],[.5,.5,.5]],dtype=torch.float64)
    joint,_=cpu_front_to_back(a,c,[False,False,True],protected=K)
    road,_=cpu_front_to_back(a[1:],c[1:],[False,True],mode='road_only',protected=K)
    return joint,road


def test_readonly_joint_transmittance_and_K_guard():
    joint,road=fixture_bases();rgb=torch.full((4,5,3),.8,dtype=torch.float64);valid=torch.ones((4,5),dtype=torch.bool)
    result=shade_pixel_material(joint,rgb,valid)
    assert torch.equal(result[:,joint.protected],joint.baseline_rgb[:,joint.protected])
    assert bool((joint.eligible_weight<road.eligible_weight).any())
    assert torch.max(abs(result-(joint.baseline_rgb-joint.eligible_rgb+joint.eligible_weight[None]*.8)))<1e-15
    wrong=copy.copy(joint);wrong.eligible_weight=joint.alpha.clone()
    with pytest.raises(ValueError):shade_pixel_material(wrong,rgb,valid)
    assert torch.equal(shade_pixel_material(joint,rgb,torch.zeros_like(valid)),joint.baseline_rgb)


def test_frozen_geometry_and_stale_basis_rejected():
    joint,_=fixture_bases();rgb=torch.full((4,5,3),.6,dtype=torch.float64,requires_grad=True)
    frozen=joint.alpha.clone();joint.binding=dict(camera='A',geometry='fixed',scope='S',edit='delete10')
    with pytest.raises(ValueError):shade_pixel_material(joint,rgb,torch.ones((4,5),dtype=torch.bool),map_binding=dict(camera='B'))
    result=shade_pixel_material(joint,rgb,torch.ones((4,5),dtype=torch.bool),map_binding=joint.binding);result.sum().backward()
    assert torch.equal(frozen,joint.alpha)and joint.alpha.grad is None and float(rgb.grad.abs().max())>0
    uv=torch.tensor([[[5.,6.]]],requires_grad=True)
    with pytest.raises(ValueError):texture().sample(uv)


def test_world_asset_reload_and_new_query(tmp_path):
    model=texture();model.high.data[12:17,12:17]=.2;uv=np.array([[[12.2,13.3]],[[17.,18.]]]);expected=model.sample(uv)[0]
    p=tmp_path/'asset';model.save_asset(p,chart=dict(world_from_uv=np.eye(4)),scope_world_rows=[1,4,7],
        geometry_reference='fixed_G_s_fixture',source_ledger='explicit_final_teacher_permission',permission_mode='observed_plus_teacher_H')
    loaded,meta=SharedWorldTexture.load_asset(p)
    assert torch.equal(expected,loaded.sample(uv)[0])and meta['new_view_or_edit_requires_fresh_native_basis']
    assert loaded.sample(np.array([[[4.1,20.2]]]))[0].shape==(1,3)
    assert meta['renderer_mode']=='eligible_native_weights_pixel_world'and not meta['camera_cache_is_asset']


def runtime_parity():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from world_texture_runtime import WorldTextureRuntime
    ref=texture(.8);rng=np.random.default_rng(101)
    with torch.no_grad():
        ref.low.copy_(torch.as_tensor(rng.normal(0,.1,ref.low.shape)));ref.high.copy_(torch.as_tensor(rng.normal(0,.1,ref.high.shape)))
    runtime=WorldTextureRuntime(ref.base.numpy(),ref.valid[...,0].numpy(),ref.H_permission[...,0].numpy(),ref.latent_mask[...,0].numpy(),ref.reg_support[...,0].numpy(),
        low_mask=ref.low_mask[...,0].numpy(),high_bound=.34)
    with torch.no_grad():runtime.low.copy_(ref.low.float());runtime.high.copy_(ref.high.float())
    center=rng.uniform(8,23,(13,2));J=np.broadcast_to(np.diag([1.5,.8]),(13,2,2));uv,w,_=pixel_box_samples(center,J)
    a=ref.sample(uv,w)[0];b=runtime.sample(uv,w)[0];a.sum().backward();b.sum().backward()
    return dict(color_max=float((a-b.double()).detach().abs().max()),high_gradient_max=float((ref.high.grad-runtime.high.grad.double()).abs().max()),
        low_gradient_max=float((ref.low.grad-runtime.low.grad.double()).abs().max()),CPU_reference_dtype='float64',runtime_parameter_dtype='float32',
        high_bound=.34,GPU_executed=False)


def test_device_generic_runtime_matches_CPU_reference():
    m=runtime_parity();assert max(m[k]for k in('color_max','high_gradient_max','low_gradient_max'))<1e-6,m


def test_native_export_blocked_before_context_access():
    class ForbiddenContext:
        def __getattribute__(self,name):raise AssertionError('禁止native/context访问')
    with pytest.raises(NativeExecutionBlocked):export_native_basis(ForbiddenContext(),86,[True],Path('never_written'),authorization=None)


def test_Q_ledger_requires_bound_recipe(tmp_path):
    with pytest.raises(ValueError):SharedWorldTexture.from_permission_ledger(tmp_path/'not_loaded.npz',bound_recipe={})
