import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import copy,random
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from scipy import ndimage as ndi

from gs_color_oracle import (ColorLayout,BoundedGSColors,NativeSparseColorRead,InjectedNativeGSOperator,
    gaussian_masked_hf,source_stable_hf_mse,capture_full_state,restore_full_state)
from box_quadratic_solver import (ProjectedBoxQuadratic,box_frank_wolfe_gap,
    capture_solver_checkpoint,restore_solver_checkpoint)

torch.set_num_threads(1)


def layout():return ColorLayout(np.array([7,19,33]),np.array([True,False,True]))


def model():return BoundedGSColors(torch.tensor([[.2,.3,.4],[.8,.7,.6],[.4,.5,.6]],dtype=torch.float64),layout())


def test_global_world_identity_and_only_eligible_color_gradient():
    m=model();rgb=m();(rgb*torch.arange(9).reshape(3,3)).sum().backward()
    assert torch.equal(rgb[1],m.c0[1])and np.array_equal(m.layout.eligible_world_rows,[7,33])
    assert torch.equal(m.rgb.grad,torch.tensor([[0.,1.,2.],[6.,7.,8.]],dtype=torch.float64))
    with pytest.raises(ValueError):ColorLayout(np.array([33,7]),np.ones(2,bool))
    with pytest.raises(ValueError):ColorLayout(np.array([7,33]),np.ones(2,np.uint8))


def test_box_projection_serialization_and_readonly_exact(tmp_path):
    m=model()
    with torch.no_grad():m.rgb[0,0]=-1;m.rgb[1,2]=2
    with pytest.raises(ValueError):m()
    m.project_();p=m.payload();m.save(tmp_path/'oracle.pt');reloaded=BoundedGSColors.load(tmp_path/'oracle.pt')
    assert torch.equal(reloaded(),m())and torch.equal(m()[1],m.c0[1])
    assert m.rgb[0,0]==0 and m.rgb[1,2]==1
    bad=copy.deepcopy(p);bad['eligible_active']=np.array([False,True,True])
    with pytest.raises(ValueError):BoundedGSColors.from_payload(bad)


def test_individual_GS_sparse_read_matches_dense_forward_and_adjoint():
    m=model();W=torch.tensor([[.2,.7],[.5,0],[0,.1]],dtype=torch.float64);P0=torch.full((3,3),.3,dtype=torch.float64)
    read=NativeSparseColorRead(W.to_sparse(),P0,m.rgb.detach(),np.array([3,10,12]),m.layout,
        receipt=dict(scope_sha256=m.layout.signature,role='synthetic_unit_test'))
    x=(m.rgb.detach()+.1).requires_grad_();probe=torch.arange(9,dtype=torch.float64).reshape(3,3)/9
    value=read(x);gradient=torch.autograd.grad((value*probe).sum(),x)[0]
    assert torch.allclose(value,P0+W@(.1*torch.ones_like(x)),atol=1e-15,rtol=0)
    assert torch.allclose(gradient,W.T@probe,atol=1e-15,rtol=0)
    with pytest.raises(ValueError):NativeSparseColorRead(W.sum(1,keepdim=True).to_sparse(),P0,m.rgb.detach(),np.array([3,10,12]),m.layout,receipt=read.receipt)
    with pytest.raises(ValueError):NativeSparseColorRead(W.to_sparse(),P0,m.rgb.detach(),np.array([3,10,12]),m.layout,receipt=dict(scope_sha256=m.layout.signature,role='actual_native_perGaussian_alphaT',native_qualified=False))


def test_masked_HF_is_scipy_equivalent_and_ignores_foreign_RGB():
    rng=np.random.default_rng(101);image=rng.random((3,19,21));mask=np.ones((19,21),bool);mask[:,0:3]=False;mask[4:7,12:15]=False
    altered=image.copy();altered[:,~mask]=900
    a=torch.tensor(image,dtype=torch.float64,requires_grad=True)
    hf,mass=gaussian_masked_hf(a,mask,1)
    changed,_=gaussian_masked_hf(torch.tensor(altered),mask,1)
    n=ndi.gaussian_filter(mask.astype(float),1,mode='constant',truncate=3)
    expected=np.stack([image[c]-ndi.gaussian_filter(np.where(mask,image[c],0),1,mode='constant',truncate=3)/np.maximum(n,1e-300)for c in range(3)])
    expected[:,~mask]=0
    assert np.max(abs(hf.detach().numpy()-expected))<5e-16
    # Two separable implementations normalize/sum FP64 kernel weights in a
    # different order; compare within four machine eps, not sub-ulp equality.
    assert np.max(abs(mass.numpy()-n))<4*np.finfo(np.float64).eps and torch.equal(hf,changed)
    grad=torch.autograd.grad(hf.square().sum(),a)[0]
    assert torch.equal(grad[:,~torch.from_numpy(mask)],torch.zeros_like(grad[:,~torch.from_numpy(mask)]))


def test_stable_straight_HF_does_not_measure_fixed_alpha_texture():
    yy,xx=np.indices((19,21));alpha=torch.tensor(.975+.025*np.sin(xx),dtype=torch.float64)
    teacher=torch.full((3,19,21),.4,dtype=torch.float64);context=np.ones((19,21),bool);core=np.zeros_like(context);core[4:-4,4:-4]=True
    rgb=(alpha[None]*teacher).requires_grad_()
    loss=source_stable_hf_mse(rgb,alpha,teacher,context,core,1)
    assert loss<1e-30
    with pytest.raises(ValueError):source_stable_hf_mse(rgb,alpha-.1,teacher,context,core,1)


def toy():
    a=np.array([1.,2.,4.]);t=np.array([-.2,.4,1.2]);queried=[]
    def evaluate(x,need_gradient):
        queried.append(x.copy());d=x-t
        return .5*np.sum(a*d*d),a*d if need_gradient else None,dict(role='synthetic_convex_test')
    return evaluate,queried,.5*np.sum(a*(np.clip(t,0,1)-t)**2)


def test_projected_solver_and_conditional_FW_bound_on_box_quadratic():
    evaluate,queried,optimum=toy()
    solver=ProjectedBoxQuadratic(np.array([.7,.9,.2]),evaluate,identity=dict(source='toy'),comparison_atol=1e-14)
    for _ in range(80):
        row=solver.step()
        assert row['objective']-optimum<=row['measured_box_FW_gap']+2e-15
        assert row['objective']<=row['previous_objective']+1e-14
    assert solver.value-optimum<1e-13
    assert all(np.all((x>=0)&(x<=1))for x in queried)
    assert box_frank_wolfe_gap(solver.x,solver.gradient)['strict_numerical_certificate']is False
    bound=box_frank_wolfe_gap(solver.x,solver.gradient,gradient_linf_error=1e-10,objective_abs_error=1e-12,value=solver.value)
    assert bound['training_optimum_interval'][0]<=optimum<=bound['training_optimum_interval'][1]


def test_solver_full_restore_next_query_replay_and_identity_rejection():
    evaluate,_,_=toy();solver=ProjectedBoxQuadratic(np.array([.7,.9,.2]),evaluate,identity=dict(source='toy'),comparison_atol=1e-14)
    for _ in range(3):solver.step()
    saved=copy.deepcopy(solver.snapshot());solver.step();first=solver.snapshot()
    solver.restore(saved);solver.step();second=solver.snapshot()
    for key in ['x','gradient']:assert np.array_equal(first[key],second[key])
    for key in ['value','L','iteration','evaluations','gradient_evaluations']:assert first[key]==second[key]
    assert first['trace']==second['trace']
    assert np.array_equal(saved['x'],saved['x'].copy())and saved['iteration']==3
    bad=copy.deepcopy(saved);bad['identity']={'source':'other'}
    with pytest.raises(ValueError):solver.restore(bad)
    full=capture_solver_checkpoint(solver,role='synthetic',frames=[86,110])
    expected=(torch.rand(2),np.random.rand(2),random.random())
    solver.step();restore_solver_checkpoint(full,solver,role='synthetic',frames=[86,110])
    observed=(torch.rand(2),np.random.rand(2),random.random())
    assert torch.equal(expected[0],observed[0])and np.array_equal(expected[1],observed[1])and expected[2]==observed[2]
    with pytest.raises(ValueError):restore_solver_checkpoint(full,solver,role='formal',frames=[86,110])


def test_full_field_Adam_RNG_sampler_restore_without_any_optimizer_update():
    torch.manual_seed(101);np.random.seed(101);random.seed(101)
    m=model();opt=torch.optim.Adam([m.rgb],lr=.01)
    opt.state[m.rgb]=dict(step=torch.tensor(9.),exp_avg=torch.ones_like(m.rgb)*.3,exp_avg_sq=torch.ones_like(m.rgb)*.2)
    saved=capture_full_state(m,opt,step=3,trace=[86,110,86],sampler_state={'queue':[110]},identity={'teacher':'fixed'},role='synthetic')
    with torch.no_grad():m.rgb.fill_(.8);opt.state[m.rgb]['step'].fill_(99)
    state=restore_full_state(saved,m,opt,identity={'teacher':'fixed'},role='synthetic')
    assert state['sampler_state']=={'queue':[110]}and saved['optimizer']['state'][0]['step']==9
    first=(torch.rand(3),np.random.rand(3),random.random())
    opt.state[m.rgb]['exp_avg'].fill_(99)
    restore_full_state(saved,m,opt,identity={'teacher':'fixed'},role='synthetic')
    second=(torch.rand(3),np.random.rand(3),random.random())
    assert torch.equal(first[0],second[0])and np.array_equal(first[1],second[1])and first[2]==second[2]
    assert torch.all(opt.state[m.rgb]['exp_avg']==.3)
    with pytest.raises(ValueError):restore_full_state(saved,m,opt,identity={'teacher':'changed'},role='synthetic')


def test_passive_native_adapter_sends_individual_colors_and_detaches_fixed_alpha():
    m=model();W=torch.tensor([[.2,.3,.4],[.1,.2,.3]],dtype=torch.float64)
    class FakeContext:
        frames=(86,);active_world_rows=m.layout.active_world_rows;base_rgb=m.c0
        def camera(self,raw):return SimpleNamespace(image_height=1,image_width=2)
        def base_packet(self,raw):return dict(means3D=torch.zeros((3,3)),opacities=torch.ones((3,1)),scales=torch.ones((3,3)),rotations=torch.zeros((3,4)))
        def render_road(self,raw,colors):return dict(rgb=(W@colors).T.reshape(3,1,2),alpha=(colors.sum()*0+torch.ones((1,1,2))))
        def assert_frozen(self):pass
    adapter=InjectedNativeGSOperator(FakeContext(),m.layout,image_hw=(1,2));out=adapter.road(86,m())
    assert out['native_alpha_output_requires_grad']and not out['alpha'].requires_grad
    grad=torch.autograd.grad(out['rgb'].sum(),m.rgb)[0]
    assert torch.allclose(grad,W[:,[0,2]].sum(0)[:,None].expand(2,3))
