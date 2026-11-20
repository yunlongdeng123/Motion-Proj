import copy
import numpy as np
import pytest
from scipy import ndimage as ndi
import torch
from native_capacity_loss import NativeCapacityLoss,LossSpec,masked_gaussian,strict_core,signed_sobel

torch.set_num_threads(1)

def fixture():
    rng=np.random.default_rng(101);h,w=38,42
    alpha=rng.uniform(.95,1.,(h,w)).astype(np.float32)
    context=np.ones((h,w),bool);context[14:20,17:24]=False
    score=np.zeros_like(context);score[4:34,4:38]=True
    frame=dict(alpha=alpha,targets={k:rng.uniform(.1,.7,(3,h,w)).astype(np.float32)for k in('A','C','S')},
        contexts={k:context.copy()for k in('A','C','S')},scores={k:score.copy()for k in('A','C','S')})
    base=torch.tensor(rng.uniform(.1,.5,(3,h,w)).astype(np.float32))*torch.from_numpy(alpha)[None]
    direction=torch.tensor(rng.uniform(-.08,.08,(3,h,w)).astype(np.float32))*torch.from_numpy(alpha)[None]
    return {86:frame},base,direction

def test_masked_gaussian_and_signed_sobel_match_independent_scipy():
    frames,P,_=fixture();mask=frames[86]['contexts']['A'];x=P.double();m=torch.from_numpy(mask)
    a=masked_gaussian(x,m,2.,3.)[0].numpy()
    mass=ndi.gaussian_filter(mask.astype(float),2.,mode='constant',truncate=3.)
    expected=np.stack([ndi.gaussian_filter(np.where(mask,c,0),2.,mode='constant',truncate=3.)for c in x.numpy()])/np.maximum(mass,1e-12)
    np.testing.assert_allclose(a,expected,atol=1e-14)
    sob=signed_sobel(x,m).numpy();clean=np.where(mask[None],x.numpy(),0)
    np.testing.assert_allclose(sob[0],np.stack([ndi.sobel(c,axis=1,mode='constant')/8 for c in clean]),atol=1e-14)
    np.testing.assert_allclose(sob[1],np.stack([ndi.sobel(c,axis=0,mode='constant')/8 for c in clean]),atol=1e-14)

def test_affine_color_objective_is_quadratic_convex_and_gradient_correct():
    frames,P,d=fixture();loss=NativeCapacityLoss(frames);loss.calibrate({86:P})
    c=torch.tensor(.4,requires_grad=True);value,_=loss.objective({86:P+c*d});value.backward()
    eps=.005
    fp=float(loss.objective({86:P+(float(c.detach())+eps)*d})[0].detach());fm=float(loss.objective({86:P+(float(c.detach())-eps)*d})[0].detach())
    assert float(c.grad)==pytest.approx((fp-fm)/(2*eps),rel=.015,abs=2e-4)
    f0=float(loss.objective({86:P})[0]);f1=float(loss.objective({86:P+d})[0])
    assert float(value.detach())<=.6*f0+.4*f1+1e-5

def test_zero_context_crop_is_exact_against_full_image_scipy_terms():
    frames,P,d=fixture();frame=frames[86]
    for key in ('A','C','S'):
        frame['contexts'][key][:]=False;frame['contexts'][key][7:32,8:36]=True
    loss=NativeCapacityLoss(frames);r=loss.regions[86]['A'];values=r.terms(P+d)
    context=frame['contexts']['A'];score=frame['scores']['A'];alpha=frame['alpha']
    E=np.where(context[None],((P+d).numpy()-alpha[None]*frame['targets']['A'])/alpha[None],0)
    assert float(values['RGB'])==pytest.approx(float(np.mean(E[:,score&context]**2)),rel=2e-6)
    for sigma in (1.,2.,4.):
        mass=ndi.gaussian_filter(context.astype(float),sigma,mode='constant',truncate=3)
        sm=np.stack([ndi.gaussian_filter(c,sigma,mode='constant',truncate=3)for c in E])/np.maximum(mass,1e-12)
        mask=strict_core(context,int(3*sigma+.5))&score
        ref=float(np.mean((E-sm)[:,mask]**2)) if mask.any() else 0.
        assert float(values['HF_'+str(sigma)])==pytest.approx(ref,rel=3e-6,abs=1e-9)
    assert r.bounds==(7,32,8,36)

def test_held_teacher_labels_cannot_change_seen_values_normalizers_or_gradient():
    frames,P,d=fixture();changed=copy.deepcopy(frames)
    for k in ('A','C','S'):changed[86]['targets'][k][:,~changed[86]['contexts'][k]]=12345.
    a=NativeCapacityLoss(frames);b=NativeCapacityLoss(changed)
    assert a.calibrate({86:P})==b.calibrate({86:P})
    results=[]
    for obj in(a,b):
        c=torch.tensor(.25,requires_grad=True);v,_=obj.objective({86:P+c*d});v.backward()
        results.append((v.detach(),c.grad))
    assert all(torch.equal(x,y)for x,y in zip(*results))

def test_complete_normalizer_restore_rejects_changed_context():
    frames,P,d=fixture();a=NativeCapacityLoss(frames);state=a.calibrate({86:P});b=NativeCapacityLoss(frames);b.restore(copy.deepcopy(state))
    assert torch.equal(a.objective({86:P+d})[0],b.objective({86:P+d})[0])
    with pytest.raises(RuntimeError):a.calibrate({86:P})
    other=copy.deepcopy(frames);other[86]['contexts']['A'][10,10]=False
    with pytest.raises(ValueError):NativeCapacityLoss(other).restore(state)

def test_low_alpha_context_is_rejected_and_optional_detail_does_not_change_rgb():
    frames,P,d=fixture();bad=copy.deepcopy(frames);bad[86]['alpha'][5,5]=.01
    with pytest.raises(ValueError):NativeCapacityLoss(bad)
    loss=NativeCapacityLoss(frames,spec=LossSpec(hf_weight=0.,sobel_weight=0.));loss.calibrate({86:P})
    assert torch.equal(loss.objective({86:P+d},'rgb')[0],loss.objective({86:P+d},'detail')[0])
