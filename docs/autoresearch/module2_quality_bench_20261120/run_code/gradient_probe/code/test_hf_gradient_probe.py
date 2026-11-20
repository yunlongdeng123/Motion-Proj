import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
from scipy import ndimage as ndi
from hf_gradient_probe import masked_hf_cpu,relation

torch.set_num_threads(1)


def numpy_hf(image,mask,sigma):
    mass=ndi.gaussian_filter(mask.astype(float),sigma,mode='constant',truncate=3)
    clean=np.where(mask[...,None],image,0)
    low=np.stack([ndi.gaussian_filter(clean[...,c],sigma,mode='constant',truncate=3)for c in range(3)],-1)
    low=np.divide(low,mass[...,None],out=np.zeros_like(low),where=mass[...,None]>0)
    return np.where(mask[...,None],clean-low,0)


def test_screen_HF_matches_masked_scipy_and_has_exact_linear_adjoint():
    h,w=9,11;n=h*w;mask=np.ones((h,w),bool);mask[0,0]=False
    rng=np.random.default_rng(101);p=rng.normal(size=(h,w,3));t=rng.normal(size=(h,w,3))
    pred=torch.tensor(p.transpose(2,0,1),requires_grad=True);target=torch.tensor(t.transpose(2,0,1))
    hp,_=masked_hf_cpu(pred,mask,1);ht,_=masked_hf_cpu(target,mask,1)
    assert np.max(abs(hp.detach().numpy().transpose(1,2,0)-numpy_hf(p,mask,1)))<2e-15
    core=ndi.minimum_filter(mask.astype(np.uint8),7,mode='constant').astype(bool)
    loss=(hp-ht).square()[:,torch.from_numpy(core)].mean()
    grad=torch.autograd.grad(loss,pred)[0].numpy().transpose(1,2,0)
    matrix=np.column_stack([numpy_hf(np.repeat(np.eye(n)[:,i].reshape(h,w,1),3,-1),mask,1)[...,0].ravel()for i in range(n)])
    residual=(numpy_hf(p,mask,1)-numpy_hf(t,mask,1))*core[...,None]
    expected=np.stack([2*(matrix.T@residual[...,c].ravel())/(3*core.sum())for c in range(3)],-1).reshape(h,w,3)
    assert np.max(abs(grad-expected))<2e-15


def test_alignment_signs_zero_gradients_and_no_step():
    rgb=torch.tensor([1.,2.]);opposed=-rgb
    aligned=relation(rgb,rgb);conflict=relation(rgb,opposed);zero=relation(rgb,torch.zeros_like(rgb))
    assert aligned['cosine']> .999 and aligned['HF_derivative_along_negative_RGB_gradient']<0
    assert conflict['cosine']<-.999 and conflict['HF_derivative_along_negative_RGB_gradient']>0
    assert zero['cosine']is None and zero['status']=='zero_HF_gradient'
    assert not aligned['finite_step_test_performed']
