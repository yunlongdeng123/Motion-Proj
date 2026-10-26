"""Exact real-SH orientation control using the frozen native eval_sh basis.

This is an operator correctness control, not relighting or a learned method.
The default baseline still inherits its original world-space SH coefficients.
"""
import math,sys
from pathlib import Path
import numpy as np
import torch
SOURCE=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018/training_source/scene0008_089_137')
sys.path.insert(0,str(SOURCE))
from utils.sh_utils import eval_sh

def matrix(rotation,degree):
    R=np.asarray(rotation,dtype=float);assert np.allclose(R.T@R,np.eye(3),atol=1e-8) and np.isclose(np.linalg.det(R),1)
    K=(degree+1)**2;i=np.arange(128);z=1-2*(i+.5)/128;theta=i*np.pi*(3-np.sqrt(5));xy=np.sqrt(1-z*z);directions=np.c_[xy*np.cos(theta),xy*np.sin(theta),z]
    coefficients=torch.eye(K,dtype=torch.float64)[None].expand(128,-1,-1)
    A=eval_sh(degree,coefficients,torch.tensor(directions)).numpy();B=eval_sh(degree,coefficients,torch.tensor(directions@R)).numpy()
    D=np.linalg.lstsq(A,B,rcond=None)[0]
    assert np.max(np.abs(A@D-B))<1e-10 and np.max(np.abs(D.T@D-np.eye(K)))<1e-10
    return D

def rotate(features,rotation):
    assert features.ndim==3 and features.shape[-1]==3
    degree=math.isqrt(features.shape[1])-1;assert (degree+1)**2==features.shape[1]
    D=torch.as_tensor(matrix(rotation,degree),device=features.device,dtype=features.dtype)
    return torch.einsum('jk,nkc->njc',D,features)

if __name__=='__main__':
    from scipy.spatial.transform import Rotation
    import json
    rng=np.random.RandomState(7);R=Rotation.from_euler('xyz',[.11,-.07,.24]).as_matrix();proof=[]
    for degree in range(4):
        K=(degree+1)**2;directions=rng.randn(80,3);directions/=np.linalg.norm(directions,axis=1,keepdims=True);C=torch.tensor(rng.randn(80,K,3));new=rotate(C,R)
        expected=eval_sh(degree,C.transpose(1,2),torch.tensor(directions@R));actual=eval_sh(degree,new.transpose(1,2),torch.tensor(directions));error=float((actual-expected).abs().max())
        inverse=float((rotate(new,R.T)-C).abs().max());assert error<1e-10 and inverse<1e-10
        proof.append(dict(degree=degree,native_eval_sh_equivariance_max=error,inverse_rotation_max=inverse))
    print(json.dumps(dict(CPU_operator_control=proof,new_training_updates=0,GPU_render_verified=False,relighting_claim=False)))
