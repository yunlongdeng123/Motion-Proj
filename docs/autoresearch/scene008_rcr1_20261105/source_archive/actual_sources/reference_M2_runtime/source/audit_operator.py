"""Independent SciPy/adjoint/determinism and original SH0 optimizer checks."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,math
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import load_npz
from deterministic_csr import Matrix
from sh_asset import SHAsset,basis3
from fit_sh import project,BOUND

RUN=Path(__file__).resolve().parent.parent;PARENT=RUN.parent/'scene008_a6_complete_asset_20261105'
@torch.no_grad()
def main():
    rng=np.random.RandomState(6);A=load_npz(RUN/'m1/design_SH3.npz').astype(np.float32)
    x=rng.uniform(-.5,.5,(A.shape[1],3)).astype(np.float32);vector=torch.tensor(x,device='cuda');op=Matrix(A)
    result=op(vector);expected=A.astype(np.float64)@x.astype(np.float64)
    error=float(np.abs(result.cpu().numpy()-expected).max());assert error<1e-6,error
    for _ in range(8):assert torch.equal(result,op(vector))
    y=rng.uniform(-.5,.5,(A.shape[0],3)).astype(np.float32);opT=Matrix(A.T)
    left=float((result.double()*torch.tensor(y,device='cuda').double()).sum())
    right=float((vector.double()*opT(torch.tensor(y,device='cuda')).double()).sum())
    adjoint_error=abs(left-right)/max(1,abs(left),abs(right));assert adjoint_error<1e-6,adjoint_error
    theta=project(torch.tensor(rng.normal(0,.8,(300,16,3)).astype(np.float32),device='cuda'))
    norm=torch.linalg.vector_norm(theta[:,1:],dim=1)
    violation=float((BOUND*norm-torch.minimum(theta[:,0],1-theta[:,0])).max());assert violation<3e-7
    directions=rng.normal(size=(10000,3));directions/=np.linalg.norm(directions,axis=1)[:,None]
    theorem=float(np.abs((basis3(directions)[:,1:]**2).sum(1)-15/(4*np.pi)).max());assert theorem<4e-15
    # Re-polish the exact original bounded SH0 objective to expose any solver confound.
    a=load_npz(PARENT/'appearance/appearance_design.npz').astype(np.float32);f=Matrix(a);b=Matrix(a.T)
    with np.load(PARENT/'appearance/appearance_inputs.npz') as z:
        prior=torch.tensor(z['prior'],dtype=torch.float32,device='cuda');target=torch.tensor(z['target'],dtype=torch.float32,device='cuda')
    original=SHAsset.from_geometry(PARENT/'appearance/complete_a6_fitted.npz').rgb.cuda()
    N=len(original);M=a.shape[0];D=torch.tensor(2*(np.asarray(abs(a).T@np.asarray(abs(a).sum(1)).ravel()).ravel()/M+.001/N),device='cuda')[:,None]
    objective=lambda q:float((f(q)-target).double().square().sum()/M+.001*(q-prior).double().square().sum()/N)
    gradient=lambda q:2*b(f(q)-target)/M+.002/N*(q-prior)
    x=original.clone();y=x.clone();t=1.;before=objective(x)
    for _ in range(2000):
        next_x=(y-gradient(y)/D).clamp(0,1);tn=(1+math.sqrt(1+4*t*t))/2;y=next_x+(t-1)/tn*(next_x-x);x=next_x;t=tn
    polished=objective(x)
    record=dict(status='PASS',deterministic_CSR_CPU_FP64_maxabs=error,exact_repeated_runs=8,relative_adjoint_error=adjoint_error,
      cone_violation=violation,SH_addition_theorem_error=theorem,
      original_SH0_objective=before,repolished_SH0_objective=polished,
      original_BUILD_normalized_MAE=float((f(original)-target).abs().double().mean()),
      repolished_BUILD_normalized_MAE=float((f(x)-target).abs().double().mean()),
      adopted_repolished_asset=False,extra_QC_steps=2000)
    (RUN/'operator_audit.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)

if __name__=='__main__':main()
