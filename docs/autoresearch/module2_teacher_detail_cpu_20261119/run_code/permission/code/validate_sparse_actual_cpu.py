"""实际150row的CSR/dense颜色与L/H梯度/重载CPU等值，无optimizer/native。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,time
import numpy as np
import torch
import torch.nn.functional as F
from fixed_material_read_operator_snapshot import FixedReadOperator
HERE=Path(__file__).resolve().parent
def load(p):
 with np.load(p,allow_pickle=False)as d:return {k:d[k].copy()for k in d.files}
def main():
 torch.set_num_threads(2);start=time.perf_counter()
 root=HERE/'local_avg_gh31_r1';op=FixedReadOperator.load(root/'operator');v=load(root/'direct_validation.npz')
 d=load(HERE/'inputs/local_compile_inputs.npz');pos=v['query_positions']
 if not np.array_equal(op.row_ids,d['query_world_rows']):raise AssertionError('worldrow mapping')
 if not (op.mass>1e-12).all():raise AssertionError('empty common field mass')
 small=FixedReadOperator(op.base[pos],op.mass[pos],op.KL[pos],{k:m[pos]for k,m in op.KH.items()},op.low_shape,op.high_shape,op.row_ids[pos],op.metadata)
 uv=torch.from_numpy(v['sample_uv']);geom=torch.from_numpy(v['geometric_valid']);w=torch.from_numpy(v['weights'])
 def sample(a,u):
  H,W,C=a.shape;grid=torch.stack((u[...,0]*2/(W-1)-1,u[...,1]*2/(H-1)-1),-1)[None]
  return F.grid_sample(a.double().permute(2,0,1)[None],grid.double(),align_corners=True,padding_mode='zeros')[0].permute(1,2,0)
 base=torch.from_numpy(d['common_base_srgb']);valid=torch.from_numpy(d['common_material_valid'].astype(float))[...,None]
 vw=sample(valid,uv)*geom[...,None];b=sample(base*valid,uv)*geom[...,None];den=(w[None,:,None]*vw).sum(1)
 score=torch.sin(torch.arange(len(pos)*3).reshape(len(pos),3).float()+.3)
 checks={}
 for name,key in [('observed','observed_H_prior'),('teacher','mH_union')]:
  low=torch.tensor(v['probe_low'],dtype=torch.float32,requires_grad=True);high=torch.tensor(v['probe_high'],dtype=torch.float32,requires_grad=True)
  prediction=small.torch_query(low,high,name);(prediction*score).sum().backward()
  gl,gh=low.grad.clone(),high.grad.clone()
  l2=torch.tensor(v['probe_low'],dtype=torch.float32,requires_grad=True);h2=torch.tensor(v['probe_high'],dtype=torch.float32,requires_grad=True)
  gate=sample(torch.from_numpy(d[key])[...,None],uv)
  dense=((w[None,:,None]*(b+vw*(sample(l2,uv/6)+gate*sample(h2,uv)))).sum(1)/den).clamp(0,1).float()
  (dense*score).sum().backward()
  errors=dict(RGB_max=float((prediction-dense).abs().max()),low_gradient_max=float((gl-l2.grad).abs().max()),
   high_gradient_max=float((gh-h2.grad).abs().max()),finite=bool(torch.isfinite(gh).all()&torch.isfinite(gl).all()))
  if max(errors[k]for k in ['RGB_max','low_gradient_max','high_gradient_max'])>1e-6 or not errors['finite']:raise AssertionError(str(errors))
  checks[name]=errors
 result=dict(schema='actual_GH31_fixed_sparse_CPU_equivalence.v1',passed=True,rows=len(pos),full_operator_rows=len(op.row_ids),kernel_order=31,
  gates=checks,row_ids_exact=True,all_query_mass_positive=True,mass_quantiles=np.quantile(op.mass,[0,.05,.5,.95,1]).tolist(),
  CPU_only=True,CUDA_calls=0,native_calls=0,optimizer_updates=0,sparse_reload_used=True,
  not_a_future_CUDA_sparse_ABI_smoke=True,seconds=time.perf_counter()-start)
 (root/'gradient_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()

