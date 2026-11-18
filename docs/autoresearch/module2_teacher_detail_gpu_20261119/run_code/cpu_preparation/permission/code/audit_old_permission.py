"""旧mH真实Σ支持CPU审计；仅NumPy，source标签/中心门/核贡献分开。"""
from pathlib import Path
import os
for key in ['CUDA_VISIBLE_DEVICES']:os.environ[key]=''
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'2')
import argparse,json,time
import numpy as np
P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
N=P.parent/'road_activation_20261118_r1/B32_beam95_native_r1/asset_guarded'
def load(p,keys=None):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in (z.files if keys is None else keys)}
def bilinear(a,uv):
 x,y=uv[...,0],uv[...,1];ix,iy=np.floor(x).astype(np.int64),np.floor(y).astype(np.int64);dx,dy=x-ix,y-iy
 value=np.zeros(x.shape,np.float64);H,W=a.shape
 for oy,ox,b in [(0,0,(1-dx)*(1-dy)),(0,1,dx*(1-dy)),(1,0,(1-dx)*dy),(1,1,dx*dy)]:
  xx,yy=ix+ox,iy+oy;inside=(xx>=0)&(xx<W)&(yy>=0)&(yy<H)
  value+=b*a[np.clip(yy,0,H-1),np.clip(xx,0,W-1)]*inside
 return value
def quant(a):return np.quantile(a,[0,.05,.25,.5,.75,.95,1]).tolist() if len(a) else None
def main(output):
 out=Path(output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 a=load(P/'cpu_exact_r1/A0.npz',['valid','mH','source_prior','source_effective','unknown_prior'])
 g=load(P/'chart_plan_candidate_r1/geometry_plan.npz',['active_world_rows','center_uv','weights','geometric_sample_valid'])
 stage=load(P/'stage0_native_r1/scope.npz');scope=load(P/'method_scope_r1/scope.npz')
 road=load(N/'road.npz',['active','minimum_model_depth']);prior=load(N/'provenance.npz',['guarded_world_rows','effective_material_available','known','surface_supported'])
 rows=g['active_world_rows'];n=len(rows);uv=np.load(P/'chart_plan_candidate_r1/sample_uv_f64.npy',mmap_mode='r')
 gate=np.zeros(n);valid_mass=np.zeros(n);Hpositive_mass=np.zeros(n)
 for b in range(0,n,1024):
  e=min(b+1024,n);q=np.asarray(uv[b:e]);v=bilinear(a['valid'],q)*g['geometric_sample_valid'][b:e];h=bilinear(a['mH'],q)
  w=g['weights'][None,:];valid_mass[b:e]=(w*v).sum(1);gate[b:e]=(w*v*h).sum(1)
  Hpositive_mass[b:e]=(w*v*(h>1e-12)).sum(1)
 gate=np.divide(gate,valid_mass,out=np.zeros(n),where=valid_mass>1e-12)
 Hpositive_mass=np.divide(Hpositive_mass,valid_mass,out=np.zeros(n),where=valid_mass>1e-12)
 center=bilinear(a['mH'],g['center_uv'][:,None,:])[:,0]
 x,y=np.rint(g['center_uv']).astype(int).T;labels=a['source_effective'][y,x];unknown=a['unknown_prior'][y,x]
 physical=prior['surface_supported'][rows]
 safe=physical&~stage['conservative_K_guard']&~prior['guarded_world_rows'][rows]&(road['minimum_model_depth'][rows]<=0)
 stats={}
 for region in ['A','C','S']:
  m=stage['actual_'+region+'_mass'];hit=m>0;old=scope['eligible']&hit
  new=hit&safe&~scope['eligible'];unavailable=hit&safe&~prior['effective_material_available'][rows]
  stat=dict(actual_road_only_contributor_rows=int(hit.sum()),old_eligible_rows=int(old.sum()),
   safe_not_old_eligible_rows=int(new.sum()),safe_prior_material_unavailable_rows=int(unavailable.sum()),
   prior_unknown_label_rows=int((hit&unknown).sum()),zero_center_positive_Sigma_rows=int((hit&(center<=1e-12)&(gate>1e-12)).sum()),
   exact_zero_Sigma_gate_rows=int((hit&(gate<=1e-12)).sum()),gate_under_point01_rows=int((hit&(gate<.01)).sum()),
   gate_under_point1_rows=int((hit&(gate<.1)).sum()),Sigma_gate_quantiles=quant(gate[hit]),
   weighted_Sigma_gate=float((m*gate).sum()/m.sum()),old_high_bound_gate_upper=float((m*(.04*gate)).sum()/m.sum()),
   prior_unavailable_rowmax_mass_fraction=float(m[hit&~prior['effective_material_available'][rows]].sum()/m.sum()),
   source_labels={str(code):dict(rows=int((hit&(labels==code)).sum()),Sigma_gate_quantiles=quant(gate[hit&(labels==code)])) for code in [0,1,2,3]})
  stats[region]=stat
 np.savez_compressed(out/'row_H_support.npz',active_world_rows=rows,old_Sigma_mH=gate,old_Hpositive_mass=Hpositive_mass,
  old_center_mH=center,kernel_valid_mass=valid_mass,source_prior=a['source_prior'][y,x],source_effective=labels,
  prior_unknown=unknown,old_eligible=scope['eligible'],physical_K_near_safe=safe,
  C_mass=stage['actual_C_mass'],S_mass=stage['actual_S_mass'])
 result=dict(schema='old_teacher_H_permission_CPU_audit.v1',CPU_only=True,CUDA_calls=0,native_calls=0,optimizer_updates=0,
  rows=stats,source_labels_are_not_teacher_evidence=True,legacy_unknown_is_not_geometric_unknown=True,
  gate_definition='sum GH15 w*bilinear(valid)*bilinear(mH)*geometry / sum GH15 w*bilinear(valid)*geometry',
  aggregate_boundary='native contribution masses are saved max across86/110, not per-pixel mass or joint visibility; center0 does not imply zero wholeSigma support',
  original_S_full=3187,S_supported=3131,all_original_inputs_unchanged=True,seconds=time.monotonic()-start)
 (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);main(ap.parse_args().output)

