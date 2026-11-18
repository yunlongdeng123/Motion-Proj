"""新共同base/valid和H=.34的小实际Σ数值资格；无全cache扫描/无GPU。"""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,sys,time
import numpy as np
from scipy.ndimage import gaussian_filter
from teacher_detail_permission import P,Q,N,BASE,load,lowpass,sample,sha
def integrate(base,valid,gate,H,uv,geo,w):
 v,_=sample(valid.astype(float),uv);v*=geo
 b,_=sample(base*valid[...,None],uv);b*=geo[...,None]
 h,_=sample(H,uv);g,_=sample(gate,uv)
 mass=(w[None]*v).sum(1)
 rgb=(w[None,:,None]*(b+v[...,None]*g[...,None]*h)).sum(1)/mass[:,None].clip(1e-12)
 return rgb.clip(0,1),mass
def main(output):
 output=Path(output);output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 d=load(Q/'permission/final_r1/teacher_evidence.npz')
 selected=load(Q/'permission/ledger_r2/small_Sigma_hf.npz',['active_indices'])['active_indices']
 g=load(P/'cpu_exact_r1/geometry_kernel.npz',['active_world_rows','world_xyz','normal_world','sigma_m'])
 a=load(BASE,['surface_world_xyz','grid_world_row','world_from_uv']);prior=load(N/'provenance.npz',['world_keys'])
 plan15=load(P/'chart_plan_candidate_r1/geometry_plan.npz',['weights','geometric_sample_valid'])
 uv15=np.asarray(np.load(P/'chart_plan_candidate_r1/sample_uv_f64.npy',mmap_mode='r')[selected])
 sys.path.insert(0,str(P/'code'))
 from road_chart_sampling import make_exact_sample_plan
 ref=make_exact_sample_plan(a['surface_world_xyz'],a['grid_world_row'],prior['world_keys'],g['active_world_rows'][selected],
  g['normal_world'][selected],g['sigma_m'][selected],a['world_from_uv'],order=41,world_xyz=g['world_xyz'][selected])
 base,valid,gate=d['common_base_srgb'],d['common_material_valid'],d['mH_union']
 y,x=np.mgrid[:valid.shape[0],:valid.shape[1]];mask=d['common_latent_mask'];bound=.34
 baseHF=base-lowpass(base,valid,3)
 teacher_raw=np.clip(2*(d['teacher_hf_srgb']-baseHF)/bound,-.95,.95)*mask[...,None]
 fields={'base_H0':np.zeros_like(base)}
 for name,arg in [('H_nyquist_u',np.pi*x+np.pi/4),('H_nyquist_diag45',np.pi*(x+y)/np.sqrt(2)+np.pi/4)]:
  raw=.8*np.sin(arg)[...,None]*np.ones(3)*mask[...,None]
  blur=np.stack([gaussian_filter(raw[...,i],3,mode='nearest',truncate=3) for i in range(3)],-1)
  fields[name]=.5*bound*(raw-blur)
 blur=np.stack([gaussian_filter(teacher_raw[...,i],3,mode='nearest',truncate=3) for i in range(3)],-1)
 fields['teacher_HF_seed']=.5*bound*(teacher_raw-blur)
 rows=[];allpass=True
 for name,high in fields.items():
  actual,m15=integrate(base,valid,gate,high,uv15,plan15['geometric_sample_valid'][selected],plan15['weights'])
  expected,m41=integrate(base,valid,gate,high,ref['sample_uv'],ref['geometric_sample_valid'],ref['weights'])
  e=np.abs(actual-expected);reliable=m41>=.9;rowmax=e.max(1)
  high_e=e[reliable];p99=float(np.quantile(high_e,.99)) if high_e.size else None;maximum=float(high_e.max()) if high_e.size else None
  passed=bool(p99 is not None and p99<=1/255 and maximum<=4/255);allpass &=passed
  rows.append(dict(case=name,rows=len(selected),reference_mass_ge_point9=int(reliable.sum()),mass_low_rows=int((~reliable).sum()),
   RGB_abs_p99=p99,RGB_abs_max=maximum,rowmax_p99=float(np.quantile(rowmax[reliable],.99)) if reliable.any() else None,
   low_mass_abs_max=float(e[~reliable].max()) if (~reliable).any() else None,
   mass_abs_max=float(np.abs(m15-m41).max()),passed=passed))
  np.savez_compressed(output/(name+'.npz'),active_indices=selected,world_rows=g['active_world_rows'][selected],candidate_RGB=actual,reference_RGB=expected,
   candidate_mass=m15,reference_mass=m41,error=e)
 result=dict(schema='new_teacher_permission_GH_small_CPU_qualification.v1',status='PASS' if allpass else 'FAIL_ENGINEERING_INTEGRATION_PENDING',
  CPU_only=True,CUDA_calls=0,native_calls=0,optimizer_updates=0,source_rows=int(len(selected)),GH15_points=int(len(selected)*225),
  GH41_reference_points=int(len(selected)*1681),high_bound=.34,seed_H0_and_legal_Hstress=True,
  thresholds=dict(reference_mass_ge=.9,RGB_abs_p99_max=1/255,RGB_abs_max_max=4/255),cases=rows,
  source_scope='same deterministic150worldrows as ledger_r2 amplitude audit; no full34M requery. Dense truechart inverse of current fixed worldSigma.',
  interpretation='Numerical source qualification only; source-mass<.9 separate, no removed train/eval denominator; not a strict error certificate',
  old_P_numeric_not_substituted=True,GPU_future_gate_passed=allpass,seconds=time.monotonic()-start)
 (output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);main(ap.parse_args().output)

