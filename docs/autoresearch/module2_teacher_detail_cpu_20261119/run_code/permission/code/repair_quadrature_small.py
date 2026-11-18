"""确定性提高积分精度；只改数值阶数，复用同150样本/冻结mask/.34/门。"""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,sys,time
import numpy as np
from scipy.ndimage import gaussian_filter
from teacher_detail_permission import P,Q,N,BASE,load,lowpass
from qualify_new_field_small import integrate
def fields(d):
 base=d['common_base_srgb'];valid=d['common_material_valid'];y,x=np.mgrid[:valid.shape[0],:valid.shape[1]]
 mask=d['common_latent_mask'];bound=.34;raw_teacher=np.clip(2*(d['teacher_hf_srgb']-(base-lowpass(base,valid,3)))/bound,-.95,.95)*mask[...,None]
 result={'base_H0':np.zeros_like(base)}
 for name,arg in [('H_nyquist_u',np.pi*x+np.pi/4),('H_nyquist_diag45',np.pi*(x+y)/np.sqrt(2)+np.pi/4)]:
  raw=.8*np.sin(arg)[...,None]*np.ones(3)*mask[...,None]
  smooth=np.stack([gaussian_filter(raw[...,i],3,mode='nearest',truncate=3) for i in range(3)],-1)
  result[name]=.5*bound*(raw-smooth)
 smooth=np.stack([gaussian_filter(raw_teacher[...,i],3,mode='nearest',truncate=3) for i in range(3)],-1)
 result['teacher_HF_seed']=.5*bound*(raw_teacher-smooth)
 return result
def metric(actual,mass,expected,refmass):
 error=np.abs(actual-expected);good=refmass>=.9;e=error[good]
 p99=float(np.quantile(e,.99)) if e.size else None;maximum=float(e.max()) if e.size else None
 return dict(reference_mass_ge_point9=int(good.sum()),mass_low_rows=int((~good).sum()),RGB_abs_p99=p99,RGB_abs_max=maximum,
  mass_abs_max=float(np.abs(mass-refmass).max()),passed=bool(p99 is not None and p99<=1/255 and maximum<=4/255))
def main(order,output,check61):
 out=Path(output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 d=load(Q/'permission/final_r1/teacher_evidence.npz');highs=fields(d)
 indices=load(Q/'permission/ledger_r2/small_Sigma_hf.npz',['active_indices'])['active_indices']
 geo=load(P/'cpu_exact_r1/geometry_kernel.npz',['active_world_rows','world_xyz','normal_world','sigma_m'])
 a=load(BASE,['surface_world_xyz','grid_world_row','world_from_uv']);keys=load(N/'provenance.npz',['world_keys'])['world_keys']
 sys.path.insert(0,str(P/'code'));from road_chart_sampling import make_exact_sample_plan
 plan=make_exact_sample_plan(a['surface_world_xyz'],a['grid_world_row'],keys,geo['active_world_rows'][indices],
   geo['normal_world'][indices],geo['sigma_m'][indices],a['world_from_uv'],order=order,world_xyz=geo['world_xyz'][indices])
 rows=[];candidate={};baseline=[]
 for name,high in highs.items():
  reference=load(Q/'permission/numeric_r1'/(name+'.npz'))
  value,mass=integrate(d['common_base_srgb'],d['common_material_valid'],d['mH_union'],high,plan['sample_uv'],plan['geometric_sample_valid'],plan['weights'])
  m=metric(value,mass,reference['reference_RGB'],reference['reference_mass']);m['case']=name;rows.append(m)
  candidate[name]=(value,mass);baseline.append(reference)
  np.savez_compressed(out/(name+'.npz'),active_indices=indices,world_rows=geo['active_world_rows'][indices],
   candidate_RGB=value,candidate_mass=mass,reference_RGB=reference['reference_RGB'],reference_mass=reference['reference_mass'])
 spot=None
 if check61:
  ref=baseline[1];worst=np.argsort(np.abs(ref['candidate_RGB']-ref['reference_RGB']).max(1))[-16:]
  chosen=np.unique(np.r_[worst,np.linspace(0,len(indices)-1,8,dtype=int)]);ids=indices[chosen]
  dense=make_exact_sample_plan(a['surface_world_xyz'],a['grid_world_row'],keys,geo['active_world_rows'][ids],geo['normal_world'][ids],
   geo['sigma_m'][ids],a['world_from_uv'],order=61,world_xyz=geo['world_xyz'][ids])
  spotrows=[]
  for name,high in highs.items():
   value,mass=integrate(d['common_base_srgb'],d['common_material_valid'],d['mH_union'],high,dense['sample_uv'],dense['geometric_sample_valid'],dense['weights'])
   ref41=load(Q/'permission/numeric_r1'/(name+'.npz'))
   m=metric(ref41['reference_RGB'][chosen],ref41['reference_mass'][chosen],value,mass);m['case']=name;spotrows.append(m)
   np.savez_compressed(out/(name+'_61spot.npz'),sample_indices=chosen,active_indices=ids,reference61_RGB=value,reference61_mass=mass,
    candidate_RGB=candidate[name][0][chosen],candidate_mass=candidate[name][1][chosen])
  spot=dict(rows=len(chosen),cases=spotrows,all_passed=all(r['passed'] for r in spotrows))
 passed=all(r['passed'] for r in rows) and (spot is None or spot['all_passed'])
 result=dict(schema='teacher_H_numeric_order_repair_CPU.v1',status='PASS' if passed else 'FAIL_ENGINEERING_INTEGRATION_PENDING',
  candidate_order=order,reference_order=41,reference61_spot=spot,sample_rows=len(indices),quad_points=len(indices)*order*order,
  cases=rows,mask_scope_bound_samples_thresholds_unchanged=True,high_bound=.34,p99_gate=1/255,max_gate=4/255,
  CPU_only=True,CUDA_calls=0,native_calls=0,optimizer_updates=0,old_failure_preserved=str(Q/'permission/numeric_r1'),
  seconds=time.monotonic()-start,no_GPU_authorization=True)
 (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--order',type=int,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--check61',action='store_true')
 a=ap.parse_args();main(a.order,a.output,a.check61)

