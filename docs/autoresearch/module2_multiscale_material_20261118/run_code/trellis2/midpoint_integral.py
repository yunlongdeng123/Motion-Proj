"""固定物理Σ的even midpoint积分：保留硬patch及同A0/source-normalized函数。"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import os,argparse,json,time,math
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np
from exact_mesh_uv import WorldMeshUVMap
from precompute_quad_prior import bilinear_A0,sha
from audit_quadrature import load,stats
import sys
P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1');sys.path.insert(0,str(P/'code'))
from road_chart_sampling import StructuredRoadChart


def main():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU only')
 ap=argparse.ArgumentParser();ap.add_argument('--all',action='store_true');ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
 out=P/'trellis2'/('midpoint_full_r1' if args.all else 'midpoint_audit_r1')
 if out.exists():raise FileExistsError('preserve prior site')
 if args.all:
  audit=json.loads((P/'trellis2/midpoint_audit_r1/receipt.json').read_text())
  if audit['status']!='PASS_MIDPOINT_CPU_QUALIFICATION':raise ValueError('fixed541 midpoint gate not passed')
 start=time.monotonic();g=load(P/'chart_plan_candidate_r1/geometry_plan.npz');A=load(P/'cpu_initial_r1/A0.npz');cache=load(P/'trellis2/quad_prior_candidate_r1/row_queries.npz');selection=load(P/'trellis2/quadrature_audit_r1/selected_rows.npz')
 parent=Path(json.loads((P/'preparation_contract.json').read_text())['input_asset'])
 with np.load(parent/'provenance.npz',allow_pickle=False) as z:surface=z['world_xyz'][A['grid_world_row']]
 chart=StructuredRoadChart(surface,A['world_from_uv']);mapper=WorldMeshUVMap.load(P/'trellis2/world_prior_r1');bounds=(mapper.locator.minimum,mapper.locator.maximum)
 C=g['world_covariance_xy'];radius=5*np.sqrt(np.linalg.eigvalsh(C)[:,-1]);center=g['world_xyz'][:,:2]
 far=np.any((center+radius[:,None]<bounds[0])|(center-radius[:,None]>bounds[1]),axis=1)
 rows=np.arange(len(center)) if args.all else selection['local_rows']
 def integrate(row,step):
  if far[row]:return dict(T=cache['filtered_rgb_A0'][row].astype(np.float64),A=cache['filtered_rgb_A0'][row].astype(np.float64),mass=float(cache['common_valid_mass'][row]),T_fraction=0.,T_physical_mass=0.,N=0,step=0.,radius=float(radius[row]),density_mass=1.,points_in_T=0,copied_A0=True)
  R=radius[row];N=max(2,int(np.ceil(2*R/step)));N+=N%2;h=2*R/N
  axis=(np.arange(N)+.5-N/2)*h;assert N%2==0 and not np.any(axis==0)
  xx,yy=np.meshgrid(axis,axis,indexing='xy');delta=np.c_[xx.ravel(),yy.ravel()];world=center[row]+delta
  inverse=np.linalg.inv(C[row]);exponent=np.einsum('ni,ij,nj->n',delta,inverse,delta)
  weight=np.exp(-.5*exponent)*(h*h)/(2*np.pi*np.sqrt(np.linalg.det(C[row])))
  exact=chart.query(world);base,vm=bilinear_A0(A['rgb_srgb'],A['valid'],exact['uv'],exact['valid'])
  q=mapper.query(world);inside=q['valid']&exact['valid'];numerator=base.copy();numerator[inside]=vm[inside,None]*q['basecolor_rgba'][inside,:3]
  den=float(weight@vm);T=np.sum(weight[:,None]*numerator,axis=0)/max(den,1e-12);B0=np.sum(weight[:,None]*base,axis=0)/max(den,1e-12)
  material_mass=float(np.sum(weight[inside]*vm[inside]));count=int(inside.sum());copied=False
  if material_mass==0:
   T=cache['filtered_rgb_A0'][row].astype(np.float64);B0=T.copy();copied=True
  return dict(T=np.clip(T,0,1),A=np.clip(B0,0,1),mass=den,T_fraction=material_mass/max(den,1e-12),T_physical_mass=float(weight[inside].sum()),N=N,step=h,radius=float(R),density_mass=float(weight.sum()),points_in_T=count,copied_A0=copied)
 def worker(row):
  step=.0025 if cache['all225_quad_in_patch'][row] else .00125
  candidate=integrate(int(row),step)
  reference=None if args.all else integrate(int(row),.000625)
  return int(row),candidate,reference
 candidate={};reference={}
 with ThreadPoolExecutor(max_workers=args.workers) as pool:
  futures=[pool.submit(worker,int(row)) for row in rows]
  for i,f in enumerate(as_completed(futures)):
   row,c,r=f.result();candidate[row]=c
   if r is not None:reference[row]=r
   if (i+1)%256==0:print(json.dumps(dict(rows_completed=i+1,total=len(rows),seconds=time.monotonic()-start)),flush=True)
 out.mkdir(parents=True)
 def save(name,data):
  d={key:np.asarray([data[int(r)][key] for r in rows]) for key in ['T','A','mass','T_fraction','T_physical_mass','N','step','radius','density_mass','points_in_T','copied_A0']}
  np.savez_compressed(out/name,local_rows=rows,active_world_rows=g['active_world_rows'][rows],filtered_rgb_T=d['T'].astype(np.float32),filtered_rgb_A0=d['A'].astype(np.float32),
   common_valid_mass=d['mass'],material_T_weight_fraction=d['T_fraction'],physical_T_quad_mass=d['T_physical_mass'],midpoint_N=d['N'],actual_step_m=d['step'],bbox_radius_m=d['radius'],
   discrete_gaussian_density_mass=d['density_mass'],points_inside_T=d['points_in_T'],copied_same_A0=d['copied_A0'],baseline_A0_GH15=cache['filtered_rgb_A0'][rows])
  return d
 cand=save('row_queries.npz',candidate);ref=None if args.all else save('reference_0625.npz',reference)
 comparisons=None;strata=None;mass_strata=None;status='PASS_FULL_FROZEN_CACHE_AFTER_FIXED541_GATE'
 if not args.all:
  comparisons=stats(cand['T']-ref['T']);comparisons['valid_mass_abs_max']=float(np.max(np.abs(cand['mass']-ref['mass'])))
  strata={}
  for name in ['core','boundary','actual_maxSigma','low_mass','global_observed_paint_A0_only','actual_S_visible']:
   positions=np.searchsorted(rows,selection[name]);strata[name]=stats(cand['T'][positions]-ref['T'][positions])
  low=ref['mass']<.9;mass_strata={k:dict(rows=int(m.sum()),error=stats(cand['T'][m]-ref['T'][m]) if m.any() else None) for k,m in [('below09',low),('atleast09',~low)]}
  status='PASS_MIDPOINT_CPU_QUALIFICATION' if comparisons['p99_gate_1code'] and comparisons['max_gate_4code'] else 'FAIL_MIDPOINT_CPU_QUALIFICATION'
 noncopy=~cand['copied_A0'];density=cand['density_mass'][noncopy]
 receipt=dict(schema='T2_even_midpoint_physical_Sigma_CPU.v1',status=status,rows=len(rows),fixed_selection_source=str(P/'trellis2/quadrature_audit_r1/selected_rows.npz'),eligible_filter_applied=False,
  core_step_upper_m=.0025,boundary_step_upper_m=.00125,reference_step_upper_m=None if args.all else .000625,
  radius_policy='worldXY axis-aligned box halfwidth=5*sqrt(maxeig(physical Sigma)); even N=ceil(2R/step) rounded up to even, actual h=2R/N; no point on center zero axis',
  weights='exp(-.5 delta^T Sigma^-1 delta)*cellarea/(2pi sqrt(detSigma)); same source-valid/geometry normalized numerator and denominator',
  probability_tail_upper_bound_outside_box=4*.5*math.erfc(5/math.sqrt(2)),tail_bound_role='probability bound only; does not bound source-normalized RGB at vanishing material mass',
  copied_same_A0_rows=int(cand['copied_A0'].sum()),density_mass_min_max=[float(density.min()),float(density.max())] if len(density) else None,
  comparisons=comparisons,strata=strata,material_mass_strata=mass_strata,
  A0_delta_vs_cached_GH15_on_noncopied=stats(cand['A'][noncopy]-cache['filtered_rgb_A0'][rows[noncopy]]) if noncopy.any() else None,
  source_mesh_or_texture_modified=False,geometry_opacity_Sigma_active_modified=False,hard_patch_width_unchanged=True,no_feather_or_visual_tuning=True,
  authoritative_source=str(P/'trellis2/world_prior_r1'),plan_sha256=sha(P/'chart_plan_candidate_r1/geometry_plan.npz'),
  runtime_write_mask_not_applied=True,new_inference_calls=0,GPU_operations=0,optimizer_updates=0,workers=args.workers,seconds=time.monotonic()-start,human_verdict=None)
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
