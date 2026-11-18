"""固定seed分层的TRELLIS真实perface积分CPU审计；不筛eligible、不改结果。"""
from pathlib import Path
import os,sys,json,time,hashlib
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np
from exact_mesh_uv import WorldMeshUVMap
from precompute_quad_prior import bilinear_A0
P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1');sys.path.insert(0,str(P/'code'))
from road_chart_sampling import StructuredRoadChart

def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def stats(error):
 e=np.abs(error);m=e.max(1)
 return dict(rows=len(e),mean=float(e.mean()),absRGB_p99=float(np.quantile(e,.99)),rowmax_p99=float(np.quantile(m,.99)),maximum=float(m.max()),
  p99_gate_1code=bool(np.quantile(m,.99)<=1/255),max_gate_4code=bool(m.max()<=4/255))

def main():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU only')
 out=P/'trellis2/quadrature_audit_r1'
 if out.exists():raise FileExistsError('preserve old audit')
 start=time.monotonic();cache=load(P/'trellis2/quad_prior_candidate_r1/row_queries.npz');plan=load(P/'chart_plan_candidate_r1/geometry_plan.npz');A=load(P/'cpu_initial_r1/A0.npz');kernel=load(P/'cpu_initial_r1/geometry_kernel.npz');scope=load(P/'stage0_native_r1/scope.npz')
 parent=Path(json.loads((P/'preparation_contract.json').read_text())['input_asset'])
 with np.load(parent/'provenance.npz',allow_pickle=False) as z:surface=z['world_xyz'][A['grid_world_row']]
 chart=StructuredRoadChart(surface,A['world_from_uv']);mapper=WorldMeshUVMap.load(P/'trellis2/world_prior_r1')
 active=plan['active_world_rows'];rev=np.empty(A['grid_world_row'].size,np.int64);rev[A['grid_world_row'].ravel()]=np.arange(len(rev));paint=A['observed_marking_detail_support'].ravel()[rev[active]]
 inside=~cache['exterior'];sigma=kernel['sigma_m'][:,0];rng=np.random.default_rng(101)
 pools=dict(core=np.flatnonzero(cache['all225_quad_in_patch']),boundary=np.flatnonzero(cache['boundarymix']),
  actual_maxSigma=np.flatnonzero(inside&(sigma==sigma[inside].max())),low_mass=np.flatnonzero(inside&(cache['common_valid_mass']<.9)),
  global_observed_paint_A0_only=np.flatnonzero((paint>=.5)&~inside),actual_S_visible=np.flatnonzero(inside&(scope['actual_S_mass']>0)))
 amounts=dict(core=128,boundary=128,actual_maxSigma=96,low_mass=64,global_observed_paint_A0_only=64,actual_S_visible=64)
 samples={k:np.sort(rng.choice(v,min(len(v),amounts[k]),replace=False)) for k,v in pools.items()}
 rows=np.unique(np.concatenate(list(samples.values())));assert len(rows)>=256
 out.mkdir(parents=True)
 np.savez_compressed(out/'selected_rows.npz',local_rows=rows,world_rows=active[rows],**{k:v for k,v in samples.items()},eligible_at_selection=scope['eligible'][rows],conservative_K_guard=scope['conservative_K_guard'][rows],sigma_m=sigma[rows],common_valid_mass_GH15=cache['common_valid_mass'][rows])
 results={};mass={};base={};inverse_errors={}
 for order in [15,41,61]:
  x,w=np.polynomial.hermite.hermgauss(order);xx,yy=np.meshgrid(x,x,indexing='xy');wx,wy=np.meshgrid(w,w,indexing='xy');nodes=np.c_[xx.ravel(),yy.ravel()]*np.sqrt(2);weight=(wx*wy).ravel()/np.pi
  color=np.zeros((len(rows),3),np.float64);b0=color.copy();den=np.zeros(len(rows));error=0.
  for begin in range(0,len(rows),32):
   rr=rows[begin:begin+32];world=plan['world_xyz'][rr,None,:2]+np.einsum('nij,qj->nqi',plan['chol_world_xy'][rr],nodes)
   exact=chart.query(world);uv=exact['uv'];geo=exact['valid'];mid=len(weight)//2;uv[:,mid]=plan['center_uv'][rr];geo[:,mid]=True
   numerator,vm=bilinear_A0(A['rgb_srgb'],A['valid'],uv,geo);q=mapper.query(world);insideq=q['valid']&geo
   value=np.where(insideq[...,None],q['basecolor_rgba'][...,:3],0);mixed=numerator.copy();mixed[insideq]=vm[insideq,None]*value[insideq]
   dm=np.sum(vm*weight,axis=1);den[begin:begin+len(rr)]=dm
   b0[begin:begin+len(rr)]=np.sum(numerator*weight[None,:,None],axis=1)/np.maximum(dm[:,None],1e-12)
   color[begin:begin+len(rr)]=np.sum(mixed*weight[None,:,None],axis=1)/np.maximum(dm[:,None],1e-12)
   if insideq.any():error=max(error,float(np.nanmax(q['worldXY_reconstruction_error'][insideq])))
  results[order]=np.clip(color,0,1);base[order]=np.clip(b0,0,1);mass[order]=den;inverse_errors[str(order)]=error
  np.savez_compressed(out/f'GH{order}.npz',local_rows=rows,world_rows=active[rows],filtered_rgb_T=results[order],filtered_rgb_A0=base[order],common_valid_mass=den)
  print(json.dumps(dict(order=order,rows=len(rows),seconds=time.monotonic()-start)),flush=True)
 comparisons={};strata={}
 for name,a,b in [('15_vs41',15,41),('15_vs61',15,61),('41_vs61',41,61)]:
  comparisons[name]=stats(results[a]-results[b]);comparisons[name]['valid_mass_abs_max']=float(np.max(np.abs(mass[a]-mass[b])))
 for name,rr in samples.items():
  pos=np.searchsorted(rows,rr);strata[name]=dict(selected=len(rr),population=len(pools[name]),scope_role='A0-only white paint negative control, no T2 paint patch exists' if name=='global_observed_paint_A0_only' else 'before eligible/K filtering',
   **{k:stats(results[a][pos]-results[b][pos]) for k,a,b in [('15_vs41',15,41),('15_vs61',15,61),('41_vs61',41,61)]})
 low=mass[61]<.9;high=~low
 split={label:dict(rows=int(mask.sum()),GH15_vs61=stats(results[15][mask]-results[61][mask]) if mask.any() else None) for label,mask in [('low_mass_below09',low),('mass_atleast09',high)]}
 cache_error=float(np.max(np.abs(results[15].astype(np.float32)-cache['filtered_rgb_T'][rows])))
 receipt=dict(schema='TRELLIS2_perface_quadrature_independent_dense_CPU.v1',seed=101,selected_rows=len(rows),eligible_filter_applied=False,
  selected_in_runtime_eligible=int(scope['eligible'][rows].sum()),selected_K_guarded=int(scope['conservative_K_guard'][rows].sum()),
  selection={k:dict(population=len(pools[k]),selected=len(v)) for k,v in samples.items()},
  observed_paint_inside_patch_rows=int((inside&(paint>=.5)).sum()),T2_node_luma_max=.522848071539402,
  comparisons=comparisons,strata=strata,material_mass_strata=split,quadrature_levels=[15,41,61],source_geometry_inverse_max_m=inverse_errors,
  GH15_reproduction_cache_max=cache_error,gate=dict(p99=1/255,max=4/255,statistic='conservative per-row maximum-channel error p99 plus overall max'),
  status='PASS_GH15_DENSE_COMPARISON' if all(comparisons[k]['p99_gate_1code'] and comparisons[k]['max_gate_4code'] for k in ['15_vs41','15_vs61']) else 'FAIL_GH15_DENSE_COMPARISON',
  reference_convergence_note='41_vs61 reported separately; finite-order comparisons are engineering evidence, not an analytic integral certificate',
  numerical_GH15_pass_for_A0_H_not_inherited=True,no_visual_selection=True,new_inference_calls=0,GPU_operations=0,optimizer_updates=0,seconds=time.monotonic()-start,human_verdict=None)
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
