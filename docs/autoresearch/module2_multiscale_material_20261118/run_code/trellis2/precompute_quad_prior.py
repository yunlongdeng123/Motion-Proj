"""共享exact-chart GH15的TRELLIS逐quad只读派生缓存；不训练、不渲染。"""
from pathlib import Path
import os,argparse,json,hashlib,time
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np
from exact_mesh_uv import WorldMeshUVMap
P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def bilinear_A0(rgb,valid,uv,geometry_valid):
 shape=uv.shape[:-1];u=np.nan_to_num(uv.reshape(-1,2),nan=-1e9,posinf=-1e9,neginf=-1e9)
 lo=np.floor(u).astype(np.int64);t=u-lo;H,W=valid.shape;base=np.zeros((len(u),3),np.float64);mass=np.zeros(len(u),np.float64)
 for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
  x=lo[:,0]+dx;y=lo[:,1]+dy;weight=(t[:,0] if dx else 1-t[:,0])*(t[:,1] if dy else 1-t[:,1]);inside=(x>=0)&(x<W)&(y>=0)&(y<H)
  ii=np.flatnonzero(inside);w=weight[ii]*valid[y[ii],x[ii]];mass[ii]+=w;base[ii]+=w[:,None]*rgb[y[ii],x[ii]]
 geo=geometry_valid.reshape(-1);mass[~geo]=0;base[~geo]=0
 return base.reshape(*shape,3),mass.reshape(shape)

def main():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU only')
 ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,default=P/'chart_plan_candidate_r1');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists():raise FileExistsError('preserve old cache')
 start=time.monotonic()
 with np.load(a.plan/'geometry_plan.npz',allow_pickle=False) as z:g={k:z[k].copy() for k in z.files}
 uv=np.load(a.plan/'sample_uv_f64.npy',mmap_mode='r');assert g['order']==15 and uv.shape==(len(g['world_xyz']),225,2)
 with np.load(P/'cpu_initial_r1/A0.npz',allow_pickle=False) as z:rgb=z['rgb_srgb'].astype(np.float64);valid=z['valid'].astype(np.float64)
 mapper=WorldMeshUVMap.load(P/'trellis2/world_prior_r1')
 with np.load(P/'trellis2/input_r1/reference_sources.npz',allow_pickle=False) as z:ref_unknown=z['unknown'].copy()
 N=len(g['world_xyz']);Q=len(g['weights']);w=g['weights'];baseCI=np.zeros((N,3),np.float32);trellisCI=baseCI.copy();mass=np.zeros(N);inside_mass=np.zeros(N);physical_mass=np.zeros(N)
 full=np.zeros(N,bool);boundary=full.copy();exterior=full.copy();maxerr=0.;records={k:[] for k in ['row_index','quad_index','fixed_face_id','generated_uv','barycentric','generated_alpha','colocated_reference_unknown_mass']}
 for begin in range(0,N,512):
  end=min(begin+512,N);world=g['world_xyz'][begin:end,None,:2]+np.einsum('nij,qj->nqi',g['chol_world_xy'][begin:end],g['standard_nodes'])
  q=mapper.query(world);geo=g['geometric_sample_valid'][begin:end];inside=q['valid']&geo
  base,vm=bilinear_A0(rgb,valid,uv[begin:end],geo);den=np.sum(vm*w,axis=1)
  color=q['basecolor_rgba'][...,:3];color=np.where(inside[...,None],color,0)
  num=base.copy();num[inside]=vm[inside,None]*color[inside]
  bc=np.sum(base*w[None,:,None],axis=1)/np.maximum(den[:,None],1e-12)
  tc=np.sum(num*w[None,:,None],axis=1)/np.maximum(den[:,None],1e-12)
  baseCI[begin:end]=np.clip(bc,0,1);trellisCI[begin:end]=np.clip(tc,0,1)
  mass[begin:end]=den;inside_mass[begin:end]=np.sum(vm*w*inside,axis=1)/np.maximum(den,1e-12);physical_mass[begin:end]=np.sum(w*inside,axis=1)
  full[begin:end]=inside.all(1);exterior[begin:end]=~inside.any(1);boundary[begin:end]=inside.any(1)&~inside.all(1)
  rr,cc=np.nonzero(inside)
  if len(rr):
   fid=q['fixed_face_id'][rr,cc];bary=q['barycentric'][rr,cc];ru=np.sum(ref_unknown[mapper.faces[fid]]*bary,axis=1)
   for name,value in [('row_index',(rr+begin).astype(np.int32)),('quad_index',cc.astype(np.uint16)),('fixed_face_id',fid),('generated_uv',q['generated_uv'][rr,cc]),('barycentric',bary),('generated_alpha',q['basecolor_rgba'][rr,cc,3].astype(np.float32)),('colocated_reference_unknown_mass',np.clip(ru,0,1).astype(np.float32))]:records[name].append(value)
   maxerr=max(maxerr,float(np.max(q['worldXY_reconstruction_error'][inside])))
  if begin%16384==0:print(json.dumps(dict(processed=end,total=N,seconds=time.monotonic()-start)),flush=True)
 if not np.array_equal(baseCI[exterior],trellisCI[exterior]):raise AssertionError('world exterior must equal same A0 filter')
 a.output.mkdir(parents=True)
 np.savez_compressed(a.output/'row_queries.npz',active_world_rows=g['active_world_rows'],filtered_rgb_T=trellisCI,filtered_rgb_A0=baseCI,
  common_valid_mass=mass,material_T_weight_fraction=inside_mass,physical_T_quad_mass=physical_mass,all225_quad_in_patch=full,boundarymix=boundary,exterior=exterior,
  center_world_xyz=g['world_xyz'],weights=w,order=np.asarray(15))
 np.savez_compressed(a.output/'per_quad_sources.npz',**{k:np.concatenate(v) for k,v in records.items()})
 record=dict(schema='TRELLIS2_shared_GH15_derived_CPU_cache.v1',status='PASS_CPU_FROZEN_FIELD_FILTER',plan=str(a.plan),plan_sha256=sha(a.plan/'geometry_plan.npz'),sample_uv_sha256=sha(a.plan/'sample_uv_f64.npy'),
  authority=str(P/'trellis2/world_prior_r1/mesh_uv_source.npz'),native_renderer_called=False,GPU_operations=0,optimizer_updates=0,seconds=time.monotonic()-start,
  active_rows=N,order=15,nodes_per_row=Q,full225_patch_rows=int(full.sum()),boundarymix_rows=int(boundary.sum()),exterior_rows=int(exterior.sum()),
  common_valid_mass_quantiles=np.quantile(mass,[0,.01,.5,.99,1]).tolist(),perquad_worldXY_inverse_max_m=maxerr,
  alpha_policy='generated alpha recorded only; no Gaussian opacity/coverage change',
  integration='same exact-chart uv cache, geometry-valid mask, bilinear A0*valid and common valid mass. Inside actual TRELLIS mesh use validmass*T2_RGB, exterior use A0 numerator; same GH15 weights/denominator',
  source_policy='authoritative fixed mesh/generated face UV/baseColor retained; row RGB is derived frozen cache, not a free ci parameter',
  exterior_same_A0_bytes_exact=True,qualification_not_expanded=True,runtime_write_mask_status='PENDING main actual A/S|(C&~known) and fresh K; unavailable rows require same legacy fallback',
  fullfilter_scope='all225 registered quadrature nodes, not an infinite Gaussian support claim',reference_unknown_mass_role='co-located reference diagnostic; not neural causal attribution or GT',
  final_native_render_pending=True,human_verdict=None)
 (a.output/'receipt.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
if __name__=='__main__':main()
