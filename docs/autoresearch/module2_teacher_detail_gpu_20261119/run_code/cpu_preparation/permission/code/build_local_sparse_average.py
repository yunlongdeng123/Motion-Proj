"""local CPU exact-GH31 -> fixed sparse average read. No GPU/native/update."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import argparse,hashlib,json,time
import numpy as np
from road_chart_sampling_snapshot import local_chart_plan,StructuredRoadChart,make_exact_sample_plan
from fixed_material_read_operator_snapshot import ReadCompiler,FixedReadOperator,bilinear
HERE=Path(__file__).resolve().parent
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def load(path):
 with np.load(path,allow_pickle=False)as d:return {k:d[k].copy()for k in d.files}
def direct(d,uv,valid,w,low,high,gate):
 v=bilinear(d['common_material_valid'],uv)[...,0]*valid
 mass=(w[None]*v).sum(1);base=bilinear(d['common_base_srgb']*d['common_material_valid'][...,None],uv)*valid[...,None]
 L=bilinear(low,uv/6);H=bilinear(high,uv);mH=bilinear(d[gate],uv)
 return ((w[None,:,None]*(base+v[...,None]*(L+mH*H))).sum(1)/mass[:,None].clip(1e-12)).clip(0,1)
def main(output,probe=False,chunk=256):
 out=Path(output);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
 inp=HERE/'inputs';identity=json.loads((inp/'local_inputs.json').read_text())
 for filename,key in [('local_compile_inputs.npz','sha256'),('geometry_for_consumers.npz','geometry_sha256')]:
  if sha(inp/filename)!=identity[key]:raise ValueError('download/source mismatch:'+filename)
 d=load(inp/'local_compile_inputs.npz');g=load(inp/'geometry_for_consumers.npz')
 n=len(d['query_world_rows']);chosen=np.linspace(0,n-1,64,dtype=int)if probe else np.arange(n)
 rows=d['query_world_rows'][chosen];normal=d['normal_world'][chosen];sigma=d['sigma_m'][chosen];xyz=d['query_world_xyz'][chosen]
 plan=local_chart_plan(g['world_xyz'],g['world_row_grid'],d['world_keys'],rows,normal,sigma,g['world_from_uv'],order=31,world_xyz=xyz)
 chart=StructuredRoadChart(g['world_xyz'],g['world_from_uv'])
 compiler=ReadCompiler(d['common_base_srgb'],d['common_material_valid'],{'observed':d['observed_H_prior'],'teacher':d['mH_union']},low_stride=6)
 selected=np.unique(np.linspace(0,len(rows)-1,min(150,len(rows)),dtype=int))
 uv_small=[];valid_small=[];small_positions=[]
 error=0.;unmapped=0
 for b in range(0,len(rows),chunk):
  e=min(b+chunk,len(rows))
  world=plan['world_xyz'][b:e,None,:2]+np.einsum('nij,qj->nqi',plan['chol_world_xy'][b:e],plan['standard_nodes'])
  q=chart.query(world);uv=q['uv'];valid=q['valid'];uv[:,len(plan['weights'])//2]=plan['center_uv'][b:e];valid[:,len(plan['weights'])//2]=True
  compiler.add(uv,plan['weights'],valid)
  error=max(error,float(q['worldXY_reconstruction_error'][q['valid']].max(initial=0)))
  unmapped+=int((~valid).sum())
  take=selected[(selected>=b)&(selected<e)]
  if len(take):uv_small.append(uv[take-b].copy());valid_small.append(valid[take-b].copy());small_positions.extend(take.tolist())
  if b//chunk%16==0 or e==len(rows):
   progress=dict(rows_done=e,total_rows=len(rows),seconds=time.perf_counter()-start,CPU_only=True,query_order=31)
   (out/'progress.json').write_text(json.dumps(progress)+'\n');print(json.dumps(progress),flush=True)
 if error>=1e-7:raise AssertionError('exact chart inverse failed')
 op=compiler.finish(rows,dict(mode='Gaussian_footprint_average',kernel_order=31,quadrature_points_per_row=961,
   fixed_common_scope=89887,high_bound=.34,units='world metres; sigma metres; exact UV texelcentres',
   same_geometry_as_P=True,readonly_active_rows_retain_original_RGB=True,
   source_files={str(inp/f):sha(inp/f)for f in ['local_compile_inputs.npz','geometry_for_consumers.npz']},
   compiler_source_sha256=sha(HERE/'fixed_material_read_operator_snapshot.py'),chart_source_sha256=sha(HERE/'road_chart_sampling_snapshot.py'),
   CPU_only=True,GPU_authorized=False,native_calls=0,optimizer_updates=0))
 op.save(out/'operator')
 # Direct original quadrature on fixed real rows, two nonzero fields and both masks.
 uv=np.concatenate(uv_small);valid=np.concatenate(valid_small);positions=np.asarray(small_positions)
 H,W=d['common_material_valid'].shape;yy,xx=np.mgrid[:H,:W]
 lh,lw=op.low_shape;ly,lx=np.mgrid[:lh,:lw]
 low=np.stack([.01*np.sin(lx/3+ly/5+j)for j in range(3)],-1)
 high=np.stack([.02*np.cos(xx/2+yy/3+j)for j in range(3)],-1)
 checks={}
 for gate,key in [('observed','observed_H_prior'),('teacher','mH_union')]:
  expected=direct(d,uv,valid,plan['weights'],low,high,key)
  actual=op.numpy_query(low,high,gate)[positions]
  err=float(np.abs(expected-actual).max());checks[gate]=err
  if err>1e-10:raise AssertionError('sparse != same quadrature')
 reloaded=FixedReadOperator.load(out/'operator')
 for gate in ['observed','teacher']:
  if not np.array_equal(op.numpy_query(low,high,gate),reloaded.numpy_query(low,high,gate)):raise AssertionError('sparse reload mismatch')
 np.savez_compressed(out/'direct_validation.npz',query_positions=positions,query_world_rows=rows[positions],sample_uv=uv,
  geometric_valid=valid,weights=plan['weights'],probe_low=low,probe_high=high)
 np.savez_compressed(out/'row_mapping.npz',query_world_rows=rows,active_local_indices=d['active_local_indices'][chosen],full_active_world_rows=d['active_world_rows'])
 result=dict(schema='local_exact_GH31_sparse_average_CPU.v1',completed=True,probe_only=probe,query_rows=len(rows),active_total=len(d['active_world_rows']),
   order=31,points=len(rows)*961,worldXY_inverse_max_m=error,unmapped_points=unmapped,
   direct_quadrature_checked_rows=len(positions),sparse_vs_quadrature_max=checks,sparse_reload_exact=True,
   mode='Gaussian_footprint_average',KH_already_contains_gate_do_not_apply_mH_again=True,
   native_calls=0,CUDA_calls=0,optimizer_updates=0,GPU_authorized=False,no_training_controller=True,
   operator_files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p))for p in sorted((out/'operator').iterdir())if p.is_file()},
   seconds=time.perf_counter()-start)
 (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--probe-only',action='store_true');p.add_argument('--chunk',type=int,default=256)
 a=p.parse_args();main(a.output,a.probe_only,a.chunk)

