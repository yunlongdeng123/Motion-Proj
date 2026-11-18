"""共同89887可写行的GH31流式CPU plan；readonly不重算，无GPU/训练控制器。"""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,sys,time
import numpy as np
from teacher_detail_permission import P,Q,N,BASE,load,sha
def main(output,chunk=512):
 out=Path(output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 numeric=Q/'permission/numeric31_r1/receipt.json';spot=Q/'permission/numeric31_r1/reference61_spot.json'
 if json.loads(numeric.read_text())['status']!='PASS' or json.loads(spot.read_text())['status']!='PASS':raise ValueError('GH31 numeric qualification missing')
 final=Q/'permission/final_r1';d=load(final/'teacher_evidence.npz',['active_world_rows','common_write_active','common_material_valid','world_xyz','world_row_grid'])
 a=load(BASE,['surface_world_xyz','grid_world_row','world_from_uv'])
 # Small global geometry export for CPU prior integrators; crop cannot replace global denominator.
 np.savez_compressed(final/'geometry_for_consumers.npz',world_xyz=d['world_xyz'],world_row_grid=d['world_row_grid'],
  common_material_valid=d['common_material_valid'],world_from_uv=a['world_from_uv'])
 selected=np.flatnonzero(d['common_write_active']);rows=d['active_world_rows'][selected]
 if len(rows)!=89887:raise ValueError('locked common scope changed')
 g=load(P/'cpu_exact_r1/geometry_kernel.npz',['active_world_rows','world_xyz','normal_world','sigma_m'])
 if not np.array_equal(g['active_world_rows'],d['active_world_rows']):raise ValueError('active row identity differs')
 keys=load(N/'provenance.npz',['world_keys'])['world_keys']
 sys.path.insert(0,str(P/'code'))
 from road_chart_sampling import local_chart_plan,StructuredRoadChart,make_exact_sample_plan
 plan=local_chart_plan(a['surface_world_xyz'],a['grid_world_row'],keys,rows,g['normal_world'][selected],g['sigma_m'][selected],
  a['world_from_uv'],order=31,world_xyz=g['world_xyz'][selected])
 chart=StructuredRoadChart(a['surface_world_xyz'],a['world_from_uv'])
 count=len(rows);q=len(plan['weights']);middle=q//2
 uv=np.lib.format.open_memmap(out/'sample_uv_f64.npy',mode='w+',dtype=np.float64,shape=(count,q,2))
 valid=np.lib.format.open_memmap(out/'geometric_sample_valid.npy',mode='w+',dtype=bool,shape=(count,q))
 errors=0.;unmapped=0
 for begin in range(0,count,chunk):
  end=min(begin+chunk,count)
  world=plan['world_xyz'][begin:end,None,:2]+np.einsum('nij,qj->nqi',plan['chol_world_xy'][begin:end],plan['standard_nodes'])
  result=chart.query(world);uv[begin:end]=result['uv'];valid[begin:end]=result['valid']
  uv[begin:end,middle]=plan['center_uv'][begin:end];valid[begin:end,middle]=True
  errors=max(errors,float(result['worldXY_reconstruction_error'][result['valid']].max(initial=0)))
  unmapped+=int((~valid[begin:end]).sum())
  if begin//chunk%16==0 or end==count:
   uv.flush();valid.flush()
   (out/'progress.json').write_text(json.dumps(dict(rows_done=end,total_rows=count,seconds=time.monotonic()-start,CPU_only=True))+'\n')
 if errors>=1e-7:raise ValueError('exact physical chart inverse failed')
 # Fixed source-helper replay protects compact row mapping and stream assembly.
 check=np.linspace(0,count-1,64,dtype=int)
 rebuilt=make_exact_sample_plan(a['surface_world_xyz'],a['grid_world_row'],keys,rows[check],g['normal_world'][selected[check]],
  g['sigma_m'][selected[check]],a['world_from_uv'],order=31,world_xyz=g['world_xyz'][selected[check]])
 error=float(np.abs(uv[check]-rebuilt['sample_uv']).max())
 valid_exact=np.array_equal(valid[check],rebuilt['geometric_sample_valid'])
 if error!=0 or not valid_exact:raise AssertionError('stream cache/source-helper mismatch')
 numbers={key:value for key,value in plan.items() if not isinstance(value,dict)}
 numbers.update(query_world_rows=rows,active_local_indices=selected,full_active_world_rows=d['active_world_rows'],
  normal_world=g['normal_world'][selected],sigma_m=g['sigma_m'][selected])
 np.savez_compressed(out/'geometry_plan.npz',**numbers)
 elapsed=time.monotonic()-start
 recipe=dict(schema='teacher_detail_common_average_query_plan.v1',completed=True,order=31,quadrature_points_per_row=q,
  query_rows=count,full_active_rows=len(d['active_world_rows']),sample_uv='sample_uv_f64.npy',
  geometric_sample_valid='geometric_sample_valid.npy',geometry_plan='geometry_plan.npz',
  mapping='query row j -> active_local_indices[j] -> full_active_world_rows[index] == query_world_rows[j]',
  only_common_writable_rows_queried=True,readonly_RGB_policy='retain N-parent original ci; no field requery for readonly rows',
  units='world metres, sigma metres, UV texel centres float64',source='actual canonical XY triangles; affine only seed; exact source world-key centers',
  mapped_points=count*q,unmapped_points=unmapped,worldXY_inverse_max_error_m=errors,source_helper_replay_rows=len(check),
  source_helper_UV_error_max=error,source_helper_valid_exact=valid_exact,high_bound=.34,shader_geometry_opacity_Sigma_active_frozen=True,
  CPU_only=True,CUDA_calls=0,native_calls=0,optimizer_updates=0,controller_created=False,GPU_authorized=False,
  numerical_qualification=str(numeric),independent61_spot=str(spot),seconds=elapsed,
  outputs_sha256={name:sha(out/name) for name in ['sample_uv_f64.npy','geometric_sample_valid.npy','geometry_plan.npz']},
  source_sha256={str(path):sha(path) for path in [P/'code/road_chart_sampling.py',final/'teacher_evidence.npz',numeric,spot]})
 (out/'recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
 r=json.loads((final/'recipe.json').read_text());r['average_query_plan']=str(out);r['average_kernel_order']=31;r['average_points_per_row']=q
 r['average_query_plan_recipe_sha256']=sha(out/'recipe.json');(final/'recipe.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(dict(completed=True,rows=count,order=31,points=count*q,world_error_m=errors,source_replay_UV_error=error,seconds=elapsed)),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--chunk',type=int,default=512)
 a=ap.parse_args();main(a.output,a.chunk)

