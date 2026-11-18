"""固定A/C/S及fullS3187的TRELLIS最终native列评价；不把patch误作整街。"""
from pathlib import Path
import os,argparse,json
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np
from exact_mesh_uv import WorldMeshUVMap
P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_capacity_cpu_20261117_r1')

def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

def prepare(output):
 if output.exists():raise FileExistsError('preserve old evaluation domains')
 output.mkdir(parents=True)
 mapper=WorldMeshUVMap.load(P/'trellis2/world_prior_r1')
 with np.load(P/'trellis2/input_r1/reference_sources.npz',allow_pickle=False) as z:ref=z['valid'].copy()
 stats=[]
 for raw in [86,110]:
  target_path=P/f'stage0_native_r1/{raw:03d}_targets.npz';t=load(target_path)
  old=load(C/f'supervision_v4_candidate_r1/{raw:03d}.npz');q=mapper.query(old['ground_world_xyz'][...,:2]);patch=q['valid']
  ref_mass=np.full(patch.shape,np.nan);faces=q['fixed_face_id'][patch]
  ref_mass[patch]=np.clip(np.sum(q['barycentric'][patch]*ref[mapper.faces[faces]],axis=1),0,1)
  new=t['fullS']&~old['S_target_shadow_photo']&(t['fixed_alpha']>.9)
  domains=dict(A=t['A'],C=t['C'],S_photo=t['S'],S_full=t['fullS'],S_oldphoto=old['S_target_shadow_photo'],S_new909=new,S_remaining56=t['fullS']&(t['fixed_alpha']<=.9))
  for name in ['A','C','S_photo','S_full','S_new909']:
   domains[name+'_patch']=domains[name]&patch
   domains[name+'_exterior']=domains[name]&~patch
   domains[name+'_patch_reference_full']=domains[name]&patch&(ref_mass>=1-1e-9)
   domains[name+'_patch_reference_mixed']=domains[name]&patch&(ref_mass>1e-9)&(ref_mass<1-1e-9)
   domains[name+'_patch_reference_unknown']=domains[name]&patch&(ref_mass<=1e-9)
  np.savez_compressed(output/f'{raw:03d}_domains.npz',**domains,patch_geometry_point=patch,colocated_reference_valid_mass=ref_mass,
   fixed_face_id=q['fixed_face_id'],generated_uv=q['generated_uv'],barycentric=q['barycentric'])
  stats.append(dict(raw=raw,source_targets=str(target_path),counts={k:int(v.sum()) for k,v in domains.items()}))
 assert sum(r['counts']['S_full'] for r in stats)==3187
 assert sum(r['counts']['S_new909'] for r in stats)==909
 assert sum(r['counts']['S_remaining56'] for r in stats)==56
 write(output/'domains_receipt.json',dict(schema='T2_fixed_native_error_domains.v1',rows=stats,fullS3187=3187,new909=909,remaining56=56,
  source_reference='A real visible, C frozen physical-hole pseudo, S frozen target-shadow pseudo; not hidden GT',
  patch_scope='actual worldXY query inside fixed generated geometry; pixel point domain only, not proof of full Gaussian footprint or joint alphaT visibility',
  reference_validity='co-located input reference support at source face corners; not neural causal attribution and never an error-evaluation exclusion',
  source_RGB_frames=[86,110],diagnostic_RGB_not_consumed=[134,179,198],GPU_operations=0,optimizer_updates=0,human_verdict=None))
 print(json.dumps(stats))


def metrics(pred,target,alpha,mask):
 n=int(mask.sum())
 if not n:return dict(pixels=0,MAE=None,RMSE=None,MSE=None,gap=0,thin=0,supported=0)
 e=pred[:,mask].astype(np.float64)-(alpha[None]*target)[:,mask].astype(np.float64)
 return dict(pixels=n,MAE=float(np.mean(np.abs(e))),RMSE=float(np.sqrt(np.mean(e*e))),MSE=float(np.mean(e*e)),
  gap=int((mask&(alpha<.05)).sum()),thin=int((mask&(alpha>=.05)&(alpha<=.9)).sum()),supported=int((mask&(alpha>.9)).sum()))


def evaluate(args):
 mapping=json.loads(args.renders.read_text());baseline=json.loads(args.baseline.read_text()) if args.baseline else None
 rows=[];global_error={}
 for raw in [86,110]:
  native_path=Path(mapping[f'{raw:03d}'] if f'{raw:03d}' in mapping else mapping[str(raw)]);render=load(native_path)
  targets=load(P/f'stage0_native_r1/{raw:03d}_targets.npz');d=load(args.domains/f'{raw:03d}_domains.npz');alpha=render[args.alpha_key]
  if alpha.ndim==3:alpha=alpha[0]
  if not np.array_equal(alpha,targets['fixed_alpha']):raise ValueError('native column changed fixed coverage')
  prediction=render[args.road_key]
  if prediction.shape!=(3,320,480) or not np.isfinite(prediction).all():raise ValueError('expected actual native premult road RGB CHW')
  result={}
  for name,mask in d.items():
   if mask.dtype!=bool or name=='patch_geometry_point':continue
   role='A' if name.startswith('A') else 'C' if name.startswith('C') else 'S'
   value=metrics(prediction,targets['target_'+role],alpha,mask);result[name]=value
   if value['pixels']:
    old=global_error.setdefault(name,dict(pixels=0,abs_sum=0.,sq_sum=0.));old['pixels']+=value['pixels'];old['abs_sum']+=value['MAE']*3*value['pixels'];old['sq_sum']+=value['MSE']*3*value['pixels']
  delta={}
  if baseline:
   base=load(Path(baseline.get(f'{raw:03d}',baseline.get(str(raw)))))
   e=np.abs(prediction.astype(np.float64)-base[args.road_key].astype(np.float64)).max(0)
   delta=dict(road_RGB_max=float(e.max()),fullS_RGB_delta_mean=float(e[targets['fullS']].mean()),patch_point_RGB_delta_mean=float(e[d['patch_geometry_point']].mean()),exterior_point_RGB_delta_max=float(e[~d['patch_geometry_point']].max()))
   if 'joint_rgb' in render and 'joint_rgb' in base and 'K' in render:
    K=render['K'];j=np.abs(render['joint_rgb'].astype(np.float64)-base['joint_rgb'].astype(np.float64)).max(0);delta.update(K_pixels=int(K.sum()),joint_K_max=float(j[K].max()))
  rows.append(dict(raw=raw,native_render=str(native_path),fixed_alpha_exact=True,metrics=result,counterfactual_vs_A0=delta))
 pooled={k:dict(pixels=v['pixels'],MAE=v['abs_sum']/(3*v['pixels']),RMSE=float(np.sqrt(v['sq_sum']/(3*v['pixels']))),MSE=v['sq_sum']/(3*v['pixels'])) for k,v in global_error.items()}
 if pooled['S_full']['pixels']!=3187 or pooled['S_new909']['pixels']!=909:raise ValueError('full denominator changed')
 args.output.mkdir(parents=True,exist_ok=False)
 write(args.output/'native_error_metrics.json',dict(schema='T2_actual_native_fixed_domain_error.v1',rows=rows,pooled=pooled,
  metric='actual native black-background premult sRGB P vs SAME fixed alpha*T_sRGB; no R/alpha or exposure fit',
  fullS_denominator=3187,teacher_status='C/S pseudo, A observed; not hidden GT',
  boundary='patch geometric point mask is not final Gaussian kernel support. Exterior image pixels may receive boundary Gaussian effects; evaluated separately, never silently omitted',
  quality_or_production_pass_claimed=False,diagnostic_frames_are_not_teachers=True,GPU_operations=0,optimizer_updates=0,human_verdict=None))
 print(json.dumps(dict(output=str(args.output),pooled=pooled)))


def main():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU only')
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','evaluate']);ap.add_argument('--output',type=Path,required=True)
 ap.add_argument('--domains',type=Path,default=P/'trellis2/eval_domains_r1');ap.add_argument('--renders',type=Path);ap.add_argument('--baseline',type=Path)
 ap.add_argument('--road-key',default='road_rgb');ap.add_argument('--alpha-key',default='road_alpha');a=ap.parse_args()
 if a.command=='prepare':prepare(a.output)
 else:
  if not a.renders:raise ValueError('actual native render map required')
  evaluate(a)
if __name__=='__main__':main()
