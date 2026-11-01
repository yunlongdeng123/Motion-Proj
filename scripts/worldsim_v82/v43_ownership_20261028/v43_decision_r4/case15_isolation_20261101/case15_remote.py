import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,pathlib,shutil,time
import numpy as np
H=pathlib.Path('/root/autodl-tmp/codex_tmp/v43_case15_isolation_20261101_r1');O=H/'evidence';O.mkdir(exist_ok=True)
rows=json.loads((H/'shortlist.json').read_text(encoding='utf8'))
P=pathlib.Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
L=pathlib.Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/baseline_engineering_20261018/full5_pipeline/inputs/scene0008_089_137')
report={'task_id':'WS-V43-CASE15-ISOLATION-20261101-R1','CPU_only':True,'rows':[],'npz_inventory':[]}
def copy(p,target):
 if p.is_file():target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);return True
 return False
for r in rows:
 key=r['scene_key']+'_'+r['alias'];entry={'case':key,'target_track':r['track_id']};a=r.get('cpu_input_audit')
 if a:
  prep=pathlib.Path(a['preparation_path']);root=prep.parent;actor_idx=r['dynamic_actor_index'];cp=pathlib.Path(a['checkpoint']);run=cp.parent
  copy(prep,O/key/'preparation.json');copy(root/'transform_matrix.txt',O/key/'transform_matrix.txt');copy(run/'cameras.json',O/key/'cameras.json')
  for name in ['effective_config.json','returned.json','dynamic_asset_alignment.json']:
   copy(run/name,O/key/name)
  configp=run/'effective_config.json'
  entry['config']={k:v for k,v in json.loads(configp.read_text()).items() if k in ['source_path','model_path','time_duration','cycle','including_dynamic','env_map_res','white_background','sh_degree','resolution_scales']} if configp.is_file() else None
  entry['bundle_files']={p.name:{'exists':p.is_file(),'bytes':p.stat().st_size if p.is_file() else None} for p in [cp,run/'dynamic_chkpnt30000.pth',run/'env_light_chkpnt30000.pth',run/'training_state30000.pth']}
  w=r['native_window'];anchor=[w['first_raw_frame'],(w['first_raw_frame']+w['last_raw_frame'])//2,w['last_raw_frame']]
  entry['mask_copied']=[]
  for f in a['expected_raw_window'][:1]:pass
  pa=json.loads(prep.read_text());actor=next(x for x in pa['actors'] if x['actor_index']==actor_idx and x['raw_track_id']==r['track_id']);entry['prepared_actor_count']=len(pa['actors'])
  for j,rec in enumerate(actor['frames']):
   raw=w['first_raw_frame']+j
   if raw not in anchor:continue
   fid=rec['frame_id'];m=root/'dynamic_actor_assets'/('actor_%d'%actor_idx)/'body_masks'/(fid+'.png')
   if copy(m,O/key/(str(raw)+'_mask.png')):entry['mask_copied'].append(raw)
  asset=root/'dynamic_actor_assets'/('actor_%d'%actor_idx)
  entry['asset_top_files']=[{'name':p.name,'bytes':p.stat().st_size} for p in asset.iterdir() if p.is_file()]
  for p in root.glob('*.json'):
   if p.stat().st_size<300000 and any(x in p.name for x in ['actor','preparation','manifest']):copy(p,O/key/p.name)
 else:entry['native_asset_for_target']='not registered; scene G may still exist for static candidates'
 report['rows'].append(entry)
for ctx in sorted({r['context_id'] for r in rows if r['scene_key'] in ['waymo-train-212','waymo-train-537','waymo-train-564']}):
 run=P/'native_scene_training_v1'/ctx/'formal30k'
 for folder in ['train/ours_30000','renders','static_time_30000','evaluation_30000']:
  p=run/folder
  if p.exists():
   files=list(p.glob('*.npz')); report['npz_inventory'].append({'path':str(p),'count':len(files),'first':str(files[0]) if files else None})
   if files:
    with np.load(files[0]) as z:report['npz_inventory'][-1]['arrays']={k:list(z[k].shape) for k in z.files}
 for p in P.glob('actor_view_support_v*/'+ctx+'.json'):copy(p,O/'existing_view_support'/p.parent.name/(ctx+'.json'))
R4=pathlib.Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/v43_decision_r4_cpu_20261031/run_r2_maskfix/real_first_scene_native30k')
for name in ['candidate_manifest.json','camera_track_context.json','tasks.json','shared_routes.json']:copy(R4/name,O/'existing_r4_real'/name)
GPU=pathlib.Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/v43_decision_r4_gpu_20261031/run_r1/real_first_scene_native30k')
for p in GPU.rglob('gpu_manifest.json'):copy(p,O/'existing_r4_real'/('gpu_'+p.parent.name+'_manifest.json'))
report['existing_r4_real_GPU_NPZ']=[str(p) for p in GPU.rglob('*.npz')][:20]
src=pathlib.Path('/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82/v43_ownership_20261028/v43_decision_r4/native_shared_route.py');copy(src,O/'native_shared_route.py')
native=pathlib.Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018/training_source/scene0008_089_137')
copy(native/'gaussian_renderer/__init__.py',O/'native_gaussian_renderer.py');copy(native/'scene/gaussian_model.py',O/'native_gaussian_model.py')
(H/'remote_inventory.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({'cases':len(rows),'native_inputs':sum('mask_copied' in x for x in report['rows']),'npz':report['npz_inventory']},ensure_ascii=False),flush=True)
