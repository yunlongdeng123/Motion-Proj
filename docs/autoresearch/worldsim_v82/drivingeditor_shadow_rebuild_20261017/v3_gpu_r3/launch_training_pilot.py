from pathlib import Path
import json,subprocess,os,datetime,hashlib
root=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
gate=json.loads((root/'benchmark_r3_controller.json').read_text())
if gate['status']!='FINISHED':raise ValueError('benchmark not finished')
if json.loads((root/'tracer_probe_r3.json').read_text())['status']!='PASSED':raise ValueError('tracer gate failed')
if (root/'E0_full_s0').exists():raise FileExistsError('do not relaunch training')
code=root/'implementation_r3/shadow_rebuild_v3'
helper=root/'E0_pilot_controller.py'
helper.write_text('''from pathlib import Path
import sys,json,subprocess,datetime
root=Path(__file__).resolve().parent
code=root/'implementation_r3/shadow_rebuild_v3'
python='/root/autodl-tmp/envs/drivingeditor/bin/python'
state=dict(status='RUNNING_1000_STEP_GATE',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
try:
 subprocess.run([python,'run.py','train','--data',str(root/'data_r2'),'--config',str(code/'configs/scene0008.json'),'--output',str(root/'E0_full_s0'),'--allow-partial-masks','--max-updates','1000','--execute'],cwd=code,check=True)
 sys.path.insert(0,str(code))
 import torch,numpy as np
 from shadow_rebuild.config import Config,LossWeights
 from shadow_rebuild.data import Dataset
 from shadow_rebuild.model import SceneModel
 from shadow_rebuild.train import seed_from_state,write
 from shadow_rebuild.multi_evaluate import actor_only
 from shadow_rebuild.evaluate import save_png
 torch.set_num_threads(2)
 checkpoint=torch.load(root/'E0_full_s0/latest.pt',map_location='cpu',weights_only=False)
 raw=checkpoint['config'];raw['weights']=LossWeights(**raw['weights']);config=Config(**raw).validate()
 model=SceneModel(seed_from_state(checkpoint['state']),config).cuda();model.load_state_dict(checkpoint['state']);model.requires_grad_(False)
 dataset=Dataset(root/'data_r2','cuda',True)
 evidence=json.loads((root/'actor_evidence_r2/actor_evidence.json').read_text())
 selected=[a for a in evidence['actors'] if a['eligible_delete']]
 out=root/'E0_preview_001000';out.mkdir(exist_ok=False);records=[]
 with torch.no_grad():
  for a in selected:
   fid=Path(a['preview']).stem.split('_',1)[1]
   i=next(i for i,r in enumerate(dataset.rows) if r['frame_id']==fid)
   frame=dataset.frame(i);g,p=model.buffers(frame,len(dataset.actor_ids),'cuda',isolated=False)
   dynamic,alpha=actor_only(frame.camera,g,'cuda',config.max_features)
   body,body_alpha=actor_only(frame.camera,g,'cuda',config.max_features,a['actor_index'])
   _,static=model.buffers(frame,len(dataset.actor_ids),'cuda',{k:{'delete':True} for k in range(len(dataset.actor_ids))},isolated=False)
   images=dict(GT=frame.rgb,canonical=p['canonical'],static=static['canonical'],dynamic=dynamic,alpha=alpha[...,None].expand_as(dynamic),target=body,target_alpha=body_alpha[...,None].expand_as(body))
   for name,value in images.items():save_png(out/f'a{a["actor_index"]:02d}_{fid}_{name}.png',value)
   mask=frame.masks[...,a['actor_index']]
   error=(p['canonical']-frame.rgb).abs()
   records.append(dict(actor=a['actor_index'],frame_id=fid,body_pixels=int(mask.sum()),body_linear_rgb_mae=float(error[mask].mean()),coverage_mean=float(p['alpha'].mean()),gaussians=len(model.field.xyz)))
 write(out/'preview.json',dict(status='REQUIRES_ROOT_VISUAL_REVIEW',global_step=checkpoint['global_step'],records=records,quality_validated=False,light_stage_not_started=True))
 state.update(status='WAITING_1000_STEP_VISUAL_REVIEW',preview=str(out))
except BaseException as e:
 state.update(status='FAILED',error=repr(e));raise
finally:
 state['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 (root/'E0_pilot_controller.json').write_text(json.dumps(state,indent=2))
''')
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
if gpu.strip():raise ValueError('GPU occupied: '+gpu)
cmd=['flock','-n','/root/autodl-tmp/drivingeditor_repro/campaign/gpu_slot.lock','/root/autodl-tmp/envs/drivingeditor/bin/python',str(helper)]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
with (root/'E0_pilot.log').open('x') as log:
 p=subprocess.Popen(cmd,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,command=cmd,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),code_snapshot='implementation_r3',
            source_sha256=hashlib.sha256((root/'source_r3.zip').read_bytes()).hexdigest(),max_updates_before_visual_review=1000)
(root/'E0_pilot_launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
