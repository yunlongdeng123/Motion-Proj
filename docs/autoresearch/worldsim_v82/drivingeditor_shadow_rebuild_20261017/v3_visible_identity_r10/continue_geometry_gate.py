"""Run only an explicitly reviewed geometry interval, preserving its parent checkpoint."""
from pathlib import Path
import argparse,json,subprocess,os,datetime,shutil
p=argparse.ArgumentParser();p.add_argument('--parent',type=int,required=True);p.add_argument('--gate',type=int,required=True);p.add_argument('--review',required=True);p.add_argument('--worker',action='store_true');a=p.parse_args()
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');py='/root/autodl-tmp/envs/drivingeditor/bin/python';run='E6_visible_identity_s0';prefix=f'E6_gate{a.gate}'
if not 9000<a.parent<a.gate<=29000:raise ValueError('geometry only; no implicit material')
review=json.loads((r/a.review).read_text())
if review.get('source_step')!=a.parent or review.get('next_gate')!=a.gate or review.get('decision')!='CONTINUE_GEOMETRY':raise ValueError('missing concrete review')
state=r/(prefix+'_controller.json')
def write(status,**kw):
 temp=state.with_suffix('.tmp');temp.write_text(json.dumps(dict(status=status,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw),indent=2));temp.replace(state)
if not a.worker:
 if (r/(prefix+'_launch.json')).exists():raise FileExistsError('no duplicate')
 active=json.loads((r/'active_experiment.json').read_text());previous=r/active['training_controller']
 if not json.loads(previous.read_text())['status'].startswith('WAITING_'):raise ValueError('previous controller not at review gate')
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 if gpu:raise RuntimeError('other GPU tasks first: '+gpu)
 command=['flock','-n','/root/autodl-tmp/drivingeditor_repro/campaign/gpu_slot.lock',py,str(r/'continue_geometry_gate.py'),'--parent',str(a.parent),'--gate',str(a.gate),'--review',a.review,'--worker']
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
 with (r/(prefix+'.log')).open('xb') as log:worker=subprocess.Popen(command,cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 receipt=dict(pid=worker.pid,parent_step=a.parent,next_gate=a.gate,review=a.review,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(r/(prefix+'_launch.json')).write_text(json.dumps(receipt))
 active.update(training_output=run,training_launch=prefix+'_launch.json',training_controller=prefix+'_controller.json',training_log=prefix+'.log',controller_script='continue_geometry_gate.py',parent_global_step=a.parent,next_visual_gate_global_step=a.gate,preview_path=f'E6_preview_{a.gate:06d}',discarded_updates_charged=1000,final_working_global_step=44000)
 (r/'active_experiment.json').write_text(json.dumps(active,indent=2));print(json.dumps(receipt));raise SystemExit(0)
try:
 import torch
 saved=torch.load(r/run/'latest.pt',map_location='cpu',weights_only=False)
 if saved['global_step']!=a.parent or saved['stage']!='geometry' or saved['config']['geometry_steps']!=29000:raise ValueError('checkpoint phase/budget differs')
 snapshot=r/run/f'gate_{a.parent:06d}.pt'
 if not snapshot.exists():shutil.copyfile(r/run/'latest.pt',snapshot)
 del saved
 write('RUNNING',phase='reviewed_geometry_continuation',parent_global_step=a.parent,next_gate=a.gate,discarded_updates_charged=1000)
 subprocess.run([py,str(r/'implementation_r10/shadow_rebuild_v3/run.py'),'train','--data',str(r/'data_r5_ray_visible_road'),'--config',str(r/run/'resolved_config.json'),
  '--output',str(r/run),'--resume',str(r/run/'latest.pt'),'--execute','--allow-partial-masks','--max-updates',str(a.gate-a.parent)],cwd=r,check=True)
 write('RUNNING',phase='same7_preview',global_step=a.gate,charged_updates=a.gate+1000)
 subprocess.run([py,str(r/'checkpoint_preview_r6.py'),'--run',run,'--step',str(a.gate),'--implementation','implementation_r10'],cwd=r,check=True)
 write('RUNNING',phase='full49_canonical_geometry_review',global_step=a.gate)
 subprocess.run([py,str(r/'geometry_sequence_review_r6.py'),'--run',run,'--step',str(a.gate),'--implementation','implementation_r10','--data','data_r5_ray_visible_road','--output',f'E6_geometry49_{a.gate:06d}'],cwd=r,check=True)
 write('RUNNING',phase='common_R9_attribution',global_step=a.gate)
 subprocess.run([py,str(r/'fullscene_ownership_audit_common_r9.py'),'--run',run,'--step',str(a.gate),'--implementation','implementation_r10','--data','data_r5_ray_visible_road','--certified-road','--all-identities','--output',f'commonR9_E6_{a.gate:06d}'],cwd=r,check=True)
 write(f'WAITING_{a.gate}_STEP_GEOMETRY_REVIEW',global_step=a.gate,charged_updates=a.gate+1000,material_started=False)
except BaseException as exc:
 write('FAILED',error=repr(exc));raise
