from pathlib import Path
import json,subprocess,os,datetime
root=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
gate=json.loads((root/'root_1000_step_gate.json').read_text())
if gate['decision']!='CONTINUE_GEOMETRY_TRAINING':raise ValueError('root visual gate not passed')
if json.loads((root/'E0_pilot_controller.json').read_text())['status']!='WAITING_1000_STEP_VISUAL_REVIEW':raise ValueError('pilot not finished')
if (root/'E0_gate5000_launch.json').exists():raise FileExistsError('inspect existing controller, never relaunch')
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
if gpu.strip():raise ValueError('GPU occupied: '+gpu)
code=root/'implementation_r3/shadow_rebuild_v3'
helper=root/'E0_gate5000_controller.py'
helper.write_text('''from pathlib import Path
import json,subprocess,datetime
root=Path(__file__).resolve().parent;code=root/'implementation_r3/shadow_rebuild_v3'
python='/root/autodl-tmp/envs/drivingeditor/bin/python'
state=dict(status='RUNNING',phase='geometry_to_5000',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
def record():
 state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 tmp=root/'E0_gate5000_controller.json.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(root/'E0_gate5000_controller.json')
try:
 record()
 subprocess.run([python,'run.py','train','--data',str(root/'data_r2'),'--config',str(code/'configs/scene0008.json'),'--output',str(root/'E0_full_s0'),'--allow-partial-masks','--resume',str(root/'E0_full_s0/latest.pt'),'--max-updates','4000','--execute'],cwd=code,check=True)
 state.update(phase='preview_5000');record()
 subprocess.run([python,str(root/'checkpoint_preview.py'),'--step','5000'],cwd=code,check=True)
 state.update(status='WAITING_5000_STEP_VISUAL_REVIEW',phase='geometry')
except BaseException as e:
 state.update(status='FAILED',error=repr(e));raise
finally:
 record()
''')
cmd=['flock','-n','/root/autodl-tmp/drivingeditor_repro/campaign/gpu_slot.lock','/root/autodl-tmp/envs/drivingeditor/bin/python',str(helper)]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
with (root/'E0_gate5000.log').open('x') as log:
 p=subprocess.Popen(cmd,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,command=cmd,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),code_snapshot='implementation_r3',resumed_global_step=1000,
            next='root review at5000 before completing45k budget; only7 admitted Actors for65stream49frame evaluation')
(root/'E0_gate5000_launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
