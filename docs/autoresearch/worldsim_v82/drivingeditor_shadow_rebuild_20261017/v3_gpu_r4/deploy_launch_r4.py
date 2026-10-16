from pathlib import Path
import os,subprocess,json,zipfile,datetime
root=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
if json.loads((root/'E0_gate5000_controller.json').read_text())['status']!='WAITING_5000_STEP_VISUAL_REVIEW':raise ValueError('E0 must be stopped at5000')
if (root/'E1_balanced_s0').exists() or (root/'E1_repair1000_launch.json').exists():raise FileExistsError('do not duplicate repair run')
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
if gpu.strip():raise ValueError('GPU occupied')
code=root/'implementation_r4';code.mkdir(exist_ok=False)
with zipfile.ZipFile(root/'source_r4.zip') as z:
 if any('..' in Path(n).parts or n.startswith('/') for n in z.namelist()):raise ValueError('unsafe archive')
 z.extractall(code)
code=code/'shadow_rebuild_v3'
helper=root/'E1_repair1000_controller.py'
helper.write_text('''from pathlib import Path
import json,subprocess,datetime
root=Path(__file__).resolve().parent;code=root/'implementation_r4/shadow_rebuild_v3';python='/root/autodl-tmp/envs/drivingeditor/bin/python'
state=dict(status='RUNNING',phase='supervision_repair_5000_to6000',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
def record():
 state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 tmp=root/'E1_repair1000_controller.json.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(root/'E1_repair1000_controller.json')
try:
 record()
 subprocess.run([python,'run.py','train','--data',str(root/'data_r2'),'--config',str(code/'configs/scene0008_balanced.json'),'--output',str(root/'E1_balanced_s0'),'--allow-partial-masks','--fork-supervision-from',str(root/'E0_full_s0/latest.pt'),'--max-updates','1000','--execute'],cwd=code,check=True)
 state.update(phase='preview_and_layer_diagnostic');record()
 subprocess.run([python,str(root/'checkpoint_preview_r4.py'),'--implementation','implementation_r4','--run','E1_balanced_s0','--step','6000'],cwd=code,check=True)
 subprocess.run([python,str(root/'diagnose_layer_r4.py'),'--implementation','implementation_r4','--run','E1_balanced_s0','--step','6000','--suffix','r4'],cwd=code,check=True)
 state.update(status='WAITING_6000_STEP_DYNAMIC_LAYER_REVIEW',phase='geometry')
except BaseException as e:
 state.update(status='FAILED',error=repr(e));raise
finally:record()
''')
cmd=['flock','-n','/root/autodl-tmp/drivingeditor_repro/campaign/gpu_slot.lock','/root/autodl-tmp/envs/drivingeditor/bin/python',str(helper)]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
with (root/'E1_repair1000.log').open('x') as log:p=subprocess.Popen(cmd,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,command=cmd,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_step=5000,stop_at_step=6000,
            code_snapshot='implementation_r4',data='data_r2',priority='dynamic branch leakage before material/shadow',total_budget_45000_unchanged=True)
(root/'E1_repair1000_launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
