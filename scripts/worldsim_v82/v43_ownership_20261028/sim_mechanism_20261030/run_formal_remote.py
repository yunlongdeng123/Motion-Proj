"""一次正式M0运行；强制真实smoke收据、保留所有现场，无电源操作。"""
from pathlib import Path
import subprocess, json, time, hashlib, shutil
root=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030')
python='/root/autodl-tmp/envs/drivingeditor/bin/python'
code=root/'code/reconstruction'
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
smoke=read(root/'smoke_r2/smoke_summary.json')
assert smoke['status']=='PASS_REAL_NATIVE_SMOKE' and smoke['approximately_ten_minutes_contract']
assert read(root/'dataset_r2/validation.json')['status']=='passed'
assert shutil.disk_usage(root).free>25*1024**3
run=root/'formal_r3_10k'
assert not run.exists(), 'Preserve earlier run; do not overwrite'
command=[python,'-u',str(code/'native_worker.py'),'--scene',str(root/'native_scene_r3'),'--output',str(run),'--steps','10000','--checkpoint-every','5000']
write(root/'formal_launch_contract.json',{'command':command,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(code.glob('*.py'))},'contract':read(root/'EXPERIMENT_CONTRACT.json'),'smoke':str(root/'smoke_r2/smoke_summary.json'),'started_unix':time.time()})
start=time.monotonic()
with (root/'formal_r3_10k.log').open('x') as stream:
 result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
receipt={'returncode':result.returncode,'wall_seconds':time.monotonic()-start,'command':command}
write(root/'formal_execution.json',receipt)
if result.returncode:raise RuntimeError('Formal native train failed; preserve log and checkpoints')
subprocess.run([python,str(code/'inspect_bundle.py'),'--run',str(run)],check=True)
event=run/'checkpoint_events/iteration_010000.json'
command=[python,'-u',str(code/'evaluate_edits.py'),'--event',str(event),'--output',str(root/'edit_r3_10k'),'--reference-root',str(root/'dataset_r2/privileged')]
start=time.monotonic()
with (root/'edit_r3_10k.log').open('x') as stream:
 result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
write(root/'edit_execution.json',{'returncode':result.returncode,'wall_seconds':time.monotonic()-start,'command':command})
if result.returncode:raise RuntimeError('Native edit evaluation failed; preserve output')
print(json.dumps({'formal_updates':10000,'native_edit_views':12,'power_actions':0}))
