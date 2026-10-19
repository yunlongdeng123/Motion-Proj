"""只读核对全部训练结束、完整节点、GPU/全局锁与当前worker/helper。"""
from pathlib import Path
import json,subprocess

W=Path(__file__).resolve().parent
O=W.parent.parent/'outputs/gn11_stage1_v2_20261019'
REMOTE=r'''
from pathlib import Path
import datetime,fcntl,hashlib,json,os,shutil,subprocess
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v2_20261019')
def read(p):return json.loads(p.read_text())
issues=[];runs={};registry=read(C/'ablations/registry.json')
if registry['status']!='THREE_ABLATIONS_TRAINING_COMPLETE_AWAITING_ZERO_UPDATE49_AND_REPORT' or registry['pending_arms']:
    issues.append('authorized controller is not in final training-complete stage')
for name,campaign in [('full',C)]+[(a,C/'ablations'/a) for a in ['real_only','prior_only','ownership_only']]:
    s=read(campaign/'run30k/status.json');audits=[]
    if s['status']!='complete' or s['iteration']!=30000 or any(s['optimizer_step_counts'][k]!=30000 for k in ['static','dynamic','environment']):
        issues.append(name+' actual complete30K counters absent')
    for step in range(5000,30001,5000):
        p=campaign/'audits'/('step_%06d.json'%step)
        if not p.exists():issues.append(name+' audit missing '+str(step));continue
        a=read(p)
        if a['status']!='PASSED_CPU_NODE_DURABILITY' or a['step']!=step or a['sampler_steps']!=step or a['GPU_restore_performed'] or a['training_updates']:
            issues.append(name+' actual audit differs '+str(step))
        else:
            current={}
            for filename in a['files']:
                st=(campaign/'run30k'/filename).stat();current[filename]={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
            if current!=a['files']:issues.append(name+' checkpoint metadata changed after audit '+str(step))
            else:audits.append(step)
    manifest=read(campaign/'source_bundle/candidate_manifest.json')
    engine_ok=True
    for rel,expected in manifest['files'].items():
        if hashlib.sha256((campaign/'engine'/rel).read_bytes()).hexdigest()!=expected['candidate_sha256']:
            engine_ok=False
            issues.append(name+' loaded candidate bytes differ '+rel)
    runs[name]={'status':s,'CPU_audited_steps':audits,'engine_manifest_verified':engine_ok}
jobs=[]
markers=['run_candidate.py','run_ablation.py','run_road_ablation_group.py','audit_v2_checkpoint.py','evaluate_v2_full49.py','package_v2_full49.py']
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace').strip()
        if str(C) in cmd and any(x in cmd for x in markers):
            state=next(x for x in (p/'status').read_text().splitlines() if x.startswith('State:'))
            jobs.append({'pid':int(p.name),'state':state,'command':cmd})
    except (FileNotFoundError,PermissionError,ProcessLookupError):pass
if jobs:issues.append('current authorized controller/worker/evaluator/helper still exists')
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
if gpu:issues.append('GPU compute app still active')
md={k:(Path('/sys/block/md0/md')/k).read_text().strip() for k in ['array_state','degraded','sync_action']}
if md['array_state'] not in ['active','clean'] or md['degraded']!='0' or md['sync_action']!='idle':issues.append('storage not healthy for new GPU work')
free=shutil.disk_usage(C).free
if free<2e9:issues.append('insufficient export/archive reserve')
lock_available=False
with (C.parent/'gpu_slot.lock').open('r+') as lock:
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock_available=True;fcntl.flock(lock,fcntl.LOCK_UN)
    except BlockingIOError:issues.append('original global GPU lock held')
out={'status':'PASSED_ALL_FOUR_TRAININGS_ENDED_GPU_IDLE_AND_LOCK_AVAILABLE' if not issues else 'NOT_READY_FOR_FINAL_GPU_EVALUATION',
    'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'issues':issues,'runs':runs,'registry_status':registry['status'],
    'active_jobs':jobs,'GPU_compute_apps':gpu,'md0':md,'disk_free_bytes':free,'global_lock_available':lock_available,
    'GPU_allocations':0,'training_updates':0,'remote_files_written':False}
print(json.dumps(out))
'''

if __name__=='__main__':
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','wm-vgpu-1011','/usr/bin/python3','-'],
        input=REMOTE,capture_output=True,text=True,encoding='utf-8',check=True)
    result=json.loads(p.stdout)
    for f in [W/'final_evaluation_preflight.json',O/'final_evaluation_preflight.json']:
        f.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','checked_utc','issues','active_jobs','GPU_compute_apps','global_lock_available']}))
