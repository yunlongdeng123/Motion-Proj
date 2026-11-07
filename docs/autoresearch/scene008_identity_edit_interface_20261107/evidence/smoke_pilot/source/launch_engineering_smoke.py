from pathlib import Path
import os,sys,json,subprocess,time,hashlib
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_full_dynamic_edit_20261107_r1')
entry=ROOT/'source/train_full5.py'
assert entry.is_file() and not (ROOT/'smoke').exists()
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
assert free>=12288,(free,apps)
command=[sys.executable,'-B',str(entry),'--mode','smoke','--smoke-policy','long540','--smoke-seconds','540','--output',str(ROOT/'smoke')]
env=os.environ.copy()
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[name]='1'
with (ROOT/'smoke.log').open('x') as log:
    proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,env=env,start_new_session=True)
receipt=dict(status='STARTED_REAL_GPU_FULL_DYNAMIC_EDIT_LIFECYCLE_SMOKE',pid=proc.pid,unix=time.time(),command=command,
    train_entry_sha256=hashlib.sha256(entry.read_bytes()).hexdigest(),gpu_free_before_MiB=free,gpu_apps_before=apps,
    dynamic_actor_ids=[5,6,18,26,37],delete_actor_ids=[37],smoke_updates_accounted_separately=True,
    formal_auto_release=False,shutdown=False,reset_card=False)
(ROOT/'smoke_launch_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
(ROOT/'smoke.pid').write_text(str(proc.pid)+'\n')
print(json.dumps(receipt,ensure_ascii=False))
