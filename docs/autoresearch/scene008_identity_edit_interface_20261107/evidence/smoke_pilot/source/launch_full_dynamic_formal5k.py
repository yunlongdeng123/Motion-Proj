from pathlib import Path
import json,hashlib,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent.parent
gate=json.loads((ROOT/'smoke_lifecycle_gate.json').read_text())
assert gate['status']=='PASS_REAL_LONG540_FULL_DYNAMIC_SAVE_PILOT_FULL_RESTORE_NEXT_ENTRY'
assert gate['exact_full_restore_differences']==[] and gate['smoke_updates']==4681 and gate['resume_extra_updates']==2
assert not (ROOT/'formal5k').exists()
entry=ROOT/'source/train_full5.py'
env=os.environ.copy()
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[key]='1'
command=[sys.executable,'-B',str(entry),'--mode','formal','--stop-iteration','5000','--output',str(ROOT/'formal5k')]
with (ROOT/'formal5k.log').open('x') as log:
    proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,env=env,start_new_session=True)
receipt=dict(status='RELEASED_FRESH_FORMAL5000_AFTER_REAL_SMOKE_LIFECYCLE_PASS',pid=proc.pid,unix=time.time(),command=command,
    source_iteration=0,formal_stop=5000,learning_rate_horizon=30000,source_smoke_not_resumed=True,
    dynamic_actor_ids=[5,6,18,26,37],training_delete_actor_ids=[],default_query_delete_actor_ids=[37],
    engineering_updates_separate=4683,ten_k_condition='Root preauthorized exact5K->10K if three-frame5K remains visibly blurred',
    training_entry_sha256=hashlib.sha256(entry.read_bytes()).hexdigest(),shutdown=False,reset_card=False)
(ROOT/'formal5k_launch_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
(ROOT/'formal5k.pid').write_text(str(proc.pid)+'\n')
print(json.dumps(receipt,ensure_ascii=False))
