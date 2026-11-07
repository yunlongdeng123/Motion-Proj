from pathlib import Path
import json,hashlib,os,sys,subprocess,time
ROOT=Path(__file__).resolve().parent.parent
def read(p):return json.loads(p.read_text())
gate=read(ROOT/'formal5k_pilot_gate.json');decision=read(ROOT/'five_k_visual_decision.json')
assert gate['status']=='PASS_COMPLETE_FORMAL_NODE_AND_THREE_FRAME_QUERY_ENGINEERING'
assert gate['iteration']==5000 and decision['proceed_exact_10k'] and decision['reason']=='FIVE_K_STILL_VISIBLY_BLURRED'
assert not (ROOT/'formal10k').exists()
checkpoint=ROOT/'formal5k/chkpnt5000.pth'
command=[sys.executable,'-B',str(ROOT/'source/train_full5.py'),'--mode','extend','--checkpoint',str(checkpoint),
    '--stop-iteration','10000','--output',str(ROOT/'formal10k')]
env=os.environ.copy()
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[key]='1'
with (ROOT/'formal10k.log').open('x') as log:
    proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,env=env,start_new_session=True)
receipt=dict(status='RELEASED_EXACT5000_TO10000_UNDER_PREAUTHORIZED_BLUR_CONDITION',pid=proc.pid,unix=time.time(),command=command,
    source_iteration=5000,new_phase_updates_cap=5000,total_formal_stop=10000,learning_rate_horizon=30000,
    exact_full_restore_required=True,from_original_fresh5k=True,source_smoke_not_resumed=True,
    decision=decision,engineering_updates_separate=4683,dynamic_actor_ids=[5,6,18,26,37],training_delete_actor_ids=[],
    further_training_auto_release=False,shutdown=False,reset_card=False)
(ROOT/'formal10k_launch_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
(ROOT/'formal10k.pid').write_text(str(proc.pid)+'\n')
print(json.dumps(receipt,ensure_ascii=False))
