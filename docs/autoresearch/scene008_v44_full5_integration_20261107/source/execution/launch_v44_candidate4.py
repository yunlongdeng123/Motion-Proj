from pathlib import Path
import json,sys,subprocess,time,hashlib,os
ROOT=Path(__file__).resolve().parent.parent
assert (ROOT/'contract/execution_contract.json').is_file() and not (ROOT/'candidate4').exists()
command=[sys.executable,'-B',str(ROOT/'source/render_v44_candidate4.py')]
env=os.environ.copy()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
with (ROOT/'candidate4.log').open('x') as log:p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,env=env,start_new_session=True)
receipt=dict(status='STARTED_CANDIDATE4_ONLY',pid=p.pid,unix=time.time(),command=command,raws=[105,113,130,137],
    contract_sha256=hashlib.sha256((ROOT/'contract/execution_contract.json').read_bytes()).hexdigest(),
    all49_auto_release=False,new_training_updates=0,shutdown=False,reset_card=False)
(ROOT/'contract/candidate_launch_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
