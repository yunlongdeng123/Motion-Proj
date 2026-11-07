from pathlib import Path
import argparse,json,subprocess,sys,time,hashlib,os
ROOT=Path(__file__).resolve().parent.parent
ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['formal5k','formal10k'],required=True)
ap.add_argument('--scope',choices=['pilot','all49'],default='pilot');a=ap.parse_args()
model=ROOT/a.phase;iteration=5000 if a.phase=='formal5k' else 10000
returned=json.loads((model/'returned.json').read_text());assert returned['iteration']==iteration and returned['status']=='PASS_COMPLETE_BOUNDARY'
label=a.phase+'_keep_'+a.scope;out=ROOT/label;assert not out.exists()
entry=ROOT/'source/evaluate_reconstruction.py'
command=[sys.executable,'-B',str(entry),'--model',str(model),'--iteration',str(iteration),'--output',str(out),'--scope',a.scope]
if a.scope=='all49':
    pilot=ROOT/(a.phase+'_keep_pilot');summary=json.loads((pilot/'summary.json').read_text())
    assert summary['status']=='PASS_NATIVE_KEEP_RECONSTRUCTION_PILOT' and summary['iteration']==iteration
    command+=['--pilot-reference',str(pilot)]
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=12288,free
env=os.environ.copy()
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[key]='1'
with (ROOT/(label+'.log')).open('x') as log:
    proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,env=env,start_new_session=True)
receipt=dict(status='STARTED_NATIVE_KEEP_RECONSTRUCTION_EVALUATION',pid=proc.pid,unix=time.time(),command=command,
    iteration=iteration,scope=a.scope,optimization_updates=0,gpu_free_before_MiB=free,
    entry_sha256=hashlib.sha256(entry.read_bytes()).hexdigest(),new_editing_queries=False,shutdown=False,reset_card=False)
(ROOT/(label+'_launch_receipt.json')).write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False))
