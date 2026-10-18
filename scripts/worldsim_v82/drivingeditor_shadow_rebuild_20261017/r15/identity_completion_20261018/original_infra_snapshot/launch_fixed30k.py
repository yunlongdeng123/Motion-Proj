"""One-shot same-run 300->30000 handoff, never a competing auto-retry queue."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
BASE=Path('/root/autodl-tmp/drivingeditor_repro/campaign/instance_editor_20261013')
sys.path.insert(0,str(BASE/'infra'))
from priority import inspect_priority

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def main():
    if '--run' not in sys.argv: raise SystemExit('Explicit --run required')
    proof=read(BASE/'metadata/edit_smoke_300_summary_r1.json')
    if proof['engineering_contract'].get('passed') is not True: raise ValueError('Real 6x49 edit contract required')
    if read(BASE/'metadata/actual_cuda_contract_r1.json').get('status')!='PASS': raise ValueError('Actual CUDA occlusion/gradient test required')
    if read(BASE/'metadata/runtime_restore_300_preflight_r1/receipt.json').get('runtime_restoration_verified') is not True:
        raise ValueError('Actual full runtime restore required')
    decision=read(BASE/'metadata/engineering_smoke_resume_decision_r1.json')
    if decision.get('continue_same_fixed_trajectory_to_total_30000') is not True or decision.get('human_quality_verdict') is not None:
        raise ValueError('Root engineering review must allow handoff; no fabricated human verdict')
    admission=inspect_priority()
    if not admission['allowed']: raise RuntimeError(json.dumps(admission))
    launch=BASE/'queue_gmc30k_r1'
    output=BASE/'runs/gmc_two_actor_fixed30k_r1'
    if launch.exists() or output.exists(): raise FileExistsError('Refuse duplicate continuation or existing assets')
    launch.mkdir()
    command=['nice','-n','10',sys.executable,'-B','-u',str(BASE/'infra/train.py'),
        '--spec',str(BASE/'specs/scene0008_023_071.json'),'--resume',str(BASE/'queue_gmc_r1/gpu_smoke/checkpoint_000300.pt'),
        '--iterations','30000','--seed','0','--output',str(output),'--run']
    with (launch/'worker.log').open('x',encoding='utf-8') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
            env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4'),start_new_session=True)
    receipt={'pid':child.pid,'command':command,'utc':datetime.now(timezone.utc).isoformat(),
        'admission':admission,'output':str(output),'resume_step':300,'first_step':301,'target_total_steps':30000,
        'new_updates_budget':29700,'checkpoint_steps':[10000,20000,30000],
        'background_frozen':True,'trajectory_fixed':True,'new_SGCM_clones':False,
        'CPU_max_threads':4,'nice':10,'other_tasks_modified':False,'automatic_retries':False}
    (launch/'launch.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt))

if __name__=='__main__': main()
