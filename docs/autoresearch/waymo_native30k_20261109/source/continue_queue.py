"""Bounded 10K->30K reconstruction-only queue; 300s checks, no editing or power actions."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time

TASK=Path(__file__).resolve().parent.parent
PY='/root/autodl-tmp/envs/drivingeditor/bin/python'
JOBS=[
    dict(scene='065',root='/root/autodl-tmp/drivingeditor_repro/campaign/waymo_val065_allframes10k_20261109_r1',count=199,keys=[60,120,133,173]),
    dict(scene='029',root='/root/autodl-tmp/drivingeditor_repro/campaign/waymo_val029_allframes10k_20261109_r1',count=198,keys=[130,155,160,190])]

def read(p): return json.loads(Path(p).read_text())
def write(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()
def status(stage,**kw):
    write(TASK/'status.json',dict(stage=stage,unix=time.time(),controller_pid=os.getpid(),automatic_shutdown=False,new_editing_allowed=False,**kw))
def run(job,name,cmd,timeout=10800):
    root=Path(job['root']); out=root/'continuation30k_20261109_r1'
    started=time.time()
    with (TASK/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(scene=job['scene'],stage=name,command=cmd,unix=started))+'\n')
    with (out/(name+'.log')).open('x') as log:
        child=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        status(name,scene=job['scene'],child_pid=child.pid,command=cmd)
        while True:
            try: code=child.wait(timeout=300);break
            except subprocess.TimeoutExpired:
                progress=out/name/'progress.json'
                row=dict(scene=job['scene'],stage=name,child_pid=child.pid,unix=time.time(),elapsed_seconds=time.time()-started,
                    progress=read(progress) if progress.exists() else None,
                    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip())
                with (TASK/'periodic_checks.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                write(TASK/'latest_periodic_check.json',row)
                if time.time()-started>timeout:
                    child.terminate()
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    raise TimeoutError(name)
        if code:raise RuntimeError(f'{job["scene"]} {name} returned {code}; see {out/name}.log')
    return time.time()-started
def train(job,stage,mode,ckpt):
    root=Path(job['root']);out=root/'continuation30k_20261109_r1'
    run(job,stage,[PY,'-B',str(root/'source/train_continue30k.py'),'--mode',mode,
        '--checkpoint',str(ckpt),'--output',str(out/stage),'--stop-iteration','30000',
        '--smoke-seconds','600','--smoke-policy','long540'])
def render(job,name,model,iteration,raws):
    root=Path(job['root']);out=root/'continuation30k_20261109_r1'
    run(job,name,[PY,'-B',str(root/'source/render_v1.py'),'--engine',str(root/'engine'),
        '--model',str(model),'--iteration',str(iteration),'--output',str(out/name),'--raws',*map(str,raws)],timeout=3600)
    s=read(out/name/'summary.json');assert s['status']=='PASS_V1_ORIGINAL_NO_EDIT_RENDER'
    assert not s['editing_package_imported'] and s['updates']==0 and len(s['rows'])==len(raws)
def prepare(job):
    root=Path(job['root']);out=root/'continuation30k_20261109_r1';out.mkdir(exist_ok=False)
    wrapper=root/'source/train_continue30k.py'
    assert not wrapper.exists()
    shutil.copy2(TASK/'source/train_continue30k.py',wrapper)
    cfg=read(root/'formal10k/effective_config.json')
    assert cfg['iterations']==cfg['position_lr_max_steps']==30000
    assert len(cfg['followup_scene_spec']['frame_ids'])==job['count']
    assert read(root/'formal10k/returned.json')['iteration']==10000
    assert not read(root/'formal10k/training_frame_coverage.json')['missing']
    paths=[root/'formal10k'/f'{prefix}10000.pth' for prefix in ['chkpnt','dynamic_chkpnt','env_light_chkpnt','training_state']]
    proof=dict(scene=job['scene'],source_iteration=10000,target_iteration=30000,formal_additional_updates=20000,
        learning_rate_horizon=30000,all_frames_train=job['count'],heldout=0,method_changes={},input_changes={},
        source_checkpoint_files={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in paths},
        source_wrapper_sha256=sha(wrapper),automatic_shutdown=False,new_editing_allowed=False)
    write(out/'contract.json',proof)
def smoke(job):
    root=Path(job['root']);out=root/'continuation30k_20261109_r1'
    train(job,'smoke','continuation_smoke',root/'formal10k/chkpnt10000.pth')
    rec=read(out/'smoke/returned.json');step=rec['iteration']
    assert rec['elapsed_seconds']>=600 and rec['source_iteration']==10000
    assert read(out/'smoke/restore_audit.json')['status']=='PASS_EXACT_FULL_RESTORE'
    for prefix in ['chkpnt','dynamic_chkpnt','env_light_chkpnt','training_state']:assert (out/'smoke'/f'{prefix}{step}.pth').is_file()
    render(job,'smoke_v1',out/'smoke',step,job['keys'])
    train(job,'smoke_restore','resume2step',out/'smoke'/f'chkpnt{step}.pth')
    audit=read(out/'smoke_restore/restore_audit.json')
    assert audit['status']=='PASS_EXACT_FULL_RESTORE' and not audit['differences']
    assert read(out/'smoke_restore/resume_entry.json')['completed_extra_updates']==2
    write(out/'smoke_gate.json',dict(status='PASS_REAL_CONTINUATION_SMOKE_SAVE_V1_EXACT_RESTORE_NEXT_ENTRY',
        source_iteration=10000,smoke_stop=step,elapsed_seconds=rec['elapsed_seconds'],engineering_additional_updates=step-10000+2,
        formal_additional_updates=0,automatic_shutdown=False))
def formal(job):
    root=Path(job['root']);out=root/'continuation30k_20261109_r1'
    assert read(out/'smoke_gate.json')['status'].startswith('PASS_REAL_')
    train(job,'formal30k','continue',root/'formal10k/chkpnt10000.pth')
    rec=read(out/'formal30k/returned.json')
    assert rec['iteration']==30000 and rec['source_iteration']==10000 and rec['actual_additional_iterations']==20000
    assert not read(out/'formal30k/training_frame_coverage.json')['missing']
    assert read(out/'formal30k/restore_audit.json')['status']=='PASS_EXACT_FULL_RESTORE'
    for step in [20000,30000]:render(job,f'v1_{step}',out/'formal30k',step,range(job['count']))
    proof=read(out/'contract.json')
    for path,row in proof['source_checkpoint_files'].items():assert sha(path)==row['sha256']
    write(out/'completed.json',dict(status='NATIVE_30K_CONTINUATION_COMPLETE_PENDING_VISUAL_REVIEW',scene=job['scene'],
        iteration=30000,formal_additional_updates=20000,original_10k_checkpoints_unchanged=True,editing_run=False,
        automatic_shutdown=False,human_verdict=None))
def main():
    TASK.mkdir(parents=True,exist_ok=True)
    lock=(TASK/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip()
    assert not active, 'Do not overlap existing GPU work: '+active
    write(TASK/'jobs.json',JOBS)
    for job in JOBS:prepare(job)
    for job in JOBS:smoke(job)
    for job in JOBS:formal(job)
    status('BOTH_NATIVE_30K_COMPLETE_PENDING_CPU_DIAGNOSIS_PAGES_GIT',jobs=JOBS)
if __name__=='__main__':
    try:main()
    except BaseException as e:status('RECOVERABLE_FAILURE',error=repr(e));raise
