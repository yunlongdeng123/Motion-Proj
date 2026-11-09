"""30-minute scheduled checks; adopt the current training PID without restarting it."""
from pathlib import Path
import fcntl
import importlib.util
import json
import os
import select
import signal
import subprocess
import time

TASK=Path(__file__).resolve().parent.parent
INTERVAL=1800
spec=importlib.util.spec_from_file_location('original_queue',TASK/'source/continue_queue.py')
Q=importlib.util.module_from_spec(spec);spec.loader.exec_module(Q)

def write(p,v):Q.write(p,v)
def status(stage,**kw):
    write(TASK/'status.json',dict(stage=stage,unix=time.time(),controller_pid=os.getpid(),
        automatic_shutdown=False,new_editing_allowed=False,check_interval_seconds=INTERVAL,**kw))
def periodic(job,name,pid,started):
    out=Path(job['root'])/'continuation30k_20261109_r1'
    p=out/name/'progress.json'
    row=dict(scene=job['scene'],stage=name,child_pid=pid,unix=time.time(),elapsed_seconds=time.time()-started,
        progress=Q.read(p) if p.exists() else None,check_interval_seconds=INTERVAL,
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip())
    with (TASK/'periodic_checks_30min.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    write(TASK/'latest_periodic_check.json',row)
def run(job,name,cmd,timeout=10800):
    root=Path(job['root']);out=root/'continuation30k_20261109_r1';started=time.time()
    with (TASK/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(scene=job['scene'],stage=name,command=cmd,unix=started,check_interval_seconds=INTERVAL))+'\n')
    with (out/(name+'.log')).open('x') as log:
        child=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        status(name,scene=job['scene'],child_pid=child.pid,command=cmd,next_scheduled_check_unix=started+INTERVAL)
        while True:
            try:code=child.wait(timeout=min(INTERVAL,max(1,timeout-(time.time()-started))));break
            except subprocess.TimeoutExpired:
                periodic(job,name,child.pid,started)
                if time.time()-started>=timeout:
                    child.terminate()
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    raise TimeoutError(name)
        if code:raise RuntimeError(f'{job["scene"]} {name} returned {code}; see {out/name}.log')
    return time.time()-started
def finish_formal(job):
    root=Path(job['root']);out=root/'continuation30k_20261109_r1'
    rec=Q.read(out/'formal30k/returned.json')
    assert rec['iteration']==30000 and rec['source_iteration']==10000 and rec['actual_additional_iterations']==20000
    assert not Q.read(out/'formal30k/training_frame_coverage.json')['missing']
    assert Q.read(out/'formal30k/restore_audit.json')['status']=='PASS_EXACT_FULL_RESTORE'
    for step in [20000,30000]:Q.render(job,f'v1_{step}',out/'formal30k',step,range(job['count']))
    for path,row in Q.read(out/'contract.json')['source_checkpoint_files'].items():assert Q.sha(path)==row['sha256']
    write(out/'completed.json',dict(status='NATIVE_30K_CONTINUATION_COMPLETE_PENDING_VISUAL_REVIEW',scene=job['scene'],
        iteration=30000,formal_additional_updates=20000,original_10k_checkpoints_unchanged=True,editing_run=False,
        automatic_shutdown=False,human_verdict=None))
def main():
    previous=Q.read(TASK/'status.json')
    assert previous['stage']=='formal30k' and previous['scene']=='065'
    old=previous['controller_pid'];child=previous['child_pid'];started=previous['unix']
    original_cmd=Path('/proc')/str(old)/'cmdline'
    assert str(TASK/'source/continue_queue.py').encode() in original_cmd.read_bytes()
    actual_cmd=(Path('/proc')/str(child)/'cmdline').read_bytes().split(b'\0')[:-1]
    assert actual_cmd==[x.encode() for x in previous['command']]
    child_start=(Path('/proc')/str(child)/'stat').read_text().split()[21]
    fd=os.pidfd_open(child);poll=select.poll();poll.register(fd,select.POLLIN)
    write(TASK/'schedule_handover.json',dict(previous_status=previous,new_controller_pid=os.getpid(),
        same_training_pid=child,training_process_start_ticks=child_start,training_restart=False,
        check_interval_seconds=INTERVAL,unix=time.time()))
    # Terminate only the old controller. The GPU training process remains intact.
    os.kill(old,signal.SIGTERM)
    lock=(TASK/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    assert (Path('/proc')/str(child)/'stat').read_text().split()[21]==child_start
    Q.status=status;Q.run=run
    now=time.time()
    write(TASK/'scheduled_checks.json',dict(schedule='every 30 minutes',interval_seconds=INTERVAL,
        controller_pid=os.getpid(),first_check_unix=now+INTERVAL,training_pid_unchanged=child,
        poll_gpu_before_first_interval=False,automatic_shutdown=False,new_editing_allowed=False,
        automatic_stage_advance_on_process_exit=True))
    status('formal30k',scene='065',child_pid=child,command=previous['command'],
        next_scheduled_check_unix=now+INTERVAL,training_restart=False)
    while True:
        remaining=10800-(time.time()-started)
        if remaining<=0:raise TimeoutError('Adopted formal30k deadline')
        if poll.poll(int(min(INTERVAL,remaining)*1000)):break
        periodic(Q.JOBS[0],'formal30k',child,started)
    os.close(fd)
    finish_formal(Q.JOBS[0])
    Q.formal(Q.JOBS[1])
    status('BOTH_NATIVE_30K_COMPLETE_PENDING_CPU_DIAGNOSIS_PAGES_GIT',jobs=Q.JOBS)
    write(TASK/'scheduled_checks_complete.json',dict(unix=time.time(),both_scenes_30k=True,
        native_rendering_complete=True,editing_run=False,automatic_shutdown=False))
if __name__=='__main__':
    try:main()
    except BaseException as e:status('RECOVERABLE_FAILURE',error=repr(e));raise
