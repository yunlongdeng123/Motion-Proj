"""Finite, lock-respecting extension for the user's 0/1/U visualization request."""
from identity_common import *
import subprocess,sys,time,os

def main():
    target=TASK/'tristate_queue_status.json'
    if target.exists():raise FileExistsError('Do not duplicate the finite visualization queue')
    write(target,dict(state='WAITING_FOR_OLD5_TRAIN_AND_RENDER',controller_pid=os.getpid()))
    while True:
        q=read(TASK/'queue_status.json')
        if q['state']=='NEEDS_ATTENTION_ASSETS_PRESERVED':
            write(target,dict(state='BLOCKED_BY_EXISTING_QUEUE_FAILURE',controller_pid=os.getpid(),original=q));return
        if q['state']=='ALL_GPU_WORK_COMPLETE_AWAITING_DELIVERY':break
        time.sleep(30)
    cmd=[sys.executable,str(TASK/'code/project_tristate.py')]
    write(target,dict(state='RUNNING',phase='frozen_init_4k_tristate_projection',controller_pid=os.getpid(),command=cmd))
    with (TASK/'project_tristate.log').open('a') as log:
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        write(TASK/'tristate_active_child.json',dict(pid=child.pid,command=cmd))
        code=child.wait()
    write(target,dict(state='ALL_VISUALIZATION_GPU_WORK_COMPLETE_AWAITING_DELIVERY' if code==0 else 'NEEDS_ATTENTION_ASSETS_PRESERVED',
        returncode=code,controller_pid=os.getpid(),shutdown_executed=False))
    if code:raise RuntimeError('Frozen projection failed; assets preserved')

if __name__=='__main__':main()
