"""One finite queue: prior comparison -> real GPU smoke -> fresh R15 4K -> 49-frame edits."""
from pathlib import Path
import os,sys,json,time,subprocess,datetime
Q=Path(__file__).resolve().parent
TASK=Q.parent.parent.parent/'identity_completion_20261018'

def write(value):
    value.update(controller_pid=os.getpid(),at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    p=Q/'queue_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)

def main():
    if (Q/'queue_status.json').exists():raise FileExistsError('Do not start a duplicate repair controller')
    write(dict(state='WAITING_FOR_OLD5_AND_FROZEN_PROJECTION',shutdown_allowed=False))
    try:
        while True:
            old=json.loads((TASK/'queue_status.json').read_text());tri=json.loads((TASK/'tristate_queue_status.json').read_text())
            if 'ATTENTION' in old['state'] or 'ATTENTION' in tri['state'] or 'BLOCKED' in tri['state']:
                raise RuntimeError('Existing comparison queue needs attention; repair not started')
            if old['state']=='ALL_GPU_WORK_COMPLETE_AWAITING_DELIVERY' and tri['state']=='ALL_VISUALIZATION_GPU_WORK_COMPLETE_AWAITING_DELIVERY':break
            time.sleep(30)
        for phase,args in [('smoke',['run_job.py','smoke']),('train4k',['run_job.py','train']),('evaluate49',['evaluate_repair.py'])]:
            cmd=[sys.executable,str(Q/args[0]),*args[1:]]
            write(dict(state='RUNNING',phase=phase,command=cmd,shutdown_allowed=False))
            with (Q/(phase+'.log')).open('a') as log:
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=Q)
                (Q/'active_child.json').write_text(json.dumps(dict(pid=child.pid,phase=phase,command=cmd)))
                code=child.wait()
            if code:raise RuntimeError(f'{phase} failed with code {code}; preserve all assets')
        write(dict(state='ALL_REPAIR_GPU_WORK_COMPLETE_AWAITING_REVIEW_AND_DELIVERY',shutdown_allowed=False))
    except BaseException as e:
        write(dict(state='NEEDS_ATTENTION_ASSETS_PRESERVED',error=repr(e),shutdown_allowed=False));raise

if __name__=='__main__':main()
