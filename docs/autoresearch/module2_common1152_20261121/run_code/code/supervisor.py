"""Detached finite supervisor: smoke, fixed training, export, CPU bench, bundle."""
from pathlib import Path
import os,subprocess,sys,time,traceback
import json,shutil
from runtime1152 import ROOT,REPO,write,sha,status

def read(p):return json.loads(Path(p).read_text())
def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    start=time.time();pid=os.getpid();contract=ROOT/'contract.json';run_id=str(int(start))
    write(ROOT/'launch_receipt.json',dict(schema='common1152_detached_launch.v1',started=True,PID=pid,
        start_time=start,contract=str(contract),contract_SHA256=sha(contract),
        formal_requires_actual_smoke=True,formal_endpoints=11,GPU_keep_on=True,
        status=str(ROOT/'status.json'),final_bundle=str(ROOT/'final_bundle.tar.gz')))
    for attempt in range(1,3):
        log=ROOT/f'worker_{run_id}_attempt{attempt}.log'
        status('bootstrap',PID=pid,supervisor_PID=pid,attempt=attempt,complete=False)
        with log.open('w') as stream:
            process=subprocess.Popen([sys.executable,'-u',str(ROOT/'code/train_pipeline.py'),'--resume'],
                cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            write(ROOT/'worker_pid.json',dict(PID=process.pid,supervisor_PID=pid,attempt=attempt,log=str(log)))
            code=process.wait()
        if code==0 and (ROOT/'receipt.json').exists() and read(ROOT/'receipt.json').get('complete'):return
        failure=ROOT/'failure.json';failures=ROOT/'failures';failures.mkdir(exist_ok=True)
        if failure.exists():shutil.copy2(failure,failures/f'{run_id}_attempt{attempt}.json')
        if attempt==1:
            # A failed smoke never becomes a warm start; preserve it before one clean replay.
            smoke=ROOT/'smoke'
            if smoke.exists() and not ((smoke/'receipt.json').exists() and read(smoke/'receipt.json').get('verified')):
                target=ROOT/('smoke_failed_'+run_id+'_attempt1')
                if not target.exists():smoke.rename(target)
            continue
        write(ROOT/'status.json',dict(state='failed',status='failed',phase='worker_failed',complete=False,
            PID=pid,worker_exit_code=code,attempts=attempt,log=str(log),
            error=failure.read_text() if failure.exists() else log.read_text()[-8000:],GPU_keep_on=True))
        raise SystemExit(code or 1)

if __name__=='__main__':main()
