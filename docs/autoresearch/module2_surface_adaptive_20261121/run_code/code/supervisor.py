"""Detached finite owner; preserve failed attempts and restore exact stage states."""
from pathlib import Path
import json,hashlib,os,shutil,subprocess,sys,time
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_surface_adaptive_20261121_r1')
def read(p):return json.loads(p.read_text())
def write(p,d):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def main():
    run=str(int(time.time()));pid=os.getpid();contract=ROOT/'contract.json'
    write(ROOT/'launch_receipt.json',dict(started=True,PID=pid,contract=str(contract),contract_SHA256=hashlib.sha256(contract.read_bytes()).hexdigest(),
        formal_requires_real_append_smoke=True,coverage_updates=600,appearance_updates=1200,GPU_keep_on=True))
    for attempt in range(1,3):
        log=ROOT/f'worker_{run}_{attempt}.log'
        write(ROOT/'status.json',dict(state='running',status='running',phase='bootstrap',supervisor_PID=pid,attempt=attempt,complete=False))
        with log.open('w') as stream:
            p=subprocess.Popen([sys.executable,'-u',str(ROOT/'code/run_adaptive.py')],cwd='/root/autodl-tmp/motion_proj_v82_gn11_v42',
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            write(ROOT/'worker_pid.json',dict(PID=p.pid,supervisor_PID=pid,attempt=attempt,log=str(log)));code=p.wait()
        if code==0 and (ROOT/'receipt.json').exists() and read(ROOT/'receipt.json').get('complete'):return
        dest=ROOT/'failures'/f'{run}_{attempt}';dest.mkdir(parents=True,exist_ok=True)
        for name in ['failure.json','status.json']:
            q=ROOT/name
            if q.exists():shutil.copy2(q,dest/name)
        smoke=ROOT/'smoke'
        if smoke.exists() and not (smoke/'receipt.json').exists():smoke.rename(dest/'smoke')
        if attempt==2:
            write(ROOT/'status.json',dict(state='failed',status='failed',phase='worker_failed',complete=False,PID=pid,exit_code=code,log=str(log),GPU_keep_on=True));raise SystemExit(code or 1)
if __name__=='__main__':main()
