"""Detached fixed R2 bootstrap; preserve failures, never retry protection violations."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_surface_residual_20261121_r2')
def read(p):return json.loads(p.read_text())
def write(p,d):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def main():
    stamp=str(int(time.time()));contract=ROOT/'contract.json'
    write(ROOT/'launch_receipt.json',dict(started=True,PID=os.getpid(),contract=str(contract),contract_SHA256=hashlib.sha256(contract.read_bytes()).hexdigest(),
        formal_requires_real_append_smoke=True,coverage_updates=400,appearance_updates=1200,GPU_keep_on=True,stages=['smoke','coverage400','appearance1200','export','CPUbench','pack']))
    for attempt in range(1,3):
        log=ROOT/f'worker_{stamp}_{attempt}.log'
        write(ROOT/'status.json',dict(state='running',status='running',phase='bootstrap',complete=False,supervisor_PID=os.getpid(),attempt=attempt))
        with log.open('w') as stream:
            p=subprocess.Popen([sys.executable,'-u',str(ROOT/'code/run_residual.py')],cwd='/root/autodl-tmp/motion_proj_v82_gn11_v42',stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            write(ROOT/'worker_pid.json',dict(PID=p.pid,supervisor_PID=os.getpid(),attempt=attempt,log=str(log)));code=p.wait()
        if code==0 and (ROOT/'receipt.json').exists() and read(ROOT/'receipt.json').get('complete'):return
        dest=ROOT/'failures'/f'{stamp}_{attempt}';dest.mkdir(parents=True,exist_ok=True)
        for name in ['failure.json','status.json','resource_ledger.json']:
            q=ROOT/name
            if q.exists():shutil.copy2(q,dest/name)
        protection=(ROOT/'failure.json').exists() and 'PROTECTION_GATE' in (ROOT/'failure.json').read_text()
        if attempt==2 or protection:
            write(ROOT/'status.json',dict(state='failed',status='failed',phase='protection_rejected' if protection else 'worker_failed',complete=False,
                PID=os.getpid(),exit_code=code,log=str(log),GPU_keep_on=True));raise SystemExit(code or 1)
        # A failed smoke artifact is immutable; an engineering retry starts a new discarded smoke.
        smoke=ROOT/'smoke'
        if smoke.exists() and not (smoke/'receipt.json').exists():smoke.rename(dest/'smoke')
if __name__=='__main__':main()
