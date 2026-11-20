"""Caller-owned authorization after numeric result/lease completion; no training."""
from pathlib import Path
import argparse,json,subprocess,time,traceback
from export_numeric_public_1920 import inspect,execute
from export_public_endpoints import read,write,file_sha

def maybe_read(p):
    try:return read(p)
    except(FileNotFoundError,json.JSONDecodeError):return {}

def main():
    p=argparse.ArgumentParser();p.add_argument('--asset',required=True);p.add_argument('--authority',required=True);p.add_argument('--completion',required=True);p.add_argument('--main-lease',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);auth=out.parent/'numeric_authorization_r1.json';lease=out.parent/'numeric_gpu_lease_r1.json';owner='native_capacity_numeric_public_r1';deadline=time.monotonic()+3600
    print('WAIT_NUMERIC_COMPLETION',a.completion,flush=True)
    while True:
        r=maybe_read(a.completion);ls=maybe_read(a.main_lease)
        if(r.get('complete')is True or r.get('completed')is True)and ls.get('active')is False:
            active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
            if not active:break
            print('WAIT_COMPUTE_PROCESS',active,flush=True)
        if time.monotonic()>deadline:raise TimeoutError('numeric source not ready; no GPU work started')
        time.sleep(10)
    plan=inspect(a.asset,a.authority,a.completion)
    if auth.exists()or lease.exists():raise FileExistsError('preserve existing numeric public lease/authorization')
    write(auth,dict(GPU_authorized=True,optimizer_updates=0,max_native_forwards=40,GPU_keep_on=True,
        root_authorization='Autonomous G_NUMERIC publicfiveview export after numericformal and its GPUlease finish; no repeated approval',
        completion=a.completion,completion_SHA256=file_sha(a.completion),main_lease=a.main_lease,main_lease_inactive=True,
        exporter=str(Path(__file__).with_name('export_numeric_public_1920.py')),exporter_SHA256=file_sha(Path(__file__).with_name('export_numeric_public_1920.py'))))
    write(lease,dict(gpu_authorized=True,active=True,owner=owner,authorization=str(auth),phase='numeric_public1920',formal_allowed=False))
    print('NUMERIC_PUBLIC_STARTED',time.time(),flush=True)
    try:execute(plan,out,lease,owner,auth)
    except Exception:
        write(out.parent/'numeric_launch_failed_r1.json',dict(error=traceback.format_exc(),time=time.time()));raise
    print('NUMERIC_PUBLIC_COMPLETED',str(out/'receipt.json'),flush=True)
if __name__=='__main__':main()
