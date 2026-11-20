"""Wait CPU-only for root's formal completion, then run the authorized export."""
from pathlib import Path
import json,subprocess,sys,time,traceback

C=Path('/root/autodl-tmp/drivingeditor_repro/campaign');B=C/'road_teacher_detail_cpu_20261119_r1/gpu_20261119/native_capacity_20261120_r1'
PUBLIC=B/'public1920';FORMAL=B/'formal_r1';MAIN_LEASE=B/'launch_r1/gpu_lease.json';OWNER='native_capacity_public_r1'

def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

def main():
    sys.path.insert(0,str(PUBLIC/'code'))
    from export_public_endpoints import inspect,execute,file_sha
    deadline=time.monotonic()+3600;announced=False
    while True:
        result=read(FORMAL/'results.json')if(FORMAL/'results.json').exists()else{}
        main_lease=read(MAIN_LEASE)
        if result.get('completed')is True and main_lease.get('active')is False:
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
            if not apps:break
            print('WAIT_COMPUTE_PROCESS',apps,flush=True)
        elif not announced:
            print('WAIT_FORMAL_COMPLETION_AND_LEASE',str(FORMAL/'results.json'),str(MAIN_LEASE),flush=True);announced=True
        if time.monotonic()>deadline:raise TimeoutError('formal readiness wait expired without GPU work')
        time.sleep(10)
    plan=inspect(FORMAL);authorization=PUBLIC/'authorization_r1.json';lease=PUBLIC/'gpu_lease.json'
    if authorization.exists()or lease.exists():raise FileExistsError('preserve existing public authorization/lease')
    write(authorization,dict(schema='native_capacity_public1920_authorization.v1',GPU_authorized=True,
        root_authorization='Root explicitly authorized autonomous export once formal completed and main lease inactive; no repeated approval',
        optimizer_updates=0,max_native_forwards=100,frames=plan['frames'],arms=plan['arms'],GPU_keep_on=True,
        formal_results=str(FORMAL/'results.json'),formal_results_SHA256=file_sha(FORMAL/'results.json'),
        main_lease=str(MAIN_LEASE),main_lease_inactive=True,main_lease_SHA256=file_sha(MAIN_LEASE),
        exporter=str(PUBLIC/'code/export_public_endpoints.py'),exporter_SHA256=file_sha(PUBLIC/'code/export_public_endpoints.py')))
    write(lease,dict(gpu_authorized=True,active=True,owner=OWNER,authorization=str(authorization),phase='actual_public1920_export',formal_allowed=False))
    print('PUBLIC_STARTED',time.time(),flush=True)
    try:execute(plan,PUBLIC/'export_r1',lease,OWNER,authorization)
    except Exception:
        write(PUBLIC/'launch_failed_r1.json',dict(error=traceback.format_exc(),time=time.time()));raise
    print('PUBLIC_COMPLETED',str(PUBLIC/'export_r1/receipt.json'),flush=True)

if __name__=='__main__':main()
