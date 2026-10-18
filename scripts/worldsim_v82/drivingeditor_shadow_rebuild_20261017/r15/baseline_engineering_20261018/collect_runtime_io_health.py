"""Read proc/sysfs and cached status only; write evidence locally, never on AutoDL."""
from pathlib import Path
import datetime,json,subprocess,shutil

W=Path(__file__).parent
O=W.parents[1]/'outputs/baseline_engineering_20261018'
REMOTE=r'''
import datetime,json,os,subprocess
from pathlib import Path
E=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/baseline_engineering_20261018')
now=datetime.datetime.now(datetime.timezone.utc)
def text(path):
 try:return Path(path).read_text()
 except Exception as e:return 'UNAVAILABLE: '+str(e)
def proc(pid):
 base=Path('/proc')/str(pid)
 return dict(pid=pid,status=text(base/'status').splitlines()[:12],wchan=text(base/'wchan'),io=text(base/'io'),cmdline=text(base/'cmdline').replace(chr(0),' '))
p=E/'direct_instance30k_parallel_resume/status.json'
a=json.loads(p.read_text());mtime=datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc)
worker=proc(79624);blocked=('md_write_start' in worker['wchan'] and (now-mtime).total_seconds()>300)
receipt=dict(at=now.isoformat(),hostname=os.uname().nodename,status='BLOCKED_HOST_STORAGE_WRITE_WAIT' if blocked else 'NO_PERSISTENT_MD_WRITE_WAIT_OBSERVED',
 A_reported_status=a,A_status_mtime_utc=mtime.isoformat(),A_status_age_seconds=(now-mtime).total_seconds(),
 A_worker=worker,guardian=proc(79622),B_worker=proc(79625),mdstat=text('/proc/mdstat'),array_state=text('/sys/block/md0/md/array_state').strip(),
 sync_completed=text('/sys/block/md0/md/sync_completed').strip(),sync_speed=text('/sys/block/md0/md/sync_speed').strip(),degraded=text('/sys/block/md0/md/degraded').strip(),io_pressure=text('/proc/pressure/io'),
 interpretation='Main A thread waits in the host md write path; hardware failure and provider maintenance cause not established.',
 training_updates=0,remote_mutations=0,A_process_preserved=True,shutdown_performed=False,checkpoint_replay_started=False)
try:receipt['gpu_csv']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
except Exception as e:receipt['gpu_error']=str(e)
print(json.dumps(receipt))
'''

def main():
 r=subprocess.run(['ssh','-o','ConnectTimeout=10','wm-vgpu-1011','/usr/bin/python3','-'],input=REMOTE,text=True,capture_output=True,timeout=30,check=True)
 data=json.loads(r.stdout)
 name='runtime_io_health.json';out=W/name
 out.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');shutil.copy2(out,O/name)
 history=W/'runtime_io_health_history.jsonl'
 with history.open('a',encoding='utf-8') as f:f.write(json.dumps(data,ensure_ascii=False)+'\n')
 print(json.dumps({k:data[k] for k in ['at','status','A_status_mtime_utc','A_status_age_seconds','array_state','gpu_csv']}))
if __name__=='__main__':main()
