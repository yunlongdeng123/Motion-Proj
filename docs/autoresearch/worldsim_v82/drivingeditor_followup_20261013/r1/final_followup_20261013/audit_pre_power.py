"""Read-only process and vendor shutdown audit. No signal or deletion."""
import datetime,fcntl,json,os,re,subprocess
from pathlib import Path
F=Path('/root/autodl-tmp/drivingeditor_repro/campaign/followup_scene_prep_20261013')
def command(pid):
    try:return [x.decode(errors='replace') for x in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if x]
    except (FileNotFoundError,PermissionError):return []
def redacted(args):
    out=[];sensitive=False
    for text in args:
        if sensitive:out.append('[redacted]');sensitive=False;continue
        if text.lower() in ('--token','--password','--passwd','--secret','--api-key'):sensitive=True
        text=re.sub(r'(?i)((?:token|password|passwd|secret|api[_-]?key)[=:])\S+',r'\1[redacted]',text)
        text=re.sub(r'://[^/\s:@]+:[^/\s@]+@','://[redacted]@',text)
        out.append(text)
    return out
owned_ids={124555,96504,108901,120796,120825,120901,122026,146207,147156,58770,56518}
for meta in ['red_terminal_20261013_1056','red_terminal_20261013_1056_attempt2']:
    p=F/'metadata'/meta/'status.json'
    if p.exists():
        d=json.loads(p.read_text());owned_ids.add(d['collector_pid']);owned_ids.update(x['pid'] for x in d.get('stages',[]) if x.get('pid'))
matrix=Path('/root/autodl-tmp/drivingeditor_repro/campaign/hybrid30k_20261012')
for p in (matrix/'runs').glob('*/status.json'):
    d=json.loads(p.read_text());owned_ids.add(d['pid'])
owned=[]
for pid in sorted(owned_ids):
    args=command(pid)
    try:state=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]
    except FileNotFoundError:state='absent'
    owned.append({'pid':pid,'state':state,'command':redacted(args),'running':bool(args) and state not in ('Z','X')})
procs=[];supervisors=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    pid=int(p.name);args=command(pid)
    if not args:continue
    row={'pid':pid,'command':redacted(args)}
    try:
        stat=(p/'stat').read_text().rsplit(')',1)[1].split();row.update(state=stat[0],ppid=int(stat[1]))
    except (FileNotFoundError,PermissionError):continue
    names=[Path(x).name for x in args[:2]]
    if 'supervisord' in names:supervisors.append(row)
    procs.append(row)
nv=subprocess.run(['/usr/bin/nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,timeout=10,close_fds=False)
assert nv.returncode==0,nv.stderr
slot_free=False
with (F.parent/'gpu_slot.lock').open('a') as lock:
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);slot_free=True;fcntl.flock(lock,fcntl.LOCK_UN)
    except BlockingIOError:pass
result={'status':'READONLY_PRE_POWER_AUDIT','observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':'wm-vgpu-1011','audit_pid':os.getpid(),
        'owned_processes':owned,'any_owned_process_running':any(x['running'] for x in owned),
        'all_processes':procs,'supervisord_candidates':supervisors,'GPU_compute_processes':nv.stdout.strip(),'shared_GPU_slot_free':slot_free,
        'shutdown_vendor_script':Path('/usr/bin/shutdown').read_text(),'signals_sent':False,'deletion_performed':False,'human_verdict':None}
print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
