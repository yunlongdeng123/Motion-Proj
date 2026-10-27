"""Independent 1s NVML samples; no worker start/stop, no deletion."""
import os,json,time,subprocess,shutil
from pathlib import Path
from segmentation import ROOT,OLD,write
def read(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None
def identity(pid,start=None):
    try:
        parts=Path('/proc/%s/stat'%pid).read_text().split();return dict(pid=pid,start_ticks=parts[21],state=parts[2],alive=parts[2]!='Z' and (start is None or str(start)==parts[21]))
    except FileNotFoundError:return dict(pid=pid,alive=False)
old=read(OLD/'gpu_peak.json') or {};historical=old.get('peak_used_mib',0);peak=0;ticks=Path('/proc/self/stat').read_text().split()[21]
write(ROOT/'monitor_gap.json',dict(previous_global_monitor=old,previous_identity=identity(old.get('pid',0),old.get('start_ticks')),v43_monitor_pid=os.getpid(),v43_start_ticks=ticks,history_preserved=True,history_not_contiguous_across_shutdown=True))
with (ROOT/'gpu_samples.jsonl').open('a') as log:
    while not (ROOT/'STOP_MONITOR').exists():
        r=dict(time=time.time(),checked_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid(),start_ticks=ticks,kind='NVML_1s_sample_not_allocator_peak',historical_previous_peak_mib=historical)
        q=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
        if q.returncode==0:
            used,util=map(int,q.stdout.strip().split(','));peak=max(peak,used);r.update(used_mib=used,utilization=util,v43_run_peak_mib=peak,historical_max_mib=max(historical,peak))
        else:r['error']=q.stderr
        log.write(json.dumps(r)+'\n');log.flush();write(ROOT/'gpu_peak.json',r)
        if int(r['time'])%20==0:
            dispatcher=read(ROOT/'actor_campaign_dispatch.json');nodes={}
            for p in (ROOT/'actor_campaign_v1').glob('*/*/status.json'):
                s=read(p);nodes[str(p.parent.relative_to(ROOT))]=dict(status=s,identity=identity(s['pid'],s['start_ticks']),latest_complete_node=read(p.parent/'latest_complete_node.json'))
            snapshot=dict(checked_utc=r['checked_utc'],dispatcher=dispatcher,nodes=nodes,gpu=r,disk=dict(zip(['total','used','free'],shutil.disk_usage(str(ROOT)))),quota_policy=read(ROOT/'authorization.json'),monitor_identity=identity(os.getpid(),ticks))
            write(ROOT/'monitor_state.json',snapshot)
            with (ROOT/'monitor_history.jsonl').open('a') as h:h.write(json.dumps(snapshot)+'\n')
        time.sleep(1)
