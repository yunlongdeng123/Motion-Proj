from pathlib import Path
import time,json,subprocess,os
p=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_v42_controlled_20261026')
peak=0;started=time.time()
with (p/'gpu_memory_samples.jsonl').open('a') as f:
    while not (p/'stop_memory_watch').exists():
        try:
            value=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=5).strip().split(',')
            used,util=map(int,value);peak=max(peak,used);r=dict(time=time.time(),used_mib=used,utilization=util,peak_mib=peak,pid=os.getpid(),monitor_started=started,pre_monitor_gap='initial read-only controls and first component smoke had CUDA allocator peaks only')
            f.write(json.dumps(r)+'\n');f.flush();(p/'gpu_memory_peak.json').write_text(json.dumps(r,indent=2))
        except Exception as e:f.write(json.dumps(dict(time=time.time(),error=str(e)))+'\n');f.flush()
        time.sleep(1)
