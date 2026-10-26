"""追加式NVML采样；进程身份、历史峰值与采样缺口分开记录。"""
import argparse,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
old=json.loads((root/'gpu_peak.json').read_text()) if (root/'gpu_peak.json').exists() else {}
peak=old.get('peak_used_mib',0);start=time.time();ticks=Path('/proc/self/stat').read_text().split()[21]
with (root/'gpu_samples.jsonl').open('a') as log:
    while not (root/'stop_gpu_watch').exists():
        row=dict(time=time.time(),pid=os.getpid(),start_ticks=ticks)
        try:
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=8)
            used,util=map(int,raw.strip().split(','));peak=max(peak,used)
            row.update(used_mib=used,utilization=util,peak_used_mib=peak,kind='NVML_1s_sample_not_allocator_peak')
            tmp=root/'gpu_peak.tmp';tmp.write_text(json.dumps(row));tmp.replace(root/'gpu_peak.json')
        except Exception as e:row['error']=str(e)
        log.write(json.dumps(row)+'\n');log.flush();time.sleep(1)
