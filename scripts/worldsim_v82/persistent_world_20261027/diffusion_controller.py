"""权重和GPU依赖满足后继续冻结扩散基线，不和原生大模型加载竞争显存。"""
import json,os,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
status=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],phase='waiting_paired_evaluation')
def save():
    p=ROOT/'diffusion_queue_status.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(status,indent=2));q.replace(p)
save()
try:
    while True:
        paired=json.loads((ROOT/'paired_eval_status.json').read_text()) if (ROOT/'paired_eval_status.json').exists() else {}
        weights=json.loads((ROOT/'download_status.json').read_text()) if (ROOT/'download_status.json').exists() else {}
        if paired.get('evaluation_complete') and weights.get('status')=='complete':break
        if paired.get('phase')=='engineering_repair_needed':raise RuntimeError('先修配对评价接线，保留本队列')
        time.sleep(15)
    command=['/root/autodl-tmp/envs/motionproj/bin/python',str(ROOT/'code/diffusion_baseline.py')]
    with (ROOT/'diffusion_baseline.log').open('a') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=str(ROOT/'code'))
        status.update(phase='diffusion_inference',child=dict(pid=process.pid,start_ticks=Path('/proc/%s/stat'%process.pid).read_text().split()[21],command=command));save()
        assert process.wait()==0
    status.update(phase='diffusion_outputs_report_pending',complete=True);status.pop('child');save()
except Exception:
    status.update(phase='engineering_repair_needed',error=traceback.format_exc());save();raise
