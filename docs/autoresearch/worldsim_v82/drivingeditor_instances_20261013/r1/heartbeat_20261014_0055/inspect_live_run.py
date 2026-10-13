"""Read-only actual instance run snapshot; no Torch/GPU context or mutations."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

BASE=Path('/root/autodl-tmp/drivingeditor_repro/campaign/instance_editor_20261013')
RUN=BASE/'runs/gmc_two_actor_fixed30k_r1'

def lines(path):
    result=[]; partial=[]
    for index,line in enumerate(path.read_text(encoding='utf-8').splitlines()):
        if not line: continue
        try: result.append(json.loads(line))
        except json.JSONDecodeError: partial.append(index+1)
    return result,partial

launch=json.loads((BASE/'queue_gmc30k_r1/launch.json').read_text())
status=json.loads((RUN/'status.json').read_text())
pid=launch['pid']; proc=Path('/proc')/str(pid)
samples,partial_samples=lines(RUN/'memory_samples.jsonl')
phases,partial_phases=lines(RUN/'memory_phases.jsonl')
steps,partial_steps=lines(RUN/'steps.jsonl')
summary=json.loads((RUN/'memory_summary.json').read_text()) if (RUN/'memory_summary.json').exists() else None
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True,timeout=8)
snapshot={'inspected_utc':datetime.now(timezone.utc).isoformat(),'run':str(RUN),'status':status,'launch':launch,
    'pid_alive':proc.exists(),'cmdline':(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode() if proc.exists() else None,
    'checkpoints':[{'name':p.name,'bytes':p.stat().st_size} for p in sorted(RUN.glob('checkpoint_*.pt'))],
    'NV_samples':len(samples),'sampling_error_rows':sum(bool(r['errors']) for r in samples),
    'sampled_PID_peak_bytes':max((r.get('process_used_bytes') or 0 for r in samples),default=0),
    'sampled_device_peak_bytes':max((r.get('device_used_bytes') or 0 for r in samples),default=0),
    'latest_sample':samples[-1] if samples else None,'completed_memory_phases':phases,'memory_summary':summary,
    'logged_checkpoint_step_rows':[r for r in steps if r['step']%10000==0],
    'partial_JSONL_line_numbers':{'samples':partial_samples,'phases':partial_phases,'steps':partial_steps},
    'GPU_compute_query_returncode':gpu.returncode,'GPU_compute_processes_csv':gpu.stdout,
    'sampled_peaks_are_lower_bounds':True,
    'active_training_pytorch_peak': 'available only when owned monitor closes phase/run' if summary is None else 'see memory_summary',
    'inspection_started_GPU_or_training':False}
print(json.dumps(snapshot,ensure_ascii=False))
