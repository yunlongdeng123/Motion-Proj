"""Compact, read-only monitor for this experiment family; no Torch/GPU import."""
import datetime,json,os
from pathlib import Path
base=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_BC_20261016_v1')
rows=[]
for path in sorted(base.glob('*_launch.json')):
    launch=json.loads(path.read_text());out=Path(launch['output']);pid=launch['pid'];proc=Path('/proc')/str(pid)/'cmdline'
    command=proc.read_bytes().replace(b'\0',b' ').decode(errors='replace') if proc.exists() else ''
    alive=any(name in command for name in ['run_experiments.py','run_plane_refinement.py','run_deferred_refinement.py','run_union_refinement.py']) and str(out) in command
    status=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
    failed=json.loads((out/'initialization_failure.json').read_text()) if (out/'initialization_failure.json').exists() else None
    log=Path(launch['console_log'])
    tail=log.read_bytes()[-1800:].decode(errors='replace') if log.exists() else ''
    rows.append({'run':out.name,'pid':pid,'owned_process_alive':alive,'state':status.get('state'),
      'phase':status.get('phase'),'arm':status.get('arm'),'step':status.get('step'),
      'updated_utc':status.get('updated_utc'),'completed_arms':[x['arm'] for x in status.get('completed_arms',[])],
      'arm_metrics':status.get('arm_metrics'),'gates':status.get('gates'),
      'error':status.get('error') or (failed or {}).get('error'),
      'gradient_smoke_exists':(out/'GPU_gradient_smoke.json').is_file(),
      'comparison_exists':(out/'comparison.json').is_file(),
      'output':str(out),'console_tail':tail})
print(json.dumps({'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runs':rows},ensure_ascii=False,indent=2))
