"""One bounded read-only postprocessing status snapshot; stdlib only."""
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone
BASE=Path('/root/autodl-tmp/drivingeditor_repro/campaign/instance_editor_20261013')
def read(path):
    return json.loads(path.read_text()) if path.is_file() else None
def process(pid):
    root=Path('/proc')/str(pid)
    try:
        return {'pid':pid,'present':True,'cmdline':(root/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')}
    except FileNotFoundError:
        return {'pid':pid,'present':False}
data={'UTC':datetime.now(timezone.utc).isoformat(),
      'training_status':read(BASE/'runs/gmc_two_actor_fixed30k_r1/status.json'),
      'owned_processes':[process(pid) for pid in (37747,65212,65834,65835)],
      'post30k_r1':read(BASE/'post30k_r1/status.json'),
      'post30k_r2':read(BASE/'post30k_r2/status.json'),
      'priority_yield':read(BASE/'post30k_r2/evaluation/priority_yield.json'),
      'partial_observed_FP32_count':len(list((BASE/'post30k_r2/evaluation/observed').glob('*.npz'))),
      'partial_fixed_FP32_count':len(list((BASE/'post30k_r2/evaluation/fixed_camera').glob('*.npz'))),
      'GPU_compute_snapshot':subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
                                           '--format=csv,noheader,nounits'],capture_output=True,text=True).stdout}
print(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False))
