import json
from pathlib import Path
import numpy as np
R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1')
rows=[]
for p in sorted((R/'rollouts').glob('*/*/summary.json')):
    row=json.loads(p.read_text());row['tag']=p.parent.name
    row['runtime_qa']=json.loads((p.parent/'runtime_qa.json').read_text())
    row['official_eval']=json.loads((p.parent/'official_eval.json').read_text())
    if row['tag']!='native_r1':
        base=json.loads((p.parents[1]/'native_r1/trace.json').read_text());now=json.loads((p.parent/'trace.json').read_text());n=min(len(base),len(now))
        diffs=[np.linalg.norm(np.array(a['post_info']['ego_pos'])-np.array(b['post_info']['ego_pos'])) for a,b in zip(base[:n],now[:n])]
        row['common_prefix_max_position_difference_m']=float(max(diffs));row['first_step_position_difference_m']=float(diffs[0])
        row['caution']='Separate GPU renders can diverge; use identical-pose replay to isolate collision-channel effects.'
    rows.append(row)
(R/'rollout_summary.json').write_text(json.dumps(rows,indent=2))
for r in rows:print(r['scene'],r['tag'],r['steps'],r['collision'],r['route_completion'],r.get('common_prefix_max_position_difference_m'))
