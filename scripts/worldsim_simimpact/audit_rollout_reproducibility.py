import json,pickle
from pathlib import Path
import numpy as np
R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1')
rows=[]
for folder in sorted((R/'rollouts').iterdir()):
    with (folder/'native_r1/initial_observation.pkl').open('rb') as f:base,info=pickle.load(f)
    for run in sorted(folder.glob('density_*')):
        with (run/'initial_observation.pkl').open('rb') as f:obs,_=pickle.load(f)
        diffs={}
        for k in base['rgb']:
            d=obs['rgb'][k].astype(float)-base['rgb'][k].astype(float)
            diffs[k]={'MAE':float(np.mean(abs(d))),'max_abs':float(abs(d).max()),'changed_fraction':float(np.mean(d!=0))}
        a=json.loads((folder/'native_r1/trace.json').read_text())[0];b=json.loads((run/'trace.json').read_text())[0]
        rows.append({'scene':folder.name,'control':run.name,'initial_rgb_difference':diffs,
            'initial_trajectory_max_difference':float(np.max(np.abs(np.array(a['trajectory'])-np.array(b['trajectory']))))})
(R/'rollout_reproducibility.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
