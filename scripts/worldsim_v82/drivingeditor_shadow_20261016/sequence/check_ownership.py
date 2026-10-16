import os,json
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np
root=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1')
old=root/'shadow_BC_20261016_v1';out=root/'shadow_sequence_20261016_v2/ownership_audit.json'
with np.load(old/'training_data_v2/training_data.npz') as f:p=f['plane_model'];s=float(f['world_scale'])
result={}
for name in ['selection_v1','selection_union_v1']:
    with np.load(old/name/'selection.npz') as f:
        result[name]={}
        for key in ['receiver_xyz_anchor','artifact_xyz_anchor']:
            h=(f[key]@p[:3]+p[3])/s
            result[name][key]={'points':len(h),'height_quantiles_m':np.quantile(h,[0,.01,.1,.5,.9,.99,1]).tolist(),'abs_height_fraction_gt075':float((abs(h)>.75).mean())}
out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
