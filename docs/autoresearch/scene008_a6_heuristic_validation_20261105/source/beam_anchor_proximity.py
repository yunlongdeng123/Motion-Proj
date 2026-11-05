"""Check how sparse first-return lines relate to PVD and observed anchors."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial import cKDTree
import torch

RUN=Path(__file__).resolve().parent.parent
C=RUN.parent
HELD=np.array([89,95,101,105,113,121,125,130,137])
RADII=(.01,.03,.05,.10)
files=sorted((RUN/'first_return/witnesses').glob('*.npz'))
assert len(files)==199
chunks=[]
for file in files:
    with np.load(file,allow_pickle=False) as f:
        if int(file.stem) in HELD:
            continue
        chunks.append({k:f[k].copy() for k in ('origin_local','unit_dir','max_free_distance','raw')})
beam={k:np.concatenate([p[k] for p in chunks]) for k in chunks[0]}
with np.load(C/'scene008_instances_20261105_r1/observations/a6.npz',allow_pickle=False) as f:
    xyz=f['xyz_local_m']
    observed=xyz[f['rgb_valid']&~np.isin(f['source_raw_frame'],HELD)]
    cutoff=float(xyz[:,2].min())-.03
checkpoint=C/'scene008_instances_20261105_r1/gpu_formal_r1/d/checkpoint000000.pt'
pvd=torch.load(checkpoint,map_location='cpu',weights_only=False)['pvd']
point=pvd['anchor'].numpy()
sigma=pvd['sigma'].numpy()
near=cKDTree(xyz).query(point,workers=1)[0]
lower=(point[:,0]<0)&(point[:,2]<=cutoff)&(near>np.maximum(.12,3*sigma))
point=point[lower]
assert len(point)==431


def classify(points):
    origin,unit,end=(beam[k] for k in ('origin_local','unit_dir','max_free_distance'))
    relative=points[:,None,:]-origin[None]
    distance_along=(relative*unit[None]).sum(2)
    closest=np.minimum(np.maximum(distance_along,0.),end[None])
    distance=np.linalg.norm(relative-closest[:,:,None]*unit[None],axis=2)
    unique=np.unique(beam['raw'])
    output={}
    for r in RADII:
        match=distance<=r
        witnesses=np.stack([match[:,beam['raw']==raw].any(1) for raw in unique],axis=1)
        counts=witnesses.sum(1)
        low=np.where(witnesses,unique[None],10000).min(1)
        high=np.where(witnesses,unique[None],-10000).max(1)
        repeated=(counts>=2)&(high-low>=3)
        output[str(r)]=dict(with_one_frame=int((counts>=1).sum()),
                            with_two_separated_frames=int(repeated.sum()),
                            total_points=len(points))
    return output

out=dict(status='PASS',build_beams=len(beam['raw']),build_raw_ids=np.unique(beam['raw']).tolist(),
         pvd_lower=classify(point),observed_surface_control=classify(observed),
         thresholds_m=RADII,
         interpretation='Distance to a finite first-return free segment is a proximity probe; no finite-radius free tube is certified.',
         scope='PVD proposal and inherited anchor geometry include full scene context. Probe uses BUILD beam witnesses only.')
(RUN/'beam_anchor_proximity.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,indent=2),flush=True)
