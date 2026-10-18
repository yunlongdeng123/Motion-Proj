"""Read-only CPU extraction of real repaired4K Gaussian evidence and geometry."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='2'
from pathlib import Path
import json,base64
import numpy as np
import torch
torch.set_num_threads(2)
Q=Path(__file__).resolve().parent;D=Q.parent
ck=Q/'train/checkpoint_004000.pt';before=(ck.stat().st_size,ck.stat().st_mtime_ns)
saved=torch.load(ck,map_location='cpu',weights_only=False);assert saved['global_step']==4000
s={k[6:]:v.numpy() for k,v in saved['state'].items() if k.startswith('field.')}
half=np.asarray(json.loads((D/'box_half_sizes.json').read_text())['half_sizes_m'])
poses=np.load(D/'data/trajectory.npz')['poses'][24]
owner=s['owner'];state=s['identity_evidence_status'];xyz=s['xyz'];sigma=np.exp(s['log_scales'])
world=xyz.copy()
for a in range(43):
    t=owner==a;world[t]=xyz[t]@poses[a,:3,:3].T+poses[a,:3,3]
rng=np.random.default_rng(20261018)
def counts(x):
    u,n=np.unique(x,return_counts=True);return {str(int(k)):int(v) for k,v in zip(u,n)}
def sample(ids,n,keep_unknown=False):
    rare=ids[state[ids]==0] if keep_unknown else np.empty(0,dtype=np.int64)
    if len(rare)>n//2:rare=rng.choice(rare,n//2,replace=False)
    rest=np.setdiff1d(ids,rare,assume_unique=True)
    if len(rest)>n-len(rare):rest=rng.choice(rest,n-len(rare),replace=False)
    return np.sort(np.r_[rare,rest])
def panel(scope,rows,coordinates,positions):
    ids=sample(rows,8000,keep_unknown=True)
    arr=np.column_stack((positions[ids],state[ids],owner[ids],sigma[ids].max(-1),s['source_frame'][ids],
        s['source_raw_row'][ids],s['source_candidate_owner'][ids],s['identity_probability'][ids])).astype('<f4')
    return dict(stage='repair4k',scope=scope,coordinates=coordinates,total=len(rows),counts=counts(state[rows]),
        sample_policy='deterministic up to8000; preserve rare unknown up to4000; counts are full population',
        points=dict(rows=len(ids),stride=12,data=base64.b64encode(arr.tobytes()).decode()))
panels=[panel('all',np.arange(len(owner)),'world@0008113',world)]
stats=[]
for a in [6,37,5,26,18]:
    rows=np.flatnonzero(owner==a);p=panel('a'+str(a),rows,'Actor-local meters',xyz);p['half_box_m']=half[a].tolist();panels.append(p)
    local=(world-poses[a,:3,3])@poses[a,:3,:3]
    # Background context is stored separately, then combined with target by the page.
    bg=np.flatnonzero((np.abs(local)<np.array([8,8,4])).all(-1)&(owner!=a))
    p=panel('a'+str(a)+'context',bg,'background in target Actor frame@raw113',local);p['half_box_m']=half[a].tolist();panels.append(p)
    drift=np.linalg.norm(xyz[rows]-s['geometry_origin'][rows],axis=-1)
    stats.append(dict(actor=a,points=len(rows),counts=counts(state[rows]),
        exact_outside=int((np.abs(xyz[rows])>half[a]).any(-1).sum()),
        max_origin_displacement_m=float(drift.max()),unknown_sampled=int((state[sample(rows,8000,True)]==0).sum())))
assert before==(ck.stat().st_size,ck.stat().st_mtime_ns)
out=Q/'tristate_repair';out.mkdir(exist_ok=True)
r=dict(status='READ_ONLY_CPU_EXTRACTION_COMPLETE',checkpoint_step=4000,checkpoint=str(ck),
    checkpoint_file_unchanged=True,parameter_updates=0,total_points=len(owner),status_counts=counts(state),actors=stats,
    reference_frame='0008113',coordinates='meters; Actor-local for assets; background posed at raw113',
    center_only=True,rendered_Gaussian_support_projection=False,human_verdict=None)
(out/'receipt.json').write_text(json.dumps(r,indent=2));(out/'viewer_panels.json').write_text(json.dumps(dict(panels=panels),separators=(',',':')))
print(json.dumps(r),flush=True)
