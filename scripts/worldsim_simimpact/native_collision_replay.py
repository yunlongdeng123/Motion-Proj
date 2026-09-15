"""Run the official static-collision kernel along released logged camera paths.

This is a geometry-consumer probe, not a policy rollout and not ground-truth collision.
"""
import json,pickle,sys
from pathlib import Path
import numpy as np
import torch
R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1')
sys.path.insert(0,'/root/autodl-tmp/external/worldsim_simimpact/HUGSIM')
from sim.utils.score_calculator import bg_collision_det
torch.set_num_threads(4)
rows=[]
for scene in ['scene-0013','scene-0038','scene-0041']:
    d=list((R/'assets/extracted').glob(f'**/{scene}/cfg.yaml'))[0].parent
    params,_=torch.load(d/'scene.pth',map_location='cpu',weights_only=False)
    xyz=params[1].detach().cuda(); semantics=params[4].detach().argmax(-1).cuda()
    opacity=params[7].detach().sigmoid().flatten().cuda()
    mask=(semantics>1)&(semantics!=10)&(opacity>0.8)
    points=xyz[mask]
    with (d/'ground_param.pkl').open('rb') as f:poses,height,_=pickle.load(f)
    verts=torch.tensor([[.5,0,.5],[.5,0,-.5],[.5,1,.5],[.5,1,-.5],[-.5,0,-.5],[-.5,0,.5],[-.5,1,-.5],[-.5,1,.5]],device='cuda')
    verts*=torch.tensor([1.6,1.5,3.0],device='cuda')
    frames=[]
    for i,p in enumerate(poses):
        t=torch.as_tensor(p,device='cuda',dtype=torch.float32)
        box=verts@t[:3,:3].T+t[:3,3]
        hit=bg_collision_det(points,box)
        frames.append({'frame':i,'collision':bool(hit),'position':p[:3,3].tolist()})
    row={'scene':scene,'source':'released ground_param camera poses, no closed-loop policy',
        'points':int(len(xyz)),'collision_points':int(len(points)),'poses':len(poses),
        'collision_frames':sum(x['collision'] for x in frames),'frames':frames}
    rows.append(row); print({k:v for k,v in row.items() if k!='frames'},flush=True)
(R/'native_collision_replay.json').write_text(json.dumps(rows,indent=2))
