"""在完全相同的已执行闭环位姿上比较几何，隔离渲染/策略数值差异。"""
import json,sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1')
sys.path.insert(0,'/root/autodl-tmp/external/worldsim_simimpact/HUGSIM')
from sim.utils.score_calculator import bg_collision_det
torch.set_num_threads(4)
canonical=np.array([[.5,0,.5],[.5,0,-.5],[.5,1,.5],[.5,1,-.5],[-.5,0,-.5],[-.5,0,.5],[-.5,1,-.5],[-.5,1,.5]])*np.array([1.6,1.5,3.])
rows=[]
for name in ['scene-0013','scene-0038','scene-0041']:
    trace=json.loads((R/'rollouts'/name/'native_r1/trace.json').read_text())
    infos=[trace[0]['pre_info']]+[t['post_info'] for t in trace]
    verts=[torch.tensor(canonical@Rotation.from_euler('XYZ',i['ego_rot']).as_matrix().T+np.array(i['ego_pos']),dtype=torch.float32,device='cuda') for i in infos]
    folder=R/'geometry_swap'/name
    for p in sorted(folder.glob('*.npz')):
        if p.name=='identity.npz':protocols=[('identity',1),('native_stride2',2),('native_stride4',4)]
        else:protocols=[(p.stem,1)]
        points=torch.from_numpy(np.load(p)['points'].astype(np.float32)).cuda()
        for label,stride in protocols:
            pts=points[::stride];counts=[];flags=[]
            for v in verts:
                mask=torch.ones(len(pts),device='cuda',dtype=torch.bool)
                for j in [1,2,5]:
                    axis=v[j]-v[0];proj=pts@axis
                    mask&=(v[0]@axis<proj)&(proj<v[j]@axis)
                count=int(mask.sum().item());counts.append(count)
                official=bool(bg_collision_det(pts,v));flags.append(official)
                if official!=(count>100):raise RuntimeError('Official rule disagreement')
            event=next((i for i,f in enumerate(flags) if f),None)
            rows.append({'scene':name,'geometry':label,'points':len(pts),'poses':len(verts),'counts':counts,
                'first_collision_time_s':None if event is None else infos[event]['timestamp'],
                'terminal_pose_collision':flags[-1],'scope':'Identical executed native closed-loop poses; collision readout replay, not a new policy rollout'})
            print(name,label,'first',rows[-1]['first_collision_time_s'],'last_count',counts[-1],flush=True)
(R/'paired_collision_geometry.json').write_text(json.dumps(rows,indent=2))
