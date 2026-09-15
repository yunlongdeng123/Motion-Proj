"""使用官方closed_loop.py的post-step记录约定；保留原pre-step诊断记录。"""
import sys,json,pickle
from pathlib import Path
import numpy as np
import torch
import open3d as o3d
R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1')
sys.path.insert(0,'/root/autodl-tmp/external/worldsim_simimpact/HUGSIM')
from sim.utils.sim_utils import traj_transform_to_global
from sim.utils.score_calculator import hugsim_evaluate
torch.set_num_threads(4)
for name in ['scene-0013','scene-0038','scene-0041']:
    p=R/'rollouts'/name/'native_r1';tr=json.loads((p/'trace.json').read_text());frames=[]
    for t in tr:
        info=t['post_info'];traj=np.array(t['trajectory'])
        frames.append({'time_stamp':info['timestamp'],'is_key_frame':True,'ego_box':info['ego_box'],
            'obj_boxes':info['obj_boxes'],'obj_names':['car']*len(info['obj_boxes']),
            'planned_traj':{'traj':traj_transform_to_global(traj[:,:2],info['ego_box']),'timestep':.5},
            'collision':info['collision'],'rc':info['rc']})
    payload=[{'type':'closeloop','frames':frames}]
    with (p/'official_post_data.pkl').open('wb') as f:pickle.dump(payload,f)
    ground=np.asarray(o3d.io.read_point_cloud(str(p/'ground.ply')).points);scene=np.asarray(o3d.io.read_point_cloud(str(p/'scene.ply')).points)
    result=hugsim_evaluate(payload,ground,scene)
    (p/'official_post_eval.json').write_text(json.dumps(result,indent=2))
