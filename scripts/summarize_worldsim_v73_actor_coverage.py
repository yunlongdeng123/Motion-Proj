"""仅按build输入和只读轨迹整理Actor覆盖，避免一直在近距离密集Actor上试验。"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

parser=argparse.ArgumentParser()
parser.add_argument('--native-run',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
torch.set_num_threads(4)
scenes=torch.load(args.native_run/'build_observations.pt',map_location='cpu',weights_only=False)
metadata=Path('/root/autodl-tmp/data/worldsim_v4/drivestudio_raw_trainval/v1.0-trainval')
categories={r['token']:r['name'] for r in json.loads((metadata/'category.json').read_text())}
instances={r['token']:categories[r['category_token']] for r in json.loads((metadata/'instance.json').read_text())}
rows=[]
for scene in scenes:
    actors=defaultdict(lambda:{'points':[],'views':[],'trajectory':{},'nearest_camera_m':float('inf')})
    for view in scene['views']:
        owners=np.asarray(view['owners'])
        for owner in set(o for o,a in zip(owners,view['actor_mask']) if a):
            take=torch.tensor(owners==owner)&~view['diagnostic_mask']
            actor=actors[owner]
            actor['points'].append(view['points_actor_m'][take])
            actor['views'].append(view['camera_id'])
            pose=view['world_from_actor'][owner]
            actor['trajectory'][view['camera_time_us']]=pose[:3,3].tolist()
            distance=np.linalg.norm(pose[:3,3]-view['world_from_camera'][:3,3])
            actor['nearest_camera_m']=min(actor['nearest_camera_m'],float(distance))
    for owner,actor in actors.items():
        stamps=sorted(actor['trajectory'])
        dt=(stamps[-1]-stamps[0])/1e6
        distance=np.linalg.norm(np.asarray(actor['trajectory'][stamps[-1]])-actor['trajectory'][stamps[0]])
        rows.append({'scene':scene['scene_id'],'log_id':scene['log_id'],'role':scene['role'],'owner':owner,
            'category':instances.get(owner),'unique_projected_build_points':len(torch.unique(torch.cat(actor['points']),dim=0)),
            'observed_views':len(actor['views']),'camera_channels':sorted(set(actor['views'])),
            'nearest_camera_m':actor['nearest_camera_m'],'window_duration_s':dt,
            'translation_speed_mps':float(distance/dt) if dt>0 else None})
summary={'native_run':str(args.native_run),'source':'build projected non-diagnostic points and known trajectory only',
 'selection_boundary':'no predictions, heldout metrics, source or external confirmation used',
 'point_ownership':'annotation box plus 0.1m, overlaps excluded; proxy membership, not per-point instance ground truth',
 'actors':rows,'scenes':[{k:v for k,v in s.items() if k!='views'} for s in scenes]}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'actors':len(rows),'scene_count':len(scenes),'motion_over_2mps':sum((r['translation_speed_mps'] or 0)>2 for r in rows),
 'build_below_100':sum(r['unique_projected_build_points']<100 for r in rows),
 'mechanism_actor':[r for r in rows if r['owner']=='b7b2cf1e7e214595bdf0d9b3dc59e471']},indent=2))
