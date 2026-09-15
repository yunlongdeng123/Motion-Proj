"""分开保存BUILD度量锚点与QUERY参考，均不进入官方模型前向。"""
import json
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1')
M=Path('/root/autodl-tmp/data/worldsim_v4/drivestudio_raw_trainval/v1.0-trainval')
src=json.loads((R/'raw_source_availability.json').read_text())
needed={x['filename'] for s in src.values() for x in s['records'] if '/LIDAR_TOP/' in x['filename']}
sd={x['filename']:x for x in json.loads((M/'sample_data.json').read_text()) if x['filename'] in needed}
ego={x['token']:x for x in json.loads((M/'ego_pose.json').read_text())}
cal={x['token']:x for x in json.loads((M/'calibrated_sensor.json').read_text())}
def mat(rec):
    t=np.eye(4);t[:3,:3]=Quaternion(rec['rotation']).rotation_matrix;t[:3,3]=rec['translation'];return t
for name,s in src.items():
    out=R/'reference'/name;out.mkdir(parents=True,exist_ok=True)
    records=[]
    for i,token in enumerate(s['samples']):
        rec=next(x for x in s['records'] if x['sample_token']==token and '/LIDAR_TOP/' in x['filename'])
        d=sd[rec['filename']];trans=mat(ego[d['ego_pose_token']])@mat(cal[d['calibrated_sensor_token']])
        p=np.fromfile(R/'raw_initial'/d['filename'],dtype=np.float32).reshape(-1,5)[:,:3]
        p=p@trans[:3,:3].T+trans[:3,3]
        role='BUILD' if i<2 else 'QUERY'
        np.savez_compressed(out/f'{role}_{i:02d}.npz',points_world=p,lidar_origin_world=trans[:3,3])
        records.append({'index':i,'role':role,'filename':d['filename'],'timestamp_us':d['timestamp'],'world_from_lidar':trans.tolist()})
    (out/'manifest.json').write_text(json.dumps(records,indent=2));print(name,flush=True)
