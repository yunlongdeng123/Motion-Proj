import os
"""只取冻结三个日志的前九关键帧状态/标注，用于日志回放策略评价。"""
import json
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
R=Path(os.environ.get('SIMIMPACT_RUN_ROOT','/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1'))
M=Path('/root/autodl-tmp/data/worldsim_v4/drivestudio_raw_trainval/v1.0-trainval')
src=json.loads((R/'raw_source_availability.json').read_text());names=sorted(src)
scenes={x['name']:x for x in json.loads((M/'scene.json').read_text()) if x['name'] in names}
samples={x['token']:x for x in json.loads((M/'sample.json').read_text())};chosen={}
for name,s in scenes.items():
    seq=[];token=src[name]['samples'][0]
    for i in range(9):
        seq.append(samples[token]);token=samples[token]['next']
    chosen[name]=seq
tokens={x['token'] for s in chosen.values() for x in s}
sd={x['sample_token']:x for x in json.loads((M/'sample_data.json').read_text()) if x['is_key_frame'] and x['sample_token'] in tokens and '/CAM_FRONT/' in x['filename']}
ego={x['token']:x for x in json.loads((M/'ego_pose.json').read_text())}
ann={x['token']:x for x in json.loads((M/'sample_annotation.json').read_text()) if x['sample_token'] in tokens}
instances={x['token']:x for x in json.loads((M/'instance.json').read_text())}
categories={x['token']:x['name'] for x in json.loads((M/'category.json').read_text())}
def T(rec):
    a=np.eye(4);a[:3,:3]=Quaternion(rec['rotation']).rotation_matrix;a[:3,3]=rec['translation'];return a
for name,seq in chosen.items():
    t0=T(ego[sd[seq[0]['token']]['ego_pose_token']]);inv=np.linalg.inv(t0);records=[]
    for i,s in enumerate(seq):
        d=sd[s['token']];ep=inv@T(ego[d['ego_pose_token']]);boxes=[]
        for a in [a for a in ann.values() if a['sample_token']==s['token']]:
            pose=inv@T(a);category=categories[instances[a['instance_token']]['category_token']]
            if category.startswith(('vehicle.','human.','movable_object.barrier','movable_object.trafficcone')):
                yaw=float(np.arctan2(pose[1,0],pose[0,0]));w,l,h=a['size']
                boxes.append({'instance':a['instance_token'],'category':category,'box':[float(pose[0,3]),float(pose[1,3]),float(pose[2,3]),w,l,h,yaw],
                    'lidar_points':a['num_lidar_pts']})
        records.append({'index':i,'time_s':(d['timestamp']-sd[seq[0]['token']]['timestamp'])/1e6,
            'pose':ep.tolist(),'xy_yaw':[float(ep[0,3]),float(ep[1,3]),float(np.arctan2(ep[1,0],ep[0,0]))],'boxes':boxes})
    vel=np.array(records[1]['xy_yaw'][:2])/records[1]['time_s']
    out={'scene':name,'world_from_ego_initial':t0.tolist(),'initial_velocity_xy_mps':vel.tolist(),'records':records,
        'reference':'Dataset ego poses and annotated actors; neither a complete static collision map nor independent validation log'}
    p=R/'lidar_policy'/f'{name}_log_reference.json';p.write_text(json.dumps(out,indent=2));print(name,len(records),vel,flush=True)
