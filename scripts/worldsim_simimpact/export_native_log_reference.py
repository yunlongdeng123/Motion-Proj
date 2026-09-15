"""补齐当前日志的评价标注，覆盖完整执行终点；不送入闭环策略。"""
import argparse,json
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
N=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-LIDAR-01/20260915-r1')
I=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-INTERACTION-01/20260915-r1')
M=N/'data/v1.0-trainval'
ap=argparse.ArgumentParser();ap.add_argument('--scene',default='scene-0004');args=ap.parse_args()
meta=json.loads((N/'metadata'/f'{args.scene}.json').read_text())
inp=json.loads((I/'inputs'/f'{args.scene}.json').read_text());token=inp['views'][0]['sample_token']
seq=[]
while token:seq.append(meta['sample'][token]);token=meta['sample'][token]['next']
tokens={s['token'] for s in seq}
ann=[a for a in json.loads((M/'sample_annotation.json').read_text()) if a['sample_token'] in tokens]
inst_ids={a['instance_token'] for a in ann}
inst={a['token']:a for a in json.loads((M/'instance.json').read_text()) if a['token'] in inst_ids}
cats={a['token']:a['name'] for a in json.loads((M/'category.json').read_text())}
def matrix(a):
    t=np.eye(4);t[:3,:3]=Quaternion(a['rotation']).rotation_matrix;t[:3,3]=a['translation'];return t
first=meta['sample_data'][seq[0]['data']['CAM_FRONT']]
t0=first['timestamp'];anchor=matrix(meta['ego_pose'][first['ego_pose_token']]);inv=np.linalg.inv(anchor)
records=[]
for i,sample in enumerate(seq):
    sd=meta['sample_data'][sample['data']['CAM_FRONT']];pose=inv@matrix(meta['ego_pose'][sd['ego_pose_token']]);boxes=[]
    for a in [a for a in ann if a['sample_token']==sample['token']]:
        cat=cats[inst[a['instance_token']]['category_token']]
        if not cat.startswith(('vehicle.','human.','movable_object.barrier','movable_object.trafficcone')):continue
        p=inv@matrix(a);w,l,h=a['size']
        boxes.append({'instance':a['instance_token'],'category':cat,'box':[float(p[0,3]),float(p[1,3]),float(p[2,3]),w,l,h,float(np.arctan2(p[1,0],p[0,0]))],
            'lidar_points':a['num_lidar_pts']})
    records.append({'index':i,'time_s':(sd['timestamp']-t0)/1e6,'sample_annotation_time_s':(sample['timestamp']-t0)/1e6,
        'pose':pose.tolist(),'xy_yaw':[float(pose[0,3]),float(pose[1,3]),float(np.arctan2(pose[1,0],pose[0,0]))],'boxes':boxes})
initial=json.loads((I/'lidar_policy'/f'{args.scene}_log_reference.json').read_text())
out={**initial,'records':records,'reference':'Full remaining current log, evaluation only. Annotation interpolation retains the previous front-camera keyframe time convention; original sample times are also saved. Not complete static collision truth.'}
dest=N/'closed_loop/reference';dest.mkdir(parents=True,exist_ok=True)
p=dest/f'{args.scene}.json'
if p.exists():raise RuntimeError(f'Preserve existing reference {p}')
p.write_text(json.dumps(out,indent=2));print(p,len(records),'keyframes',flush=True)
