"""只用窗口前的记录位姿补齐策略状态；在新批策略执行前冻结修正。"""
import json
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-INTERACTION-01/20260915-r1');M=Path('/root/autodl-tmp/data/worldsim_v4/drivestudio_raw_trainval/v1.0-trainval')
src=json.loads((R/'raw_source_availability.json').read_text());samples={s['token']:s for s in json.loads((M/'sample.json').read_text())}
past={};tokens=set()
for n,s in src.items():
 t=s['samples'][0];seq=[t,samples[t]['prev']];seq.append(samples[seq[-1]]['prev']);past[n]=seq;tokens.update(seq)
sd={d['sample_token']:d for d in json.loads((M/'sample_data.json').read_text()) if d['is_key_frame'] and d['sample_token'] in tokens and '/CAM_FRONT/' in d['filename']}
ego={e['token']:e for e in json.loads((M/'ego_pose.json').read_text())}
def T(e):
 m=np.eye(4);m[:3,:3]=Quaternion(e['rotation']).rotation_matrix;m[:3,3]=e['translation'];return m
rows=[]
for name,seq in past.items():
 now=T(ego[sd[seq[0]]['ego_pose_token']]);inv=np.linalg.inv(now)
 xyz=[(inv@T(ego[sd[t]['ego_pose_token']]))[:2,3] for t in seq];ts=np.array([sd[t]['timestamp'] for t in seq])/1e6
 dt1=ts[0]-ts[1];dt2=ts[1]-ts[2];v=(xyz[0]-xyz[1])/dt1;old=(xyz[1]-xyz[2])/dt2;a=(v-old)/((dt1+dt2)/2)
 p=R/'lidar_policy'/f'{name}_log_reference.json';r=json.loads(p.read_text())
 r['initial_velocity_xy_mps_forward_difference_legacy']=r['initial_velocity_xy_mps'];r['initial_velocity_xy_mps']=v.tolist();r['initial_acceleration_xy_mps2']=a.tolist()
 r['status_source']='Backward finite differences from current and two previous CAM_FRONT ego poses; no future state input'
 p.write_text(json.dumps(r,indent=2));rows.append({'scene':name,'velocity':v.tolist(),'acceleration':a.tolist(),'past_samples':seq});print(rows[-1],flush=True)
(R/'causal_status_amendment.json').write_text(json.dumps({'reason':'Official NAVSIM consumes velocity and acceleration. Original HUGSIM client zeros acceleration and previously used forward difference. Correct new-batch status before any policy result, using available past poses.',
 'scope':'Six-log interaction batch; initial three-scene results retained with their original status convention',
 'policy_outputs_existed_before_amendment':bool(list((R/'lidar_policy/policy_outputs').glob('*/*.json'))),'rows':rows},indent=2))
