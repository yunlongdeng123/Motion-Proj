"""一次固定路线提示对照；不按模型输出选择命令，也不覆盖首个失败基线。"""
import json,shutil,time
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-FRESH-NATIVE-01/20260915-r1');B=R/'baseline';C=R/'baseline_route_r1';P=Path('/root/autodl-tmp/motion_proj/scripts/worldsim_simimpact')
if C.exists():raise RuntimeError('Preserve route control')
assert json.loads((B/'registration.json').read_text())['completed']
C.mkdir();shutil.copy2(__file__,C/'prepare_source_snapshot.py');L=C/'lidar_policy';L.mkdir()
reg={'task_id':'WS-SIM-FRESH-NATIVE-ROUTE-CONTROL-01','run_id':C.name,'seed':20260915,'start_unix':time.time(),
    'motivation':'The first real-input baseline has meanADE1.667m/FDE4.642m and one overlap; the inherited client ignores route and hard-codes straight while this log turns. This is not reconstruction evidence.',
    'protocol':'One known-route oracle: use the next20m of the recorded spatial ego route (arc length, not future motion speed). At20m lookahead, ego-local lateral offset>2.5m means left, <-2.5m right, otherwise straight. One-hot NAVSIM order left/straight/right/unknown. No alternative lookaheads, thresholds, or outcome-selected commands.',
    'extra_information':'This is a GT-derived high-level route hint, explicitly extra information. No future numeric trajectory, velocity, actor boxes, RGB or LiDAR are sent to the policy. All current sensors and causal kinematic state are byte-identical source data to the first baseline.',
    'implementation':'Official TransFuser feature builder already consumes driving_command; override only this previously fixed field. Same weights, preprocessing, steering initialization, controller and evaluation.',
    'admission':'Same meanADE<=1m,meanFDE<=2m,and no executed or recorded overlap across eight starts. Original failed admission remains unchanged; any fit uses this explicitly separate route-informed protocol.',
    'stop_rule':'If this one ordinary input correction does not qualify the real baseline, do not fit this source or tune the command rule; investigate a native-dataset planner instead.',
    'failure_ledger_refs':['V74-H2-F21'],'failure_ledger_delta':'pending','human_verdict':None,'prepared':False}
(C/'registration.json').write_text(json.dumps(reg,indent=2))
(C/'inputs').symlink_to(B/'inputs',target_is_directory=True);(C/'lidar_raycast.log').write_text('')
for name in ['scans','assets']:(L/name).symlink_to(B/'lidar_policy'/name,target_is_directory=True)
for name in ['probe_transfuser_lidar.py','simulate_transfuser_pdm.py']:shutil.copy2(P/name,C/name)
base_reg=json.loads((B/'registration.json').read_text());out=[]
def T(x):
    p=np.eye(4);p[:3,:3]=Quaternion(x['rotation']).rotation_matrix;p[:3,3]=x['translation'];return p
for row in base_reg['rows']:
    meta=json.loads((R/'metadata'/f'{row["scene"]}.json').read_text());samples=meta['sample'];sd=meta['sample_data'];ego=meta['ego_pose'];token=row['sample_token'];route=[]
    while token:
        sample=samples[token];d=sd[sample['data']['CAM_FRONT']];route.append(T(ego[d['ego_pose_token']])[:3,3]);token=sample['next']
    route=np.asarray(route);cum=np.r_[0,np.cumsum(np.linalg.norm(np.diff(route[:,:2],axis=0),axis=1))]
    assert cum[-1]>=20,(row['case_id'],cum[-1])
    ref=json.loads((B/'lidar_policy'/f'{row["case_id"]}_log_reference.json').read_text());inv=np.linalg.inv(ref['world_from_ego_initial']);world_point=np.array([np.interp(20,cum,route[:,k]) for k in range(3)]);local=inv[:3,:3]@world_point+inv[:3,3]
    command_index=0 if local[1]>2.5 else (2 if local[1]<-2.5 else 1);onehot=np.eye(4,dtype=int)[command_index].tolist()
    ref.update(driving_command_onehot=onehot,driving_command_source='GT-derived static spatial route,20m arc-length lookahead and2.5m lateral threshold. Extra route oracle; no outcome-based command selection.')
    (L/f'{row["case_id"]}_log_reference.json').write_text(json.dumps(ref,indent=2));out.append({**row,'route_lookahead_local_xyz':local.tolist(),'driving_command_onehot':onehot,'route_available_m':float(cum[-1])});print('FROZEN_ROUTE_COMMAND',row['case_id'],onehot,local.tolist(),flush=True)
reg.update(rows=out,expected_policy_forwards=len(out),expected_vehicle_forecasts=len(out),prepared=True,prepared_unix=time.time());(C/'registration.json').write_text(json.dumps(reg,indent=2))
