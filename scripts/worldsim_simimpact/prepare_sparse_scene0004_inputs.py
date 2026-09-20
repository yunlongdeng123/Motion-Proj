"""已完成SplatAD的六相机对照，冻结裁剪控制和各条件的独立时序记忆。"""
import copy,json,time,shutil
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-SPARSE-NATIVE-01/20260915-r1');N=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-LIDAR-01/20260915-r1')
O=R/'scene0004_inputs_r1';O.mkdir(exist_ok=False)
pair_root=R/'scene0004_sensor_pairs_r1';pair_reg=json.loads((pair_root/'registration.json').read_text());assert pair_reg['completed'] and pair_reg['actual_camera_renders']==60
pairs={(x['original_index'],x['camera']):x for x in json.loads((pair_root/'pairs.json').read_text())}
meta=json.loads((N/'metadata/scene-0004.json').read_text());seq=sorted(meta['sample'].values(),key=lambda s:s['timestamp']);sd=meta['sample_data'];ego=meta['ego_pose'];cal=meta['calibrated_sensor']
conditions=['real_original','real_common','rendered_common'];reg={'task_id':'WS-SIM-SPARSE-NATIVE-SENSOR-01','created_unix':time.time(),'seed':20260915,'scene':'scene-0004','expected_forward_count':30,'conditions':conditions,'warmup_original_indices':[0,1],'evaluation_original_indices':list(range(2,10)),
 'temporal_protocol':'Reset all official instance banks and motion instance queue between conditions; within each condition process original indices0..9 chronologically. Same known ego poses/calibration/timestamps for every condition.',
 'scope':'Frozen existing-scene replay, not a new log or feedback loop. First compare real-original versus identical-common-support real to isolate the native CAM_BACK bottom80 crop; then compare reconstructed-common against real-common.',
 'preprocessing':'Retain official SparseDrive augmentation based on1600x900→704x256. Native820-height back image uses identical top-left support in both common conditions; official PIL crop pads the unavailable bottom after resize identically. No geometric resize stretch.',
 'status':'Ten-zero CAN placeholder is the unused official test-pipeline field. Source audit: ego_status read only in training loss; inference queue uses predicted ego status. No CAN effect is claimed.',
 'route':'Official future-six-keyframe high-level command from LiDAR-local final x±2m, explicitly extra GT route information; identical across three conditions.',
 'information':'Existing complete scene SplatAD has RGB+LiDAR and actor supervision. Source split is mixed per-camera train/heldout; no equal-information or independent-generalization claim.',
 'stop_rule':'If either real baseline is poor or crop control explains discrepancy, do not promote it as reconstruction geometry harm. RGB degradation alone still requires local asset intervention and appearance controls.',
 'human_verdict':None,'failure_ledger_refs':['V74-H2-F18','V74-H2-F19','V74-H2-F22'],'failure_ledger_delta':'pending','prepared':False}
(O/'registration.json').write_text(json.dumps(reg,indent=2))
def T(x):
    t=np.eye(4);t[:3,:3]=Quaternion(x['rotation']).rotation_matrix;t[:3,3]=x['translation'];return t
def world_sensor(tok):
    d=sd[tok];return T(ego[d['ego_pose_token']])@T(cal[d['calibrated_sensor_token']])
order=['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT'];base=[];refs=[]
for i,s in enumerate(seq[:10]):
    wl=world_sensor(s['data']['LIDAR_TOP']);inv=np.linalg.inv(wl);projections=[];intrinsics=[]
    for camera in order:
        d=sd[s['data'][camera]];c=cal[d['calibrated_sensor_token']];k=np.eye(4);k[:3,:3]=c['camera_intrinsic'];projections.append((k@np.linalg.inv(world_sensor(d['token']))@wl).tolist());intrinsics.append(c['camera_intrinsic'])
        assert np.max(np.abs(np.array(pairs[i,camera]['intrinsics'])[0]-np.array(c['camera_intrinsic'])))<1e-3
    future=np.array([(inv@world_sensor(ss['data']['LIDAR_TOP']))[:3,3] for ss in seq[i:i+7]])
    cmd=[1,0,0] if future[-1,0]>=2 else [0,1,0] if future[-1,0]<=-2 else [0,0,1]
    base.append({'sample_token':s['token'],'original_index':i,'timestamp':s['timestamp']/1e6,'lidar2img':projections,'cam_intrinsic':intrinsics,'lidar2global':wl.tolist(),'ego_status':[0.]*10,'gt_ego_fut_cmd':cmd})
    refs.append({'original_index':i,'sample_token':s['token'],'future_lidar_origin_xyz':future[1:].tolist(),'future_timestamps_s':[(ss['timestamp']-s['timestamp'])/1e6 for ss in seq[i+1:i+7]],'evaluation_only':True})
inputs=[]
for condition in conditions:
    key={'real_original':'source_image','real_common':'real_common_image','rendered_common':'render_image'}[condition]
    for x in base:
        row=copy.deepcopy(x);row['condition']=condition;row['img_filename']=[pairs[x['original_index'],camera][key] for camera in order];assert all(Path(p).is_file() for p in row['img_filename']);inputs.append(row)
(O/'policy_inputs.json').write_text(json.dumps(inputs,indent=2));(O/'evaluation_references.json').write_text(json.dumps(refs,indent=2));shutil.copy2(__file__,O/'source_snapshot.py');reg.update(prepared=True,prepared_unix=time.time(),input_count=len(inputs));(O/'registration.json').write_text(json.dumps(reg,indent=2));print('PAIRED_INPUTS_READY',len(inputs),flush=True)
