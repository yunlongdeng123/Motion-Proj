"""冻结一个已选新日志的真实六相机原生规划基线；未来轨迹与框仅存评价文件。"""
import json, time, zipfile, shutil
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.splits import create_splits_scenes

R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-SPARSE-NATIVE-01/20260915-r1')
Q=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-FRESH-NATIVE-01/20260915-r1')
O=R/'real_inputs_r2';O.mkdir(exist_ok=False)
meta=json.loads((Q/'metadata/scene-0002.json').read_text());samples=meta['sample'];sd=meta['sample_data'];ego=meta['ego_pose'];cal=meta['calibrated_sensor']
seq=sorted(samples.values(),key=lambda s:s['timestamp']);assert len(seq)>=17
reg={'task_id':'WS-SIM-SPARSE-NATIVE-01','stage':'real_inputs_r2','created_unix':time.time(),'seed':20260915,
 'scene':'scene-0002','expected_forward_count':10,'warmup_original_indices':[1,2],'evaluation_original_indices':list(range(3,11)),
 'preparation_correction':'r1 stopped before any model forward: original index0 precedes available CAN. Drop only that warmup frame; preserve all eight registered evaluation starts. r1 remains archived.',
 'frozen_before_forward':True,'sensor_input':'Original six RGB cameras only; official camera order/preprocessing/calibrations and causal model instance memory. No LiDAR range input.',
 'status_input':'Latest CAN pose and steer messages at or before sample timestamp; no zero fallback. Official converter uses nearest message; causal timestamp selection is an explicit adapter difference.',
 'route_input':'Official SparseDrive command from next six keyframe LiDAR-origin displacement: lidar x>=2 right, x<=-2 left, else straight. Extra GT-derived route bit, no full future motion in policy.',
 'evaluation':'Native 3s six-waypoint LiDAR-frame L2 at 1/2/3s and across all six; separate vehicle tracking requires explicit coordinate/origin adaptation. Do not compare directly to 4s TransFuser metrics.',
 'decision':'First validate native real-input baseline on all eight frozen starts; do not tune commands, start scene fitting, or infer reconstruction harm in this preparation stage.',
 'training_overlap':'Checked against official nuScenes splits, reported below; this is fresh to our simulation scan, not necessarily held out from model training.',
 'human_verdict':None,'failure_ledger_refs':['V74-H2-F21'],'failure_ledger_delta':'pending'}
reg['official_nuscenes_split']=[k for k,v in create_splits_scenes().items() if 'scene-0002' in v]
(O/'registration.json').write_text(json.dumps(reg,indent=2))
with zipfile.ZipFile('/root/autodl-pub/nuScenes/CANbusexpansion/can_bus.zip') as z:
    wanted=[n for n in z.namelist() if n.endswith(('scene-0002_pose.json','scene-0002_steeranglefeedback.json'))]
    assert len(wanted)==2,wanted
    can={Path(n).stem.split('scene-0002_')[1]:json.loads(z.read(n)) for n in wanted}
    (O/'can_source').mkdir()
    for n in wanted:(O/'can_source'/Path(n).name).write_bytes(z.read(n))
def T(rec):
    t=np.eye(4);t[:3,:3]=Quaternion(rec['rotation']).rotation_matrix;t[:3,3]=rec['translation'];return t
def world_sensor(tok):
    d=sd[tok];return T(ego[d['ego_pose_token']])@T(cal[d['calibrated_sensor_token']])
order=['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']
inputs=[];references=[]
for i in range(1,11):
    s=seq[i]
    wl=world_sensor(s['data']['LIDAR_TOP']);il=np.linalg.inv(wl);images=[];projections=[];intrinsics=[]
    for camera in order:
        d=sd[s['data'][camera]];c=cal[d['calibrated_sensor_token']];k=np.eye(4);k[:3,:3]=c['camera_intrinsic']
        projections.append((k@np.linalg.inv(world_sensor(d['token']))@wl).tolist());intrinsics.append(c['camera_intrinsic']);image=Q/'data'/d['filename'];assert image.exists();images.append(str(image))
    status=[];ages={}
    for kind in ['pose','steeranglefeedback']:
        msgs=can[kind];idx=int(np.searchsorted([m['utime'] for m in msgs],s['timestamp'],side='right')-1);assert idx>=0
        m=msgs[idx];ages[kind]=(s['timestamp']-m['utime'])/1e6;assert 0<=ages[kind]<.1
        status.extend(m['accel']+m['rotation_rate']+m['vel'] if kind=='pose' else [m['value']])
    future=np.array([(il@world_sensor(ss['data']['LIDAR_TOP']))[:3,3] for ss in seq[i:i+7]])
    assert len(future)==7
    cmd=[1,0,0] if future[-1,0]>=2 else [0,1,0] if future[-1,0]<=-2 else [0,0,1]
    inputs.append({'sample_token':s['token'],'original_index':i,'timestamp':s['timestamp']/1e6,'img_filename':images,'lidar2img':projections,'cam_intrinsic':intrinsics,'lidar2global':wl.tolist(),'ego_status':status,'gt_ego_fut_cmd':cmd,'CAN_age_seconds':ages})
    references.append({'sample_token':s['token'],'original_index':i,'future_lidar_origin_xyz':future[1:].tolist(),'future_timestamps_s':[(ss['timestamp']-s['timestamp'])/1e6 for ss in seq[i+1:i+7]],'evaluation_only':True})
(O/'policy_inputs.json').write_text(json.dumps(inputs,indent=2));(O/'evaluation_references.json').write_text(json.dumps(references,indent=2));shutil.copy2(__file__,O/'source_snapshot.py')
reg.update(prepared=True,prepared_unix=time.time(),camera_files=len(inputs)*6,max_CAN_age_seconds=max(max(r['CAN_age_seconds'].values()) for r in inputs));(O/'registration.json').write_text(json.dumps(reg,indent=2))
print(json.dumps(reg,indent=2),flush=True)
