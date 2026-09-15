"""复用已完成的四模型资产，固定历史扫描隔离当前几何/缺失对检测的影响。"""
import json,time,shutil
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
N=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-LIDAR-01/20260915-r1')
I=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-INTERACTION-01/20260915-r1')
O=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-FF-PERCEPTION-01/20260915-r1')
if O.exists():raise RuntimeError(f'Preserve {O}')
O.mkdir(parents=True);shutil.copy2(__file__,O/'prepare_source_snapshot.py')
reg={'task_id':'WS-SIM-FF-PERCEPTION-01','run_id':'20260915-r1','scenes':['scene-0004','scene-0061'],'seed':20260915,
     'selection':'Two already exposed discovery logs with all ten raw past scans available; not new or independent confirmation.',
     'scope':'Four existing six-view BUILD-scale reconstruction meshes, current first-intersection scan only. This is a sensor/perception isolation test, not a new-pose complete simulator.',
     'information':'Known metric poses, BUILD LiDAR global scale, original current-beam intensity, and the same nine real past sweeps are extra inputs shared by all methods. fill_missing additionally restores GT ranges on absent current returns; no claim of same-information advantage.',
     'conditions':['real']+[f'{m}_{c}' for m in ['vggt','omega512','dvgt1','pi3x'] for c in ['full','fill_missing']],
     'evaluation':'Frozen CenterPoint native nuScenes protocol; do not infer safety from detection changes. Same confidence0.3/0.5, center2m/IoU0.5, front32m/lateral16m, currentGTpoints>=5.',
     'stop_rule':'One finite probe, no new geometry model inference. Only consistent actual downstream loss may justify local asset repair; no threshold/parameter sweeps to manufacture a failure.',
     'failure_ledger_refs':['V74-H2-F16','V74-H2-F17','V74-H2-F18'],'failure_ledger_delta':'pending','human_verdict':None,'prepared':False,'start_unix':time.time()}
(O/'registration.json').write_text(json.dumps(reg,indent=2))
nusc=NuScenes(version='v1.0-trainval',dataroot=str(N/'data'),verbose=False)
def pose(table,token):
    r=nusc.get(table,token);p=np.eye(4);p[:3,:3]=Quaternion(r['rotation']).rotation_matrix;p[:3,3]=r['translation'];return p
def world(sd):return pose('ego_pose',sd['ego_pose_token'])@pose('calibrated_sensor',sd['calibrated_sensor_token'])
frames=[];checks=[]
for scene in reg['scenes']:
    views=json.loads((I/f'inputs/{scene}.json').read_text())['views'];s=nusc.get('sample',views[0]['sample_token']);sd=nusc.get('sample_data',s['data']['LIDAR_TOP']);inv=np.linalg.inv(world(sd))
    raw=np.fromfile(N/'data'/sd['filename'],np.float32).reshape(-1,5)
    ranges=np.linalg.norm(raw[:,:3],axis=1);valid=np.isfinite(raw[:,:3]).all(1)&(ranges>1)&(ranges<80);current=raw[valid,:4].copy()
    z=np.load(I/f'lidar_policy/scans/{scene}/real.npz');gt=z['points'];T=inv@np.array(views[0]['world_from_ego_camera']);gt_lidar=gt@T[:3,:3].T+T[:3,3]
    assert len(gt)==len(current),(scene,len(gt),len(current))
    delta=float(np.max(np.abs(gt_lidar-current[:,:3])));assert delta<.001,(scene,delta)
    checks.append({'scene':scene,'current_points':len(current),'existing_real_to_raw_lidar_max_abs_difference_m':delta})
    past=[];sources=[];sd_past=sd
    for j in range(1,10):
        if not sd_past['prev']:raise RuntimeError('No causal past sweep')
        sd_past=nusc.get('sample_data',sd_past['prev']);p=np.fromfile(N/'data'/sd_past['filename'],np.float32).reshape(-1,5)[:,:4].copy();p=p[~((abs(p[:,0])<1)&(abs(p[:,1])<1))]
        rel=inv@world(sd_past);p[:,:3]=p[:,:3]@rel[:3,:3].T+rel[:3,3];age=(sd['timestamp']-sd_past['timestamp'])/1e6
        past.append(np.c_[p,np.full(len(p),age)]);sources.append({'token':sd_past['token'],'age_s':age,'points':len(p)})
    history=np.concatenate(past).astype(np.float32);dest=O/scene;dest.mkdir();arrays={}
    def save(name,p):
        path=dest/f'{name}.npz';p=np.concatenate([np.c_[p,np.zeros(len(p))],history]).astype(np.float32);np.savez_compressed(path,points=p);arrays[name]=str(path)
    save('real',current)
    for method in ['vggt','omega512','dvgt1','pi3x']:
        f=np.load(I/f'lidar_policy/scans/{scene}/{method}_six_build_scale.npz')['first_range'];finite=np.isfinite(f)
        xyz=z['origin']+z['directions']*f[:,None];xyz=xyz@T[:3,:3].T+T[:3,3]
        save(f'{method}_full',np.c_[xyz[finite],current[finite,3]])
        p=current.copy();p[finite,:3]=xyz[finite];save(f'{method}_fill_missing',p)
        checks[-1][method]={'finite_current_returns':int(finite.sum()),'missing_current_returns':int((~finite).sum())}
    frames.append({'scene':scene,'sample_token':s['token'],'points':arrays,'history':sources})
reg.update(prepared=True,beam_alignment_checks=checks,end_unix=time.time(),expected_detector_forwards=sum(len(f['points']) for f in frames))
(O/'registration.json').write_text(json.dumps(reg,indent=2));(O/'input_manifest.json').write_text(json.dumps({'registration':reg,'frames':frames},indent=2));print(json.dumps(checks,indent=2),flush=True)
