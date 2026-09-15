"""先核查两个实际检测翻转的可观测支持，不根据几何误差再挑案例。"""
import json,time,shutil
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial import cKDTree
I=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-INTERACTION-01/20260915-r1')
F=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-FF-PERCEPTION-01/20260915-r1')
N=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-LIDAR-01/20260915-r1')
O=F/'candidate_support_r1'
if O.exists():raise RuntimeError(f'Preserve {O}')
O.mkdir();shutil.copy2(__file__,O/'source_snapshot.py')
candidates=[{'scene':'scene-0004','method':'omega512','class':'car','instance':'f54238bdd48443fcb4a019271dbca2f8'},
            {'scene':'scene-0004','method':'pi3x','class':'pedestrian','instance':'acb2c4d8335344ad8a5a504513b6f40f'}]
reg={'task_id':'WS-SIM-FF-CANDIDATE-SUPPORT-01','run_id':O.name,'selected':candidates,
     'selection':'Only native CenterPoint center-distance misses remaining at score0.3 in both full and missing-restored six-view scans; real baseline matched. Selection fixed before support/geometric inspection.',
     'scope':'Read-only observed-support audit on already exposed nuScenes train frame. Not independent confirmation or an asset intervention.',
     'failure_ledger_refs':['V74-H2-F19'],'failure_ledger_delta':'pending','human_verdict':None,'completed':False,'start_unix':time.time()}
(O/'registration.json').write_text(json.dumps(reg,indent=2))
rows=json.loads((F/'detection_r1/summary.json').read_text());meta=json.loads((N/'metadata/scene-0004.json').read_text());views=json.loads((I/'inputs/scene-0004.json').read_text())['views']
def pose(t,k):
    r=meta[t][k];p=np.eye(4);p[:3,:3]=Quaternion(r['rotation']).rotation_matrix;p[:3,3]=r['translation'];return p
realrow=next(r for r in rows if r['scene']=='scene-0004' and r['condition']=='real');sd=meta['sample_data'][realrow['lidar_token']]
ego=np.array(views[0]['world_from_ego_camera']);W=pose('ego_pose',sd['ego_pose_token'])@pose('calibrated_sensor',sd['calibrated_sensor_token']);to_lidar=np.linalg.inv(W)@ego
z=np.load(I/'lidar_policy/scans/scene-0004/real.npz');points=z['points']@to_lidar[:3,:3].T+to_lidar[:3,3]
out=[]
for c in candidates:
    g=next(g for g in realrow['eligible_gt'] if g['instance']==c['instance']);center=np.array(g['center_lidar']);rot=Quaternion(g['rotation']).rotation_matrix;half=np.array(g['wlh'])[[1,0,2]]/2
    local=(points-center)@rot;mask=(abs(local)<half).all(1);inside=np.flatnonzero(mask);stats={**c,'gt':g,'measured_inside_returns':len(inside),'observed_local_coordinate_span_m':np.ptp(local[mask],axis=0).tolist() if mask.any() else None,'methods':[]}
    pcfile=O/(c['method']+'_'+c['class']+'.npz');arrays={'gt_target_points_ego':z['points'][mask],'target_ray_indices':inside}
    for method in ['vggt','omega512','dvgt1','pi3x']:
        for variant in ['six','twelve']:
            p=np.load(I/f'lidar_policy/scans/scene-0004/{method}_{variant}_build_scale.npz');first=p['first_range'];valid=mask&np.isfinite(first);err=first[valid]-z['ranges'][valid]
            pp=p['origin']+p['directions'][valid]*first[valid,None];rr={'method':method,'variant':variant,'target_rays':len(inside),'returned':int(valid.sum()),'missing':int((mask&~np.isfinite(first)).sum())}
            if len(err):rr.update(median_signed_error_m=float(np.median(err)),mean_absolute_error_m=float(abs(err).mean()),error_quantiles_m=np.quantile(err,[.1,.5,.9]).tolist(),early_0p2=int((err<-.2).sum()),late_0p2=int((err>.2).sum()))
            stats['methods'].append(rr);arrays[method+'_'+variant]=pp
    for camera in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT']:
        v=next(v for v in views[:6] if v['camera']==camera);T=np.linalg.inv(np.array(v['world_from_camera']))@W
        xyz=T[:3,:3]@points[mask].T+T[:3,3:4];uv=np.array(v['intrinsics_original'])@xyz;uv=uv[:2]/uv[2:];visible=(xyz[2]>.1)&(uv[0]>=0)&(uv[0]<v['original_wh'][0])&(uv[1]>=0)&(uv[1]<v['original_wh'][1]);stats.setdefault('projected_support',{})[camera]={'in_fov':int(visible.sum())}
        if visible.any():stats['projected_support'][camera]['xyxy']=[*uv[:,visible].min(1),*uv[:,visible].max(1)]
    stats['real_match']=next(m for m in realrow['scores']['0.3']['center2m']['matches'] if m['instance']==c['instance'])
    arrays['lidar_from_ego']=to_lidar;np.savez_compressed(pcfile,**arrays);out.append(stats)
(O/'summary.json').write_text(json.dumps(out,indent=2));reg.update(completed=True,end_unix=time.time());(O/'registration.json').write_text(json.dumps(reg,indent=2))
for r in out:print(json.dumps(r,indent=2),flush=True)
