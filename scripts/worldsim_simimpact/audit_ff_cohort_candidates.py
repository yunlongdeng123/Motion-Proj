"""只对已按下游规则冻结的新目标检查参考支持并导出真实 RGB / BEV。"""
import json,shutil,tarfile
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
ROOT=Path('/root/autodl-tmp/runs/worldsim_simimpact')
O=ROOT/'WS-SIM-FF-COHORT-PERCEPTION-01/20260915-r1';I=ROOT/'WS-SIM-INTERACTION-01/20260915-r1'
E=O/'evidence';D=O/'candidate_support_r1';D.mkdir(exist_ok=False)
shutil.copy2(__file__,D/'source_snapshot.py')
decision=json.loads((E/'decision.json').read_text());objects=json.loads((E/'objects.json').read_text());meta=json.loads((O/'metadata.json').read_text())
def pose(tab,token):
    x=meta[tab][token];p=np.eye(4);p[:3,:3]=Quaternion(x['rotation']).rotation_matrix;p[:3,3]=x['translation'];return p
supports=[]
for j,c in enumerate(decision['selected_new_targets']):
    scene=c['scene'];method=c['method'];r=next(r for r in objects if r['scene']==scene and r['condition']=='real' and r['instance']==c['instance']);g=r['gt']
    views=meta['views'][scene];sd=meta['sample_data'][meta['sequences'][scene][0]];world=pose('ego_pose',sd['ego_pose_token'])@pose('calibrated_sensor',sd['calibrated_sensor_token']);ego=np.asarray(views[0]['world_from_ego_camera']);T=np.linalg.inv(world)@ego
    z=np.load(I/f'lidar_policy/scans/{scene}/real.npz');points=z['points']@T[:3,:3].T+T[:3,3]
    rot=Quaternion(g['rotation']).rotation_matrix;half=np.asarray(g['wlh'])[[1,0,2]]/2;local=(points-g['center_lidar'])@rot;mask=(abs(local)<half).all(1)
    height_fraction=(local[:,2]+half[2])/(2*half[2]);body=mask&(height_fraction>.2)&(height_fraction<.8)
    s={**c,'gt':g,'real_detection':r['scores'],'actual_inside_returns':int(mask.sum()),'middle_height_returns':int(body.sum()),
       'local_observed_span_m':np.ptp(local[mask],axis=0).tolist() if mask.any() else None,'models':[],
       'reference_boundary':'Inside an annotated box is not sufficient proof of opaque body support; middle-height counts and visible RGB retained. No reference interpolation invents missing surfaces.'}
    prefix=f'{j}_{scene}_{method}';s['prefix']=prefix
    arrays={'gt_points_ego':z['points'][mask],'target_ray_indices':np.flatnonzero(mask),'lidar_from_ego':T,'origin':z['origin']}
    for m in ['vggt','omega512','dvgt1','pi3x']:
        for v in ['six','twelve']:
            f=np.load(I/f'lidar_policy/scans/{scene}/{m}_{v}_build_scale.npz')['first_range'];valid=mask&np.isfinite(f);err=f[valid]-z['ranges'][valid]
            info={'method':m,'variant':v,'returned':int(valid.sum()),'missing':int((mask&~np.isfinite(f)).sum())}
            if len(err):info.update(median_signed_error_m=float(np.median(err)),MAE_m=float(abs(err).mean()),error_q10_q50_q90_m=np.quantile(err,[.1,.5,.9]).tolist())
            s['models'].append(info);arrays[f'{m}_{v}_points_ego']=z['origin']+z['directions'][valid]*f[valid,None]
    w,l,h=g['wlh'];corners=np.array([[x*l/2,y*w/2,z*h/2] for x in [-1,1] for y in [-1,1] for z in [-1,1]])@rot.T+g['center_lidar'];inv=np.linalg.inv(T);arrays['gt_corners_ego']=corners@inv[:3,:3].T+inv[:3,3]
    options=[]
    for v in views[:6]:
        cam=np.linalg.inv(np.asarray(v['world_from_camera']))@world;xyz=points[mask]@cam[:3,:3].T+cam[:3,3];uv=xyz@np.asarray(v['intrinsics_original']).T;uv=uv[:,:2]/uv[:,2:]
        visible=(xyz[:,2]>.1)&(uv[:,0]>=0)&(uv[:,0]<v['original_wh'][0])&(uv[:,1]>=0)&(uv[:,1]<v['original_wh'][1]);options.append((int(visible.sum()),v,cam))
    count,v,cam=max(options,key=lambda x:x[0]);s['camera']=v['camera'];s['visible_target_returns']=count
    xyz=corners@cam[:3,:3].T+cam[:3,3];uv=xyz@np.asarray(v['intrinsics_original']).T;uv=uv[:,:2]/uv[:,2:];s['image_box_xyxy']=[*uv.min(0),*uv.max(0)]
    shutil.copy2(v['image'],D/f'{prefix}_rgb.jpg');np.savez_compressed(D/f'{prefix}_points.npz',**arrays)
    supports.append(s)
(D/'support.json').write_text(json.dumps(supports,indent=2))
with tarfile.open(O/'candidate_support_r1.tar.gz','w:gz') as t:
    for p in D.iterdir():t.add(p,arcname=p.name)
for s in supports:print(s['scene'],s['method'],s['class'],'inside/body',s['actual_inside_returns'],s['middle_height_returns'],[r for r in s['models'] if r['method']==s['method']],flush=True)
