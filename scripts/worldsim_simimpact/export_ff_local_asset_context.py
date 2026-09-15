"""导出实际局部网格实验的 RGB 场景、目标点和几何记录。"""
import json,shutil,tarfile
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
I=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-INTERACTION-01/20260915-r1')
F=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-FF-PERCEPTION-01/20260915-r1')
O=F/'local_asset_context_r2'
if O.exists():raise RuntimeError(f'Preserve {O}')
O.mkdir();shutil.copy2(__file__,O/'source_snapshot.py')
support=json.loads((F/'candidate_support_r1/summary.json').read_text());views=json.loads((I/'inputs/scene-0004.json').read_text())['views'];ego=np.array(views[0]['world_from_ego_camera']);z=np.load(I/'lidar_policy/scans/scene-0004/real.npz')
for c in support:
    method=c['method'];points=np.load(F/'candidate_support_r1'/(method+'_'+c['class']+'.npz'));lidar_from_ego=points['lidar_from_ego'];gt_indices=points['target_ray_indices'];g=c['gt'];w,l,h=g['wlh'];rot=Quaternion(g['rotation']).rotation_matrix
    corners=np.array([[x*l/2,y*w/2,z*h/2] for x in [-1,1] for y in [-1,1] for z in [-1,1]])@rot.T+g['center_lidar']
    inv=np.linalg.inv(lidar_from_ego);corners_ego=corners@inv[:3,:3].T+inv[:3,3]
    camera='CAM_FRONT' if c['class']=='car' else 'CAM_FRONT_LEFT';v=next(v for v in views if v['camera']==camera);cam=np.linalg.inv(np.array(v['world_from_camera']))@ego;xyz=corners_ego@cam[:3,:3].T+cam[:3,3];uv=xyz@np.array(v['intrinsics_original']).T;uv=uv[:,:2]/uv[:,2:];c['image_box_xyxy']=[*uv.min(0),*uv.max(0)];c['camera']=camera
    shutil.copy2(v['image'],O/(method+'_context.jpg'));arrays={'gt':z['points'][gt_indices],'target_ray_indices':gt_indices,'lidar_from_ego':lidar_from_ego,'gt_box_corners_ego':corners_ego}
    for variant in ['six','twelve']:
        source=np.load(I/f'lidar_policy/scans/scene-0004/{method}_{variant}_build_scale.npz');old=source['first_range'];finite=np.isfinite(old)
        for condition in ['original','local_radial','same_faces_deleted']:
            f=old if condition=='original' else np.load(F/f'local_asset_inputs_r2/{method}_{variant}/{condition}.npz')['first_range'];valid=finite[gt_indices]&np.isfinite(f[gt_indices]);sel=gt_indices[valid]
            arrays[variant+'_'+condition]=z['origin']+z['directions'][sel]*f[sel,None]
    np.savez_compressed(O/(method+'_points.npz'),**arrays)
(O/'support.json').write_text(json.dumps(support,indent=2));shutil.copy2(F/'local_asset_inputs_r2/geometry_audit.json',O/'geometry_audit.json')
with tarfile.open(F/'local_asset_context_r2.tar.gz','w:gz') as t:
    for p in O.iterdir():t.add(p,arcname=p.name)
print('LOCAL_ASSET_CONTEXT_EXPORTED',flush=True)
