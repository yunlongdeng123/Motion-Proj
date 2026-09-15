"""用原生闭环实际位姿回查碰撞点；稀疏LiDAR近邻仅作参考，不当作完整表面真值。"""
import json,pickle,sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1')
M=Path('/root/autodl-tmp/data/worldsim_v4/drivestudio_raw_trainval/v1.0-trainval')
torch.set_num_threads(4)
src=json.loads((R/'raw_source_availability.json').read_text())
needed={x['filename'] for s in src.values() for x in s['records'] if '/LIDAR_TOP/' in x['filename']}
sd={x['filename']:x for x in json.loads((M/'sample_data.json').read_text()) if x['filename'] in needed}
ego={x['token']:x for x in json.loads((M/'ego_pose.json').read_text())}
cal={x['token']:x for x in json.loads((M/'calibrated_sensor.json').read_text())}
def mat(rec):
    t=np.eye(4);t[:3,:3]=Quaternion(rec['rotation']).rotation_matrix;t[:3,3]=rec['translation'];return t
canonical=np.array([[.5,0,.5],[.5,0,-.5],[.5,1,.5],[.5,1,-.5],[-.5,0,-.5],[-.5,0,.5],[-.5,1,-.5],[-.5,1,.5]])*np.array([1.6,1.5,3.])
def in_box(points,verts):
    origin=verts[0];ans=np.ones(len(points),dtype=bool)
    for j in [1,2,5]:
        axis=verts[j]-origin;pr=points@axis
        ans&=(origin@axis<pr)&(pr<verts[j]@axis)
    return ans
rows=[]
for name,s in src.items():
    run=R/'rollouts'/name/'native_r1';out=R/'collision_audit'/name;out.mkdir(parents=True,exist_ok=True)
    model_dir=next((R/'assets/extracted').glob(f'**/{name}/scene.pth')).parent
    meta=json.loads((model_dir/'meta_data.json').read_text());inv=np.array(meta['inv_pose'])
    pars,_=torch.load(model_dir/'scene.pth',map_location='cpu',weights_only=False)
    pars=[x.detach() if torch.is_tensor(x) else x for x in pars]
    xyz=pars[1].numpy();labels=pars[4].argmax(-1).numpy();opacity=pars[7].sigmoid().numpy().ravel()
    eligible=(labels>1)&(labels!=10)&(opacity>.8)
    points=xyz[eligible];point_labels=labels[eligible]
    lidar=[];timestamps=[]
    for rec in s['records']:
        if rec['filename'] not in needed:continue
        d=sd[rec['filename']];trans=inv@mat(ego[d['ego_pose_token']])@mat(cal[d['calibrated_sensor_token']])
        p=np.fromfile(R/'raw_initial'/d['filename'],dtype=np.float32).reshape(-1,5)[:,:3]
        p=p@trans[:3,:3].T+trans[:3,3];lidar.append(p);timestamps.append(d['timestamp'])
    lidar=np.concatenate(lidar);tree=cKDTree(lidar)
    trace=json.loads((run/'trace.json').read_text());counts=[]
    for item in trace:
        inf=item['post_info'];rotation=Rotation.from_euler('XYZ',inf['ego_rot']).as_matrix()
        verts=canonical@rotation.T+np.array(inf['ego_pos'])
        mask=in_box(points,verts);counts.append(int(mask.sum()))
    collide=points[mask];clabels=point_labels[mask]
    distances=tree.query(collide)[0] if len(collide) else np.array([])
    local=(collide-np.array(inf['ego_pos']))@rotation
    # This bbox spans y=0..1.5 below the camera in the native camera convention.
    cam_height=pickle.load((model_dir/'ground_param.pkl').open('rb'))[1]
    with (run/'initial_observation.pkl').open('rb') as f:_,init=pickle.load(f)
    camera=init['cam_params']['CAM_FRONT'];intr=camera['intrinsic']
    K=np.array([[intr['W']/(2*np.tan(intr['fovx']/2)),0,intr['cx']],
        [0,intr['H']/(2*np.tan(intr['fovy']/2)),intr['cy']],[0,0,1.]])
    # cam_rect is read from the resolved simulator config for exact projection.
    from omegaconf import OmegaConf
    cfg=OmegaConf.load(run/'resolved_config.yaml');rect=np.eye(4)
    if 'cam_rect' in cfg.camera:
        rect[:3,:3]=Rotation.from_euler('XYZ',cfg.camera.cam_rect.rot,degrees=True).as_matrix();rect[:3,3]=cfg.camera.cam_rect.trans
    T=np.eye(4);T[:3,:3]=rotation;T[:3,3]=inf['ego_pos'];c2w=T@rect
    pcam=(collide-c2w[:3,3])@c2w[:3,:3];uvz=pcam@K.T;uv=uvz[:,:2]/uvz[:,2:3]
    near=np.linalg.norm(points[:,[0,2]]-np.array(inf['ego_pos'])[[0,2]],axis=1)<15
    lnear=np.linalg.norm(lidar[:,[0,2]]-np.array(inf['ego_pos'])[[0,2]],axis=1)<15
    np.savez_compressed(out/'geometry.npz',background=points[near],background_labels=point_labels[near],
        collision_points=collide,collision_labels=clabels,collision_uv=uv,collision_camera_z=pcam[:,2],
        lidar=lidar[lnear],verts=verts,path=np.array([i['post_info']['ego_pos'] for i in trace]),
        ground_path=pickle.load((model_dir/'ground_param.pkl').open('rb'))[0][:,:3,3])
    row={'scene':name,'native_terminal':json.loads((run/'summary.json').read_text()),'counts_by_step':counts,
        'collision_centers':len(collide),'label_counts':{str(k):int((clabels==k).sum()) for k in np.unique(clabels)},
        'camera_height_m':float(cam_height),'collision_height_above_nominal_ground_quantiles_m':np.quantile(cam_height-local[:,1],[0,.1,.5,.9,1]).tolist(),
        'nearest_raw_lidar_distance_quantiles_m':np.quantile(distances,[0,.1,.5,.9,1]).tolist(),
        'lidar_inside_same_box':int(in_box(lidar,verts).sum()),'reference':'first 8 keyframe raw LiDAR; sparse, moving objects not yet removed; not full collision GT',
        'geometry_rule':'official online center-count >100; no threshold change',
        'observed_rule_agreement':all((c>100)==bool(t['post_info']['collision']) for c,t in zip(counts,trace))}
    (out/'audit.json').write_text(json.dumps(row,indent=2));rows.append(row);print(name,row['collision_centers'],row['label_counts'],row['collision_height_above_nominal_ground_quantiles_m'],flush=True)
(R/'native_collision_audit.json').write_text(json.dumps(rows,indent=2))
