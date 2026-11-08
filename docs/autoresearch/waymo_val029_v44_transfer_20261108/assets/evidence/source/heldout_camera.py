"""Reuse the BUILD normalization and source pinhole calibration without recomputing PCA."""
import numpy as np
from editable_scene import ROOT,BANK
def from_source(scene,raw,calib):
    import torch
    from scene.cameras import Camera
    d=scene.observation(raw);lines=calib.read_text().splitlines();c={s.split(':')[0]:np.fromstring(s.split(':')[1],sep=' ') for s in lines};K=c['P0'].reshape(3,4)[:,:3];lidar_cam=np.eye(4);lidar_cam[:3]=c['Tr_velo_to_cam_0'].reshape(3,4);pose=scene.T@np.array(d['ego_world']).reshape(4,4)@np.linalg.inv(lidar_cam);scale=np.linalg.det(scene.T[:3,:3])**(1/3);pose[:3,:3]/=scale;w=np.linalg.inv(pose)
    return Camera(raw,w[:3,:3].T.copy(),w[:3,3].copy(),cx=K[0,2]/4,cy=K[1,2]/4,fx=K[0,0]/4,fy=K[1,1]/4,image=torch.zeros((3,320,480)),image_name=f'0029{raw:03d}',resolution=(480,320),timestamp=(raw-98.5)*.02)
def held_camera(scene,raw):return from_source(scene,raw,ROOT/'heldout/calib'/f'{raw:03d}.txt')
def audit():
    from editable_scene import Scene
    scene=Scene(False,False);errors=[]
    for raw in [0,130,155,160,190,197]:
        native=scene.camera(raw);made=from_source(scene,raw,BANK/'calib'/f'0029{raw:03d}.txt');errors.append(float((native.world_view_transform-made.world_view_transform).abs().max()))
    assert max(errors)<2e-5,errors
    (ROOT/'heldout/camera_audit.json').write_text(__import__('json').dumps(dict(BUILD_camera_transform_maxabs=errors,recompute_PCA=False,heldout_fitting=False),indent=2))
if __name__=='__main__':audit()
