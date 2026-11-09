"""Reuse the BUILD normalization and source pinhole calibration without recomputing PCA."""
import numpy as np
from editable_scene import ROOT,BANK
def from_source(scene,raw,calib):
    import torch
    from scene.cameras import Camera
    d=__import__('json').loads((ROOT/'heldout/observations'/f'{raw:03d}.json').read_text()) if raw in range(79,119) else __import__('json').loads((BANK/'observations'/f'{raw:03d}.json').read_text());lines=calib.read_text().splitlines();c={s.split(':')[0]:np.fromstring(s.split(':')[1],sep=' ') for s in lines};K=c['P0'].reshape(3,4)[:,:3];lidar_cam=np.eye(4);lidar_cam[:3]=c['Tr_velo_to_cam_0'].reshape(3,4);pose=scene.T@np.array(d['ego_world']).reshape(4,4)@np.linalg.inv(lidar_cam);scale=np.linalg.det(scene.T[:3,:3])**(1/3);pose[:3,:3]/=scale;w=np.linalg.inv(pose)
    return Camera(raw,w[:3,:3].T.copy(),w[:3,3].copy(),cx=K[0,2]/4,cy=K[1,2]/4,fx=K[0,0]/4,fy=K[1,1]/4,image=torch.zeros((3,320,480)),image_name=f'0065{raw:03d}',resolution=(480,320),timestamp=(raw-99)*.02)
def held_camera(scene,raw):return from_source(scene,raw,ROOT/'heldout/calib'/f'{raw:03d}.txt')
