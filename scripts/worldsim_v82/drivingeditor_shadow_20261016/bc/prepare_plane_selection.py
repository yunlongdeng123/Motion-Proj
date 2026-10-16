"""Project existing local receiver kernels to supported road, preserving anchor EWA."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from shadow_math import project_gaussians
from prepare_e0 import rays_to_plane

base=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_BC_20261016_v1')
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--selection-source',default=str(base/'selection_v1/selection.npz'));a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=False)
torch.set_num_threads(2)
with np.load(a.selection_source) as f:sel={k:f[k] for k in f.files}
with np.load(base/'training_data_v2/training_data.npz') as f:plane=f['plane_model'];scale=float(f['world_scale'])
cam=json.loads((base/'training_data_v2/camera_metadata.json').read_text())[12]['camera']
parent=base.parent/'rigid_runs/scene0008_089_137/full30k/checkpoint_030000.pt'
cap=torch.load(parent,map_location='cpu',weights_only=False)['background'];ids=sel['receiver_ids'];npoints=len(ids)
arr=lambda x:x.detach().numpy().astype('float32',copy=False)
state=dict(xyz=sel['receiver_xyz_anchor'],scales=arr(cap[4][ids].exp()),rotation=arr(cap[5][ids]),
    opacity=np.ones(npoints,dtype='float32'),colors=np.ones((npoints,3),dtype='float32'),mask=np.ones(npoints,dtype=bool))
old=project_gaussians(state,cam,True);assert np.array_equal(old['ids'],np.arange(npoints))
xyz=rays_to_plane(old['uv'],cam,plane)
normal=plane[:3]/np.linalg.norm(plane[:3]);seed=np.eye(3)[np.argmin(abs(normal))]
u=np.cross(normal,seed);u/=np.linalg.norm(u);v=np.cross(normal,u);basis=np.stack([u,v],axis=1)
view=np.asarray(cam['world_view_transform']);cxyz=xyz@view[:3,:3]+view[3,:3];z=cxyz[:,2]
t=math.tan(-.5);fx=cam['width']/(2*t);fy=cam['height']/(2*t)
x=np.minimum(1.3*t,np.maximum(-1.3*t,cxyz[:,0]/z))*z
y=np.minimum(1.3*t,np.maximum(-1.3*t,cxyz[:,1]/z))*z
j=np.zeros((npoints,2,3));j[:,0,0]=fx/z;j[:,1,1]=fy/z;j[:,0,2]=-fx*x/(z*z);j[:,1,2]=-fy*y/(z*z)
mapping=j@view[:3,:3].T@basis;inv=np.linalg.inv(mapping)
c=old['conic'];det=c[:,0]*c[:,2]-c[:,1]**2
cov2=np.stack([c[:,2]/det,-c[:,1]/det,-c[:,1]/det,c[:,0]/det],1).reshape(-1,2,2)-np.eye(2)*.3
val,vec=np.linalg.eigh(cov2);cov2=(vec*val.clip(1e-8)[:,None,:])@vec.transpose(0,2,1)
tan=inv@cov2@inv.transpose(0,2,1)
world=basis@tan@basis.T+(normal[:,None]*normal[None,:])*(.001*scale)**2
val,vec=np.linalg.eigh(world);vec[np.linalg.det(vec)<0,:,0]*=-1
q=Rotation.from_matrix(vec).as_quat()[:,[3,0,1,2]].astype('float32')
new=dict(state,xyz=xyz.astype('float32'),scales=np.sqrt(val.clip(1e-12)).astype('float32'),rotation=q)
actual=project_gaussians(new,cam,True);assert np.array_equal(actual['ids'],old['ids'])
centre_error=float(abs(actual['uv']-old['uv']).max())
relative=float(np.median(abs(actual['conic']-old['conic'])/(abs(old['conic'])+.01)))
assert centre_error<.001 and relative<.01,(centre_error,relative)
sel.update(receiver_xyz_anchor=new['xyz'],receiver_scaling=new['scales'],receiver_rotation=new['rotation'])
np.savez_compressed(out/'selection.npz',**sel)
receipt={'schema':'supported_road_kernel_projection_v1','receiver_points':npoints,'new_points_added':0,
  'anchor_projected_center_max_error_px':centre_error,'median_relative_conic_error':relative,
  'receiver_plane_abs_height_max_m':float(abs(new['xyz']@plane[:3]+plane[3]).max()/scale),
  'centre_displacement_median_m':float(np.median(np.linalg.norm(new['xyz']-state['xyz'],axis=1)/scale)),
  'normal_sigma_m':.001,'scope':'existing selected receivers only; off-plane artifact gates and unselected legacy branch unchanged',
  'geometry_is_LiDAR_supported_plane_hypothesis_not_dense_GT':True}
(out/'selection_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
