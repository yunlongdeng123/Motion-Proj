"""Read-only guard audit: LiDAR-road projection can be camera-occluded by actual body points."""
from pathlib import Path
import sys,json
import numpy as np
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
sys.path.insert(0,str(r/'implementation_r8/shadow_rebuild_v3'))
from shadow_rebuild.data import Dataset
from shadow_rebuild.prepare import project,transform
from shadow_rebuild.train import write
d=Dataset(r/'data_r4_road_identity','cpu',True)
banks={a:np.load(r/'initialization_source_audit_r7'/f'a{a:02d}_actual_banks.npz')['body'] for a in [5,6,12,18,25,26,37]};records=[]
for index in d.train_indices:
 f=d.frame(index);h,w=f.depth.shape;depth=np.full((h,w),np.inf);owners=np.full((h,w),-1,np.int16)
 for actor,points in banks.items():
  if not bool(f.valid[actor]):continue
  world=transform(points,f.poses[actor].numpy());ij,z,valid=project(world,np.asarray(d.rows[index]['intrinsic']),np.asarray(d.rows[index]['world_from_camera']),(h,w))
  ij,z=ij[valid],z[valid]
  for dy in [-1,0,1]:
   for dx in [-1,0,1]:
    xx,yy=ij[:,0]+dx,ij[:,1]+dy;ok=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
    np.minimum.at(depth,(yy[ok],xx[ok]),z[ok])
 cert=f.road_certified.numpy();blocked=cert&(depth+.1<f.depth.numpy())
 records.append(dict(frame_id=f.frame_id,certificate_pixels=int(cert.sum()),body_occlusion_candidates=int(blocked.sum())))
write(r/'road_ray_occlusion_r8.json',dict(read_only=True,training_updates=0,certified_pixels=sum(x['certificate_pixels'] for x in records),
 potentially_camera_occluded=sum(x['body_occlusion_candidates'] for x in records),
 method='actual train-time body LiDAR bank transported rigidly;1pixel support, body depth+0.1m < actual road depth; rejection candidate only, no fabricated body mask',frames=records))
print(json.dumps(dict(certified_pixels=sum(x['certificate_pixels'] for x in records),potentially_camera_occluded=sum(x['body_occlusion_candidates'] for x in records))))
