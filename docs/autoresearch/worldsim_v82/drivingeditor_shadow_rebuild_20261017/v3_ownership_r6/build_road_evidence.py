"""真实训练 LiDAR 道路表面证据，恢复旧 OBB guard 内的实际可见路面。

不裁 Actor 高度。仅实际点投影、已知 body 排除、已有静态路面局部几何一致、
相同世界位置在多时刻重复测到者，才作为道路正证据。未知/遮挡不产生负标签。
"""
from pathlib import Path
import json,sys,time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
sys.path.insert(0,str(r/'implementation_r4/shadow_rebuild_v3'))
from shadow_rebuild.data import Dataset
from shadow_rebuild.prepare import project,transform
from shadow_rebuild.train import write
d=Dataset(r/'data_r2','cpu',True);out=r/'road_evidence_r5';out.mkdir(exist_ok=False)
seed=d.seed;ids=np.flatnonzero(seed['road']);xyz=seed['xyz'][ids];tree=cKDTree(xyz)
normals=Rotation.from_quat(seed['quaternion'][ids][:,[1,2,3,0]]).as_matrix()[...,2]
origin=np.loadtxt(Path(d.manifest['source_scene'])/'pose'/(d.rows[0]['frame_id']+'.txt'))[:3,3]
allpoints=[];sources=[];pointids=[];pixels=[];stats=[];start=time.monotonic()
for j,index in enumerate(d.train_indices):
 row=d.rows[index];fid=row['frame_id'];f=d.frame(index);scene=Path(d.manifest['source_scene'])
 raw=np.fromfile(scene/'velodyne'/(fid+'.bin'),np.float32).reshape(-1,6)[:,:3]
 pose=np.loadtxt(scene/'pose'/(fid+'.txt'));pose[:3,3]-=origin
 world=transform(raw,pose);ij,z,good=project(world,np.array(row['intrinsic']),np.array(row['world_from_camera']),f.depth.shape)
 rows=np.flatnonzero(good&(z<80));yy,xx=ij[rows,1],ij[rows,0]
 depth=f.depth.numpy();body=f.masks.numpy().any(-1)
 rows=rows[(depth[yy,xx]>0)&(np.abs(depth[yy,xx]-z[rows])<.15)&(~body[yy,xx])]
 dist,near=tree.query(world[rows],k=3,workers=1)
 # Agreement with three measured road surface anchors, not a fixed world/Actor height threshold.
 delta=world[rows,None]-xyz[near]
 signed=np.abs(np.einsum('nki,nki->nk',delta,normals[near]))
 agreement=(normals[near]*normals[near[:,0]][:,None]).sum(-1)
 keep=(dist[:,-1]<2.)&(signed.max(-1)<.06)&(np.abs(agreement).min(-1)>.95)
 rows=rows[keep];allpoints.append(world[rows]);sources.append(np.full(len(rows),index,np.int16));pointids.append(rows);pixels.append(ij[rows])
 stats.append(dict(frame_id=fid,candidates=len(rows)))
 print(json.dumps(stats[-1]),flush=True)
points=np.concatenate(allpoints);frames=np.concatenate(sources);lidar_ids=np.concatenate(pointids);uv=np.concatenate(pixels)
atlas_tree=cKDTree(points);support=np.zeros(len(points),np.uint8)
# Count distinct actual training timestamps in a 10cm ball; no test points/colors enter decisions.
for begin in range(0,len(points),20000):
 neighbors=atlas_tree.query_ball_point(points[begin:begin+20000],.10,workers=1)
 support[begin:begin+len(neighbors)]=[min(255,len(np.unique(frames[v]))) for v in neighbors]
certified=support>=3
np.savez_compressed(out/'actual_lidar_atlas.npz',xyz=points.astype(np.float32),frame_index=frames,lidar_row=lidar_ids,
                    uv=uv,support_frames=support,certified=certified)
for stat,index in zip(stats,d.train_indices):
 take=(frames==index)&certified;mask=np.zeros(d.frame(index).depth.shape,bool);pix=uv[take];mask[pix[:,1],pix[:,0]]=True
 np.savez_compressed(out/(stat['frame_id']+'.npz'),road=mask)
 stat.update(certified_points=int(take.sum()),certified_pixels=int(mask.sum()))
write(out/'summary.json',dict(status='COMPLETE',train_only=True,actual_points=len(points),certified_points=int(certified.sum()),
                            frames=stats,elapsed_seconds=time.monotonic()-start,
                            method='actual LiDAR rows; known-body exclusion; agreement with 3 existing measured road local surface normals within6cm; world-space10cm repeated>=3trainframes',
                            limitation='conservative geometric road-surface certificate, not human road semantic GT; no negative inference from missing support'))
print(json.dumps(dict(status='COMPLETE',actual_points=len(points),certified_points=int(certified.sum()),elapsed_seconds=time.monotonic()-start)))
