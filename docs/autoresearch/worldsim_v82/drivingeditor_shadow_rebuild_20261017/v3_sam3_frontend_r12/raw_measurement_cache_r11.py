"""Common raw measurement pool with explicit source rows and withheld-return split."""
from pathlib import Path
import sys,json,time
import numpy as np
from PIL import Image
sys.path.insert(0,'/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3/implementation_r10/shadow_rebuild_v3')
from shadow_rebuild.prepare import transform,box_pose,project,zbuffer
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');root=r/'ownership_controls_r11';out=root/'raw_measurements';out.mkdir(exist_ok=False)
m=json.loads((r/'data_r2/manifest.json').read_text());scene=Path(m['source_scene']);origin=np.array(m['original_world_origin_m']);actors={name:j for j,name in enumerate(m['actor_ids'])};records=[];start=time.monotonic()
if not (root/'shared_sam_masks_r2/summary.json').exists():raise RuntimeError('complete real SAM receipt required')
for i,row in enumerate(m['frames']):
 fid=row['frame_id'];raw=np.fromfile(scene/'velodyne'/(fid+'.bin'),np.float32).reshape(-1,6)[:,:3];pose=np.loadtxt(scene/'pose'/(fid+'.txt'));pose[:3,3]-=origin;world=transform(raw,pose);N=len(raw);raw_rows=np.arange(N,dtype=np.int32)
 labels=json.loads((Path(m['annotations'])/(fid+'.json')).read_text())['laser_labels'];owner=np.full(N,-1,np.int16);count=np.zeros(N,np.int16)
 for label in labels:
  bp,size=box_pose(label,pose);inside=(np.abs(transform(world,np.linalg.inv(bp)))<=size/2+.03).all(-1);count[inside]+=1;owner[inside]=actors.get(label['id'],-2)
 owner[count>1]=-2;K=np.array(row['intrinsic']);camera=np.array(row['world_from_camera']);ij,z,in_image=project(world,K,camera,(320,480));depth=zbuffer(world,K,camera,(320,480));visible=in_image.copy();ids=np.flatnonzero(in_image);visible[ids]=z[ids]<=depth[ij[ids,1],ij[ids,0]]+.15
 rgb=np.asarray(Image.open(r/'data_r2'/row['rgb']).convert('RGB'));color=np.full((N,3),.5,np.float32);ids=np.flatnonzero(visible);color[ids]=rgb[ij[ids,1],ij[ids,0]]/255.
 sup=np.load(root/'shared_sam_masks_r2'/(fid+'.npz'));masks=sup['masks'];known=sup['mask_known'];pixel_owner=np.where(masks.any(-1),masks.argmax(-1),-1).astype(np.int16);semantic=np.full(N,-2,np.int16);semantic[ids]=pixel_owner[ij[ids,1],ij[ids,0]]
 body=np.zeros(N,bool)
 for actor in range(len(actors)):
  take=ids[owner[ids]==actor]
  if bool(known[actor]):body[take]=masks[ij[take,1],ij[take,0],actor]
 build=(raw_rows%5!=0)&(row['split']=='train');withheld=(raw_rows%5==0)|(row['split']=='test')
 camera_xyz=transform(world,np.linalg.inv(camera));uvw=camera_xyz@K.T;uv=(uvw[:,:2]/np.maximum(uvw[:,2:3],1e-6)).astype(np.float32)
 np.savez_compressed(out/(fid+'.npz'),world=world.astype(np.float32),raw_row=raw_rows,candidate_owner=owner,box_overlap_count=count,uv=uv,pixel=ij.astype(np.int16),camera_z=z.astype(np.float32),in_image=in_image,lidar_projection_frontmost=visible,
  observed_rgb=color,semantic_owner=semantic,sameframe_body_support=body,BUILD=build,QUERY=withheld)
 rec=dict(frame_id=fid,split=row['split'],actual_points=N,BUILD_points=int(build.sum()),QUERY_points=int(withheld.sum()),visible_candidate_points=int((visible&(owner>=0)).sum()));records.append(rec)
 (out/'progress.json').write_text(json.dumps(dict(frame=i+1,total=49,seconds=time.monotonic()-start)));print(json.dumps(rec),flush=True)
(out/'manifest.json').write_text(json.dumps(dict(status='COMMON_ACTUAL_MEASUREMENT_CACHE_COMPLETE',raw_frames=records,actor_ids=m['actor_ids'],coordinate_origin_m=origin.tolist(),
 source_scene=str(scene),mask_bundle='shared_sam_masks_r2',BUILD='train46 only, actual raw row modulo5 !=0',QUERY='all raw101/113/125 returns plus rowmod5==0 in other46; never seed/depth/tracking/reference construction',
 visibility_caveat='LiDAR projection frontmost is not complete camera visibility; SAM/otherbody/depth/motion checks still required',seconds=time.monotonic()-start),indent=2));print('RAW_CACHE_COMPLETE',flush=True)
