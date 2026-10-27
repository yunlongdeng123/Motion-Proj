"""Measured actor-local seeds, retaining every identity-qualified source return.

No GT box filters points. Track pose defines actor coordinates; dimensions only
define a later growth envelope, preserving initially outside measured seeds.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,sys,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from segmentation import OLD,ROOT,CONTEXTS,write
def main():
    for ctx in CONTEXTS:
        out=ROOT/'datasets'/ctx
        if (out/'manifest.json').exists():continue
        out.mkdir(parents=True,exist_ok=False);source=OLD/'motion_scene_inputs_v1'/ctx
        sensor=json.loads((source/'sensor_preparation.json').read_text());review=json.loads((source/'native_preparation/selection_reviewed.json').read_text());track=review['track']
        ids=np.load(source/'dynamic_actor_assets/actor_0/source_lidar_indices.npz');T=np.loadtxt(source/'transform_matrix.txt')
        xyz=[];colors=[];origins=[];frames=[];sizes=[]
        for i,f in enumerate(sensor['frames']):
            fid=f['frame_id'];points=np.fromfile(source/'velodyne'/(fid+'.bin'),np.float32).reshape(-1,6)[ids[fid],:3].astype(np.float64)
            pose=np.loadtxt(source/'pose'/(fid+'.txt'));annotations=json.loads((source/'annotations'/(fid+'.json')).read_text())['laser_labels'];b=next(x for x in annotations if x['id']==track)['box']
            center=np.array([b['center_x'],b['center_y'],b['center_z']])@pose[:3,:3].T+pose[:3,3];R=pose[:3,:3]@Rotation.from_euler('z',b['heading']).as_matrix()
            world=points@pose[:3,:3].T+pose[:3,3];local=(world-center)@R
            calib={l.split(':')[0]:np.fromstring(l.split(':')[1],sep=' ') for l in (source/'calib'/(fid+'.txt')).read_text().splitlines()};V=calib['Tr_velo_to_cam_0'].reshape(3,4);K=calib['P0'].reshape(3,4)[:,:3];cam=points@V[:3,:3].T+V[:3,3];uv=cam[:,:2]/cam[:,2:]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
            image=np.array(Image.open(source/'image/image_0'/(fid+'.png')).convert('RGB'));pix=np.floor(uv).astype(int);pix[:,0]=pix[:,0].clip(0,image.shape[1]-1);pix[:,1]=pix[:,1].clip(0,image.shape[0]-1)
            color=image[pix[:,1],pix[:,0]]/255.;xyz.append(local);colors.append(color);origins.extend((i,int(j)) for j in ids[fid])
            mask=np.array(Image.open(source/'dynamic_actor_assets/actor_0/body_masks'/(fid+'.png')).resize((480,320),Image.Resampling.NEAREST))>0
            # Actual target returns are sparse depth supervision, never box masks.
            p=np.floor(uv/4).astype(int);valid=(p[:,0]>=0)&(p[:,0]<480)&(p[:,1]>=0)&(p[:,1]<320)&(cam[:,2]>.1);p=p[valid];z=cam[valid,2];depth=np.full((320,480),np.nan,np.float32)
            for j in np.argsort(z)[::-1]:depth[p[j,1],p[j,0]]=z[j]
            np.savez_compressed(out/('%03d.npz'%f['raw']),rgb=np.array(Image.fromarray(image).resize((480,320),Image.Resampling.LANCZOS)),body=mask,depth=depth)
            frames.append(dict(raw=f['raw'],frame_id=fid,model_time=i*.02-.48,actor_center_world=center.tolist(),actor_rotation_world=R.tolist(),source_lidar_points=len(points)))
            sizes.append([b['length'],b['width'],b['height']])
        xyz=np.concatenate(xyz).astype('float32');color=np.concatenate(colors).astype('float32');origins=np.array(origins,np.int64)
        # Spatial deduplication is independent of box coordinates or quality.
        voxel=np.floor(xyz/.06).astype(np.int64);_,keep=np.unique(voxel,axis=0,return_index=True);keep=np.sort(keep);seed=xyz[keep]
        dist,_=cKDTree(seed).query(seed,k=2);sigma=np.clip(dist[:,1]*.6,.025,.12).astype('float32')
        np.savez_compressed(out/'seeds.npz',xyz=seed,rgb=color[keep],sigma=sigma,source_frame_index=origins[keep,0],source_lidar_row=origins[keep,1],box_half=np.median(sizes,axis=0)/2,transform=T)
        source_hashes={str(source/'dynamic_actor_assets/actor_0/source_lidar_indices.npz'):hashlib.sha256((source/'dynamic_actor_assets/actor_0/source_lidar_indices.npz').read_bytes()).hexdigest()}
        write(out/'manifest.json',dict(context=ctx,track=track,frames=frames,raw_real_returns=len(xyz),actor_local_voxel_seeds=len(seed),voxel_m=.06,box_filter_applied=False,initial_outside_box_rows=int((np.abs(seed)>np.median(sizes,axis=0)/2).any(1).sum()),source_hashes=source_hashes,source_identity_review=str(source/'native_preparation/selection_reviewed.json'),four_fields=dict(semantic='vehicle from reviewed target mask and measured return',instance=track,motion='rigid source ActorTrack, stationary status independent of identity',evidence='source frame and exact LiDAR row, no generated evidence'),quality_accepted=False,human_verdict=None))
        print('DATASET',ctx,len(seed),flush=True)
if __name__=='__main__':main()
