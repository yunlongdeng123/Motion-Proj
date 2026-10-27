"""Retrieve existing masks, source LiDAR depth and annotation boxes; no inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,base64,zlib,hashlib,time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
contexts=['14705303724557273004_3105_000_3125_000','4295449061847708198_3769_000_3789_000','473735159277431842_630_095_650_095']
def read(p):return json.loads(p.read_text())
def ap(m,p):return np.asarray(p)@m[:3,:3].T+m[:3,3]
def packed(arrays):
    buf=b'';meta={}
    for k,a in arrays.items():
        a=np.ascontiguousarray(a);v=a.tobytes();meta[k]={'offset':len(buf),'bytes':len(v),'dtype':a.dtype.str,'shape':list(a.shape)};buf+=v
    return {'arrays':meta,'data':base64.b64encode(zlib.compress(buf,6)).decode()}
out={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'GPU_used':False,'model_modified':False,'contexts':[]}
for ctx in contexts:
    p=R/'motion_scene_inputs_v1'/ctx;sensor=read(p/'sensor_preparation.json');review=read(p/'native_preparation/selection_reviewed.json');track=review['track'];L=np.linalg.inv(np.loadtxt(p/'pose'/(sensor['frames'][0]['frame_id']+'.txt')));frames={};hashes={}
    for item in sensor['frames']:
        raw=item['raw'];fid=item['frame_id'];pose=np.loadtxt(p/'pose'/(fid+'.txt'));cal=p/'calib'/(fid+'.txt');fields={l.split(':')[0]:np.fromstring(l.split(':')[1],sep=' ') for l in cal.read_text().splitlines()};V=np.eye(4);V[:3]=fields['Tr_velo_to_cam_0'].reshape(3,4);K=fields['P0'].reshape(3,4)[:,:3].copy();K[:2]/=4
        bp=p/'dynamic_actor_assets/actor_0/body_masks'/(fid+'.png');mask=np.array(Image.open(bp).resize((480,320),Image.Resampling.NEAREST))>0
        points=np.fromfile(p/'velodyne'/(fid+'.bin'),np.float32).reshape(-1,6);cam=ap(V,points[:,:3]);uv=cam[:,:2]/np.maximum(cam[:,2:],1e-8)*[K[0,0],K[1,1]]+[K[0,2],K[1,2]];good=(cam[:,2]>.1)&(uv[:,0]>=0)&(uv[:,0]<480)&(uv[:,1]>=0)&(uv[:,1]<320);ids=np.flatnonzero(good);ids=ids[np.argsort(cam[ids,2])];ij=np.floor(uv[ids]).astype(int);pix=ij[:,1]*480+ij[:,0];_,first=np.unique(pix,return_index=True);depth=np.full(480*320,np.nan,np.float32);depth[pix[first]]=cam[ids[first],2]
        ann=p/'annotations'/(fid+'.json');boxes=[]
        for lab in read(ann)['laser_labels']:
            if lab['type']!='TYPE_VEHICLE':continue
            b=lab['box'];c=np.array([b['center_x'],b['center_y'],b['center_z']]);size=np.array([b['length'],b['width'],b['height']]);rot=Rotation.from_euler('z',b['heading']).as_matrix();q=np.array([[1 if (n>>j)&1 else -1 for j in range(3)] for n in range(8)])*size/2@rot.T+c;cc=ap(V,q);proj=cc[:,:2]/cc[:,2:]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]] if np.all(cc[:,2]>.1) else None;visible=proj is not None and proj[:,0].max()>=0 and proj[:,0].min()<480 and proj[:,1].max()>=0 and proj[:,1].min()<320
            boxes.append({'track':lab['id'],'target':lab['id']==track,'corners_local':ap(L@pose,q).tolist(),'uv_sensor':proj.tolist() if proj is not None else None,'in_front_image':bool(visible)})
        frames[str(raw)]={'source_c2w':(L@pose@np.linalg.inv(V)).tolist(),'vehicle_boxes':boxes,'observations':packed({'mask':mask.astype('uint8').ravel(),'interior':np.minimum(distance_transform_edt(mask),255).astype('uint8').ravel(),'lidar_z':depth.astype('<f2')})}
        for f in [bp,cal,ann]:hashes[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    aliases={t:('TARGET' if t==track else 'V%02d'%(i+1)) for i,t in enumerate(sorted({b['track'] for f in frames.values() for b in f['vehicle_boxes']}))}
    for f in frames.values():
        for b in f['vehicle_boxes']:b['label']=aliases[b['track']]
    out['contexts'].append({'context':ctx,'target_track':track,'frames':frames,'source_hashes':hashes,'semantic_source':'existing reviewed target DEVA body mask; no dense class segmentation; boxes audit only'})
print(json.dumps(out,separators=(',',':'),allow_nan=False))
