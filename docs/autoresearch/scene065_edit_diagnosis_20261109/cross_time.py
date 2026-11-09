"""Audit existing source rules beyond the clipped chart; no model/asset updates."""
from pathlib import Path
import importlib.util,json,os
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree
CAM=Path('/root/autodl-tmp/drivingeditor_repro/campaign');R=CAM/'waymo_val065_delete30k_20261109_r1';O=CAM/'waymo_val065_delete30k_diagnosis_20261109_r1'
load=lambda p:json.loads(p.read_text())
job=next(x for x in load(CAM/'waymo_allframes_20261109_r1/edit_jobs.json')['jobs'] if x['scene']=='065')
B=Path(job['bank']);M=Path(job['masks']);S=Path(job['semantics']);man=load(R/'final_manifest.json');ref=np.load(R/'world_road/reference.npz');track=next(x['track_id'] for x in man['actors'] if x['actor_id']==10)
spec=importlib.util.spec_from_file_location('road',CAM/'waymo_allframes_20261109_r1/public_api/drivingeditor_dynamic_edit/v44_road.py');road=importlib.util.module_from_spec(spec);spec.loader.exec_module(road)
receipt=load(B/'receipt.json');raws=receipt['build_raws'];ds={raw:load(B/'observations'/f'{raw:03d}.json') for raw in raws}
boxes=[next(x for x in d['boxes'] if x['track']==track) for d in ds.values() if any(x['track']==track for x in d['boxes'])]
A=np.array(boxes[0]['world_from_actor']).reshape(4,4);A[:3,3]=np.median([np.array(x['world_from_actor']).reshape(4,4)[:3,3] for x in boxes],axis=0);invA=np.linalg.inv(A)
xyz=ref['local_xyz'];coef=np.linalg.lstsq(np.c_[xyz[:,:2],np.ones(len(xyz))],xyz[:,2],rcond=None)[0]
queries=[];parts=[]
for raw in [119,198]:
    frame=next(x for x in man['frames'] if x['key']==raw);c=np.load(O/f'coverage{raw}.npz');mask=c['target']&~c['inside'];yy,xx=np.where(mask)
    fid=next(x['id'] for x in receipt['frames'] if x['raw']==raw);cal={s.split(':')[0]:np.fromstring(s.split(':')[1],sep=' ') for s in (B/'calib'/f'{fid}.txt').read_text().splitlines()}
    K=cal['P0'].reshape(3,4)[:,:3]/4;K[2,2]=1;C=np.eye(4);C[:3]=cal['Tr_velo_to_cam_0'].reshape(3,4);C=C@np.linalg.inv(np.array(ds[raw]['ego_world']).reshape(4,4))@A;invC=np.linalg.inv(C)
    rays=np.c_[xx+.5,yy+.5,np.ones(len(xx))]@np.linalg.inv(K).T@invC[:3,:3].T;origin=invC[:3,3];t=(coef[2]+coef[:2]@origin[:2]-origin[2])/(rays[:,2]-rays[:,:2]@coef[:2]);hit=origin+t[:,None]*rays;world=hit@A[:3,:3].T+A[:3,3]
    parts.append(dict(raw=raw,start=sum(len(q) for q in queries),count=len(world),pixels=np.c_[xx,yy].tolist()));queries.append(world)
world=np.concatenate(queries);qlocal=world@invA[:3,:3].T+invA[:3,3];N=len(world)
valid=np.zeros((len(raws),N),bool);rgbs=np.zeros((len(raws),N,3),np.float32);pts=[]
for i,raw in enumerate(raws):
    d=ds[raw];cloud=np.fromfile(B/d['lidar_path'],dtype='<f4').reshape(-1,6)[:,:3];E=np.array(d['ego_world']).reshape(4,4);w=cloud@E[:3,:3].T+E[:3,3]
    uv,z=road.project(w,d);p=np.floor(uv/4).astype(int);inside=(z>.1)&(p[:,0]>=0)&(p[:,0]<480)&(p[:,1]>=0)&(p[:,1]<320);ids=np.flatnonzero(inside);p=p[ids]
    nearest=np.full((320,480),np.inf);np.minimum.at(nearest,(p[:,1],p[:,0]),z[ids]);sem=np.asarray(Image.open(S/f'{raw:03d}.png').resize((480,320),Image.Resampling.NEAREST));guard=binary_dilation(np.asarray(Image.open(M/f'{raw:03d}/dynamic_union.png'))>0,iterations=1)
    good=(sem[p[:,1],p[:,0]]==0)&~guard[p[:,1],p[:,0]]&(z[ids]<=nearest[p[:,1],p[:,0]]+.3);l=w[ids[good]]@invA[:3,:3].T+invA[:3,3];plane=abs(l[:,2]-(np.c_[l[:,:2],np.ones(len(l))]@coef))<.12;pts.append(l[plane,:2])
    uv,z=road.project(world,d);p=np.floor(uv/4).astype(int);inside=(z>.1)&(p[:,0]>=0)&(p[:,0]<480)&(p[:,1]>=0)&(p[:,1]<320);ids=np.flatnonzero(inside);p=p[ids]
    good=(sem[p[:,1],p[:,0]]==0)&~guard[p[:,1],p[:,0]]&np.isfinite(nearest[p[:,1],p[:,0]])&(abs(z[ids]-nearest[p[:,1],p[:,0]])<.3);ids=ids[good];p=np.floor(uv[ids]/4).astype(int)
    im=np.asarray(Image.open(B/d['image_path']).resize((480,320),Image.Resampling.BILINEAR),np.float32)/255;valid[i,ids]=True;rgbs[i,ids]=road.linear(im[p[:,1],p[:,0]])
    if i%50==0:print('AUDIT',raw,flush=True)
dist,_=cKDTree(np.concatenate(pts)).query(qlocal[:,:2]);valid&=dist[None]<=.6
luma=rgbs@np.array([.2126,.7152,.0722]);import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore');q75=np.nanquantile(np.where(valid,luma,np.nan),.75,axis=0)
upper=valid&(luma>=q75[None]-.025);depths=[]
for raw,d in ds.items():depths.append(road.project(world,d)[1])
score=1/np.maximum(np.asarray(depths),.1)**2;chosen=np.argmax(np.where(upper,score,-1),axis=0);colour=rgbs[chosen,np.arange(N)];agree=upper&(abs(rgbs-colour[None]).max(2)<.08);real=agree.sum(0)>=3
records=[]
for part in parts:
    a=part['start'];b=a+part['count'];counts=valid[:,a:b].sum(0);c=real[a:b]
    rec=dict(raw=part['raw'],outside_pixels=part['count'],measured_road_supported=int((dist[a:b]<=.6).sum()),at_least_one_source=int((counts>=1).sum()),three_agreeing_sources=int(c.sum()),no_legal_source=int((counts==0).sum()),example=None)
    if c.any():
        k=a+int(np.flatnonzero(c)[0]);sources=np.asarray(raws)[agree[:,k]];rec['example']=dict(pixel_xy=part['pixels'][k-a],world_xyz=world[k].tolist(),reference_raws=sources.tolist())
    records.append(rec)
    np.savez_compressed(O/f'outside_sources{part["raw"]}.npz',pixels=np.array(part['pixels']),real=c,valid=valid[:,a:b],agree=agree[:,a:b],source_raw=np.array(raws),world_xyz=world[a:b],colour_linear=colour[a:b])
result=dict(records=records,training_updates=0,role='posthoc source-admissibility audit under the existing local-plane approximation, not hidden-image ground truth or new asset',projection='query native pinhole; source inherited Brown projection; no calibration changes',automatic_shutdown=False)
(O/'cross_time.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
