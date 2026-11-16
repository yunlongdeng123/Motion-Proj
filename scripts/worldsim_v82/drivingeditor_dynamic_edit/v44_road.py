"""Measured world-coordinate road chart from an explicit input bank. Unknown cells stay unknown; inferred asphalt is labelled.

New-scene adapter for the inherited 2cm road representation. The automatic
local robust plane replaces the old scene-specific road chart preparation.
It must be reported as a transfer adapter, not byte-identical preprocessing.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,time,warnings
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.ndimage import binary_dilation,gaussian_filter
from .v44_handoff_policy import handoff_domains,target_mask_from_labels,rebind_other_vehicles

def linear(x):return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
def srgb(x):return np.where(x<=.0031308,x*12.92,1.055*np.maximum(x,1e-10)**(1/2.4)-.055)
def project(world,d):
    C=np.array(d['camera_pose_world']).reshape(4,4)@np.array(d['camera_extrinsic']).reshape(4,4);inv=np.linalg.inv(C);c=world@inv[:3,:3].T+inv[:3,3];z=c[:,0]
    fx,fy,cx,cy,k1,k2,p1,p2,k3=d['camera_intrinsic_Brown'];x=-c[:,1]/np.maximum(z,.1);y=-c[:,2]/np.maximum(z,.1);r=x*x+y*y;q=1+k1*r+k2*r*r+k3*r*r*r
    uv=np.c_[fx*(x*q+2*p1*x*y+p2*(r+2*x*x))+cx,fy*(y*q+p1*(r+2*y*y)+2*p2*x*y)+cy]
    return uv,z
def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bank','masks','semantics','native-eval','transform','output']:p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--track-id',required=True);p.add_argument('--actor-id',required=True,type=int)
    a=p.parse_args();BANK=a.bank;TRACK=a.track_id;target=a.actor_id
    if target<=0:raise ValueError('Positive registered actor ID required')
    started=time.monotonic();out=a.output;out.mkdir(exist_ok=False);(out/'views').mkdir()
    receipt=json.loads((BANK/'receipt.json').read_text());frame_ids={r['raw']:r['id'] for r in receipt['frames']}
    obs=lambda raw:json.loads((BANK/'observations'/f'{raw:03d}.json').read_text())
    raws=json.loads((BANK/'receipt.json').read_text())['build_raws'];ds={r:obs(r) for r in raws};boxes=[next(b for b in d['boxes'] if b['track']==TRACK) for d in ds.values() if any(b['track']==TRACK for b in d['boxes'])]
    A=np.array(boxes[0]['world_from_actor']).reshape(4,4);A[:3,3]=np.median([np.array(b['world_from_actor']).reshape(4,4)[:3,3] for b in boxes],axis=0);inv=np.linalg.inv(A)
    centres=np.array([np.array(b['world_from_actor']).reshape(4,4)[:3,3] for b in boxes]);local=centres@inv[:3,:3].T+inv[:3,3];half=np.max([b['dimensions'] for b in boxes],axis=0)/2
    lo=local[:,:2].min(0)-half[:2]-[.70,.45];hi=local[:,:2].max(0)+half[:2]+[.70,.45]
    cell=.02;axes=[np.arange(np.floor(lo[i]/cell),np.ceil(hi[i]/cell)+1,dtype=int) for i in range(2)];kx,ky=np.meshgrid(*axes,indexing='ij');keys=np.c_[kx.ravel(),ky.ravel()];xy=(keys+.5)*cell;shape=kx.shape
    samples=[];frame_cache={}
    for raw,d in ds.items():
        cloud=np.fromfile(BANK/d['lidar_path'],dtype='<f4').reshape(-1,6)[:,:3];E=np.array(d['ego_world']).reshape(4,4);w=cloud@E[:3,:3].T+E[:3,3];l=w@inv[:3,:3].T+inv[:3,3]
        uv,z=project(w,d);p=np.floor(uv/4).astype(int);inside=(z>.1)&(p[:,0]>=0)&(p[:,0]<480)&(p[:,1]>=0)&(p[:,1]<320);ids=np.flatnonzero(inside);p=p[ids];nearest=np.full((320,480),np.inf);np.minimum.at(nearest,(p[:,1],p[:,0]),z[ids])
        semantic=np.array(Image.open(a.semantics/f'{raw:03d}.png').resize((480,320),Image.Resampling.NEAREST));labels=np.array(Image.open(a.masks/f'{raw:03d}/labels.png'));foreground=np.array(Image.open(a.masks/f'{raw:03d}/dynamic_union.png'))>0;guard=binary_dilation(foreground,iterations=1)
        good=(semantic[p[:,1],p[:,0]]==0)&~guard[p[:,1],p[:,0]]&(z[ids]<=nearest[p[:,1],p[:,0]]+.3);ids=ids[good];near=(l[ids,:2]>=lo-1).all(1)&(l[ids,:2]<=hi+1).all(1)&(abs(l[ids,2]+half[2])<.8)
        samples.append(l[ids[near]]);frame_cache[raw]=(nearest,semantic,guard)
    pts=np.concatenate(samples);assert len(pts)>=30,'No measured local road support'
    # Robust measured tangent plane, fit only to independently road-labelled returns.
    X=np.c_[pts[:,:2],np.ones(len(pts))];coef=np.linalg.lstsq(X,pts[:,2],rcond=None)[0]
    for _ in range(5):
        residual=X@coef-pts[:,2];weight=np.minimum(1,.05/np.maximum(abs(residual),1e-8));coef=np.linalg.lstsq(X*weight[:,None]**.5,pts[:,2]*weight**.5,rcond=None)[0]
    clean=abs(X@coef-pts[:,2])<.12;tree=cKDTree(pts[clean,:2]);dist,_=tree.query(xy);xyz=np.c_[xy,np.c_[xy,np.ones(len(xy))]@coef];world=xyz@A[:3,:3].T+A[:3,3];N=len(world)
    rgbs=np.zeros((len(raws),N,3),np.float32);valid=np.zeros((len(raws),N),bool);score=np.zeros_like(valid,dtype=np.float32);offsets=np.zeros_like(rgbs)
    for i,raw in enumerate(raws):
        d=ds[raw];uv,z=project(world,d);pix=np.floor(uv/4).astype(int);inside=(z>.1)&(pix[:,0]>=0)&(pix[:,0]<480)&(pix[:,1]>=0)&(pix[:,1]<320);ids=np.flatnonzero(inside);p=pix[ids];nearest,semantic,guard=frame_cache[raw]
        good=(semantic[p[:,1],p[:,0]]==0)&~guard[p[:,1],p[:,0]]&np.isfinite(nearest[p[:,1],p[:,0]])&(abs(z[ids]-nearest[p[:,1],p[:,0]])<.3)&(dist[ids]<=.6);ids=ids[good]
        image=np.array(Image.open(BANK/d['image_path']).resize((480,320),Image.Resampling.BILINEAR),np.float32)/255;p=pix[ids];rgbs[i,ids]=linear(image[p[:,1],p[:,0]]);valid[i,ids]=True;score[i,ids]=1/np.maximum(z[ids],.1)**2
        native=np.load(a.native_eval/f'{raw:03d}'/'keep.npz')['rgb'].transpose(1,2,0);offsets[i,ids]=linear(native[p[:,1],p[:,0]])-rgbs[i,ids]
        if i%30==0:print('ROAD_SOURCE',raw,int(valid[i].sum()),flush=True)
    luma=rgbs@np.array([.2126,.7152,.0722],np.float32)
    with warnings.catch_warnings():warnings.simplefilter('ignore');q75=np.nanquantile(np.where(valid,luma,np.nan),.75,axis=0)
    upper=valid&(luma>=q75[None]-.025);chosen=np.argmax(np.where(upper,score,-1),axis=0);reference=rgbs[chosen,np.arange(N)].copy();agree=upper&(abs(rgbs-reference[None]).max(2)<.08);real=agree.sum(0)>=3
    colour=srgb(reference);asphalt=real&(colour.max(1)<.65)&(np.ptp(colour,axis=1)<.18);marked=real&~asphalt
    inferred=np.zeros(N,bool)
    if asphalt.sum()>=8:
        ids=np.flatnonzero(asphalt);distance,near=cKDTree(xy[ids]).query(xy,k=8);inferred=~real&(distance[:,0]<=.60)&(dist<=.60);weight=1/np.maximum(distance,.02)**2;reference[inferred]=(reference[ids[near[inferred]]]*weight[inferred,:,None]).sum(1)/weight[inferred].sum(1)[:,None]
    # Fixed, world-stable clean-neighbour native colour adaptation; inherited cap/sigmas.
    safe=agree&(abs(offsets).max(2)<.05);den=gaussian_filter(safe.sum(0).reshape(shape).astype(float),(.80/cell,.40/cell));correction=np.stack([gaussian_filter((offsets[:,:,c]*safe).sum(0).reshape(shape),(.80/cell,.40/cell))/np.maximum(den,1e-8) for c in range(3)],-1).reshape(N,3);correction=np.clip(correction,-.025,.025);uncalibrated=reference.copy();reference=np.clip(reference+correction,0,1)
    active=(real|inferred)&~marked;T=np.loadtxt(a.transform);scale=float(np.linalg.det(T[:3,:3])**(1/3));model=world@T[:3,:3].T+T[:3,3];normal=np.array([-coef[0],-coef[1],1.]);normal/=np.linalg.norm(normal);normal_world=A[:3,:3]@normal
    np.savez_compressed(out/'reference.npz',active=active,real=real,inferred=inferred,marked=marked,local_xyz=xyz,world_xyz=world,model_xyz=model,reference_srgb=srgb(reference).astype(np.float32),reference_linear=reference,uncalibrated_linear=uncalibrated,correction_linear=correction,normal_world=np.tile(normal_world,(N,1)),gaussian_sigma_m=np.tile([cell*.65,cell*.65,.003],(N,1)),road_opacity=np.full(N,.99,np.float32),source_valid=valid,source_agree=agree,best_source_raw=np.where(real,np.array(raws)[chosen],-1),source_raw=np.array(raws),keys=keys,model_from_world=T,meter_to_model=scale)
    # Query maps use exactly the same pinhole model as the native renderer.
    for raw,d in ds.items():
        lines=(BANK/'calib'/f'{frame_ids[raw]}.txt').read_text().splitlines();cal={s.split(':')[0]:np.fromstring(s.split(':')[1],sep=' ') for s in lines};K=cal['P0'].reshape(3,4)[:,:3]/4;K[2,2]=1;C=np.eye(4);C[:3]=cal['Tr_velo_to_cam_0'].reshape(3,4);C=C@np.linalg.inv(np.array(d['ego_world']).reshape(4,4))@A;invC=np.linalg.inv(C);yy,xx=np.mgrid[:320,:480];rays=np.stack((xx+.5,yy+.5,np.ones_like(xx)),2)@np.linalg.inv(K).T;rays=rays@invC[:3,:3].T;origin=invC[:3,3];depth=(coef[2]+coef[:2]@origin[:2]-origin[2])/(rays[:,:,2]-rays[:,:,:2]@coef[:2]);hit=origin+depth[:,:,None]*rays;key=np.floor(hit[:,:,:2]/cell).astype(int)-[axes[0][0],axes[1][0]];inside=(key>=0).all(2)&(key<shape).all(2)&(depth>0);key=np.clip(key,[0,0],np.array(shape)-1);index=key[:,:,0]*shape[1]+key[:,:,1];nearest,semantic,guard=frame_cache[raw];labels=np.array(Image.open(a.masks/f'{raw:03d}/labels.png'));foreground=np.array(Image.open(a.masks/f'{raw:03d}/dynamic_union.png'))>0;road=inside&active[index]&~guard&(semantic==0);reveal=inside&active[index]&(labels==target);fit=(road|reveal)*np.where(inferred[index],.25,1.);marking=inside&marked[index];keep=np.ones((320,480),np.float32);keep[road|reveal]=0;keep[foreground&(labels!=target)]=1
        target_mask=target_mask_from_labels(labels,[target])
        other=rebind_other_vehicles(foreground&(labels!=target),labels,[target])
        domains=handoff_domains(dict(fit=fit.astype(np.float32),keep=keep,inside=inside,other=other,marking=marking),target_mask)
        np.savez_compressed(out/'views'/f'{raw:03d}.npz',index=index,inside=inside,fit=domains['fit'],keep=domains['keep'],marking=marking,other=other,
            target=target_mask,target_actor_ids=np.array([target],np.int64),candidate_support=domains['candidate_support'],left=domains['left'],unknown_target=domains['unknown_target'],
            reference_linear=reference[index].transpose(2,0,1).astype(np.float32),real_domain=(road|reveal)&real[index],inferred_domain=(road|reveal)&inferred[index],observed_road=road)
    (out/'views'/'binding.json').write_text(json.dumps(dict(schema='v44.handoff_views.v1',delete_actor_ids=[target],track_ids=[TRACK],policy='full_target_candidate_and_preservation_v1',reference_added=False),indent=2)+'\n')
    report=dict(status='PASS_BUILD_ONLY_WORLD_ROAD',frames=raws,measured_points=len(pts),plane_inliers=int(clean.sum()),plane_mae_m=float(abs(X[clean]@coef-pts[clean,2]).mean()),cells=N,real_cells=int(real.sum()),inferred_cells=int(inferred.sum()),active_cells=int(active.sum()),unknown_cells=int((~active).sum()),marking_cells=int(marked.sum()),grid_m=cell,geometry_adapter='automatic robust local measured road tangent plane; differs from old scene-specific road chart',inference='eight measured asphalt neighbours within0.60m; marked, confidence0.25; not observed truth',colour_policy='3 agreeing real source views; best resolution in upper radiance cluster',heldout_frames=receipt['holdout_raws'],target_track=TRACK,target_actor=target,source_bank=str(BANK),seconds=time.monotonic()-started)
    (out/'receipt.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
