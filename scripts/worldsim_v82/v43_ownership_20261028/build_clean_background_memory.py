"""Measured, clean cross-time background samples. CPU only; first scene only.

Source pixels must be confident static semantics, outside dilated people/vehicle
masks. A surface voxel needs three distinct frames and two temporal thirds.
Query supervision remains sparse: absent evidence is not filled or called GT.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,time,argparse
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation,distance_transform_edt

CTX='14705303724557273004_3105_000_3125_000'
OLD=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
ROOT=OLD.parent/'v43_ownership_20261028'

def project(world,source,f):
    fid=f['frame_id'];pose=np.loadtxt(source/'pose'/(fid+'.txt'))
    sensor=(world-pose[:3,3])@pose[:3,:3]
    cal={l.split(':')[0]:np.fromstring(l.split(':')[1],sep=' ') for l in (source/'calib'/(fid+'.txt')).read_text().splitlines()}
    V=cal['Tr_velo_to_cam_0'].reshape(3,4);K=cal['P0'].reshape(3,4)[:,:3]/4
    cam=sensor@V[:3,:3].T+V[:3,3];z=cam[:,2]
    uv=np.floor(cam[:,:2]/np.maximum(z[:,None],1e-4)*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]).astype(int)
    return uv,z

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--fullclip',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    source=ROOT/'target_unknown_v2/fullclip_inputs' if a.fullclip else OLD/'motion_scene_inputs_v1'/CTX;ds=ROOT/'datasets'/CTX;query_frames=json.loads((ds/'manifest.json').read_text())['frames'];frames=json.loads((source/'manifest.json').read_text())['frames'] if a.fullclip else query_frames;xyz=[];colors=[];frameids=[];records=[]
    for i,f in enumerate(frames):
        fid=f['frame_id'];raw=f['raw'];pts=np.fromfile(source/'velodyne'/(fid+'.bin'),np.float32).reshape(-1,6)[:,:3].astype('float64');pose=np.loadtxt(source/'pose'/(fid+'.txt'));world=pts@pose[:3,:3].T+pose[:3,3];uv,z=project(world,source,f)
        valid=(z>.1)&(uv[:,0]>=0)&(uv[:,0]<480)&(uv[:,1]>=0)&(uv[:,1]<320);ind=np.flatnonzero(valid);pix=uv[ind]
        sem=np.load((ROOT/'target_unknown_v2/fullclip_semantics' if a.fullclip else ROOT/'segformer_b0_attempt3'/CTX)/('%03d.npz'%raw));body=np.load(ds/('%03d.npz'%raw))['body'] if (ds/('%03d.npz'%raw)).exists() else np.zeros((320,480),bool);vehicle=np.isin(sem['labels'],np.arange(12,20))|body
        if a.fullclip:
            # Annotated boxes only EXCLUDE potential foreground from clean
            # background observations. They never label Gaussian ownership.
            for obj in json.loads((source/'annotations'/(fid+'.json')).read_text()):
                b=obj['box'];h=b.get('heading',0);c,s=np.cos(h),np.sin(h);rot=np.array([[c,-s,0],[s,c,0],[0,0,1]]);corners=np.array([[x,y,z] for x in [-.5,.5] for y in [-.5,.5] for z in [-.5,.5]])*[b['length'],b['width'],b['height']];corners=corners@rot.T+[b['center_x'],b['center_y'],b['center_z']];cw=corners@pose[:3,:3].T+pose[:3,3];pp,zz=project(cw,source,f)
                if (zz>.1).all():
                    lo=np.maximum(pp.min(0),[0,0]);hi=np.minimum(pp.max(0)+1,[480,320])
                    if np.all(hi>lo):vehicle[lo[1]:hi[1],lo[0]:hi[0]]=True
        safe=(~binary_dilation(vehicle,iterations=5))&(sem['labels']>=1)&(sem['labels']<=10)&(sem['confidence']>=.8)
        # Keep only the nearest actual return in each observed pixel.
        depth=np.full((320,480),np.inf);np.minimum.at(depth,(pix[:,1],pix[:,0]),z[ind]);keep=safe[pix[:,1],pix[:,0]]&(z[ind]<=depth[pix[:,1],pix[:,0]]+.03);ind=ind[keep];pix=uv[ind]
        image_path=source/'rgb'/('%03d.png'%raw) if a.fullclip else source/'image/image_0'/(fid+'.png');image=np.asarray(Image.open(image_path).convert('RGB').resize((480,320),Image.Resampling.LANCZOS),dtype='float32')/255
        xyz.append(world[ind]);colors.append(image[pix[:,1],pix[:,0]]);frameids.append(np.full(len(ind),i,np.int16));records.append(dict(raw=raw,clean_returns=len(ind)));print('CLEAN_MEMORY',raw,len(ind),flush=True)
    xyz=np.concatenate(xyz);colors=np.concatenate(colors);fi=np.concatenate(frameids);origin=xyz.min(0);voxel=np.floor((xyz-origin)/.1).astype('int32');_,inverse=np.unique(voxel,axis=0,return_inverse=True);N=int(inverse.max()+1)
    # Deduplicate within each source frame before testing repeated support.
    pairs=np.unique(np.stack((inverse,fi),1),axis=0);cnt=np.bincount(pairs[:,0],minlength=N);bits=np.zeros(N,np.uint8);np.bitwise_or.at(bits,pairs[:,0],1<<np.minimum(2,pairs[:,1]*3//len(frames)))
    bitcount=np.array([bin(v).count('1') for v in range(8)])
    first=np.full(N,len(frames),np.int32);last=np.full(N,-1,np.int32);np.minimum.at(first,pairs[:,0],pairs[:,1]);np.maximum.at(last,pairs[:,0],pairs[:,1])
    # Global thirds become arbitrarily stricter when a clip is lengthened.
    # Surface-local repeated visibility: >=3 distinct frames spanning >=0.5s.
    # Original frame cadence is 10Hz. Keep global-third bits for auditing only.
    good=(cnt>=3)&((last-first)>=5)
    weight=np.bincount(inverse,minlength=N);pos=np.stack([np.bincount(inverse,weights=xyz[:,j],minlength=N)/weight for j in range(3)],1);rgb=np.stack([np.bincount(inverse,weights=colors[:,j],minlength=N)/weight for j in range(3)],1)
    variance=np.stack([np.bincount(inverse,weights=colors[:,j]**2,minlength=N)/weight-rgb[:,j]**2 for j in range(3)],1).clip(0)
    # Large appearance disagreement abstains; no invented clean background.
    good&=(variance.mean(1)<.02);pos=pos[good];rgb=rgb[good];support=cnt[good];timebits=bits[good];selected=np.zeros(len(pos),bool);query_rows=[]
    for f in query_frames:
        raw=f['raw'];uv,z=project(pos,source,f);ob=np.load(ds/('%03d.npz'%raw));body=ob['body'];domain=binary_dilation(body,iterations=5)
        ok=(z>.1)&(uv[:,0]>=0)&(uv[:,0]<480)&(uv[:,1]>=0)&(uv[:,1]<320);inds=np.flatnonzero(ok);pix=uv[inds];hit=domain[pix[:,1],pix[:,0]];inds=inds[hit];pix=uv[inds]
        # A road surface can sit at the car's base. Reject surfaces in front of
        # nearby measured target returns, allowing depth calibration tolerance.
        v=np.isfinite(ob['depth']);dist,near=distance_transform_edt(~v,return_indices=True);nearz=ob['depth'][tuple(near)];visible=(dist[pix[:,1],pix[:,0]]>5)|(z[inds]>=nearz[pix[:,1],pix[:,0]]-.3);inds=inds[visible];pix=uv[inds];selected[inds]=True
        # 3x3 surfel footprint, z-buffered. Sparse coverage is saved explicitly.
        image=np.zeros((320,480,3),np.float32);qd=np.full((320,480),np.inf,np.float32);sid=np.full((320,480),-1,np.int32)
        for dy in [-1,0,1]:
            for dx in [-1,0,1]:
                pp=pix+[dx,dy];on=(pp[:,0]>=0)&(pp[:,0]<480)&(pp[:,1]>=0)&(pp[:,1]<320);ii=inds[on];pp=pp[on];order=np.argsort(z[ii])[::-1]
                # Vector assignment can repeat pixels; sorted nearest is last.
                pp=pp[order];ii=ii[order];better=z[ii]<qd[pp[:,1],pp[:,0]];pp=pp[better];ii=ii[better];qd[pp[:,1],pp[:,0]]=z[ii];image[pp[:,1],pp[:,0]]=rgb[ii];sid[pp[:,1],pp[:,0]]=ii
        valid=np.isfinite(qd)&domain
        np.savez_compressed(a.output/('%03d.npz'%raw),rgb=image,depth=qd,valid=valid,body=body,surface_id=sid)
        Image.fromarray(np.round(image*255).astype('uint8')).save(a.output/('%03d.png'%raw));query_rows.append(dict(raw=raw,target_body_pixels=int(body.sum()),measured_background_target_pixels=int((valid&body).sum()),measured_background_domain_pixels=int(valid.sum())))
    np.savez_compressed(a.output/'all_surfaces.npz',xyz=pos.astype('float64'),rgb=rgb.astype('float32'),support_frames=support,time_bits=timebits)
    ids=np.flatnonzero(selected);np.savez_compressed(a.output/'seeds.npz',xyz=pos[ids].astype('float32'),rgb=rgb[ids].astype('float32'),sigma=np.full(len(ids),.07,np.float32),support_frames=support[ids],time_bits=timebits[ids],surface_id=ids)
    report=dict(schema='clean-world-background-memory-v2',context=CTX,source_frames=records,queries=query_rows,voxel_m=.1,clean_return_samples=len(xyz),repeated_surface_voxels=len(pos),target_related_surface_seeds=len(ids),source_requirement='>=3 distinct clean frames spanning >=0.5s per world surface; confident static semantics; excludes people/all vehicles +5px and annotated boxes in fullclip; source zbuffer; color disagreement abstention',global_thirds_retained_for_audit_only=True,training_updates=0,GPU_calls=0,pseudo_ground_truth_not_GT=True,source_reconstruction_seen_all_49_frames=True,unobserved_background_not_filled=True,wall_seconds=time.monotonic()-started)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['source_frames','queries']}),flush=True)

if __name__=='__main__':main()
