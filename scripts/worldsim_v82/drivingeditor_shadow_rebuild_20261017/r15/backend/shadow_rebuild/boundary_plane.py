"""用实际轮廓法向定位任意方向的3D切面，避免只检查最长轴漏检。"""
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation


def propose_contour_planes(state,dataset,audit_dir,road_dir,indices):
    values=lambda k:state['field.'+k].detach().cpu().numpy()
    xyz=values('xyz')[indices];s=np.exp(values('log_scales')[indices]);owner=values('owner')[indices]
    q=values('quaternion')[indices];basis=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
    cov=(basis*s[:,None,:]**2)@basis.transpose(0,2,1)
    votes=[[] for _ in indices];steps=np.linspace(-2,2,33)
    for index in dataset.train_indices:
        f=dataset.frame(index);measure=np.load(audit_dir/(f.frame_id+'.npz'))
        lookup=np.zeros((len(values('xyz')),4),np.float32);lookup[measure['rows']]=measure['evidence'];ev=lookup[indices]
        road=np.load(road_dir/(f.frame_id+'.npz'))['road'];road_distance=distance_transform_edt(~road)
        for actor in np.unique(owner):
            if actor<0 or not bool(f.valid[actor]) or not bool(f.mask_known[actor]):continue
            rows=np.flatnonzero((owner==actor)&(ev[:,0]>.02)&(ev[:,2]>.02))
            if not len(rows):continue
            mask=f.masks[...,actor].numpy();signed=distance_transform_edt(mask)-distance_transform_edt(~mask)
            gy,gx=np.gradient(signed);pose=f.poses[actor].numpy();rotation=pose[:3,:3]
            world=xyz[rows]@rotation.T+pose[:3,3]
            camera=world@f.camera.view.numpy()[:3,:3].T+f.camera.view.numpy()[:3,3]
            uv,depth=f.camera.project(torch.as_tensor(world,dtype=f.camera.K.dtype));uv=uv.numpy();depth=depth.numpy()
            h,w=road.shape;pixel=np.floor(uv+.5).astype(int);xx,yy=pixel.T
            valid=(depth>.1)&(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
            for j,row in enumerate(rows):
                if not valid[j]:continue
                x,y=xx[j],yy[j];normal=np.array([gx[y,x],gy[y,x]])
                if np.linalg.norm(normal)<.1:continue
                normal/=np.linalg.norm(normal)
                cx,cy,z=camera[j];fx,fy=float(f.camera.K[0,0]),float(f.camera.K[1,1])
                jac=np.array([[fx/z,0,-fx*cx/z**2],[0,fy/z,-fy*cy/z**2]])@f.camera.view.numpy()[:3,:3]@rotation
                plane=jac.T@normal
                std=np.sqrt(plane@cov[row]@plane)
                if std<.375 or abs(signed[y,x])>2.5*std+1:continue
                white=(basis[row].T@plane)*s[row]/std
                v=cov[row]@plane/std
                samples=(xyz[row]+steps[:,None]*v)@rotation.T+pose[:3,3]
                pix,zz=f.camera.project(torch.as_tensor(samples,dtype=f.camera.K.dtype));pix=np.floor(pix.numpy()+.5).astype(int)
                px,py=pix.T
                if (zz.numpy()<=.1).any() or (px<0).any() or (px>=w).any() or (py<0).any() or (py>=h).any():continue
                body=mask[py,px]
                if body[0]==body[-1]:continue
                crossing=np.flatnonzero(body[1:]!=body[:-1])
                if len(crossing)!=1:continue
                exterior=0 if not body[0] else -1
                if road_distance[py[exterior],px[exterior]]>3:continue
                cut=float((steps[crossing[0]]+steps[crossing[0]+1])/2)
                if abs(cut)>1.5:continue
                weight=float(min(ev[row,0],ev[row,2]))
                votes[row].append(dict(white=white,cut=cut,weight=weight,frame=f.frame_id,projected_sigma=float(std)))
    result=[];directions=[];cuts=[];provenance=[]
    for j,items in enumerate(votes):
        if len(items)<3:continue
        # Stable 3D boundary orientation in canonical coordinates; align signs before voting.
        ref=max(items,key=lambda x:x['weight'])['white'];good=[]
        for item in items:
            dot=float(item['white']@ref)
            if abs(dot)<.9:continue
            sign=1 if dot>=0 else -1
            good.append(dict(**item,aligned_white=item['white']*sign,aligned_cut=item['cut']*sign))
        if len(good)<3:continue
        weights=np.array([x['weight'] for x in good]);vv=np.array([x['aligned_cut'] for x in good]);order=np.argsort(vv)
        cut=vv[order[np.searchsorted(np.cumsum(weights[order]),weights.sum()/2)]];good=[x for x in good if abs(x['aligned_cut']-cut)<=.4]
        if len(good)<3:continue
        direction=sum(x['aligned_white']*x['weight'] for x in good);direction/=np.linalg.norm(direction)
        result.append(int(indices[j]));directions.append(direction);cuts.append(float(cut))
        provenance.append(dict(parent_row=int(indices[j]),whitened_direction=direction.tolist(),cut_sigma=float(cut),
                               evidence=[{k:v for k,v in x.items() if k not in ['white','aligned_white']} for x in good]))
    return np.array(result,np.int64),np.asarray(directions,np.float32).reshape(-1,3),np.array(cuts,np.float32),provenance
