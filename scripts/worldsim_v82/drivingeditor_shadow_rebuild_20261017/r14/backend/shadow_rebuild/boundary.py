"""多视角可见 body 边界与已认证道路支持，提出主轴切分位置。"""
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation


def propose_boundary_splits(state,dataset,audit_dir,road_dir,indices):
    """No center-only/height tests. Unknown views cannot vote exterior.

    The 3D principal axis with largest projected extent is sampled through the
    actual body mask. Require an exterior endpoint near a measured road pixel,
    one boundary crossing, >=3 independent views agreeing on the axis and a
    consistent cut. Both children remain in their original owner for re-audit.
    """
    value=lambda key:state['field.'+key].detach().cpu().numpy()
    xyz=value('xyz')[indices];scales=np.exp(value('log_scales')[indices]);owners=value('owner')[indices]
    q=value('quaternion')[indices];basis=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
    votes=[[] for _ in indices];steps=np.linspace(-2,2,33)
    for index in dataset.train_indices:
        f=dataset.frame(index);fid=f.frame_id
        audit=np.load(audit_dir/(fid+'.npz'));ev=np.zeros((len(value('xyz')),4),np.float32);ev[audit['rows']]=audit['evidence']
        ev=ev[indices]
        road=np.load(road_dir/(fid+'.npz'))['road'];distance=distance_transform_edt(~road)
        for actor in np.unique(owners):
            if actor<0 or not bool(f.valid[actor]) or not bool(f.mask_known[actor]):continue
            select=np.flatnonzero((owners==actor)&(ev[:,0]>.02)&(ev[:,2]>.02))
            if not len(select):continue
            pose=f.poses[actor].numpy();r=pose[:3,:3];t=pose[:3,3]
            centers=xyz[select]@r.T+t
            directions=np.einsum('ij,njk->nik',r,basis[select])
            world=centers[:,None,None,:]+np.einsum('na,nca,s->nasc',scales[select],directions,steps)
            uv,depth=f.camera.project(__import__('torch').as_tensor(world.reshape(-1,3),dtype=f.camera.K.dtype))
            uv=uv.numpy().reshape(len(select),3,33,2);depth=depth.numpy().reshape(len(select),3,33)
            extent=np.linalg.norm(uv[:,:,24]-uv[:,:,8],axis=-1);axes=extent.argmax(-1)
            for j,row in enumerate(select):
                axis=axes[j]
                if extent[j,axis]<1.:continue
                pixels=np.floor(uv[j,axis]+.5).astype(int);xx,yy=pixels.T;h,w=road.shape
                if (depth[j,axis]<=.1).any() or (xx<0).any() or (xx>=w).any() or (yy<0).any() or (yy>=h).any():continue
                body=f.masks[...,actor].numpy()[yy,xx]
                if body[0]==body[-1]:continue
                changes=np.flatnonzero(body[1:]!=body[:-1])
                if len(changes)!=1:continue
                exterior=0 if not body[0] else -1
                if distance[yy[exterior],xx[exterior]]>3:continue
                cut=float((steps[changes[0]]+steps[changes[0]+1])/2)
                if abs(cut)>1.5:continue
                weight=float(min(ev[row,0],ev[row,2]))
                votes[row].append((int(axis),cut,weight,fid))
    rows=[];axes=[];cuts=[];provenance=[]
    for i,items in enumerate(votes):
        if not items:continue
        axis=max(range(3),key=lambda a:sum(x[2] for x in items if x[0]==a))
        chosen=[x for x in items if x[0]==axis]
        if len(chosen)<3:continue
        vals=np.array([x[1] for x in chosen]);weights=np.array([x[2] for x in chosen]);order=np.argsort(vals)
        cut=vals[order[np.searchsorted(np.cumsum(weights[order]),weights.sum()/2)]]
        consistent=np.abs(vals-cut)<=.4
        if consistent.sum()<3:continue
        rows.append(int(indices[i]));axes.append(axis);cuts.append(float(cut))
        provenance.append(dict(parent_row=int(indices[i]),axis=axis,cut_sigma=float(cut),
                               evidence=[dict(frame=x[3],cut_sigma=x[1],alphaT_weight=x[2]) for x,ok in zip(chosen,consistent) if ok]))
    return np.array(rows,np.int64),np.array(axes,np.int64),np.array(cuts,np.float32),provenance
