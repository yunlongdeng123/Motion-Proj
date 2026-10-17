"""Independent evidence state is separate from rendering owner and opacity."""
import numpy as np
import torch
from .math3d import quaternion_matrix

def evidence_probability(body_repeat,static_positive,tracked_actor,tracked_static):
    """Dirichlet evidence scores, not a calibrated ownership ground-truth probability.

    Missing/ambiguous motion supplies no static vote. Contradictions increase the
    unknown component. The hard owner decision still requires positive3D evidence.
    """
    arrays=np.broadcast_arrays(body_repeat,static_positive,tracked_actor,tracked_static)
    body,road,moving,stationary=[x.astype(bool) for x in arrays]
    evidence=np.zeros(body.shape+(3,),np.float32)+[1,1,2]
    evidence[...,0]+=7*road+10*stationary
    evidence[...,1]+=5*body+10*moving
    conflict=(road|stationary)&(body|moving)
    evidence[...,2]+=np.where(conflict,np.minimum(evidence[...,0],evidence[...,1]),0)
    return evidence/evidence.sum(-1,keepdims=True)

@torch.no_grad()
def reassign_rigid_owner(field,rows,new_owner,source_world_from_local,target_world_from_local,optimizer=None):
    """Change actual rigid owner while preserving world geometry at an evidence anchor.

    This does not certify ownership: caller must supply independent repeated
    measurements, semantic support and a recorded hysteresis gate. Anchors are
    real poses, not arbitrary time-freezing. Changed Adam rows are reset because
    their coordinate/identity interpretation changed; unaffected rows remain.
    """
    rows=torch.as_tensor(rows,dtype=torch.long,device=field.xyz.device)
    if not len(rows):return
    if not isinstance(new_owner,(int,np.integer)) or new_owner < -1:raise ValueError('unknown evidence is not a rigid background owner')
    def pose(value):
        p=torch.as_tensor(value,dtype=field.xyz.dtype,device=field.xyz.device)
        if p.shape!=(4,4) or not torch.allclose(p[3],p.new_tensor([0,0,0,1]),atol=1e-5) or not torch.allclose(p[:3,:3].T@p[:3,:3],torch.eye(3,device=p.device),atol=1e-4) or not torch.isclose(torch.linalg.det(p[:3,:3]),p.new_tensor(1.),atol=1e-4):raise ValueError('actual rigid poses required')
        return p
    src=pose(source_world_from_local);dst=pose(target_world_from_local);rotation=dst[:3,:3].T@src[:3,:3];translation=dst[:3,:3].T@(src[:3,3]-dst[:3,3])
    for key in ['xyz','seed_xyz','anchor_xyz']:getattr(field,key)[rows]=getattr(field,key)[rows]@rotation.T+translation
    from scipy.spatial.transform import Rotation
    for key in ['quaternion','anchor_quaternion']:
        old=quaternion_matrix(getattr(field,key)[rows]);matrix=rotation[None]@old
        q=Rotation.from_matrix(matrix.cpu().numpy()).as_quat()[:,[3,0,1,2]]
        getattr(field,key)[rows]=torch.as_tensor(q,device=field.xyz.device,dtype=field.xyz.dtype)
    field.owner[rows]=new_owner;field.road[rows]=False;field.reference_confidence[rows]=0
    field.view_coeff[rows]=0;field.features_rest[rows]=0
    if optimizer is not None:
        for name in field.point_parameters:
            state=optimizer.state.get(getattr(field,name),{})
            for key,value in state.items():
                if torch.is_tensor(value) and value.ndim and len(value)==len(field.xyz):value[rows]=0
    # Rebuild same-owner neighborhoods; never keep cross-owner normal links.
    from scipy.spatial import cKDTree
    owner=field.owner.cpu().numpy();points=field.xyz.detach().cpu().numpy()
    for actor in np.unique(owner):
        ids=np.flatnonzero(owner==actor);near=cKDTree(points[ids]).query(points[ids],k=min(2,len(ids)),workers=1)[1]
        field.neighbor[torch.as_tensor(ids,device=field.xyz.device)]=torch.as_tensor(ids[near[:,1] if near.ndim==2 else near],device=field.xyz.device)

@torch.no_grad()
def append_measured_candidates(field,seed,optimizer):
    """Admit positively certified real measurements from the external unknown pool.

    The caller supplies the evidence gate and frozen BUILD provenance. This
    function only makes that decision concrete in the renderer and optimizer.
    It never recovers an unobserved point by inventing a box seed, or changes
    opacity on existing points to make an asset appear cleaner.
    """
    from .field import GaussianField
    required={'source_frame','source_raw_row','source_world','source_candidate_owner',
              'identity_probability','identity_evidence_status'}
    if not required.issubset(seed):raise ValueError('actual measurement/evidence provenance required')
    n=len(seed['xyz'])
    if n==0:return
    if np.any(np.asarray(seed['source_frame'])<0) or np.any(np.asarray(seed['source_raw_row'])<0):raise ValueError('unknown source is not a measured candidate')
    if np.any(np.asarray(seed['owner'])<0) or np.any(np.asarray(seed['identity_evidence_status'])!=1):raise ValueError('this operation only admits positively certified Actor measurements')
    probability=np.asarray(seed['identity_probability'])
    if probability.shape!=(n,3) or not np.isfinite(probability).all() or not np.allclose(probability.sum(-1),1,atol=1e-5):raise ValueError('normalized evidence distribution required')
    count=len(field.xyz);device=field.xyz.device
    added=GaussianField(seed,field.config).to(device)
    added.sgcm_lineage+=int(field.sgcm_lineage.max())+1
    point_buffers={name:buffer for name,buffer in field.named_buffers() if buffer.ndim and len(buffer)==count}
    for name,buffer in point_buffers.items():
        if not hasattr(added,name) or getattr(added,name).shape[1:]!=buffer.shape[1:]:raise ValueError('candidate buffer mismatch: '+name)
    replacements={}
    for name in field.point_parameters:
        old=getattr(field,name);new=torch.nn.Parameter(torch.cat((old.detach(),getattr(added,name).detach()),0),requires_grad=old.requires_grad)
        setattr(field,name,new);replacements[old]=new
    for name,buffer in point_buffers.items():
        extra=getattr(added,name).clone()
        if name=='neighbor':extra+=count
        setattr(field,name,torch.cat((buffer,extra),0))
    for group in optimizer.param_groups:
        for i,old in enumerate(group['params']):
            if old not in replacements:continue
            new=replacements[old];group['params'][i]=new;state=optimizer.state.pop(old,{})
            optimizer.state[new]={key:torch.cat((value,torch.zeros((n,)+value.shape[1:],dtype=value.dtype,device=value.device)),0)
                                  if torch.is_tensor(value) and value.ndim and len(value)==count else value
                                  for key,value in state.items()}
    # Admission changes local adjacency, but preserves all learned old parameters,
    # point evidence, optimizer moments and global non-point state exactly.
    from scipy.spatial import cKDTree
    owner=field.owner.cpu().numpy();points=field.xyz.detach().cpu().numpy()
    for actor in np.unique(owner):
        ids=np.flatnonzero(owner==actor);near=cKDTree(points[ids]).query(points[ids],k=min(2,len(ids)),workers=1)[1]
        field.neighbor[torch.as_tensor(ids,device=device)]=torch.as_tensor(ids[near[:,1] if near.ndim==2 else near],device=device)
