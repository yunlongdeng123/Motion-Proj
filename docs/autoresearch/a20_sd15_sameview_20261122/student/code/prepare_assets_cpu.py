"""Independent actor support and activated-teacher adapter. CPU only.

An unapproved opaque-anchor proposal is reviewable but never training-ready.
No old native-owner21 packet and no glass-return triangle patch is admitted.
"""
from pathlib import Path
import argparse
import importlib.util
import sys
import numpy as np
from common import bank, cpu_environment, read, ref, sha, synthetic_specs, validate_contract, validate_packet, write
cpu_environment()


def load_observation_helpers(path):
    spec=importlib.util.spec_from_file_location('_distill_cpu_observation',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def area_sample(mesh, count, seed):
    rng=np.random.default_rng(seed)
    ids=rng.choice(len(mesh.faces),count,p=np.asarray(mesh.area_faces)/mesh.area)
    uv=rng.random((count,2));s=np.sqrt(uv[:,0])
    bary=np.column_stack([1-s,s*(1-uv[:,1]),s*uv[:,1]])
    xyz=(np.asarray(mesh.triangles)[ids]*bary[:,:,None]).sum(1)
    return xyz,ids,bary,np.asarray(mesh.face_normals)[ids]


def normal_quaternion(normal):
    from scipy.spatial.transform import Rotation
    z=normal/np.linalg.norm(normal,axis=1)[:,None]
    auxiliary=np.broadcast_to([0.,0.,1.],z.shape).copy();auxiliary[np.abs(z[:,2])>.9]=[0.,1.,0.]
    x=np.cross(auxiliary,z);x/=np.linalg.norm(x,axis=1)[:,None];y=np.cross(z,x)
    return Rotation.from_matrix(np.stack([x,y,z],axis=2)).as_quat()[:,[3,0,1,2]]


def scene_domains(mesh, args, obs):
    result={}
    for raw in (86,110):
        mapping,arrays,refs=obs.read_frame(args.conditioning,raw)
        depth,face=obs.raster_mesh(mesh,mapping,arrays['visible_mask'].shape)
        result[raw]=dict(mapping=mapping,arrays=arrays,refs=refs,depth=depth,face=face)
    return result


def classify(xyz, domains, obs):
    protected=np.zeros(len(xyz),bool);rgb_sum=np.zeros((len(xyz),3));count=np.zeros(len(xyz))
    per_frame=[]
    for raw,frame in domains.items():
        uv,z=obs.project(xyz,frame['mapping']);arrays=frame['arrays']
        v,inside=obs.pixel_lookup(uv,arrays['visible_mask']);guard,_=obs.pixel_lookup(uv,arrays['occluder_guard_mask'],1)
        front,_=obs.pixel_lookup(uv,frame['depth'],np.inf)
        # The 7.5cm band is conservative support protection, not an accepted geometry metric.
        hit=inside&v&~guard&np.isfinite(front)&(z>.1)&(np.abs(z-front)<=.075)
        rgb,_=obs.pixel_lookup(uv,arrays['rgb'])
        rgb_sum[hit]+=rgb[hit]/255.;count[hit]+=1;protected|=hit
        per_frame.append(dict(raw=raw,protected_rows=int(hit.sum()),source=frame['refs']))
    colors=np.full((len(xyz),3),.5);seen=count>0;colors[seen]=rgb_sum[seen]/count[seen,None]
    return protected,colors,per_frame


def opaque_proposal(args, domains, obs):
    g=bank(args.grouped_returns)
    if len(g['xyz_canonical_m'])!=1030 or not bool(g['all_1030_points_retained']):
        raise ValueError('All 1030 original proxy points must remain in audit')
    good=np.zeros(1030,bool)
    for raw,f in domains.items():
        ids=np.flatnonzero((g['source_raw']==raw)&(g['RGB_region_group']==2))
        uv,z=obs.project(g['xyz_canonical_m'][ids],f['mapping'])
        visible,inside=obs.pixel_lookup(uv,f['arrays']['visible_mask'])
        guard,_=obs.pixel_lookup(uv,f['arrays']['occluder_guard_mask'],1)
        good[ids]=visible&inside&~guard&(z>.1)
    proposed=np.flatnonzero(good)
    approved=np.empty(0,np.int64)
    if args.anchor_approval:
        a=read(args.anchor_approval)
        if a.get('grouped_returns_sha256')!=sha(args.grouped_returns):
            raise ValueError('Opaque-anchor selection is bound to other data')
        approved=np.asarray(a['opaque_anchor_ids'],np.int64)
        if len(np.unique(approved))!=len(approved) or not np.isin(approved,proposed).all():
            raise ValueError('Approved anchors must be a unique subset of RGB-opaque V candidates')
    return g,proposed,approved


def initialize_generated_dc(student,teacher,c,output):
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation
    cfg=c['student']['generated_DC_initialization'];ids=np.flatnonzero(student['missing'])
    source=np.flatnonzero(teacher['opacity']>=cfg['opacity_minimum'])
    if not len(source):raise ValueError('No opacity-qualified Gaussian colors')
    distance,index=cKDTree(teacher['xyz'][source]).query(student['xyz'][ids],k=cfg['nearest_k'],workers=2)
    index=source[index];delta=student['xyz'][ids,None]-teacher['xyz'][index]
    R=Rotation.from_quat(teacher['rotation'][index.reshape(-1)][:,[1,2,3,0]]).as_matrix().reshape(len(ids),cfg['nearest_k'],3,3)
    local=np.einsum('nkji,nkj->nki',R,delta)
    maha=(local*local/(teacher['scales'][index]**2+cfg['footprint_sigma_floor_m']**2)).sum(2)
    valid=(distance<=cfg['euclidean_radius_m'])&(maha<=cfg['maximum_mahalanobis_squared'])
    weight=teacher['opacity'][index]*np.exp(-.5*maha)*valid
    fallback=weight.sum(1)<=1e-20;weight[fallback]=0;weight[fallback,0]=1
    weight/=weight.sum(1)[:,None]
    student['coefficients'][ids,0]=(teacher['coefficients'][index,0]*weight[:,:,None]).sum(1)
    p=output/'generated_DC_source.npz';np.savez_compressed(p,student_row_ids=ids,
        teacher_row_ids=index.astype(np.int32),weights=weight.astype(np.float32),
        euclidean_distance_m=distance.astype(np.float32),mahalanobis_squared=maha.astype(np.float32),fallback=fallback)
    return dict(algorithm=cfg,rows=len(ids),fallback_rows=int(fallback.sum()),
        nearest_distance_m=dict(mean=float(distance[:,0].mean()),p90=float(np.quantile(distance[:,0],.9)),maximum=float(distance[:,0].max())),
        source_mapping=ref(p),generated_teacher_geometry_copied=False,artificial_color_error_added=False)


def teacher_packet(args, mesh, domains, obs):
    from scipy.spatial.transform import Rotation
    p=args.teacher_activated
    if p.suffix=='.npz':
        a=bank(p)
    else:
        import torch
        source=torch.load(p,map_location='cpu',weights_only=False)
        a={k:(v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)) for k,v in source.items()}
        if torch.cuda.is_initialized():raise RuntimeError('CPU adapter initialized CUDA')
    n=len(a['xyz'])
    if n!=410080:raise ValueError('The exact same-SLAT decoder must yield 410080 rows')
    xyz=np.asarray(a['xyz'],np.float64);scale=np.asarray(a['scales'],np.float64)
    q=np.asarray(a['rotation_wxyz'],np.float64);dc=np.asarray(a['features_dc'],np.float64)
    if args.teacher_coordinates=='raw':
        uniform=read(args.uniform_transform)['raw_to_actor_uniform_4x4']
        bounded=read(args.registration)['relative_initial_to_registered_4x4']
        T=np.asarray(bounded)@np.asarray(uniform);s=np.linalg.norm(T[:3,:3],axis=0)
        if np.ptp(s)>1e-8 or np.linalg.det(T[:3,:3])<=0:raise ValueError('Only proper similarity transforms are allowed')
        R=T[:3,:3]/s.mean();xyz=xyz@T[:3,:3].T+T[:3,3];scale*=s.mean()
        q=Rotation.from_matrix(R[None]@Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()).as_quat()[:,[3,0,1,2]]
    elif args.teacher_coordinates!='registered_actor':
        raise ValueError('Teacher coordinate provenance must be explicit')
    if args.actor_rigid_transform:
        T=rigid_transform(args.actor_rigid_transform);xyz=xyz@T[:3,:3].T+T[:3,3]
        q=Rotation.from_matrix(T[:3,:3][None]@Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()).as_quat()[:,[3,0,1,2]]
    observed,_,visibility=classify(xyz,domains,obs)
    coeff=np.zeros((n,16,3),np.float32);coeff[:,:1]=dc
    out=dict(xyz=xyz.astype(np.float32),scales=scale.astype(np.float32),rotation=q.astype(np.float32),
        coefficients=coeff,opacity=np.asarray(a['opacity'],np.float32).reshape(-1),
        observed_protected=observed,missing=~observed,is_opaque_anchor=np.zeros(n,bool))
    validate_packet(out,allow_opacity_boundary=True);np.savez_compressed(args.output/'teacher_actor.npz',**out)
    return dict(input=ref(p),coordinates=args.teacher_coordinates,activated_once=True,rows=n,visibility=visibility,
                output=ref(args.output/'teacher_actor.npz'))


def rigid_transform(path):
    source=read(path)
    T=np.asarray(source['registered_actor_to_contact_actor_4x4'],float)
    if T.shape!=(4,4) or not np.allclose(T[3],[0,0,0,1]) or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-7) or not np.isclose(np.linalg.det(T[:3,:3]),1,atol=1e-7):
        raise ValueError('Contact adjustment must be a single proper rigid transform, without scale')
    if source.get('approved') is not True:raise ValueError('Contact transform has not been approved by root')
    return T


def main(args):
    import trimesh
    c=validate_contract(read(args.contract))
    args.output.mkdir(parents=True,exist_ok=False)
    obs=load_observation_helpers(args.observation_helpers)
    mesh=trimesh.load(args.mesh,force='mesh',process=False)
    if not np.isfinite(mesh.vertices).all() or not len(mesh.faces):raise ValueError('Invalid registered teacher mesh')
    if args.actor_rigid_transform:mesh.apply_transform(rigid_transform(args.actor_rigid_transform))
    d=scene_domains(mesh,args,obs)
    s=c['student'];xyz,face,bary,normal=area_sample(mesh,s['surface_rows'],s['sampling_seed'])
    protected,real_rgb,visibility=classify(xyz,d,obs)
    jitter=s['missing_normal_jitter_m']*np.where(np.arange(len(xyz))%2,1.,-1.)
    xyz+=normal*(jitter*(~protected))[:,None]
    tangent=np.clip(np.sqrt(mesh.area/s['surface_rows'])*.65,s['initial_tangent_sigma_min_m'],s['initial_tangent_sigma_max_m'])
    scales=np.tile([tangent,tangent,tangent*s['initial_normal_sigma_ratio']],(len(xyz),1))
    coeff=np.zeros((len(xyz),16,3),np.float32);coeff[:,0]=(real_rgb-.5)/.28209479177387814
    out=dict(xyz=xyz.astype(np.float32),scales=scales.astype(np.float32),rotation=normal_quaternion(normal).astype(np.float32),
        coefficients=coeff,opacity=np.full(len(xyz),s['initial_opacity'],np.float32),
        observed_protected=protected,missing=~protected,is_opaque_anchor=np.zeros(len(xyz),bool),
        observed_appearance_prior_geometry_candidate=protected.copy(),
        surface_face_id=face,surface_barycentric=bary.astype(np.float32),surface_normal_actor=normal.astype(np.float32))
    groups,proposal,anchors=opaque_proposal(args,d,obs)
    # Anchors are exact measured positions, not a snapped/draped prior shell.
    if len(anchors):
        p=groups['xyz_canonical_m'][anchors];_,colors,_=classify(p,d,obs)
        q=np.tile([1.,0.,0.,0.],(len(p),1));cc=np.zeros((len(p),16,3));cc[:,0]=(colors-.5)/.28209479177387814
        extra=dict(xyz=p,scales=np.full((len(p),3),.012),rotation=q,coefficients=cc,opacity=np.full(len(p),.8),
            observed_protected=np.ones(len(p),bool),missing=np.zeros(len(p),bool),is_opaque_anchor=np.ones(len(p),bool),
            observed_appearance_prior_geometry_candidate=np.zeros(len(p),bool),
            surface_face_id=np.full(len(p),-1),surface_barycentric=np.zeros((len(p),3)),surface_normal_actor=np.zeros((len(p),3)))
        for k in out:out[k]=np.concatenate([out[k],extra[k].astype(out[k].dtype)])
    teacher=None;color_init=None
    if args.teacher_activated:
        teacher=teacher_packet(args,mesh,d,obs)
        color_init=initialize_generated_dc(out,bank(args.output/'teacher_actor.npz'),c,args.output)
    validate_packet(out);np.savez_compressed(args.output/'student_init.npz',**out)
    target,tface,tbary,tnormal=area_sample(mesh,s['geometry_target_rows'],s['geometry_target_sampling_seed'])
    target_observed,_,_=classify(target,d,obs)
    np.savez_compressed(args.output/'geometry_target.npz',xyz=target.astype(np.float32),missing=~target_observed,
        face_id=tface,source_mesh_sha256=np.asarray(sha(args.mesh)))
    np.savez_compressed(args.output/'all_1030_measurements_audit.npz',**groups)
    write(args.output/'opaque_anchor_proposal.json',dict(schema='a20.opaque_anchor_proposal.v1',approved=False,
        grouped_returns=ref(args.grouped_returns),opaque_candidate_ids=proposal.tolist(),
        policy='RGB group2 and original real V outside a9 guard; no teacher-distance rejection; selection needs root qualification.',
        glass_points_excluded_from_anchor_proposal=int((groups['RGB_region_group']==1).sum()),all_points_audited=1030))
    write(args.output/'camera_specs.json',dict(synthetic=synthetic_specs(c),build=list((86,110)),
        build_resolution=[1152,768],synthetic_resolution=[512,512],heldout_opened=[]))
    from scipy.spatial import cKDTree
    dd=cKDTree(target[~target_observed]).query(out['xyz'][out['missing']])[0]
    receipt=dict(schema='a20.object_centric_distillation.assets.v1',contract=ref(args.contract),source_mesh=ref(args.mesh),
        conditioning=str(args.conditioning),independent_surface_sampling=True,teacher_gaussian_support_cloned=False,
        sampling_seed=s['sampling_seed'],rows=len(out['xyz']),surface_rows=s['surface_rows'],opaque_anchor_rows=len(anchors),
        observed_rows=int(out['observed_protected'].sum()),missing_rows=int(out['missing'].sum()),visibility=visibility,
        initial_missing_normal_offset_m=s['missing_normal_jitter_m'],artificial_error_for_formal_gain=False,
        appearance_qualification='CPU candidates only; actual native teacher depth/alpha agreement is mandatory in cache entry',
        initial_discrete_target_nearest_distance_m=dict(mean=float(dd.mean()),p90=float(np.quantile(dd,.9)),maximum=float(dd.max())),
        student_init=ref(args.output/'student_init.npz'),geometry_target=ref(args.output/'geometry_target.npz'),
        teacher=teacher,generated_DC_initialization=color_init,anchor_selection=ref(args.anchor_approval) if args.anchor_approval else None,
        actor_rigid_transform=ref(args.actor_rigid_transform) if args.actor_rigid_transform else None,
        anchors_approved=False,anchor_root_approval_required_at_GPU_entry=True,training_ready=bool(args.anchor_approval and teacher),
        gpu_used=False,training_updates=0,heldout_opened=[],all_1030_retained=True)
    write(args.output/'assets_receipt.json',receipt);print(__import__('json').dumps(receipt,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('contract','mesh','conditioning','grouped-returns','observation-helpers','output'):
        p.add_argument('--'+k,type=Path,required=True)
    for k in ('anchor-approval','teacher-activated','uniform-transform','registration','actor-rigid-transform'):
        p.add_argument('--'+k,type=Path)
    p.add_argument('--teacher-coordinates',choices=['raw','registered_actor'])
    main(p.parse_args())
