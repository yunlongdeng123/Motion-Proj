"""CPU mechanism audit inspired by GaussianPro (ICML 2024), not its CUDA reproduction.

Controls: old V41 labels; fixed-surface same-verifier; image-neighbor plane
propagation; metric road-neighbor plane propagation with the SAME verifier.
All geometry, visibility, colors and denominators remain inspectable. No optimizer.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree,Delaunay
from scipy.ndimage import map_coordinates,distance_transform_edt,binary_dilation

H,W=320,480
POLICY=dict(plane_candidates_image=2,plane_candidates_metric=2,metric_radius_m=3.,image_radius_px=96.,
    patch_radius_px=2,patch_stride_px=2,ncc_min=.8,low_texture_std=.003,color_mae_max=.05,
    geometry_relative_depth=.01,geometry_roundtrip_px=2.,measured_depth_abs_m=.3,
    triangle_edge_m=1.6,triangle_edge_px=24.,triangle_normal_cos=.9,
    source_separation_frames=2,fit_validation_split='alternating blocks of five original frames',
    candidate_selection='fit sources only; validation never selects a plane',
    low_texture='no photometric depth identification; allow only two independent measured geometry witnesses + color agreement',
    pixel_convention='K inverse at pixel +0.5; projection minus0.5')

def transform(x,M):return x@M[:3,:3].T+M[:3,3]
def project(x,f):
    c=transform(x,f['w2l']);p=c@f['K'].T
    return p[:,:2]/np.maximum(c[:,2:],1e-8)-.5,c[:,2]
def sample(im,uv,order=1):
    if im.ndim==3:return np.stack([sample(im[...,k],uv,order) for k in range(im.shape[-1])],-1)
    return map_coordinates(im.astype(float),[uv[:,1],uv[:,0]],order=order,mode='constant',cval=0)
def rays(f,uv):
    c2l=np.linalg.inv(f['w2l']);r=np.c_[uv+.5,np.ones(len(uv))]@np.linalg.inv(f['K']).T@c2l[:3,:3].T
    return c2l[:3,3],r
def intersect(origin,rays_,point,normal):
    den=np.sum(rays_*normal,axis=-1)
    t=np.divide(np.sum((point-origin)*normal,axis=-1),den,out=np.full(len(rays_),-1.),where=np.abs(den)>1e-8)
    return origin+t[:,None]*rays_,t

def measured_maps(f,plane):
    """Independent of original grid support: nearest-return, small ground triangles."""
    uv,z=project(f['cloud'],f);inside=(z>.1)&(uv[:,0]>=0)&(uv[:,0]<W-1)&(uv[:,1]>=0)&(uv[:,1]<H-1)
    ids=np.flatnonzero(inside);pix=np.floor(uv[ids]+.5).astype(int);order=np.argsort(z[ids],kind='stable')
    _,first=np.unique(pix[order,1]*W+pix[order,0],return_index=True);near=ids[order[first]]
    yy,xx=np.mgrid[:H,:W];q=np.c_[xx.ravel(),yy.ravel()]
    vv=near[f['ground'][near]];tri=Delaunay(uv[vv]);si=tri.find_simplex(q);safe=np.maximum(si,0);vertices=vv[tri.simplices[safe]]
    mat=tri.transform[safe];bc=np.einsum('nij,nj->ni',mat[:,:2],q-mat[:,2]);bc=np.c_[bc,1-bc.sum(1)]
    dep=1/np.maximum(np.sum(bc/z[vertices],1),1e-9)
    normals=np.cross(f['cloud'][vertices[:,1]]-f['cloud'][vertices[:,0]],f['cloud'][vertices[:,2]]-f['cloud'][vertices[:,0]])
    nn=np.linalg.norm(normals,axis=1);normals/=np.maximum(nn[:,None],1e-9)
    normals*=np.where(normals@plane[:3]<0,-1.,1.)[:,None]
    we=np.stack([np.linalg.norm(f['cloud'][vertices[:,a]]-f['cloud'][vertices[:,b]],axis=1) for a,b in [(0,1),(1,2),(2,0)]],1)
    pe=np.stack([np.linalg.norm(uv[vertices[:,a]]-uv[vertices[:,b]],axis=1) for a,b in [(0,1),(1,2),(2,0)]],1)
    dist,hit=cKDTree(uv[near]).query(q);hit=near[hit]
    conflict=(dist<=1.5)&(~f['ground'][hit]|(np.abs(z[hit]-dep)>.3))
    valid=(si>=0)&(nn>1e-8)&(we.max(1)<=1.6)&(pe.max(1)<=24)&(np.abs(normals@plane[:3])>=.9)&~conflict&~f['guard'].ravel()
    # All RGB patch samples also need to be outside protected silhouettes.
    f['depth']=np.where(valid,dep,0).reshape(H,W).astype(np.float32)
    f['measured_valid']=valid.reshape(H,W)
    f['patch_clear']=~binary_dilation(f['guard'],iterations=3)
    return f

def source_witness(x,f,target,uv_target):
    uv,z=project(x,f);valid=sample(f['measured_valid'],uv)>=1-1e-6
    dep=sample(f['depth'],uv)
    origin,rr=rays(f,uv);actual=origin+dep[:,None]*rr
    back,zback=project(actual,target)
    _,ztarget=project(x,target)
    valid&=(z>.1)&(np.abs(dep-z)<=.3)&(np.linalg.norm(back-uv_target,axis=1)<2.)&(np.abs(zback-ztarget)/np.maximum(ztarget,.1)<.01)
    valid&=sample(f['patch_clear'],uv)>=1-1e-6
    return valid,uv

def pair_patch(x,n,f,g):
    """Plane homography implemented as ray/plane intersections, no target-car RGB."""
    uv,_=project(x,f);offset=np.array([(a,b) for a in [-2,0,2] for b in [-2,0,2]])
    points=(uv[:,None]+offset).reshape(-1,2);origin,rr=rays(f,points)
    xyz,t=intersect(origin,rr,np.repeat(x,9,axis=0),np.repeat(n,9,axis=0))
    uv2,z2=project(xyz,g)
    aa=sample(f['rgb'],points).reshape(-1,9,3);bb=sample(g['rgb'],uv2).reshape(-1,9,3)
    clear=(sample(f['patch_clear'],points)>=1-1e-6).reshape(-1,9).all(1)&(sample(g['patch_clear'],uv2)>=1-1e-6).reshape(-1,9).all(1)
    clear&=(t.reshape(-1,9)>.1).all(1)&(z2.reshape(-1,9)>.1).all(1)
    gray_a=aa.mean(2);gray_b=bb.mean(2);a=gray_a-gray_a.mean(1,keepdims=True);b=gray_b-gray_b.mean(1,keepdims=True)
    std_a=np.sqrt(np.mean(a*a,1));std_b=np.sqrt(np.mean(b*b,1))
    ncc=np.mean(a*b,1)/np.maximum(std_a*std_b,1e-9);mae=np.abs(aa-bb).mean((1,2))
    low=(std_a<POLICY['low_texture_std'])|(std_b<POLICY['low_texture_std'])
    accept=clear&(mae<=.05)&((ncc>=.8)|low)
    return accept,mae,ncc,low,aa[:,4],bb[:,4]

def verify(x,n,valid,target_i,target_uv,frames,fit):
    """First geometry-qualified source pair, fit-score selection; disjoint validation."""
    count=len(x);source=np.full((count,3),-1,np.int16);geo=np.zeros(count,np.uint8)
    for j,f in enumerate(frames):
        if j==target_i:continue
        active=np.flatnonzero(valid)
        if not len(active):continue
        good,_=source_witness(x[active],f,frames[target_i],target_uv[active]);ids=active[good];geo[ids]+=1
        if fit[j]:
            first=ids[source[ids,0]<0];source[first,0]=j
            second=ids[(source[ids,0]>=0)&(source[ids,1]<0)&(j-source[ids,0]>=2)];source[second,1]=j
        else:
            third=ids[source[ids,2]<0];source[third,2]=j
    accepted=np.zeros(count,bool);score=np.full(count,np.inf);low=np.zeros(count,bool);color=np.zeros((count,3),np.float32)
    validation=np.zeros(count,bool);validation_error=np.full(count,np.nan)
    candidates=np.flatnonzero(source[:,1]>=0)
    pairs=np.unique(source[candidates,:2],axis=0)
    for a,b in pairs:
        ids=candidates[(source[candidates,0]==a)&(source[candidates,1]==b)]
        ok,mae,ncc,l,c1,c2=pair_patch(x[ids],n[ids],frames[a],frames[b])
        accepted[ids]=ok;score[ids]=mae+.02*np.maximum(0,1-ncc);low[ids]=l;color[ids]=(c1+c2)/2
    # Validation does not change accepted, candidate score, or selected plane.
    for j in np.unique(source[:,2]):
        if j<0:continue
        ids=np.flatnonzero(accepted&(source[:,2]==j));uv,_=project(x[ids],frames[j]);c=sample(frames[j]['rgb'],uv)
        err=np.abs(color[ids]-c).mean(1);validation[ids]=err<=.05;validation_error[ids]=err
    return dict(accepted=accepted,score=score,source=source,geo=geo,color=color,low=low,validation=validation,validation_error=validation_error)

def refine_planes(x,n,valid,target_i,target_uv,frames,fit):
    """One bounded fallback: source-measured tangent-plane correction, not gate relaxation.

    Two rounds. Only fit-block unmasked ground patches inform the displacement;
    independent validation sources and target RGB are never used. Final verifier
    remains identical. This does not produce observations behind every car.
    """
    initial=x.copy();origin,rr=rays(frames[target_i],target_uv)
    for _ in range(2):
        residual=[]
        for j,f in enumerate(frames):
            if j==target_i or not fit[j]:continue
            uv,z=project(x,f);dep=sample(f['depth'],uv)
            ok=valid&(sample(f['measured_valid'],uv)>=1-1e-6)&(sample(f['patch_clear'],uv)>=1-1e-6)&(z>.1)&(np.abs(dep-z)<.6)
            o,r=rays(f,uv);measured=o+dep[:,None]*r
            delta=np.sum((measured-x)*n,1)
            residual.append(np.where(ok,delta,np.nan))
        residual=np.asarray(residual);count=np.isfinite(residual).sum(0)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning);correction=np.nanmedian(residual,axis=0)
        correction=np.where(count>=2,np.clip(np.nan_to_num(correction),-.1,.1),0)
        pos,t=intersect(origin,rr,x+n*correction[:,None],n)
        within=(np.abs(np.sum((pos-initial)*n,1))<=.2)&(np.linalg.norm(pos[:,:2]-initial[:,:2],axis=1)<=2.)&(t>.1)&(t<120)
        x=np.where(within[:,None],pos,x)
    return x

def run(a):
    start=time.perf_counter()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);(a.output/'frames').mkdir()
    meta=json.loads((a.inputs/'manifest.json').read_text());u=dict(np.load(a.inputs/'units.npz'))
    plane=np.asarray(meta['ground_plane']);plane=plane/np.linalg.norm(plane[:3])
    old_observed=u['surface_supported']&((u['base_weight']>0).sum(0)>=2)
    seeds=u['local'][old_observed];normals=u['normal_local'][old_observed];normals/=np.linalg.norm(normals,axis=1)[:,None]
    tree=cKDTree(seeds[:,:2]);frames=[]
    for i,fid in enumerate(meta['frame_order']):
        f=dict(np.load(a.inputs/f'{fid}.npz'));frames.append(measured_maps(f,plane))
        print(json.dumps(dict(stage='measured_maps',frame=fid,valid=int(frames[-1]['measured_valid'].sum()))),flush=True)
    fit=(np.arange(len(frames))//5)%2==0
    methods=['fixed_same_verifier','image_plane','metric_plane']+(['measured_refined'] if a.refine else [])
    (a.output/'protocol.json').write_text(json.dumps(dict(policy=POLICY,seed_count=len(seeds),
        controls=methods,frozen_before_results=True,refinement_enabled=a.refine,
        refinement_contract='two source-geometry corrections, fit views only, 0.1m per round/0.2m total normal displacement; unchanged final verifier',
        critical_frame='0008102',critical_pixels=[[255,240],[256,219]],all_target_pixels=(a.stride==1),
        primary_question='Does metric plane proposal improve a37/a6 over fixed geometry and screen-plane control?',
        stopping_rule='Do not tune thresholds on these results; retain negative critical-region findings.'),indent=2))
    rows=[];critical=[];all_births=[];all_sources=[];all_normals=[];all_colors=[]
    for i,(fid,f) in enumerate(zip(meta['frame_order'],frames)):
        yy,xx=np.where(f['actor_labels']>0);uv=np.c_[xx,yy].astype(float)
        # Optional development subset, never confused with full-pixel final run.
        if a.stride>1:uv=uv[(xx%a.stride==0)&(yy%a.stride==0)]
        if fid=='0008102':uv=np.unique(np.r_[uv,[[255,240],[256,219]]],axis=0)
        px=uv.astype(int);actor=f['actor_labels'][px[:,1],px[:,0]];origin,rr=rays(f,uv)
        base,t=intersect(origin,rr,np.broadcast_to(-plane[3]*plane[:3],(len(uv),3)),np.broadcast_to(plane[:3],(len(uv),3)))
        supported=f['supported'][px[:,1],px[:,0]];fixed_depth=f['fixed_surface_depth_m'][px[:,1],px[:,0]]
        fixed=origin+fixed_depth[:,None]*rr;fixed=np.where(supported[:,None],fixed,base)
        base_valid=(t>.1)&(t<120)&(base[:,0]>=3)&(base[:,0]<=90)&(np.abs(base[:,1])<=8)
        # Control A: original supported surface, no old R eligibility gate, same verifier.
        uid=f['unit_id'][px[:,1],px[:,0]]
        fixed_normal=u['normal_local'][np.maximum(uid,0)]
        proposals=[fixed];pn=[fixed_normal];valids=[base_valid&supported];names=['fixed_same_verifier']
        # GaussianPro-derived image-plane control: screen-neighbor plane hypotheses.
        suv,sz=project(seeds,f);inside=(sz>.1)&(suv[:,0]>=0)&(suv[:,0]<W)&(suv[:,1]>=0)&(suv[:,1]<H)
        ss=np.flatnonzero(inside);sd,si=cKDTree(suv[ss]).query(uv,k=2);si=ss[si]
        for k in range(2):
            pos,tt=intersect(origin,rr,seeds[si[:,k]],normals[si[:,k]])
            proposals.append(pos);pn.append(normals[si[:,k]]);valids.append(base_valid&(tt>.1)&(tt<120)&(sd[:,k]<=96));names.append('image_plane')
        # Improvement: metric neighbors propagate reliable ROAD planes into an occluded
        # query ray, without interpreting a car-shaped rendered depth as a road seed.
        md,mi=tree.query(base[:,:2],k=2)
        for k in range(2):
            pos,tt=intersect(origin,rr,seeds[mi[:,k]],normals[mi[:,k]])
            proposals.append(pos);pn.append(normals[mi[:,k]]);valids.append(base_valid&(tt>.1)&(tt<120)&(md[:,k]<=3.)&(np.linalg.norm(pos[:,:2]-base[:,:2],axis=1)<=1.));names.append('metric_plane')
        if a.refine:
            for k in [3,4]:
                pos=refine_planes(proposals[k].copy(),pn[k],valids[k],i,uv,frames,fit)
                proposals.append(pos);pn.append(pn[k]);valids.append(valids[k]);names.append('measured_refined')
        xyz=np.concatenate(proposals);ns=np.concatenate(pn);vs=np.concatenate(valids);vuv=np.tile(uv,(len(names),1))
        result=verify(xyz,ns,vs,i,vuv,frames,fit);N=len(uv)
        old=f['deletion_valid'][px[:,1],px[:,0]]
        out=dict(pixels=px,actor=actor,legacy=old,supported=supported)
        selected={}
        for name in methods:
            ks=[k for k,s in enumerate(names) if s==name]
            ids=np.array(ks)[:,None]*N+np.arange(N)
            sc=np.where(result['accepted'][ids],result['score'][ids],np.inf)
            take=np.argmin(sc,axis=0);choose=ids[take,np.arange(N)];accepted=np.isfinite(sc.min(0))
            selected[name]=(choose,accepted)
            out[name]=accepted;out[name+'_validation']=result['validation'][choose]&accepted
            out[name+'_testable']=(result['source'][choose,2]>=0)&accepted
            out[name+'_xyz']=xyz[choose].astype(np.float32);out[name+'_sources']=result['source'][choose]
            out[name+'_normal']=ns[choose].astype(np.float32)
            out[name+'_rgb']=result['color'][choose];out[name+'_geo_count']=result['geo'][choose]
            out[name+'_validation_error']=result['validation_error'][choose];out[name+'_low_texture']=result['low'][choose]&accepted
            out[name+'_proposal_valid']=np.logical_or.reduce([valids[k] for k in ks])
            out[name+'_added_vs_legacy']=accepted&~old
            out[name+'_union_legacy']=accepted|old
        # Strong incremental test: same surface verifier vs actual position change.
        added=out['metric_plane']&~out['fixed_same_verifier'];new_geometry=added&~supported
        out['metric_added_over_fixed']=added;out['metric_added_unsupported']=new_geometry
        all_births.append(out['metric_plane_xyz'][added]);all_sources.append(out['metric_plane_sources'][added])
        all_normals.append(out['metric_plane_normal'][added]);all_colors.append(out['metric_plane_rgb'][added])
        np.savez_compressed(a.output/'frames'/f'{fid}.npz',**out)
        record=dict(frame=fid,by_actor={})
        for k,act in enumerate(meta['actors']):
            mask=actor==k+1
            row={'denominator':int(mask.sum()),'legacy':int((mask&old).sum()),'original_supported':int((mask&supported).sum())}
            for name in selected:
                for suffix in ['', '_testable','_validation','_low_texture','_added_vs_legacy','_union_legacy']:
                    row[name+suffix]=int((mask&out[name+suffix]).sum())
            row['metric_added_over_fixed']=int((mask&added).sum());row['metric_added_unsupported']=int((mask&new_geometry).sum());record['by_actor'][str(act)]=row
        rows.append(record)
        if fid=='0008102':
            for point in [[255,240],[256,219]]:
                q=np.flatnonzero((px==point).all(1))[0]
                critical.append(dict(pixel=point,actor_label_index=int(actor[q]),legacy=bool(old[q]),original_supported=bool(supported[q]),methods={name:dict(accepted=bool(accept[q]),local_xyz=xyz[choose[q]].tolist(),geometry_sources=int(result['geo'][choose[q]]),sources=result['source'][choose[q]].tolist(),validation=bool(result['validation'][choose[q]])) for name,(choose,accept) in selected.items()}))
        print(json.dumps(dict(stage='frame',frame=fid,by_actor=record['by_actor'])),flush=True)
    totals={}
    for act in map(str,meta['actors']):totals[act]={key:sum(r['by_actor'][act][key] for r in rows) for key in rows[0]['by_actor'][act]}
    births=np.concatenate(all_births);sources=np.concatenate(all_sources);birth_normals=np.concatenate(all_normals);birth_rgb=np.concatenate(all_colors)
    if len(births):
        _,ii=np.unique(np.floor(births/.1).astype(int),axis=0,return_index=True);births=births[ii];sources=sources[ii];birth_normals=birth_normals[ii];birth_rgb=birth_rgb[ii]
    np.savez_compressed(a.output/'proposed_births.npz',local_xyz=births,source_indices=sources,normal_local=birth_normals,rgb=birth_rgb)
    summary=dict(version='V42_CPU_PLANE_PROPAGATION_R1',input_manifest=str(a.inputs/'manifest.json'),policy=POLICY,stride=a.stride,actors=totals,per_frame=rows,critical=critical,additional_birth_voxels_10cm=len(births),elapsed_seconds=time.perf_counter()-start,GPU_used=False,optimizer_updates=0,quality_accepted=False,human_verdict=None,
        baseline_scope='GaussianPro-inspired CPU plane hypotheses + NCC + geometric reprojection, not official CUDA ACMH/full training baseline',
        limitations=['All49 is a development set. Disjoint source blocks validate labels, not an independent scene test.',
        'Candidate seed geometry inherited from all49; no independent geometry holdout.',
        'Source first pair is chosen by geometry, not best RGB; a third disjoint-block view is verification only.',
        'Low-texture acceptance relies on measured geometry and RGB agreement; NCC does not identify its depth.',
        'Proposal coverage is not renderer contribution, clean deletion, or reconstruction quality.',
        'Unsupported/unseen locations without witnesses remain unlabelled. Proposed births are not trained Gaussians.',
        'The image control uses nearest projected road planes, not the official iterative checkerboard/random-search CUDA implementation.'])
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps({'totals':totals,'critical':critical,'elapsed':summary['elapsed_seconds']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stride',type=int,default=1);p.add_argument('--refine',action='store_true');run(p.parse_args())
