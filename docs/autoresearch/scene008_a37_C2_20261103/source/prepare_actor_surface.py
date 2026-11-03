"""Merge real a6 geometry with ONE learned Omni surface; CPU only, no fitting.

Example:
  python prepare_actor_surface.py --mesh source/prior/proposals/seed000/candidate_omni.glb
  python prepare_actor_surface.py --mesh candidate_actor_native.ply --mesh-space actor_native_m
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a37_C2_20261103_r1')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def atomic_json(path, value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)

def atomic_npz(path, **arrays):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()

def unit(v, fallback=None):
    n=np.linalg.norm(v,axis=1);good=n>1e-12;out=np.zeros_like(v,dtype=float)
    out[good]=v[good]/n[good,None]
    if fallback is not None:out[~good]=fallback[~good]
    else:out[~good]=[0,0,1]
    return out

def real_anchors(points, voxel_m=.05):
    keys,inverse=np.unique(np.floor(points/voxel_m).astype(np.int32),axis=0,return_inverse=True)
    count=np.bincount(inverse,minlength=len(keys))
    xyz=np.column_stack([np.bincount(inverse,weights=points[:,j],minlength=len(keys))/count for j in range(3)])
    tree=cKDTree(xyz);distance,index=tree.query(xyz,k=min(16,len(xyz)))
    if index.ndim==1:index=index[:,None];distance=distance[:,None]
    q=xyz[index]-xyz[:,None]
    weight=(distance<=.18).astype(float);weight[:,0]=1
    mean=(q*weight[:,:,None]).sum(1)/np.maximum(weight.sum(1),1)[:,None]
    centered=q-mean[:,None]
    covariance=np.einsum('nki,nkj,nk->nij',centered,centered,weight)/np.maximum(weight.sum(1),1)[:,None,None]
    values,vectors=np.linalg.eigh(covariance);normals=vectors[:,:,0]
    outward=unit(xyz);flip=(normals*outward).sum(1)<0;normals[flip]*=-1
    normal_confidence=np.clip((values[:,1]-values[:,0])/np.maximum(values[:,2],1e-10),0,1)
    weak=(weight.sum(1)<4)|(values[:,1]<1e-7);normals[weak]=outward[weak];normal_confidence[weak]=0
    spacing=distance[:,min(3,distance.shape[1]-1)]
    sigma=np.clip(spacing*.60,.020,.060)
    return xyz,normals,sigma,normal_confidence,inverse,keys

def load_mesh(path, space, half_span):
    import trimesh
    loaded=trimesh.load(str(path),process=False)
    if isinstance(loaded,trimesh.Scene):
        loaded=loaded.to_mesh() if hasattr(loaded,'to_mesh') else loaded.dump(concatenate=True)
    if not isinstance(loaded,trimesh.Trimesh):raise ValueError('Input must contain a learned triangular mesh')
    vertices=np.asarray(loaded.vertices,dtype=float);faces=np.asarray(loaded.faces,dtype=np.int64)
    if space=='omni_normalized':
        vertices=vertices[:,[0,2,1]]*half_span
        # y/z exchange has determinant -1. Reverse winding to preserve outward normals.
        faces=faces[:,[0,2,1]]
    if len(vertices)==0 or len(faces)==0 or not np.isfinite(vertices).all():raise ValueError('Empty/nonfinite learned mesh')
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
    keep=np.asarray(mesh.area_faces)>1e-12
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces[keep],process=False)
    if not len(mesh.faces):raise ValueError('Learned mesh has no positive-area surface')
    return mesh

def sample_surface(mesh, count, rng):
    area=np.asarray(mesh.area_faces);face=np.searchsorted(np.cumsum(area),rng.random(count)*area.sum(),side='right')
    root=np.sqrt(rng.random(count));v=rng.random(count);bary=np.c_[1-root,root*(1-v),root*v]
    xyz=np.einsum('ni,nij->nj',bary,np.asarray(mesh.triangles)[face])
    normal=unit(np.einsum('ni,nij->nj',bary,np.asarray(mesh.vertex_normals)[np.asarray(mesh.faces)[face]]),np.asarray(mesh.face_normals)[face])
    return xyz,normal,face,bary

def exact_mesh_distances(mesh, points):
    """CPU exact closest triangle using centroid spheres, no rtree dependency.

    A triangle capable of improving distance d must have centroid within d+Rmax.
    Querying that ball proves completeness, then closest_point is exact per candidate.
    """
    import trimesh
    triangles=np.asarray(mesh.triangles);centers=triangles.mean(1)
    radius=np.linalg.norm(triangles-centers[:,None],axis=2).max(1);rmax=float(radius.max())
    tree=cKDTree(centers);k=min(16,len(triangles));_,near=tree.query(points,k=k)
    if near.ndim==1:near=near[:,None]
    queries=np.repeat(points,near.shape[1],axis=0)
    closest=trimesh.triangles.closest_point(triangles[near.reshape(-1)],queries)
    best=np.linalg.norm(closest-queries,axis=1).reshape(len(points),-1).min(1)
    # Grossly wrong proposals are diagnostic failures. Avoid an expensive global
    # triangle search merely to confirm metre-scale mismatch already certified by
    # the mesh-AABB distance lower bound reported separately.
    if np.quantile(best,.9)>.30:
        return best,dict(method='closest of16 nearest-centroid actual triangles; surface-distance UPPER BOUND only, gross-mismatch early exit',
            exact=False,queries=len(points),max_triangle_centroid_radius_m=rmax)
    exact=best.copy();candidate_pairs=0;max_candidates=0
    for begin in range(0,len(points),64):
        stop=min(begin+64,len(points));balls=tree.query_ball_point(points[begin:stop],best[begin:stop]+rmax+1e-9)
        lengths=np.array([len(x) for x in balls]);candidate_pairs+=int(lengths.sum());max_candidates=max(max_candidates,int(lengths.max(initial=0)))
        if not lengths.sum():continue
        point_ids=np.repeat(np.arange(begin,stop),lengths);triangle_ids=np.concatenate(balls)
        nearest=trimesh.triangles.closest_point(triangles[triangle_ids],points[point_ids])
        np.minimum.at(exact,point_ids,np.linalg.norm(nearest-points[point_ids],axis=1))
    return exact,dict(method='exact triangle closest point, complete centroid-sphere candidate bound; observed5cm anchor centroids',exact=True,
        queries=len(points),candidate_pairs=candidate_pairs,max_candidates_per_point=max_candidates,max_triangle_centroid_radius_m=rmax)

def colors_for_anchors(xyz, observed, evidence):
    keep=evidence['valid_mask'].astype(bool)&np.isfinite(evidence['rgb_srgb']).all(1)
    p=np.asarray(evidence['canonical_xyz'][keep],float);c=np.asarray(evidence['rgb_srgb'][keep],float);raw=evidence['raw_frame'][keep]
    if not len(p):raise ValueError('No accepted actual a6 RGB evidence')
    # Unknown surfaces use neutral measured body color; never extrapolate tail-light red.
    neutral=(np.ptp(c,axis=1)<=.12)&(c.mean(1)>=.12)&(c.mean(1)<=.70)
    if neutral.sum()<20:neutral=(np.ptp(c,axis=1)<=.18)
    if not neutral.any():body=np.repeat(np.median(c.mean(1)),3)
    else:body=np.median(c[neutral],axis=0)
    result=np.tile(body,(len(xyz),1));known=np.zeros(len(xyz),bool);source_frames=np.zeros(len(xyz),np.int16)
    tree=cKDTree(p);radius=.06
    for i in np.flatnonzero(observed):
        ids=np.asarray(tree.query_ball_point(xyz[i],radius),int)
        if not len(ids):continue
        med=np.median(c[ids],axis=0);inlier=np.abs(c[ids]-med).mean(1)<=.10
        ids=ids[inlier]
        if not len(ids):continue
        distinct=len(np.unique(raw[ids]));source_frames[i]=distinct
        if distinct>=2:
            result[i]=np.median(c[ids],axis=0);known[i]=True
    return result,known,source_frames,dict(unknown_body_srgb=body.tolist(),neutral_evidence_rows=int(neutral.sum()),
        measured_color_policy='observed anchor within0.06m actual clean color samples; >=2 raw frames; per-sample RGB MAE to median<=0.10',
        inferred_color_policy='all unobserved/prior surfaces start at robust neutral observed body median; no invented red tail-light detail')

def stats(x):
    return dict(mean=float(np.mean(x)),p50=float(np.quantile(x,.5)),p90=float(np.quantile(x,.9)),p95=float(np.quantile(x,.95)),max=float(np.max(x)),
        fraction_le_004m=float(np.mean(x<=.04)),fraction_le_010m=float(np.mean(x<=.10)),fraction_le_020m=float(np.mean(x<=.20)))

def component_retention(mesh, sample_face, inside, kept):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    adjacency=np.asarray(mesh.face_adjacency)
    graph=coo_matrix((np.ones(len(adjacency),np.uint8),(adjacency[:,0],adjacency[:,1])),shape=(len(mesh.faces),len(mesh.faces)))
    count,labels=connected_components(graph,directed=False)
    rows=[]
    for component in np.argsort(np.bincount(labels))[::-1]:
        faces=labels==component;sample=labels[sample_face]==component
        rows.append(dict(component_id=int(component),faces=int(faces.sum()),area_m2=float(np.asarray(mesh.area_faces)[faces].sum()),
            sampled=int(sample.sum()),inside_full_box_samples=int((sample&inside).sum()),retained_samples=int((sample&kept).sum()),
            entirely_rejected_in_sample=bool(sample.any() and not (sample&kept).any())))
    return rows

def projection_plot(mesh, points, size, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    surface,normal,_,_=sample_surface(mesh,65536,np.random.default_rng(0))
    inside=(np.abs(surface)<=size/2+1e-6).all(1)
    light=np.array([-.4,-.6,1.]);light/=np.linalg.norm(light)
    shade=.32+.68*np.maximum(normal@light,0)
    colors=shade[:,None]*np.array([.60,.65,.70])[None]
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for column,(x,y,depth,sign,lx,ly) in enumerate([(0,1,2,1,'x length (m)','y width (m)'),(0,2,1,-1,'x length (m)','z height (m)'),(1,2,0,-1,'y width (m)','z height (m)')]):
        lo=np.minimum(surface[:,[x,y]].min(0),-size[[x,y]]/2)-.10
        hi=np.maximum(surface[:,[x,y]].max(0),size[[x,y]]/2)+.10
        for row,select in enumerate([np.ones(len(surface),bool),inside]):
            q=surface[select];col=colors[select];H,W=320,400
            scale=min((W-1)/(hi[0]-lo[0]),(H-1)/(hi[1]-lo[1]));center=(lo+hi)/2
            xy=np.round((q[:,[x,y]]-center)*scale+np.array([W/2,H/2])).astype(int);xy[:,1]=H-1-xy[:,1]
            offsets=np.array([[a,b] for a in [-1,0,1] for b in [-1,0,1]])
            pixel=np.concatenate([xy+d for d in offsets]);z=np.tile(q[:,depth]*sign,9);cc=np.tile(col,(9,1))
            ok=(pixel[:,0]>=0)&(pixel[:,0]<W)&(pixel[:,1]>=0)&(pixel[:,1]<H)
            pix=pixel[ok,1]*W+pixel[ok,0];zz=z[ok];ccc=cc[ok];order=np.argsort(-zz,kind='stable')
            unique,first=np.unique(pix[order],return_index=True);img=np.ones((H*W,3));img[unique]=ccc[order[first]];img=img.reshape(H,W,3)
            ax=axes[row,column];extent=[center[0]-W/(2*scale),center[0]+W/(2*scale),center[1]-H/(2*scale),center[1]+H/(2*scale)]
            ax.imshow(img,extent=extent,origin='upper');ax.add_patch(Rectangle((-size[x]/2,-size[y]/2),size[x],size[y],fill=False,color='#dc2626',lw=1))
            if row==1:ax.scatter(points[::10,x],points[::10,y],s=.3,c='#2563eb',alpha=.5)
            ax.set_aspect('equal');ax.set_xlabel(lx);ax.set_ylabel(ly)
            ax.set_title(('Raw learned surface' if row==0 else 'Only legal-box surface; blue = measured')+' / '+['top','side','rear'][column],fontsize=10)
    fig.suptitle('CPU orthographic sampled-surface zbuffer | clipping removes samples; no caps or shape repair',fontsize=12)
    fig.savefig(path,dpi=160);plt.close(fig)
    return dict(surface_area_inside_full_box_fraction_mc=float(inside.mean()),samples=65536,
        plot=str(path),method='uniform actual learned triangle samples; exact point-depth zbuffer splats; no generated replacement geometry')

def prepare(args):
    start=time.monotonic();rng=np.random.default_rng(0)
    root=args.root.resolve();inputs=root/'inputs';out=args.output or inputs/'actor_surface_seed0.npz'
    contract=read(inputs/'a6_prior_contract.json');half_span=float(contract['normalization_half_span_m'])
    size=np.asarray(contract['full_box_size_lwh_m'],float)
    if abs(half_span-size.max()/2)>1e-6:raise ValueError('Must normalize by fixed WHOLE annotated box, not partial measured bounds')
    geometry=np.load(inputs/'a6_geometry_minimal.npz');points=np.asarray(geometry['canonical_xyz'],np.float64)
    observed,normals,sigma,normal_confidence,inverse,keys=real_anchors(points)
    mesh=load_mesh(args.mesh,args.mesh_space,half_span)
    generated,gen_normals,face,bary=sample_surface(mesh,args.prior_samples,rng)
    inside_box=(np.abs(generated)<=size/2+1e-6).all(1)
    dist,_=cKDTree(observed).query(generated);keep=inside_box&(dist>=.04)
    components=component_retention(mesh,face,inside_box,keep)
    projection=projection_plot(mesh,points,size,out.with_name(out.stem+'_legal_views.png'))
    generated=generated[keep];gen_normals=gen_normals[keep];face=face[keep];bary=bary[keep]
    xyz=np.concatenate([observed,generated]);normal=np.concatenate([normals,gen_normals])
    prior_sigma=float(np.clip(np.sqrt(float(mesh.area)/args.prior_samples)*.65,.020,.070))
    sigma=np.r_[sigma,np.full(len(generated),prior_sigma)]
    observed_flag=np.arange(len(xyz))<len(observed);prior_flag=~observed_flag
    evidence=dict(np.load(inputs/'a6_colour_evidence.npz'));color,color_known,color_frames,color_policy=colors_for_anchors(xyz,observed_flag,evidence)
    nearest=cKDTree(xyz).query(points)[1].astype(np.int32)
    # Frozen measured geometry is the first block. A generated sample may be closer to
    # an individual noisy raw row, so use its own 5cm observed voxel anchor for constraints.
    observed_nearest=inverse.astype(np.int32)
    residual,residual_method=exact_mesh_distances(mesh,observed)
    low,high=np.asarray(mesh.bounds)
    aabb_lower_bound=np.linalg.norm(np.maximum(np.maximum(low-observed,observed-high),0),axis=1)
    geometric_reject=np.quantile(aabb_lower_bound,.9)>.30
    bottom=float(points[:,2].min());box_half=size/2;outside=(np.abs(generated)>box_half+.05).any(1)
    atomic_npz(out,canonical_xyz=xyz.astype(np.float32),canonical_normals=normal.astype(np.float32),
        sigma_tangent_m=sigma.astype(np.float32),colour_srgb=color.astype(np.float32),
        observed_anchor=observed_flag,colour_observed=color_known,prior_origin=prior_flag,
        observed_points=points.astype(np.float32),observed_nearest_anchor=observed_nearest,
        observed_nearest_any_anchor=nearest,observed_voxel_keys=keys,observed_source_raw_frame=geometry['raw_frame'],
        observed_source_cloud_row=geometry['source_cloud_row'],observed_anchor_normal_confidence=normal_confidence.astype(np.float32),
        colour_source_frame_count=color_frames,full_box_lwh_m=size,normalization_half_span_m=half_span,
        prior_sample_face=face,prior_sample_barycentric=bary.astype(np.float32),
        observed_anchor_to_learned_mesh_distance_m=residual.astype(np.float32))
    surface_status='REJECTED_LEARNED_PROPOSAL_DIAGNOSTIC_SURFACE' if geometric_reject else ('DIAGNOSTIC_ONLY_LEARNED_SURFACE_UNACCEPTED' if args.diagnostic_only else 'READY_LEGAL_BOX_LEARNED_SURFACE_QUALITY_UNASSESSED')
    receipt=dict(status=surface_status,output=str(out),output_sha256=sha(out),
        formal_eligible=not (geometric_reject or args.diagnostic_only),quality_accepted=False,
        geometric_reject=bool(geometric_reject),diagnostic_only=args.diagnostic_only,
        input_mesh=str(args.mesh.resolve()),input_mesh_sha256=sha(args.mesh),mesh_space=args.mesh_space,
        input_geometry_sha256=sha(inputs/'a6_geometry_minimal.npz'),input_colour_evidence_sha256=sha(inputs/'a6_colour_evidence.npz'),
        normalization=dict(whole_box_lwh_m=size.tolist(),half_span_m=half_span,
            transform='Omni normalized (x_length,y_height,z_width) -> native[:,[0,2,1]] * fixed half_span; reverse face winding',
            partial_bounds_recenter_or_stretch=False),
        seed=0,prior_requested_samples=args.prior_samples,prior_retained_samples=len(generated),
        prior_rejected_outside_full_box=int((~inside_box).sum()),
        prior_removed_near_observed=int((inside_box&(dist<.04)).sum()),prior_observed_distance_threshold_m=.04,
        prior_box_clipping='retain only actual learned triangle samples inside fixed raw full box +1e-6m; no stretch/recenter, no caps, no programmed car replacement',
        connected_component_sample_retention=components,projection_diagnostics=projection,
        observed_anchor_count=len(observed),observed_measurement_count=len(points),all_anchor_count=len(xyz),
        observed_geometry_fixed=True,observed_voxel_m=.05,measured_colour_anchor_count=int(color_known.sum()),
        inferred_colour_anchor_count=int((~color_known).sum()),color_policy=color_policy,
        observed_normals='local PCA16 neighbors within0.18m, oriented outward from annotated-box origin; degenerate fallback radial; estimated normals, not GT',
        observed_nearest_anchor_policy='index into first real-voxel-anchor block, preserves each actual row assigned5cm voxel; optional nearest-any field separately',
        mesh=dict(vertices=len(mesh.vertices),faces=len(mesh.faces),area_m2=float(mesh.area),
            native_bounds_m=np.asarray(mesh.bounds).tolist(),native_extent_m=np.asarray(mesh.extents).tolist(),
            is_watertight=bool(mesh.is_watertight),is_winding_consistent=bool(mesh.is_winding_consistent),
            prior_samples_outside_annotated_box_plus005m=int(outside.sum()),outside_fraction=float(outside.mean()) if len(outside) else 0),
        point_to_learned_surface_residual_m=stats(residual),point_to_learned_surface_method=residual_method,
        observed_anchor_to_mesh_AABB_distance_lower_bound_m=stats(aabb_lower_bound),
        legal_prior_bounds_m=[generated.min(0).tolist(),generated.max(0).tolist()] if len(generated) else None,
        geometry_rejection_policy='If p90 observed-anchor distance to even mesh AABB exceeds0.30m, metre-scale misalignment is certified; no scale/recenter correction or small-offset fit can create missing car body.',
        bottom_coverage=dict(raw_box_bottom_m=float(-size[2]/2),lowest_measured_m=bottom,
            no_measurement_band_m=[float(-size[2]/2),bottom],measured_anchor_count_below_band=int((observed[:,2]<bottom-1e-6).sum()),
            learned_prior_anchor_count_in_unobserved_lower_band=int(((generated[:,2]<bottom)&(generated[:,2]>=-size[2]/2)).sum()),
            learned_prior_min_z_m=float(generated[:,2].min()) if len(generated) else None,
            policy='Lower approximately0.51m/wheels have no measured geometry; all added lower geometry and color explicitly prior/inferred'),
        sigma_tangent_observed_range_m=[float(sigma[:len(observed)].min()),float(sigma[:len(observed)].max())],
        sigma_tangent_prior_m=prior_sigma,GPU_used=False,training_updates=0,wall_seconds=time.monotonic()-start,
        generated_mesh_modified=False,manual_replacement_mesh_used=False,old_assets_modified=False)
    atomic_json(out.with_name(out.stem+'_receipt.json'),receipt)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mesh',required=True,type=Path)
    parser.add_argument('--mesh-space',choices=['omni_normalized','actor_native_m'],default='omni_normalized')
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--prior-samples',type=int,default=14000)
    parser.add_argument('--diagnostic-only',action='store_true')
    args=parser.parse_args();prepare(args)
