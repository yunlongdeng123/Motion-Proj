"""Official-ground-only height profile and uncertainty; does not approve inferred geometry."""
import argparse, importlib.util, json, time
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from diagnose_nonplanar_inference_cpu import load,read,write,pin,stats,design,normals,medoids,fit,neighbourhood
from qualify_local_surface_geometry_cpu import measured_normals


def both(x):
    return dict(signed=stats(x),absolute=stats(abs(np.asarray(x))))


def angle(a,b):
    a=a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-15)
    b=b/np.maximum(np.linalg.norm(b,axis=-1,keepdims=True),1e-15)
    return np.degrees(np.arccos(np.clip(abs((a*b).sum(-1)),0,1)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    C,out=a.campaign_root,a.output;out.mkdir(parents=True,exist_ok=False);start=time.time()
    R=C/'road_underbody_local_20261123_r10';strict=R/'audit/local_surface_geometry_v1';ev=R/'audit/official_ground_geometry_v1'
    old=read(strict/'geometry_qualification_contract.json');s=load(strict/'local_surface_geometry.npz')
    contract=dict(schema='R10.official_ground_height_profile_diagnostic_contract.v1',complete=True,written_before_fitting=True,
        decision='READ_ONLY_PROFILE_DIAGNOSTIC_NOT_GEOMETRY_ACCEPTANCE',point_evidence=pin(ev/'official_ground_point_receipt.json'),
        train_raw=old['train_raw'],heldout_raw=old['heldout_raw'],
        fit='Only official ROAD18/LANE_MARKER19 original XYZ; every OBB/photo category retained; same0.1m medoid and quadratic IRLS8 Huber.03m rules as initial strict protocol',
        heldout='Original134/179/198 strict raw physical points; photo-road0 erode2 proxy qualification only, no official point labels available; never fit or trim using prediction',
        heldout_height_median_max_m=.04,heldout_height_p95_max_m=.10,
        heldout_normal_median_max_deg=10.,heldout_normal_p95_max_deg=20.,normal_evidence='PCA nearest20/.5m >=10/eigenfraction<=.05; same untrimmed fixed heldout set',
        uncertainty='512 training-frame cluster bootstraps at fixed actual medoids seed20261123, plus leave-one-frame-out with reselected medoids; conditional model sensitivity, not calibrated probability',
        interpolation='global measured hull, nearest official point, multi-frame/multi-direction support all reported; no inferred-domain PASS threshold or renderable activation',
        boundary='Actual parent XYZ triangle/finite-difference normals; no pull-to-parent correction, no C0/C1 claim',
        source_points_or_geometry_modified=False,GPU_forward=0,updates=0)
    write(out/'diagnostic_contract.json',contract)
    evidence=load(ev/'local_original_point_evidence.npz');use=evidence['ground']&np.isin(evidence['raw'],old['train_raw'])
    X=evidence['world_xyz'];raw=evidence['raw'];centre=s['fit_XY_centre'];origin=np.asarray(old['XY_fit_context_bbox'])[0]
    ids=medoids(X,np.flatnonzero(use),origin);P=X[ids];A=design(P[:,:2],centre);b,w=fit(A,P[:,2]);assert b is not None
    XY=s['world_xyz'][...,:2];shape=XY.shape[:2];Q=XY.reshape(-1,2);AQ=design(Q,centre)
    height=AQ@b;profile=np.concatenate([Q,height[:,None]],-1).reshape(*shape,3);n=normals(Q,centre,b).reshape(*shape,3)
    factors=neighbourhood(P,raw[ids],X[use],raw[use],Q)
    hull=ConvexHull(P[:,:2]);margin=-(Q@hull.equations[:,:2].T+hull.equations[:,2]).max(-1);factors['hull_margin_m']=margin
    inside=margin>=-1e-10
    inv=np.linalg.pinv(A.T@(w[:,None]*A));lev=np.einsum('ni,ij,nj->n',AQ,inv,AQ);tl=np.einsum('ni,ij,nj->n',A,inv,A)
    factors['regression_leverage']=lev;factors['leverage_over_max_training']=lev/tl.max()
    residual=A@b-P[:,2];sigma2=float((w*residual**2).sum()/max(1,w.sum()-6))
    factors['conditional_height_mean_SE_m']=np.sqrt(np.maximum(0,sigma2*lev))
    trains=np.unique(raw[ids]);rng=np.random.default_rng(20261123);bootstrap=[];rankdef=0
    for _ in range(512):
        draw=rng.integers(0,len(trains),len(trains));counts=np.bincount(draw,minlength=len(trains))
        mass=np.array([counts[np.flatnonzero(trains==r)[0]] for r in raw[ids]])
        bb,_=fit(A,P[:,2],mass)
        if bb is None:rankdef+=1
        else:bootstrap.append(bb)
    bootstrap=np.asarray(bootstrap);lo=np.zeros(len(Q));hi=lo.copy();sd=lo.copy()
    for startq in range(0,len(Q),4096):
        z=AQ[startq:startq+4096]@bootstrap.T
        lo[startq:startq+4096]=np.quantile(z,.025,axis=-1);hi[startq:startq+4096]=np.quantile(z,.975,axis=-1);sd[startq:startq+4096]=z.std(-1,ddof=1)
    factors['frame_bootstrap_SD_m']=sd;factors['frame_bootstrap_95pct_width_m']=hi-lo
    loo=[]
    for r in trains:
        ii=medoids(X,np.flatnonzero(use&(raw!=r)),origin);bb,_=fit(design(X[ii,:2],centre),X[ii,2]);assert bb is not None;loo.append(bb)
    loo=np.asarray(loo);L=AQ@loo.T
    factors['leave_one_frame_out_max_delta_m']=abs(L-height[:,None]).max(-1)
    factors['leave_one_frame_out_span_m']=np.ptp(L,axis=-1)
    measured=load(strict/'strict_measurements.npz');heldout=[];passed=True;normal_total=0
    for r in old['heldout_raw']:
        v=measured['world_xyz'][measured['source_raw']==r];dz=design(v[:,:2],centre)@b-v[:,2];ag,ok,nn=measured_normals(v,centre,b);normal_total+=len(ag)
        hs=stats(abs(dz));ns=stats(ag);check=(len(dz)>=50 and hs['median']<=.04 and hs['p95']<=.10 and len(ag)>=20 and ns['median']<=10 and ns['p95']<=20)
        passed&=check;heldout.append(dict(raw=r,signed_height_error_m=both(dz),normal_error_deg=ns,normal_unmeasurable=int((~ok).sum()),fixed_threshold_checks_pass=bool(check),official_point_semantic_qualification=False,trimmed_using_prediction=False))
    passed&=normal_total>=100
    parentpath=C/'road_shared_appearance_20261122_r7/shared_grid_geometry.npz';g=load(parentpath);u0,v0,u1,v1=s['bounds_uvxy'].astype(int)
    parent=g['world_xyz'][v0:v1,u0:u1];valid=g['geometry_valid'][v0:v1,u0:u1]
    geometric=load(R/'audit/physical_crosscheck_v1/parent_actual_geometry_normal_maps.npz');boundary=geometric['parent_boundary']
    parent_z=(design(parent[...,:2].reshape(-1,2),centre)@b).reshape(shape)
    tri=geometric['triangles'];tri_xyz=parent.reshape(-1,3)[tri];tri_n=normals(tri_xyz.mean(1)[:,:2],centre,b)
    tri_angle=angle(geometric['triangle_geometric_normal'],tri_n);goodtri=geometric['triangle_geometry_valid'];goodboundary=geometric['triangle_boundary']
    fdprofile=normals(parent[...,:2].reshape(-1,2),centre,b).reshape(parent.shape);fdangle=angle(geometric['normal_finite_difference'],fdprofile);fdgood=geometric['finite_difference_normal_valid']
    boundary_stats=dict(profile_minus_parent_Z_m={'all_supported':both(parent_z[valid]-parent[...,2][valid]),'missing_boundary':both(parent_z[boundary]-parent[...,2][boundary])},
        actual_triangle_normal_vs_profile_deg={'all_supported':stats(tri_angle[goodtri]),'missing_boundary':stats(tri_angle[goodboundary])},
        actual_finite_difference_normal_vs_profile_deg={'all_supported':stats(fdangle[fdgood]),'missing_boundary':stats(fdangle[fdgood&boundary])},
        boundary_C0_C1_matched=False,parent_normal_storage_not_used_as_physical_truth=True)
    factors={k:v.reshape(shape) for k,v in factors.items()}
    missing=~valid;candidate=missing&inside.reshape(shape)
    np.savez_compressed(out/'profile_candidate_DIAGNOSTIC_ONLY.npz',profile_world_xyz=profile,profile_normal_world=n,uv_global=s['uv_global'],
        inferred_only_candidate_mask=candidate,direct_witness_mask=s['geometry_valid'],renderable_mask=np.zeros(shape,bool),
        parent_geometry_supported=valid,source_official_ground_worldXYZ=X[use],source_raw=raw[use],source_lidar_row_id=evidence['lidar_row_id'][use],
        selected_medoid_evidence_row_ids=ids,fit_coefficients=b,fit_XY_centre=centre,bootstrap_coefficients=bootstrap,leave_one_frame_out_coefficients=loo,
        bootstrap_height_p025_m=lo.reshape(shape),bootstrap_height_p975_m=hi.reshape(shape),bounds_uvxy=s['bounds_uvxy'],spacing_m=s['spacing_m'],**factors)
    spec=importlib.util.spec_from_file_location('official_profile_projector',R/'code/local_surface_projection.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    targets=[]
    M=g['world_from_uv_xy'][:2,:2];O=g['world_from_uv_xy'][:2,3]
    for r in [86,110]:
        cp=C/f'road_shared_appearance_20261122_r7/projection_cache_v1/{r:03d}/view_cache.npz';v=load(cp)
        pr=mod.raster_surface(profile,s['uv_global'],inside.reshape(shape),v['world_to_camera'],v['camera_K'],v['projection_valid'].shape[::-1])
        oldt=load(R/f'audit/local_surface_inference_v1_replay1/{r:03d}_target_coverage_DIAGNOSTIC.npz');regions={}
        for name in ['red_diagnostic','focused_residual_candidate']:
            mask=oldt[name];visible=mask&pr['projection_valid'];queryxy=pr['atlas_uv_global'][visible]@M.T+O;qq=design(queryxy,centre)
            qheight=qq@b;zz=qq@bootstrap.T;ll=qq@loo.T;qfactor=neighbourhood(P,raw[ids],X[use],raw[use],queryxy)
            regions[name]=dict(fixed_pixels=int(mask.sum()),profile_projected_pixels=int(visible.sum()),
                factors={k:stats(value) for k,value in qfactor.items()},
                conditional_bootstrap_SD_m=stats(zz.std(-1,ddof=1)),conditional_bootstrap_95pct_width_m=stats(np.quantile(zz,.975,axis=-1)-np.quantile(zz,.025,axis=-1)),
                leave_one_frame_out_max_delta_m=stats(abs(ll-qheight[:,None]).max(-1)),
                new_minus_old_profile_height_m=both(qheight-design(queryxy,centre)@s['fit_coefficients']),
                diagnostic_projection_is_not_geometry_acceptance=True)
        np.savez_compressed(out/f'{r:03d}_profile_projection_DIAGNOSTIC.npz',**pr,red_diagnostic=oldt['red_diagnostic'],focused_residual_candidate=oldt['focused_residual_candidate'])
        targets.append(dict(raw=r,regions=regions,camera=pin(cp)))
    S=np.linalg.svd(A*np.sqrt(w[:,None]),compute_uv=False)
    result=dict(schema='R10.official_ground_profile_diagnostic.v1',complete=True,decision='PROFILE_NOT_APPROVED_FOR_RENDERING',
        official_training_points=int(use.sum()),selected_actual_medoids=len(ids),medoid_frame_counts={str(r):int((raw[ids]==r).sum()) for r in trains},
        profile_fit_coefficients=b.tolist(),profile_fit_XY_centre=centre.tolist(),weighted_residual_sigma_m=float(np.sqrt(sigma2)),
        weighted_design_condition=float(S[0]/S[-1]),weighted_column_normalized_condition=float(np.linalg.cond(A*np.sqrt(w[:,None])/np.linalg.norm(A*np.sqrt(w[:,None]),axis=0))),
        bootstrap_valid=len(bootstrap),bootstrap_rank_deficient=rankdef,bootstrap_is_conditional_not_calibrated=True,
        fixed_heldout_height_and_normal_checks_all_pass=bool(passed),heldout_official_labels_available=False,heldout=heldout,
        parent_boundary=boundary_stats,parent_missing_texels=int(missing.sum()),global_hull_missing_candidates=int(candidate.sum()),
        missing_factors={k:stats(value[candidate]) for k,value in factors.items()},target_rows=targets,
        candidate=pin(out/'profile_candidate_DIAGNOSTIC_ONLY.npz'),diagnostic_contract=pin(out/'diagnostic_contract.json'),
        actual_affine_grid_XY_vs_parent_XY_max_m=float(abs(XY-parent[...,:2]).max()),
        original_strict_or_profile_or_parent_modified=False,new_renderable_texels=0,GPU_forward=0,updates=0,elapsed_seconds=time.time()-start)
    write(out/'official_profile_statistics.json',result)
    print(json.dumps(dict(complete=True,receipt=pin(out/'official_profile_statistics.json'),heldout_pass=bool(passed),boundary=boundary_stats,targets=targets)),flush=True)


if __name__=='__main__':main()
