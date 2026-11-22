"""Explicit LOO-selected local inference candidate; distinct hypothesis and asset scopes."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import ConvexHull
from diagnose_nonplanar_inference_cpu import load,read,write,pin,stats,design,normals


def project_to_mask(world,view,mask,protection):
    E,K=view['world_to_camera'],view['camera_K'];cam=world@E[:3,:3].T+E[:3,3];q=cam@K.T
    uv=q[...,:2]/np.maximum(cam[...,2:],1e-12)-.5
    ix=np.rint(uv[...,0]).astype(int);iy=np.rint(uv[...,1]).astype(int);H,W=mask.shape
    valid=np.isfinite(uv).all(-1)&(cam[...,2]>.1)&(ix>=0)&(iy>=0)&(ix<W)&(iy<H)
    selected=np.zeros(valid.shape,bool);yy,xx=np.nonzero(valid)
    selected[yy,xx]=mask[iy[yy,xx],ix[yy,xx]]&~protection[iy[yy,xx],ix[yy,xx]]
    return selected


def lift_cells(mask,shape):
    out=np.zeros(shape,bool)
    out[:-1,:-1]|=mask;out[:-1,1:]|=mask;out[1:,:-1]|=mask;out[1:,1:]|=mask
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    C,out=a.campaign_root,a.output;out.mkdir(parents=True,exist_ok=False);start=time.time()
    R=C/'road_underbody_local_20261123_r10';CV=R/'audit/curvature_ridge_loo_v1'
    selection=read(CV/'training_only_CV_selection.json');chosen=selection['one_SE_preferred_model'];src=load(CV/'plane_and_selected_ridge_DIAGNOSTIC.npz')
    s=load(R/'audit/local_surface_geometry_v1/local_surface_geometry.npz');gpath=C/'road_shared_appearance_20261122_r7/shared_grid_geometry.npz';g=load(gpath)
    centre=src['fit_XY_centre'];b=src[chosen+'__coefficients'];world=src[chosen+'__profile_world_xyz'];normal=src[chosen+'__profile_normal_world'];shape=world.shape[:2]
    uv=src['uv_global'].astype(np.int32);assert np.array_equal(uv,src['uv_global'])
    original=s['original_geometry_supported'].astype(bool);strict=s['geometry_valid'].astype(bool);assert int(strict.sum())==837
    evpath=R/'audit/official_ground_geometry_v1/local_original_point_evidence.npz';ev=load(evpath)
    pool=read(R/'audit/local_surface_geometry_v1/geometry_qualification_contract.json');take=ev['ground']&np.isin(ev['raw'],pool['train_raw'])
    hull=ConvexHull(ev['world_xyz'][take,:2]);margin=-(world[...,:2]@hull.equations[:,:2].T+hull.equations[:,2]).max(-1)
    candidate=np.isfinite(world).all(-1)&np.isfinite(normal).all(-1)&(margin>=-1e-10)
    celluv=.25*(uv[:-1,:-1]+uv[:-1,1:]+uv[1:,:-1]+uv[1:,1:])
    M=g['world_from_uv_xy'][:2,:2];O=g['world_from_uv_xy'][:2,3];cellxy=celluv@M.T+O
    cellworld=np.concatenate([cellxy,(design(cellxy.reshape(-1,2),centre)@b).reshape(*cellxy.shape[:2],1)],-1)
    cellnormal=normals(cellxy.reshape(-1,2),centre,b).reshape(cellworld.shape)
    cellcandidate=candidate[:-1,:-1]&candidate[:-1,1:]&candidate[1:,:-1]&candidate[1:,1:]
    reconstruction=np.zeros(cellcandidate.shape,bool);scope_sources=[]
    for r in [86,110]:
        cp=C/f'road_shared_appearance_20261122_r7/projection_cache_v1/{r:03d}/view_cache.npz';view=load(cp)
        tp=R/f'audit/local_surface_inference_v1_replay1/{r:03d}_target_coverage_DIAGNOSTIC.npz';target=load(tp)
        fixedpath=R/f'fixed_eval_v1_replay1/{r:03d}/fixed_eval.npz';fixed=load(fixedpath)
        protective=fixed['body_protected']|fixed['K_protected']
        selected=project_to_mask(cellworld,view,target['focused_ground_envelope'],protective)&cellcandidate
        reconstruction|=selected
        scope_sources.append(dict(raw=r,camera=pin(cp),fixed_tight_footprint=pin(tp),body_and_K=pin(fixedpath),selected_cells=int(selected.sum())))
    assert reconstruction.any()
    ring=(distance_transform_edt(~reconstruction,sampling=.02)<=.2)&~reconstruction&cellcandidate
    asset=reconstruction|ring
    vertexreconstruction=lift_cells(reconstruction,shape);vertexasset=lift_cells(asset,shape);vertexring=vertexasset&~vertexreconstruction
    pose=np.eye(4);pose[:3,3]=np.r_[centre,design(centre[None],centre)@b]
    local=(world-pose[:3,3]).astype(np.float32);celllocal=(cellworld-pose[:3,3]).astype(np.float32)
    sourcekind=np.where(candidate,np.where(strict,1,2),0).astype(np.uint8)
    cellstrict=strict[:-1,:-1]&strict[:-1,1:]&strict[1:,:-1]&strict[1:,1:]
    cellkind=np.where(cellcandidate,np.where(cellstrict,1,2),0).astype(np.uint8)
    boots=src[chosen+'__bootstrap_coefficients'];centred=design(cellxy.reshape(-1,2),centre);cellSD=np.zeros(cellcandidate.size)
    for j in range(0,len(centred),4096):cellSD[j:j+4096]=(centred[j:j+4096]@boots.T).std(-1,ddof=1)
    path=out/'candidate_geometry.npz'
    np.savez_compressed(path,world_xyz=world,normal_world=normal,uv_global=uv,candidate_mask=candidate,
        geometry_valid=candidate,renderable_mask=np.zeros(shape,bool),direct_witness_mask=strict,
        strict_witness_mask_legacy=strict,inferred_geometry_mask=candidate&~strict,geometry_source_kind=sourcekind,
        original_geometry_supported=original,new_inferred_geometry=candidate&~original,replaced_parent_geometry=candidate&original,
        reconstruction_mask=reconstruction,blend_ring_mask=ring,asset_mask=asset,
        cell_world_xyz=cellworld,cell_normal_world=cellnormal,cell_uv_global=celluv,cell_candidate_mask=cellcandidate,
        cell_direct_witness_mask=cellstrict,cell_inferred_geometry_mask=cellcandidate&~cellstrict,cell_geometry_source_kind=cellkind,
        cell_reconstruction_mask=reconstruction,cell_blend_ring_mask=ring,cell_asset_mask=asset,
        vertex_reconstruction_mask=vertexreconstruction,vertex_blend_ring_mask=vertexring,vertex_asset_support_mask=vertexasset,
        local_xyz=local,cell_local_xyz=celllocal,world_from_patch=pose,
        height_bootstrap_SD_m=src[chosen+'__bootstrap_SD_m'],height_bootstrap_p025_m=src[chosen+'__bootstrap_height_p025_m'],height_bootstrap_p975_m=src[chosen+'__bootstrap_height_p975_m'],
        cell_height_bootstrap_SD_m=cellSD.reshape(cellcandidate.shape),bootstrap_coefficients=boots,
        fit_coefficients=b,fit_XY_centre=centre,bounds_uvxy=s['bounds_uvxy'],spacing_m=np.asarray(.02),parent_geometry_sha256=pin(gpath)['sha256'])
    schema=dict(schema='R10.explicit_local_inference_geometry_candidate.v2',complete=True,decision='CANDIDATE_FOR_CPU_PROJECTION_PENDING_GEOMETRY_AND_ROOT_ACCEPTANCE',
        candidate_geometry=pin(path),LOO_selection=pin(CV/'training_only_CV_selection.json'),chosen_model=chosen,
        shape_vertex=list(shape),shape_cell=list(cellcandidate.shape),spacing_m=.02,bounds_uvxy=s['bounds_uvxy'].tolist(),
        layout='vertex uv_global integer(H,W,2); cell_uv_global=integerUV+.5(H-1,W-1,2). All metric XYZ evaluated analytically at their own UV, not constant-Z',
        candidate_mask_role='context vertex hypothesis range; finite and inside official-ground XY hull, not asset scope',
        asset_mask_role='CELL mask: tight ground-envelope union from86/110 plus .2m world-grid ring, excluding source-view body/K; evaluation bands unchanged',
        field_aliases={'reconstruction_mask':'cell_reconstruction_mask','blend_ring_mask':'cell_blend_ring_mask','asset_mask':'cell_asset_mask'},
        geometry_source_kind={'0':'no hypothesis','1':'legacy strict measured-neighbourhood witness; coordinates are model profile, not a literal LiDAR return','2':'model-inferred profile; no direct witness at this texel'},
        direct_witness_mask_legacy_exact=True,direct_witness_vertices=int(strict.sum()),
        vertex_hypotheses=int(candidate.sum()),parent_missing_model_inferred_vertices=int((candidate&~original).sum()),
        cell_reconstruction_count=int(reconstruction.sum()),cell_blend_ring_count=int(ring.sum()),cell_asset_count=int(asset.sum()),
        cell_asset_surface_area_xy_m2=float(asset.sum()*abs(np.linalg.det(M))),
        asset_cell_strict_witness_count=int((asset&cellstrict).sum()),asset_cell_model_inferred_count=int((asset&~cellstrict).sum()),
        asset_uncertainty_conditional_SD_m=stats(cellSD.reshape(cellcandidate.shape)[asset]),
        confidence_is_not_calibrated_probability=True,GT_bottom_contact_used_for_fitting=False,
        original_parent_geometry_edited=False,existing_vehicle_edited=False,
        renderable_mask_all_false_is_review_status_only_CPU_projection_explicitly_allowed=True,
        geometry_certificate_accepted_for_zero_step=False,target_coverage_gate='PENDING',root_geometry_gate='PENDING',
        cell_centre_interpolation_authorized=False,cell_centres_analytically_evaluated=True,
        source_scope=scope_sources,CPU_only=True,GPU_forward=0,updates=0,elapsed_seconds=time.time()-start)
    write(out/'candidate_geometry_receipt.json',schema);print(json.dumps(schema),flush=True)


if __name__=='__main__':main()
