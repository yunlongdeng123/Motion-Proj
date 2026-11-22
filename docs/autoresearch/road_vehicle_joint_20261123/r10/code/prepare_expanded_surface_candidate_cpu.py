"""Image-mask-driven local plane candidate. No refit, GPU, source RGB write or approval."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull,cKDTree
from diagnose_nonplanar_inference_cpu import load,read,write,pin,stats,design,normals,medoids


def rays(view,xy,b,centre):
    E,K=view['world_to_camera'],view['camera_K'];C=-E[:3,:3].T@E[:3,3]
    direction=np.c_[xy+.5,np.ones(len(xy))]@np.linalg.inv(K).T@E[:3,:3]
    n=np.array([-b[1],-b[2],1.]);d=b[0]-b[1]*centre[0]-b[2]*centre[1]
    den=direction@n;t=(d-C@n)/den
    world=C+t[:,None]*direction
    good=np.isfinite(world).all(-1)&(t>0)&(abs(den)>1e-12)
    return world,good,t


def query_cell_mask(world,view,mask):
    E,K=view['world_to_camera'],view['camera_K'];camera=world@E[:3,:3].T+E[:3,3];q=camera@K.T;uv=q[...,:2]/np.maximum(camera[...,2:],1e-12)-.5
    u=np.rint(uv[...,0]).astype(int);v=np.rint(uv[...,1]).astype(int);H,W=mask.shape
    valid=np.isfinite(uv).all(-1)&(camera[...,2]>.1)&(u>=0)&(v>=0)&(u<W)&(v<H)
    out=np.zeros(valid.shape,bool);iy,ix=np.nonzero(valid);out[iy,ix]=mask[v[iy,ix],u[iy,ix]]
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();C,out=a.campaign_root,a.output;out.mkdir(parents=True,exist_ok=False);start=time.time()
    R=C/'road_underbody_local_20261123_r10';oldpath=R/'audit/local_surface_candidate_v2/candidate_geometry.npz';old=load(oldpath)
    b=old['fit_coefficients'];centre=old['fit_XY_centre'];boots=old['bootstrap_coefficients'];assert not b[3:].any() and boots.shape==(512,6)
    gridpath=C/'road_shared_appearance_20261122_r7/shared_grid_geometry.npz';g=load(gridpath);M=g['world_from_uv_xy'][:2,:2];O=g['world_from_uv_xy'][:2,3];inverse=np.linalg.inv(M)
    maskroot=R/'edit_footprint_masks_v2';rows=[];queries=[];boundaries=[]
    for raw in [86,110,134,179,198]:
        mp=maskroot/f'{raw:03d}/scope_masks.npz';mask=load(mp);cp=C/f'road_shared_appearance_20261122_r7/projection_cache_v1/{raw:03d}/view_cache.npz';view=load(cp)
        yy,xx=np.nonzero(mask['inside_allowed_edit']);xy=np.c_[xx,yy];world,good,depth=rays(view,xy,b,centre)
        uv=(world[:,:2]-O)@inverse.T
        rows.append(dict(raw=raw,mask=mask,view=view,xy=xy,world=world,good=good,uv=uv,depth=depth,mask_pin=pin(mp),camera_pin=pin(cp)))
        queries.append(uv[good])
        for dx,dy in [(-.5,-.5),(-.5,.5),(.5,-.5),(.5,.5)]:
            q,ok,_=rays(view,xy+[dx,dy],b,centre);boundaries.append(((q[:,:2]-O)@inverse.T)[ok])
    edge=np.concatenate(boundaries);lo=np.floor(edge.min(0)).astype(int)-11;hi=np.ceil(edge.max(0)).astype(int)+12
    u0,v0=lo;u1,v1=hi;vv,uu=np.mgrid[v0:v1,u0:u1];uv=np.stack([uu,vv],-1).astype(np.int32);shape=vv.shape
    contract=dict(schema='R10.expanded_mask_driven_geometry_contract.v1',complete=True,written_before_credibility_statistics=True,
        parent_geometry=pin(oldpath),source_masks=pin(maskroot/'receipt.json'),model_unchanged=True,all_coordinates_model_inferred=True,
        scope='Five-view inside_allowed_edit pixel squares ray-plane -> worldUV union plus exact .2m metric cell ring; no RGB selection and no hardcoded u0',
        bounds_uvxy_half_open=[int(u0),int(v0),int(u1),int(v1)],
        credibility=dict(physical_official_road_hull_required=True,nearest_official_road_max_m=2.,conditional_bootstrap_SD_max_m=.05,nearest16_official_road_median_abs_profile_residual_max_m=.10),
        physical_validation='Original8 official-labeled training frames only, full global ROAD/LANE points restricted to requested world bbox +3m; no OBB/photo/model-residual point pruning; no refitting',
        original_fit_hull='Separate477 original fitting medoids with+.4m context, not the new3m validation padding; extrapolation flagged',
        leverage_reported_not_silently_renormalized=True,heldout_not_fitted_or_used_for_screen_selection=True,
        uncertainty_is_conditional_not_calibrated_probability=True,approval=False,GPU_forward=0)
    write(out/'preparation_contract.json',contract)
    XY=uv@M.T+O;Q=XY.reshape(-1,2);AQ=design(Q,centre);Z=AQ@b;world=np.c_[Q,Z].reshape(*shape,3);normal=normals(Q,centre,b).reshape(*shape,3)
    physicalpath=R/'audit/ground_contact_prior_raw_GT_v2/GT_bottom_and_road_evidence.npz';physical=load(physicalpath);P=physical['official_training_road_worldXYZ'];pr=physical['official_training_road_source_raw']
    bbox=np.stack([Q.min(0)-3,Q.max(0)+3]);use=(P[:,:2]>=bbox[0]).all(-1)&(P[:,:2]<=bbox[1]).all(-1)&(abs(P[:,2]+20.817)<=1.5)
    P=P[use];pr=pr[use];assert len(P)>50
    hull=ConvexHull(P[:,:2]);margin=-(Q@hull.equations[:,:2].T+hull.equations[:,2]).max(-1)
    ev=load(R/'audit/official_ground_geometry_v1/local_original_point_evidence.npz');cfg=read(R/'audit/local_surface_geometry_v1/geometry_qualification_contract.json');fituse=ev['ground']&np.isin(ev['raw'],cfg['train_raw'])
    ids=medoids(ev['world_xyz'],np.flatnonzero(fituse),np.asarray(cfg['XY_fit_context_bbox'])[0]);fitP=ev['world_xyz'][ids];A=design(fitP[:,:2],centre)
    residual=A@b-fitP[:,2];weight=np.minimum(1.,.03/np.maximum(abs(residual),1e-12));inv=np.linalg.pinv(A[:,:3].T@(weight[:,None]*A[:,:3]));lev=np.einsum('ni,ij,nj->n',AQ[:,:3],inv,AQ[:,:3]);trainlev=np.einsum('ni,ij,nj->n',A[:,:3],inv,A[:,:3])
    fithull=ConvexHull(fitP[:,:2]);fitmargin=-(Q@fithull.equations[:,:2].T+fithull.equations[:,2]).max(-1)
    cov=np.cov(boots,rowvar=False);variance=np.einsum('ni,ij,nj->n',AQ,cov,AQ);SD=np.sqrt(np.maximum(variance,0))
    tree=cKDTree(P[:,:2]);distance,near=tree.query(Q,k=16);point_residual=design(P[:,:2],centre)@b-P[:,2]
    local_residual=np.median(abs(point_residual[near]),axis=-1)
    flags=dict(inside_physical_validation_hull=margin>=-1e-10,nearest_road_le2m=distance[:,0]<=2.,conditional_SD_le0p05m=SD<=.05,nearby_road_median_residual_le0p10m=local_residual<=.10)
    candidate=np.logical_and.reduce(list(flags.values())).reshape(shape)
    celluv=uv[:-1,:-1].astype(float)+.5;cellxy=celluv@M.T+O;cellQ=cellxy.reshape(-1,2);cellA=design(cellQ,centre);cellworld=np.c_[cellQ,cellA@b].reshape(*celluv.shape[:2],3);cellnormal=normals(cellQ,centre,b).reshape(cellworld.shape)
    requested=np.zeros(celluv.shape[:2],bool)
    for row in rows:requested|=query_cell_mask(cellworld,row['view'],row['mask']['inside_allowed_edit'])
    assert requested.any();nearest_to_request=cKDTree(cellworld[requested]).query(cellworld.reshape(-1,3))[0].reshape(requested.shape)
    desired_ring=(nearest_to_request<=.2)&~requested;desired=requested|desired_ring
    cellcandidate=candidate[:-1,:-1]&candidate[:-1,1:]&candidate[1:,:-1]&candidate[1:,1:]
    recon=requested&cellcandidate;ring=desired_ring&cellcandidate;asset=recon|ring
    cellSD=np.sqrt(np.maximum(np.einsum('ni,ij,nj->n',cellA,cov,cellA),0)).reshape(requested.shape)
    legacy=np.zeros(shape,bool);oldbounds=old['bounds_uvxy'].astype(int);over=(uu>=oldbounds[0])&(uu<oldbounds[2])&(vv>=oldbounds[1])&(vv<oldbounds[3])
    legacy[over]=old['strict_witness_mask_legacy'][vv[over]-oldbounds[1],uu[over]-oldbounds[0]]
    sourcepath=R/'local_shared_continuous_v1/final_atlas.npz';source=load(sourcepath);Fh,Fw=source['permission_mask'].shape
    pu=celluv[...,0].astype(int);pv=celluv[...,1].astype(int);bounds=(pu>=0)&(pv>=0)&(pu+1<Fw)&(pv+1<Fh);su=np.clip(pu,0,Fw-2);sv=np.clip(pv,0,Fh-2)
    material=bounds&source['permission_mask'][sv,su]&source['permission_mask'][sv,su+1]&source['permission_mask'][sv+1,su]&source['permission_mask'][sv+1,su+1]
    pose=np.eye(4);pose[:3,3]=np.r_[centre,design(centre[None],centre)@b]
    npz=out/'expanded_candidate_geometry.npz'
    np.savez_compressed(npz,world_xyz=world,normal_world=normal,uv_global=uv,candidate_mask=candidate,geometry_valid=candidate,renderable_mask=np.zeros(shape,bool),
        geometry_source_kind=np.where(candidate,2,0).astype(np.uint8),inferred_geometry_mask=candidate,strict_witness_mask_legacy=legacy,
        cell_world_xyz=cellworld,cell_normal_world=cellnormal,cell_uv_global=celluv,cell_candidate_mask=cellcandidate,
        reconstruction_mask=recon,blend_ring_mask=ring,asset_mask=asset,cell_reconstruction_mask=recon,cell_blend_ring_mask=ring,cell_asset_mask=asset,
        requested_cell_reconstruction_mask=requested,requested_cell_asset_mask=desired,unsupported_requested_cell_mask=desired&~cellcandidate,
        cell_geometry_source_kind=np.where(cellcandidate,2,0).astype(np.uint8),current_source_all4_permission=material,
        height_bootstrap_SD_m=SD.reshape(shape),cell_height_bootstrap_SD_m=cellSD,
        physical_validation_hull_margin_m=margin.reshape(shape),original_fit_hull_margin_m=fitmargin.reshape(shape),original_fit_leverage=lev.reshape(shape),original_fit_leverage_over_max_training=(lev/trainlev.max()).reshape(shape),
        nearest_official_road_distance_m=distance[:,0].reshape(shape),nearby_road_median_abs_profile_residual_m=local_residual.reshape(shape),
        fit_coefficients=b,fit_XY_centre=centre,bootstrap_coefficients=boots,bounds_uvxy=np.array([u0,v0,u1,v1]),spacing_m=np.asarray(.02),world_from_patch=pose,
        local_xyz=(world-pose[:3,3]).astype(np.float32),cell_local_xyz=(cellworld-pose[:3,3]).astype(np.float32),
        **{k:v.reshape(shape) for k,v in flags.items()})
    cover=[]
    for row in rows:
        q=row['uv'];cu=np.floor(q[:,0]).astype(int)-u0;cv=np.floor(q[:,1]).astype(int)-v0;ok=row['good']&(cu>=0)&(cv>=0)&(cu<asset.shape[1])&(cv<asset.shape[0]);u=np.clip(cu,0,asset.shape[1]-1);v=np.clip(cv,0,asset.shape[0]-1)
        supported=ok&asset[v,u];known=supported&material[v,u];oldcount=int(row['mask']['inside_replacement'].sum())
        camera=row['view'];E,K=camera['world_to_camera'],camera['camera_K'];cameraQ=row['world']@E[:3,:3].T+E[:3,3];screen=cameraQ@K.T;reproj=screen[:,:2]/screen[:,2,None]-.5
        error=float(abs(reproj[row['good']]-row['xy'][row['good']]).max())
        maps={};H,W=row['mask']['inside_allowed_edit'].shape
        for name,value in [('expanded_geometry_supported',supported),('expanded_material_available_current_source',known),('expanded_geometry_unknown',~supported)]:
            canvas=np.zeros((H,W),bool);canvas[row['xy'][:,1],row['xy'][:,0]]=value;maps[name]=canvas
        np.savez_compressed(out/f'{row["raw"]:03d}_requested_coverage.npz',inside_allowed_edit=row['mask']['inside_allowed_edit'],**maps)
        cover.append(dict(raw=row['raw'],inside_allowed_edit_pixels=len(q),old_replacement_pixels=oldcount,new_candidate_geometry_pixels=int(supported.sum()),current_source_reusable_pixels=int(known.sum()),
            geometry_unknown_pixels=int((~supported).sum()),source_material_extension_needed_pixels=int((supported&~known).sum()),ray_plane_reprojection_max_abs_px=error,
            requested_UV_min=q[row['good']].min(0).tolist(),requested_UV_max=q[row['good']].max(0).tolist(),mask=row['mask_pin'],camera=row['camera_pin']))
    report=dict(schema='R10.expanded_plane_model_inference_candidate.v1',complete=True,decision='CANDIDATE_ONLY_NOT_APPROVED',geometry_npz=pin(npz),
        parent_model_geometry=pin(oldpath),model_coefficients_and_512_bootstrap_byteexact=True,model_refitted=False,A20_or_old_assets_modified=False,
        shape_vertex=list(shape),shape_cell=list(asset.shape),bounds_uvxy=[int(u0),int(v0),int(u1),int(v1)],spacing_m=.02,
        all_candidate_geometry_source_kind=2,raw_requested_cells=int(desired.sum()),supported_asset_cells=int(asset.sum()),unsupported_requested_cells=int((desired&~cellcandidate).sum()),
        reconstruction_cells=int(recon.sum()),blend_ring_cells=int(ring.sum()),asset_area_xy_m2=float(asset.sum()*abs(np.linalg.det(M))),
        asset_cell_conditional_SD_m=stats(cellSD[asset]),context_conditional_SD_m=stats(SD),
        original_fit_hull_inside_context_vertices=int((fitmargin>=0).sum()),physical_validation_hull_inside_context_vertices=int((margin>=0).sum()),
        original_fit_leverage_over_max_training=stats(lev/trainlev.max()),physical_validation_points=len(P),physical_validation_source_frames=np.unique(pr).tolist(),
        candidate_independent_rejection_counts={k:int((~v).sum()) for k,v in flags.items()},current_source_reusable_asset_cells=int((asset&material).sum()),source_extension_needed_asset_cells=int((asset&~material).sum()),
        old_parent_grid_bounds_exceeded=bool(u0<0 or v0<0 or u1>Fw or v1>Fh),coverage=cover,
        prior_GT_contact_and_heldout_preserved_not_reoptimized=True,source_RGB_not_written=True,geometry_accepted_for_zero_step=False,
        physical_validation_data=pin(physicalpath),preparation_contract=pin(out/'preparation_contract.json'),CPU_only=True,GPU_forward=0,updates=0,seconds=time.time()-start)
    write(out/'expanded_candidate_receipt.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
