"""Recompute fixed edit rays and footprint using the training-selected17-frame plane.

Model support, original denominators and unknown edges remain explicit. No source
colour, old road, original actor, camera, renderable asset or GPU is changed.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree,ConvexHull
from diagnose_nonplanar_inference_cpu import load,read,write,pin,stats,design,normals
from prepare_expanded_surface_candidate_cpu import rays,query_cell_mask

RAWS=[86,110,134,179,198]


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    C,out=a.campaign_root,a.output;out.mkdir(parents=True,exist_ok=False);start=time.time();R=C/'road_underbody_local_20261123_r10';D=R/'audit/timewindow17_geometry_v1'
    select_path=D/'loo_selection/training_only_CV_selection.json';selection=read(select_path);assert selection['one_SE_preferred_model']=='plane'
    train_gate=read(D/'training_physical_gate_summary.json');assert train_gate['decision']=='PASS_TRAINING_HEIGHT_ONLY'
    diagpath=D/'loo_selection/expanded_curvature_diagnostic.json';diag=read(diagpath);selected=next(x for x in diag['models'] if x['name']=='plane');assert selected['fixed_expanded_heldout_pass']
    fitted=load(D/'loo_selection/expanded_geometry_DIAGNOSTIC.npz');b=fitted['plane__coefficients'];centre=fitted['fit_XY_centre'];boots=fitted['plane__bootstrap_coefficients'];assert boots.shape==(512,6) and not b[3:].any()
    Xtrain=load(D/'loo_selection/training_points.npz');X=Xtrain['world_xyz'];raw=Xtrain['source_raw'];assert np.array_equal(np.unique(raw),np.arange(90,171,5))
    gridpath=C/'road_shared_appearance_20261122_r7/shared_grid_geometry.npz';grid=load(gridpath);M,O=grid['world_from_uv_xy'][:2,:2],grid['world_from_uv_xy'][:2,3];iM=np.linalg.inv(M)
    oldpath=R/'audit/local_surface_expanded_candidate_v1/expanded_candidate_geometry.npz';old=load(oldpath)
    old_scope=old['cell_world_xyz'][old['requested_cell_asset_mask'],:2];context_tree=cKDTree(old_scope)
    views=[];corners=[]
    for rr in RAWS:
        mp=R/f'edit_footprint_masks_v2/{rr:03d}/scope_masks.npz';m=load(mp);cp=C/f'road_shared_appearance_20261122_r7/projection_cache_v1/{rr:03d}/view_cache.npz';v=load(cp)
        yy,xx=np.nonzero(m['inside_allowed_edit']);xy=np.c_[xx,yy];P,good,_=rays(v,xy,b,centre);q=(P[:,:2]-O)@iM.T
        views.append(dict(raw=rr,mask=m,view=v,xy=xy,world=P,good=good,uv=q,mask_pin=pin(mp),camera_pin=pin(cp)))
        for dx,dy in [(-.5,-.5),(-.5,.5),(.5,-.5),(.5,.5)]:
            PP,ok,_=rays(v,xy+[dx,dy],b,centre);corners.append(((PP[:,:2]-O)@iM.T)[ok])
    boundary=np.concatenate(corners);lo=np.floor(boundary.min(0)).astype(int)-11;hi=np.ceil(boundary.max(0)).astype(int)+12
    u0,v0=lo;u1,v1=hi;vv,uu=np.mgrid[v0:v1,u0:u1];uv=np.stack([uu,vv],-1).astype(np.int32);shape=vv.shape
    contract=dict(schema='R10.timewindow17_new_plane_footprint_contract.v1',complete=True,written_before_support_statistics=True,
        model_selection=pin(select_path),physical_model_evaluation=pin(diagpath),training_physical_gate=pin(D/'training_physical_gate_summary.json'),
        reference_pool=pin(D/'reference_pool_contract.json'),old1844_unmodified_scope_reference=pin(oldpath),
        scope='Re-intersect every fixed5view inside_allowed_edit pixel square with newly chosen plane; derive globalUV footprint and exact .2m cell-centre ring afresh',
        global_world_UV_transform_unchanged=pin(gridpath),geometry_source_kind=2,not_old_UV_mask_with_only_Z_replaced=True,
        support=dict(training_XY_convex_hull_required=True,fit_context_distance_max_m=.4,nearest_original_road_max_m=2.,
            conditional_bootstrap_SD_max_m=.05,nearest16_original_point_median_abs_profile_residual_max_m=.10),
        thresholds_reused_from_preexisting_expansion_support_protocol=True,
        out_of_support_edges='Unknown; never force-filled to meet coverage; fixed fullband denominators remain unchanged',
        geometry_approval=False,GPU_forward=0,updates=0)
    write(out/'candidate_contract.json',contract)
    XY=uv@M.T+O;Q=XY.reshape(-1,2);AQ=design(Q,centre);world=np.c_[Q,AQ@b].reshape(*shape,3);normal=normals(Q,centre,b).reshape(*shape,3)
    hull=ConvexHull(X[:,:2]);margin=-(Q@hull.equations[:,:2].T+hull.equations[:,2]).max(-1)
    d,near=cKDTree(X[:,:2]).query(Q,k=16);residual=design(X[:,:2],centre)@b-X[:,2];localres=np.median(abs(residual[near]),axis=-1)
    contextdist=context_tree.query(Q)[0];cov=np.cov(boots,rowvar=False);sd=np.sqrt(np.maximum(np.einsum('ni,ij,nj->n',AQ,cov,AQ),0))
    flags=dict(inside_training_fit_hull=margin>=-1e-10,inside_declared_XY_fit_context=contextdist<=.4+1e-10,
        nearest_road_le2m=d[:,0]<=2.,conditional_SD_le0p05m=sd<=.05,nearby_road_median_residual_le0p10m=localres<=.10)
    valid=np.logical_and.reduce(list(flags.values())).reshape(shape)
    celluv=uv[:-1,:-1].astype(float)+.5;cellxy=celluv@M.T+O;cellQ=cellxy.reshape(-1,2);AC=design(cellQ,centre)
    cellworld=np.c_[cellQ,AC@b].reshape(*celluv.shape[:2],3);cellnormal=normals(cellQ,centre,b).reshape(cellworld.shape)
    requested=np.zeros(celluv.shape[:2],bool)
    for row in views:requested|=query_cell_mask(cellworld,row['view'],row['mask']['inside_allowed_edit'])
    ringdist=cKDTree(cellworld[requested]).query(cellworld.reshape(-1,3))[0].reshape(requested.shape)
    desired_ring=(ringdist<=.2)&~requested;desired=requested|desired_ring
    cellvalid=valid[:-1,:-1]&valid[:-1,1:]&valid[1:,:-1]&valid[1:,1:]
    recon=requested&cellvalid;ring=desired_ring&cellvalid;asset=recon|ring
    cell_sd=np.sqrt(np.maximum(np.einsum('ni,ij,nj->n',AC,cov,AC),0)).reshape(asset.shape)
    coefficients=load(D/'loo_selection/all_training_model_coefficients.npz');ids=coefficients['medoid_indices'];A=design(X[ids,:2],centre)[:,:3]
    w=np.minimum(1.,.03/np.maximum(abs(A@b[:3]-X[ids,2]),1e-12));inverse=np.linalg.pinv(A.T@(w[:,None]*A))
    leverage=np.einsum('ni,ij,nj->n',AQ[:,:3],inverse,AQ[:,:3]);trainlev=np.einsum('ni,ij,nj->n',A,inverse,A)
    legacy=np.zeros(shape,bool);ob=old['bounds_uvxy'].astype(int);over=(uu>=ob[0])&(uu<ob[2])&(vv>=ob[1])&(vv<ob[3]);legacy[over]=old['strict_witness_mask_legacy'][vv[over]-ob[1],uu[over]-ob[0]]
    pose=np.eye(4);pose[:3,3]=np.r_[centre,design(centre[None],centre)@b]
    vertex_asset=np.zeros(shape,bool)
    for dy,dx in [(0,0),(0,1),(1,0),(1,1)]:vertex_asset[dy:dy+asset.shape[0],dx:dx+asset.shape[1]]|=asset
    gp=out/'candidate_geometry.npz'
    np.savez_compressed(gp,world_xyz=world,normal_world=normal,uv_global=uv,geometry_valid=valid,candidate_mask=valid,renderable_mask=np.zeros(shape,bool),
        geometry_source_kind=np.where(valid,2,0).astype(np.uint8),inferred_geometry_mask=valid,strict_witness_mask_legacy=legacy,
        cell_world_xyz=cellworld,cell_normal_world=cellnormal,cell_uv_global=celluv,cell_candidate_mask=cellvalid,
        cell_reconstruction_mask=recon,cell_blend_ring_mask=ring,cell_asset_mask=asset,reconstruction_mask=recon,blend_ring_mask=ring,asset_mask=asset,
        cell_geometry_source_kind=np.where(cellvalid,2,0).astype(np.uint8),vertex_asset_support_mask=vertex_asset,
        requested_cell_reconstruction_mask=requested,requested_cell_asset_mask=desired,unsupported_requested_cell_mask=desired&~cellvalid,
        fit_coefficients=b,fit_XY_centre=centre,bootstrap_coefficients=boots,height_bootstrap_SD_m=sd.reshape(shape),cell_height_bootstrap_SD_m=cell_sd,
        training_fit_hull_margin_m=margin.reshape(shape),declared_fit_context_distance_m=contextdist.reshape(shape),nearest_official_road_distance_m=d[:,0].reshape(shape),
        nearby_road_median_abs_profile_residual_m=localres.reshape(shape),training_fit_leverage=leverage.reshape(shape),training_fit_leverage_over_max=(leverage/trainlev.max()).reshape(shape),
        bounds_uvxy=np.asarray([u0,v0,u1,v1]),spacing_m=np.asarray(.02),world_from_patch=pose,local_xyz=(world-pose[:3,3]).astype(np.float32),
        cell_local_xyz=(cellworld-pose[:3,3]).astype(np.float32),**{k:v.reshape(shape) for k,v in flags.items()})
    coverage=[]
    for row in views:
        q=row['uv'];cu=np.floor(q[:,0]).astype(int)-u0;cv=np.floor(q[:,1]).astype(int)-v0
        inbounds=row['good']&(cu>=0)&(cv>=0)&(cu<asset.shape[1])&(cv<asset.shape[0]);pu=np.clip(cu,0,asset.shape[1]-1);pv=np.clip(cv,0,asset.shape[0]-1)
        supported=inbounds&asset[pv,pu];H,W=row['mask']['inside_allowed_edit'].shape;canvas=np.zeros((H,W),bool);canvas[row['xy'][:,1],row['xy'][:,0]]=supported
        np.savez_compressed(out/f'{row["raw"]:03d}_requested_coverage.npz',inside_allowed_edit=row['mask']['inside_allowed_edit'],new_plane_geometry_supported=canvas,
            geometry_unknown=row['mask']['inside_allowed_edit']&~canvas)
        coverage.append(dict(raw=row['raw'],fixed_inside_allowed_edit=int(len(q)),supported_candidate_asset=int(supported.sum()),geometry_unknown=int((~supported).sum()),
            new_requested_UV_min=q[row['good']].min(0).tolist(),new_requested_UV_max=q[row['good']].max(0).tolist(),
            mask=row['mask_pin'],camera=row['camera_pin']))
    result=dict(schema='R10.timewindow17_reprojected_inferred_candidate.v1',complete=True,decision='CANDIDATE_READY_MODEL_GATES_PASS_LOCAL_QA_PENDING',
        geometry_npz=pin(gp),candidate_contract=pin(out/'candidate_contract.json'),selected_model='plane',reference_frames=np.arange(90,171,5).tolist(),
        geometry_source_kind=2,all_new_geometry_model_inferred=True,geometry_accepted_for_zero_step=False,
        shape_vertex=list(shape),shape_cell=list(asset.shape),bounds_uvxy=[int(u0),int(v0),int(u1),int(v1)],spacing_m=.02,
        requested_before_support_cells=int(desired.sum()),supported_asset_cells=int(asset.sum()),geometry_unknown_requested_cells=int((desired&~cellvalid).sum()),
        reconstruction_cells=int(recon.sum()),blend_ring_cells=int(ring.sum()),asset_area_m2=float(asset.sum()*abs(np.linalg.det(M))),
        candidate_support_rejection_counts={k:int((~v).sum()) for k,v in flags.items()},asset_SD_m=stats(cell_sd[asset]),
        training_original_points=len(X),training_original_frames=np.unique(raw).tolist(),
        current_17_training_height_gate=pin(D/'training_physical_gate_summary.json'),fixed_heldout_evaluation=pin(diagpath),
        fixed_edit_rays_recomputed_from_new_plane=True,old1844_and_b5df_unchanged=True,globalUV_transform_unchanged=True,
        source_RGB_not_written=True,new_source_projection_required=True,coverage=coverage,
        GPU_forward=0,backward=0,updates=0,seconds=time.time()-start)
    write(out/'candidate_receipt.json',result);print(json.dumps(dict(complete=True,geometry=result['geometry_npz'],bounds=result['bounds_uvxy'],asset_cells=int(asset.sum()),unknown_cells=result['geometry_unknown_requested_cells'],coverage=coverage)),flush=True)


if __name__=='__main__':main()
