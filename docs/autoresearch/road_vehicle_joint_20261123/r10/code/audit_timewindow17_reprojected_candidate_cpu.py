"""Independent chosen-plane readback and calibrated five-view geometric review."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull,cKDTree
from scipy.ndimage import binary_erosion
from PIL import Image,ImageDraw
from diagnose_nonplanar_inference_cpu import load,read,write,pin,stats,design
from audit_expanded_surface_candidate_cpu import intersect
from record_limited_geometry_qa_cpu import refit_plane

RAWS=[86,110,134,179,198]


def png(p,x):Image.fromarray(np.rint(np.clip(x,0,1)*255).astype(np.uint8)).save(p)


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    C,out=a.campaign_root,a.output;out.mkdir(parents=True,exist_ok=False);start=time.time();R=C/'road_underbody_local_20261123_r10';D=R/'audit/timewindow17_geometry_v1'
    gp=D/'candidate_reprojected_v1/candidate_geometry.npz';g=load(gp);sp=D/'loo_selection/training_only_CV_selection.json';selection=read(sp)
    model=load(D/'loo_selection/expanded_geometry_DIAGNOSTIC.npz');assert selection['one_SE_preferred_model']=='plane'
    for key,ref in [('fit_coefficients','plane__coefficients'),('fit_XY_centre','fit_XY_centre'),('bootstrap_coefficients','plane__bootstrap_coefficients')]:
        assert g[key].dtype==model[ref].dtype and g[key].tobytes()==model[ref].tobytes()
    b,centre=g['fit_coefficients'],g['fit_XY_centre'];t=load(D/'loo_selection/training_points.npz');X,raw=t['world_xyz'],t['source_raw']
    mc=load(D/'loo_selection/all_training_model_coefficients.npz');ids=mc['medoid_indices']
    independent=refit_plane(np.c_[X[ids,:2]-centre,X[ids,2]]);fit_error=float(abs(np.r_[independent,np.zeros(3)]-b).max());assert fit_error<1e-9
    grid=load(C/'road_shared_appearance_20261122_r7/shared_grid_geometry.npz');M,O=grid['world_from_uv_xy'][:2,:2],grid['world_from_uv_xy'][:2,3];inv=np.linalg.inv(M)
    uv=g['uv_global'];asset=g['cell_asset_mask'];shape=g['candidate_mask'].shape
    assert np.issubdtype(uv.dtype,np.integer) and np.array_equal(g['cell_uv_global'],uv[:-1,:-1]+.5)
    assert np.array_equal(g['world_xyz'][...,:2],uv@M.T+O)
    assert np.array_equal(g['cell_world_xyz'][...,:2],g['cell_uv_global']@M.T+O)
    Q=g['world_xyz'][...,:2].reshape(-1,2);A=design(Q,centre)
    zerror=float(abs(A@b-g['world_xyz'][...,2].ravel()).max());assert zerror<1e-12
    cellQ=g['cell_world_xyz'][...,:2].reshape(-1,2);AC=design(cellQ,centre)
    czerror=float(abs(AC@b-g['cell_world_xyz'][...,2].ravel()).max());assert czerror<1e-12
    normal=np.array([-b[1],-b[2],1.]);normal/=np.linalg.norm(normal)
    assert np.array_equal(np.broadcast_to(normal,g['normal_world'].shape),g['normal_world'])
    sd=np.std(AC[::97]@g['bootstrap_coefficients'].T,axis=1,ddof=1)
    sd_error=float(abs(sd-g['cell_height_bootstrap_SD_m'].ravel()[::97]).max());assert sd_error<1e-10
    oldg=load(R/'audit/local_surface_expanded_candidate_v1/expanded_candidate_geometry.npz')
    context=cKDTree(oldg['cell_world_xyz'][oldg['requested_cell_asset_mask'],:2]).query(Q)[0]
    hull=ConvexHull(X[:,:2]);margin=-(Q@hull.equations[:,:2].T+hull.equations[:,2]).max(-1)
    d,near=cKDTree(X[:,:2]).query(Q,k=16);residual=design(X[:,:2],centre)@b-X[:,2];localres=np.median(abs(residual[near]),axis=-1)
    expected=(margin>=-1e-10)&(context<=.4+1e-10)&(d[:,0]<=2)&(g['height_bootstrap_SD_m'].ravel()<=.05)&(localres<=.10)
    assert np.array_equal(expected.reshape(shape),g['candidate_mask'])
    assert np.array_equal(asset,g['cell_reconstruction_mask']|g['cell_blend_ring_mask'])
    assert not (asset&~g['cell_candidate_mask']).any() and not (asset&(g['cell_geometry_source_kind']!=2)).any()
    assert not g['renderable_mask'].any()
    u0,v0,u1,v1=g['bounds_uvxy'].astype(int);previews=[];rows=[];contact=read(R/'audit/ground_contact_prior_raw_GT_v2/ground_contact_prior_audit.json')
    for rr in RAWS:
        folder=out/f'{rr:03d}';folder.mkdir();cp=C/f'road_shared_appearance_20261122_r7/projection_cache_v1/{rr:03d}/view_cache.npz';view=load(cp)
        mp=R/f'edit_footprint_masks_v2/{rr:03d}/scope_masks.npz';m=load(mp);H,W=m['inside_allowed_edit'].shape
        P,raygood,depth=intersect(view,(H,W),b,centre);quv=(P[...,:2]-O)@inv.T
        ix=np.floor(np.nan_to_num(quv[...,0],nan=-1e8)).astype(int)-u0;iy=np.floor(np.nan_to_num(quv[...,1],nan=-1e8)).astype(int)-v0
        inbounds=raygood&(ix>=0)&(iy>=0)&(ix<asset.shape[1])&(iy<asset.shape[0]);pu,pv=np.clip(ix,0,asset.shape[1]-1),np.clip(iy,0,asset.shape[0]-1)
        candidate=inbounds&g['cell_candidate_mask'][pv,pu];supported=inbounds&asset[pv,pu];body,K=m['actor_protected'],m['K_protected'];display=supported&~body&~K
        original=load(D/f'candidate_reprojected_v1/{rr:03d}_requested_coverage.npz')
        assert np.array_equal(supported&m['inside_allowed_edit'],original['new_plane_geometry_supported'])
        E,camK=view['world_to_camera'],view['camera_K'];camera=P@E[:3,:3].T+E[:3,3];q=camera@camK.T;screen=q[...,:2]/camera[...,2,None]-.5
        yy,xx=np.mgrid[:H,:W];check=m['inside_allowed_edit']&raygood
        rayerror=float(np.maximum(abs(screen[...,0]-xx),abs(screen[...,1]-yy))[check].max());assert rayerror<1e-6
        hatch=np.where((((xx//8)+(yy//8))%2)[...,None],.11,.17)*np.ones((H,W,3));neutral=hatch.copy();neutral[supported]=[.49,.61,.65]
        gridline=(abs(quv[...,0]-np.rint(quv[...,0]/10)*10)<.55)|(abs(quv[...,1]-np.rint(quv[...,1]/10)*10)<.55)
        neutral[supported&gridline]=[.77,.85,.85]
        photo=np.clip(view['photo_pinhole_srgb'],0,1);overlay=photo.copy();overlay[display]=.68*photo[display]+.32*neutral[display];overlay[display&gridline]=[.1,.8,.72]
        edge=supported&~binary_erosion(supported);overlay[edge&~body&~K]=[1,.6,.1]
        image=Image.fromarray(np.rint(overlay*255).astype(np.uint8));draw=ImageDraw.Draw(image);contacts=[]
        for row in contact['GT_bottom_rows']:
            if row['raw']!=rr or not row.get('present'):continue
            pts=np.asarray(row['bottom_world_corners_and_centre']);cam=pts@E[:3,:3].T+E[:3,3];sq=cam@camK.T;xy=sq[:,:2]/cam[:,2,None]-.5
            if (cam[:,2]>.1).all():draw.line([tuple(xy[k]) for k in [0,1,3,2,0]],fill=(60,235,70) if row['alias']=='a20' else (235,80,215),width=2)
            contacts.append(dict(alias=row['alias'],road_minus_GT_bottom_world_Z_m=(design(pts[:,:2],centre)@b-pts[:,2]).tolist(),GT_not_fitted=True,actor_not_moved=True))
        png(folder/'geometry_surface_only.png',neutral);image.save(folder/'geometry_photo_context.png')
        np.savez_compressed(folder/'projection_and_scope.npz',atlas_uv_global=np.where(inbounds[...,None],quv,np.nan).astype(np.float32),
            camera_depth_m=np.where(inbounds,depth,0).astype(np.float32),candidate_support=candidate,asset_scope=supported,
            inside_allowed_edit=m['inside_allowed_edit'],K_protected=K,actor_protected=body,display_visible=display)
        regions={}
        for name in ['inside_allowed_edit','full_original_band','inside_replacement','K_protected','actor_protected']:
            mask=m[name];part=dict(no_ray=int((mask&~raygood).sum()),outside_grid=int((mask&raygood&~inbounds).sum()),
                outside_model_support=int((mask&inbounds&~candidate).sum()),supported_context_not_asset=int((mask&candidate&~supported).sum()),asset_supported=int((mask&supported).sum()))
            assert sum(part.values())==int(mask.sum());regions[name]=dict(fixed_pixels=int(mask.sum()),partition=part)
        row=dict(raw=rr,camera=pin(cp),fixed_masks=pin(mp),regions=regions,ray_reprojection_max_abs_px=rayerror,GT_contact=contacts,
            geometry_shader_not_material=True,original_photo_is_display_context_only=True,behind_body_ground_not_occlusion_acceptance=True)
        write(folder/'projection_receipt.json',row);rows.append(row)
        panel=Image.new('RGB',(1152,410),(20,23,30));panel.paste(Image.fromarray(np.rint(photo*255).astype(np.uint8)).resize((576,384)),(0,26));panel.paste(image.resize((576,384)),(576,26))
        ImageDraw.Draw(panel).text((8,7),f'{rr:03d} original photo | newly reprojected17-frame plane + GT bottom',fill=(235,235,235));previews.append(panel)
    sheet=Image.new('RGB',(1152,410*5),(20,23,30))
    for j,im in enumerate(previews):sheet.paste(im,(0,410*j))
    sheet.save(out/'five_view_geometry_contactsheet.png')
    result=dict(schema='R10.independent_timewindow17_candidate_geometry_audit.v1',complete=True,decision='NUMERIC_INTEGRITY_PASS_VISUAL_PENDING',
        geometry_npz=pin(gp),selection=pin(sp),independent_plane_fit_max_abs=fit_error,vertex_Z_analytic_max_abs_m=zerror,
        cell_Z_analytic_max_abs_m=czerror,explicit512_bootstrap_SD_error_m=sd_error,all_support_masks_recomputed_exact=True,
        worldUV_transform_unchanged=True,old_assets_not_modified=True,all_geometry_source_kind=2,
        asset_cells=int(asset.sum()),requested_unknown_cells=int(g['unsupported_requested_cell_mask'].sum()),
        asset_conditional_parameter_SD_m=stats(g['cell_height_bootstrap_SD_m'][asset]),conditional_SD_not_full_model_error=True,
        asset_vertices_training_hull_outside=int((g['vertex_asset_support_mask']&(g['training_fit_hull_margin_m']<0)).sum()),
        asset_vertices_declared_fit_context_outside=int((g['vertex_asset_support_mask']&(g['declared_fit_context_distance_m']>.4)).sum()),
        rows=rows,contactsheet=pin(out/'five_view_geometry_contactsheet.png'),geometry_accepted_for_zero_step=False,
        new_source_projection_and_acceptance_required=True,root_review_pending=True,GPU_forward=0,backward=0,updates=0,seconds=time.time()-start)
    write(out/'independent_geometry_audit.json',result);print(json.dumps(dict(complete=True,receipt=pin(out/'independent_geometry_audit.json'),rows=rows)),flush=True)


if __name__=='__main__':main()
