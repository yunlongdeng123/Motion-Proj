"""Fresh CPU readback and limited independent recommendation, never a full-road success gate."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from diagnose_nonplanar_inference_cpu import load,read,write,pin,stats,design,medoids


def refit_plane(X):
    A=np.c_[np.ones(len(X)),X[:,:2]];b=np.linalg.lstsq(A,X[:,2],rcond=None)[0]
    for _ in range(8):
        w=np.minimum(1.,.03/np.maximum(abs(A@b-X[:,2]),1e-12));b=np.linalg.lstsq(A*np.sqrt(w[:,None]),X[:,2]*np.sqrt(w),rcond=None)[0]
    return b


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign-root',type=Path,required=True);a=p.parse_args();C=a.campaign_root
    R=C/'road_underbody_local_20261123_r10';D=R/'audit/local_surface_candidate_v2';path=D/'candidate_geometry.npz';g=load(path)
    producer=read(D/'candidate_geometry_receipt.json');assert producer['candidate_geometry']['sha256']==pin(path)['sha256']
    strict=load(R/'audit/local_surface_geometry_v1/local_surface_geometry.npz');selpath=R/'audit/curvature_ridge_loo_v1/training_only_CV_selection.json';selection=read(selpath)
    diagnostic=read(R/'audit/curvature_ridge_loo_v1/curvature_ridge_diagnostic.json');chosen=next(x for x in diagnostic['finalists'] if x['name']==selection['one_SE_preferred_model'])
    assert chosen['name']=='plane'
    points=load(R/'audit/official_ground_geometry_v1/local_original_point_evidence.npz');contract=read(R/'audit/local_surface_geometry_v1/geometry_qualification_contract.json')
    use=points['ground']&np.isin(points['raw'],contract['train_raw']);X=points['world_xyz'];raw=points['raw'];centre=g['fit_XY_centre'];origin=np.asarray(contract['XY_fit_context_bbox'])[0]
    ids=medoids(X,np.flatnonzero(use),origin);b=refit_plane(np.c_[X[ids,:2]-centre,X[ids,2]])
    fitdiff=float(abs(np.r_[b,np.zeros(3)]-g['fit_coefficients']).max());assert fitdiff<1e-9
    checks={};trains=np.unique(raw[use]);plane_scores=[]
    for rr in trains:
        ii=medoids(X,np.flatnonzero(use&(raw!=rr)),origin);bb=refit_plane(np.c_[X[ii,:2]-centre,X[ii,2]]);test=X[use&(raw==rr)]
        plane_scores.append(float(abs(np.c_[np.ones(len(test)),test[:,:2]-centre]@bb-test[:,2]).mean()))
    expected=next(x for x in selection['models'] if x['name']=='plane')['fold_mean_absolute_height_error_m'];cvdiff=float(abs(np.array(plane_scores)-expected).max());assert cvdiff<1e-9
    best=min(selection['models'],key=lambda x:x['mean_fold_MAE_m']);limit=best['mean_fold_MAE_m']+best['SE_fold_MAE_m'];eligible=[x for x in selection['models'] if x['mean_fold_MAE_m']<=limit+1e-15]
    preferred=max(eligible,key=lambda x:(x['kind']=='plane',x['lambda_m4']))['name'];assert preferred=='plane'
    for name in ['world_xyz','normal_world','cell_world_xyz','cell_normal_world','local_xyz','cell_local_xyz','height_bootstrap_SD_m','cell_height_bootstrap_SD_m']:
        assert np.isfinite(g[name]).all()
    normal_error=float(abs(np.linalg.norm(g['normal_world'],axis=-1)-1).max());assert normal_error<1e-12
    cellnormal_error=float(abs(np.linalg.norm(g['cell_normal_world'],axis=-1)-1).max());assert cellnormal_error<1e-12
    assert np.array_equal(g['direct_witness_mask'],strict['geometry_valid']) and int(g['direct_witness_mask'].sum())==837
    assert np.array_equal(g['cell_asset_mask'],g['cell_reconstruction_mask']|g['cell_blend_ring_mask'])
    assert not (g['cell_reconstruction_mask']&g['cell_blend_ring_mask']).any()
    assert not (g['cell_asset_mask']&~g['cell_candidate_mask']).any()
    assert not (g['cell_asset_mask']&(g['cell_geometry_source_kind']!=2)).any()
    assert np.array_equal(g['asset_mask'],g['cell_asset_mask']) and np.array_equal(g['candidate_mask'],g['geometry_valid'])
    uv=g['uv_global'];assert np.issubdtype(uv.dtype,np.integer)
    assert np.array_equal(np.diff(uv[...,0],axis=1),np.ones((uv.shape[0],uv.shape[1]-1),int))
    assert np.array_equal(np.diff(uv[...,1],axis=0),np.ones((uv.shape[0]-1,uv.shape[1]),int))
    assert np.array_equal(g['cell_uv_global'],uv[:-1,:-1]+.5)
    cellz=design(g['cell_world_xyz'][...,:2].reshape(-1,2),centre)@g['fit_coefficients']
    zerror=float(abs(cellz.reshape(g['cell_candidate_mask'].shape)-g['cell_world_xyz'][...,2]).max());assert zerror<1e-12
    pose=g['world_from_patch'];worldreplay=g['local_xyz'].astype(float)@pose[:3,:3].T+pose[:3,3];local_error=float(abs(worldreplay-g['world_xyz']).max());assert local_error<5e-7
    projectionpath=D/'five_view_geometry_v1/five_view_geometry_receipt.json';projection=read(projectionpath);assert projection['candidate']['sha256']==pin(path)['sha256']
    red=[];tight=[]
    for row in projection['rows']:
        if row['raw'] not in [86,110]:continue
        rr=row['regions']['red_diagnostic'];assert rr['fixed_pixels']==rr['explicit_asset_scope_supported'] and rr['partition']['asset_behind_body_or_K']==0
        red.append(dict(raw=row['raw'],**rr))
        original=load(R/f'audit/local_surface_inference_v1_replay1/{row["raw"]:03d}_target_coverage_DIAGNOSTIC.npz')
        pr=load(D/f'five_view_geometry_v1/{row["raw"]:03d}/projection_and_scope.npz');mask=original['focused_residual_candidate']
        tight.append(dict(raw=row['raw'],fixed_tight_pixels=int(mask.sum()),asset_supported=int((mask&pr['asset_scope_projected']).sum()),
            visible_not_body_K=int((mask&pr['display_context_visible']).sum()),not_covered=int((mask&~pr['asset_scope_projected']).sum())))
    assert chosen['fixed_original_heldout_checks_pass']
    contactpath=R/'audit/ground_contact_prior_raw_GT_v2/ground_contact_prior_audit.json';contact=read(contactpath);assert contact['raw_GT_used_directly_for_bottom_and_dimensions']
    bodycontact=[]
    for row in contact['GT_bottom_rows']:
        if not row.get('present') or row['alias']!='a20' or row['raw'] not in [86,110]:continue
        bottom=np.array(row['bottom_world_corners_and_centre'])[-1];z=float(design(bottom[None,:2],centre)@g['fit_coefficients'])
        actor=next(x for x in contact['A20_world_quantile_rows'] if x['raw']==row['raw'])
        bodycontact.append(dict(raw=row['raw'],candidate_road_Z_m=z,GT_bottom_centre_Z_m=float(bottom[2]),
            road_minus_GT_bottom_m=z-bottom[2],
            original_high_opacity_actor_mean_q01_above_GT_bottom_m=actor['mean_Z_above_GT_bottom_at_same_XY_m_quantiles']['0.01'],
            actor_mean_quantile_is_not_GT_ground=True))
    result=dict(schema='R10.independent_limited_local_geometry_review.v1',complete=True,
        decision='PASS_FOR_LIMITED_ZERO_STEP_MODEL_INFERENCE_CONTROL',candidate_geometry=pin(path),final_atlas_sha256=None,
        geometry_source_kind=2,all_approved_asset_cells_are_model_inferred=True,approved_cell_mask_key='cell_asset_mask',approved_cell_mask_count=int(g['cell_asset_mask'].sum()),
        candidate_geometry_accepted_for_zero_step=True,requires_separate_root_authorization=True,requires_separate_shared_appearance_acceptance=True,
        full_road_geometry_final_PASS=False,whole_scene_restoration_PASS=False,hidden_hole_measured_GT_claim=False,
        independent_CPU_checks=dict(fit_coefficient_max_abs=fitdiff,training_LOO_plane_fold_MAE_max_abs=cvdiff,one_SE_rule_replayed=preferred,
            all_XYZ_normal_uncertainty_finite=True,normal_unit_max_error=normal_error,cell_normal_unit_max_error=cellnormal_error,
            integer_global_UV_exact=True,cell_UV_plus_half_exact=True,cell_Z_analytic_error_m=zerror,local_world_replay_error_m=local_error,
            strict837_mask_byteexact=True,asset_partition_and_inference_labels_exact=True),
        source_physical_checks=pin(R/'audit/physical_crosscheck_v1/crosscheck_receipt.json'),official_ground_evidence=pin(R/'audit/official_ground_geometry_v1/official_ground_point_receipt.json'),
        training_selection=pin(selpath),fixed_heldout_height_normal=chosen['heldout'],training_only_selection_and_untrimmed_heldout=True,
        target_coverage=dict(red_fixed_denominators_retained=red,tight_fixed_denominators_retained=tight,all5_full_band_partitions_preserved=pin(projectionpath),
            blue_diagnostic_only_partially_covered=True,evaluation_three_color_diagnostic_zeros_mean_not_measured=True),
        conditional_geometry_uncertainty_asset_SD_m=stats(g['cell_height_bootstrap_SD_m'][g['cell_asset_mask']]),uncertainty_is_not_calibrated_probability=True,
        GT_contact_evidence=pin(contactpath),GT_contact=bodycontact,GT_or_actor_minZ_not_fitted_as_dense_truth=True,
        independent_visual_review=dict(reviewer='native_clean_photo_regions',frames=[86,110,134,179,198],original_resolution=[1152,768],
            decision='PASS_FOR_LOCAL_GEOMETRY_ZERO_STEP_PROBE',root_concurred=False,
            observations='All five native-resolution context projections inspected. No visible folding or gross frame/axis mismatch. The inferred ground grid lies behind the originally photographed A9 car that is removed in the target scene. Grid/photo overlay is a geometric diagnostic and does not demonstrate unoccluded appearance or final native occlusion.',
            contactsheet=pin(D/'five_view_geometry_v1/five_view_geometry_contactsheet.png')),
        remaining_scope='A20 original vehicle and native context remain intact for the actual joint zero-step probe. Poor old-parent geometry is not a hard fitting target. No production geometry, full blue-band coverage, or final joint image-quality conclusion follows from this limited gate.',
        original_strict_and_failed_profiles_preserved=True,new_GPU_forward=0,updates=0)
    write(D/'independent_limited_geometry_review.json',result);print(json.dumps(dict(complete=True,receipt=pin(D/'independent_limited_geometry_review.json'),contact=bodycontact,tight=tight)),flush=True)


if __name__=='__main__':main()
