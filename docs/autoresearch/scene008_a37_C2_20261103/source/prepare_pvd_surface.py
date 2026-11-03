"""CPU adapter: actual PVD complete points -> local point patches + fixed real a6.

No point generation, meshing, upsampling, recentering, or bounds fitting occurs here.
The producer must provide the actual completion and an explicit coordinate contract.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).resolve().parent))
from prepare_actor_surface import ROOT,read,sha,atomic_json,atomic_npz,real_anchors,colors_for_anchors,stats


def actual_points(path, contract_path, point_space, expected_size):
    contract=read(contract_path)
    size=np.asarray(contract['whole_box_lwh_m'],float)
    if size.shape!=(3,) or not np.allclose(size,expected_size,atol=5e-6,rtol=0):
        raise ValueError('PVD producer must retain the fixed annotated whole a6 box')
    # Missing fields fail explicitly; no inference from this partial cloud is permitted.
    if contract.get('partial_bounds_scaled') is not False or contract.get('partial_bounds_centered') is not False:
        raise ValueError('Coordinate contract must explicitly rule out partial-bounds scale/centering')
    T=np.asarray(contract['pvd_to_actor_native_4x4'],float)
    if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1],atol=1e-10):
        raise ValueError('Invalid producer PVD->native affine matrix')
    if abs(np.linalg.det(T[:3,:3]))<1e-12:raise ValueError('Singular PVD->native coordinate transform')
    if not contract.get('normalization_source'):raise ValueError('Normalization provenance must be explicit')
    with np.load(path,allow_pickle=False) as a:
        raw=np.asarray(a['points_pvd_normalized'],float) if 'points_pvd_normalized' in a.files else None
        if point_space=='actor_native_m':
            points=np.asarray(a['points_actor_native_m'],float)
        else:
            if raw is None:raise ValueError('points_pvd_normalized is required in normalized point mode')
            points=raw@T[:3,:3].T+T[:3,3]
        if points.ndim==3 and points.shape[0]==1:points=points[0]
        if raw is not None and raw.ndim==3 and raw.shape[0]==1:raw=raw[0]
        if points.ndim!=2 or points.shape[1]!=3 or len(points)<32 or not np.isfinite(points).all():
            raise ValueError('Actual completion must be a finite Nx3 array, N>=32')
        inverse_error=None
        if raw is not None:
            reconstructed=raw@T[:3,:3].T+T[:3,3]
            if reconstructed.shape!=points.shape:raise ValueError('Raw/native PVD point arrays have different shapes')
            inverse_error=float(np.abs(points-reconstructed).max())
            if inverse_error>2e-5:raise ValueError('PVD raw/native inverse transform mismatch')
        generated=np.asarray(a['generated_mask'],bool) if 'generated_mask' in a.files else None
        rows=np.asarray(a['conditioning_observed_row_indices'],np.int64) if 'conditioning_observed_row_indices' in a.files else None
        if generated is None or generated.shape!=(len(points),):raise ValueError('Actual PVD output needs per-point generated_mask')
        if len(points)!=2048 or generated.sum()!=1848 or (~generated).sum()!=200:
            raise ValueError('PVD contract is exactly200 fixed actual partial points +1848 generated =2048 complete')
        if rows is None or rows.shape!=(200,):raise ValueError('200 conditioning_observed_row_indices are required for actual-source audit')
    return points,contract,dict(point_space=point_space,producer_transform=T.tolist(),
        raw_native_inverse_max_abs_error_m=inverse_error,
        normalization_source=contract['normalization_source'],
        normalization_policy='producer-supplied official training normalization + fixed annotated whole box; adapter never infers axes, scale, or center from partial points'),dict(generated_mask=generated,conditioning_observed_row_indices=rows)


def pca_point_patches(points):
    """Estimate patches only where actual completion contains local 2-D support.

    Keep actual sample positions. A 5mm duplicate grid selects an existing point,
    never a centroid or newly fabricated completion point.
    """
    _,first=np.unique(np.floor(points/.005).astype(np.int64),axis=0,return_index=True)
    original_index=np.sort(first);p=points[original_index]
    tree=cKDTree(p);k=min(16,len(p));distance,index=tree.query(p,k=k)
    if k<4:raise ValueError('Too few distinct actual PVD points for point patches')
    spacing=np.median(distance[:,1:min(4,k)],axis=1)
    radius=np.minimum(.25,np.maximum(4*spacing,.03))
    weight=(distance<=radius[:,None]).astype(float)
    q=p[index]-p[:,None]
    mean=(q*weight[:,:,None]).sum(1)/np.maximum(weight.sum(1),1)[:,None]
    q=q-mean[:,None]
    covariance=np.einsum('nki,nkj,nk->nij',q,q,weight)/np.maximum(weight.sum(1),1)[:,None,None]
    eigenvalue,eigenvector=np.linalg.eigh(covariance);normal=eigenvector[:,:,0]
    # The normal direction comes from PCA. Its sign uses the fixed actor origin.
    flip=(normal*p).sum(1)<0;normal[flip]*=-1
    valid=(weight.sum(1)>=4)&(eigenvalue[:,1]>1e-7)&(spacing>1e-6)
    confidence=np.clip((eigenvalue[:,1]-eigenvalue[:,0])/np.maximum(eigenvalue[:,2],1e-10),0,1)
    # No positive minimum is imposed: close points cannot become oversized splats.
    sigma=np.minimum(.60*spacing,.060)
    sigma=np.minimum(sigma,.45*radius)
    return dict(points=p,normal=normal,sigma=sigma,normal_valid=valid,
        normal_confidence=confidence,knn_spacing=spacing,normal_radius=radius,original_index=original_index)


def point_projection_plot(complete, retained, observed, size, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(1,3,figsize=(11,3.8),constrained_layout=True)
    for ax,(x,y,xlabel,ylabel) in zip(axes,[(0,1,'x length (m)','y width (m)'),(0,2,'x length (m)','z height (m)'),(1,2,'y width (m)','z height (m)')]):
        ax.scatter(complete[:,x],complete[:,y],s=2,c='#9ca3af',alpha=.35,label='Actual PVD complete')
        if len(retained):ax.scatter(retained[:,x],retained[:,y],s=4,c='#c2410c',alpha=.6,label='Retained PVD patch')
        ax.scatter(observed[::10,x],observed[::10,y],s=.7,c='#2563eb',alpha=.45,label='Fixed real observations')
        ax.add_patch(Rectangle((-size[x]/2,-size[y]/2),size[x],size[y],fill=False,color='#dc2626',lw=1))
        ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.set_aspect('equal');ax.grid(alpha=.15)
    axes[0].legend(fontsize=7,loc='best')
    fig.suptitle('Actual PVD point completion in fixed a6 coordinates | local patches, no mesh or shape repair',fontsize=11)
    fig.savefig(path,dpi=180);plt.close(fig)


def prepare(args):
    start=time.monotonic();root=args.root.resolve();inputs=root/'inputs'
    output=args.output or inputs/'actor_surface_pvd_seed0.npz'
    expected=np.asarray(read(inputs/'a6_prior_contract.json')['full_box_size_lwh_m'],float)
    complete,contract,coordinate,source_audit=actual_points(args.point_cloud,args.transform_contract,args.point_space,expected)
    geometry=np.load(inputs/'a6_geometry_minimal.npz');actual=np.asarray(geometry['canonical_xyz'],float)
    conditioning_rows=source_audit['conditioning_observed_row_indices']
    if (conditioning_rows<0).any() or (conditioning_rows>=len(actual)).any():raise ValueError('Conditioning point row outside actual real observations')
    conditioning_error=float(np.abs(complete[~source_audit['generated_mask']]-actual[conditioning_rows]).max())
    if conditioning_error>2e-5:raise ValueError('PVD fixed200 conditioning points drifted from actual canonical observations')
    observed,observed_normal,observed_sigma,observed_conf,inverse,voxel_keys=real_anchors(actual)
    if len(observed)!=5976:raise ValueError('Fixed real a6 anchor contract changed')
    patch=pca_point_patches(complete);candidate=patch['points']
    inside=(np.abs(candidate)<=expected/2+1e-6).all(1)
    distance,_=cKDTree(observed).query(candidate)
    is_generated=source_audit['generated_mask'][patch['original_index']]
    valid_normal=patch['normal_valid'];keep=inside&valid_normal&is_generated&(distance>=.04)
    generated=candidate[keep]
    if not len(generated):raise ValueError('Actual PVD completion has no legal independently supported prior patch')
    xyz=np.concatenate([observed,generated]);normal=np.concatenate([observed_normal,patch['normal'][keep]])
    sigma=np.r_[observed_sigma,patch['sigma'][keep]]
    observed_flag=np.arange(len(xyz))<len(observed);prior_flag=~observed_flag
    evidence=dict(np.load(inputs/'a6_colour_evidence.npz'))
    color,color_known,color_frame,color_policy=colors_for_anchors(xyz,observed_flag,evidence)
    # Observation mapping remains in the immutable real-voxel block.
    observed_nearest=inverse.astype(np.int32)
    nn_any=cKDTree(xyz).query(actual)[1].astype(np.int32)
    complete_tree=cKDTree(complete);obs_distance=complete_tree.query(observed)[0]
    low,high=complete.min(0),complete.max(0)
    aabb_lower=np.linalg.norm(np.maximum(np.maximum(low-observed,observed-high),0),axis=1)
    bottom=float(actual[:,2].min())
    atomic_npz(output,canonical_xyz=xyz.astype(np.float32),canonical_normals=normal.astype(np.float32),
        sigma_tangent_m=sigma.astype(np.float32),colour_srgb=color.astype(np.float32),
        observed_anchor=observed_flag,colour_observed=color_known,prior_origin=prior_flag,
        observed_points=actual.astype(np.float32),observed_nearest_anchor=observed_nearest,
        observed_nearest_any_anchor=nn_any,observed_voxel_keys=voxel_keys,
        observed_source_raw_frame=geometry['raw_frame'],observed_source_cloud_row=geometry['source_cloud_row'],
        observed_anchor_normal_confidence=observed_conf.astype(np.float32),colour_source_frame_count=color_frame,
        full_box_lwh_m=expected,prior_source_complete_index=patch['original_index'][keep].astype(np.int32),
        prior_knn_spacing_m=patch['knn_spacing'][keep].astype(np.float32),
        prior_normal_radius_m=patch['normal_radius'][keep].astype(np.float32),
        prior_normal_confidence=patch['normal_confidence'][keep].astype(np.float32),
        pvd_complete_generated_mask=source_audit['generated_mask'],pvd_conditioning_observed_row_indices=conditioning_rows,
        pvd_complete_points_actor_native_m=complete.astype(np.float32),
        observed_anchor_to_pvd_complete_point_distance_m=obs_distance.astype(np.float32))
    plot=output.with_name(output.stem+'_views.png');point_projection_plot(complete,generated,actual,expected,plot)
    receipt=dict(status='READY_ACTUAL_PVD_POINT_PATCHES_QUALITY_UNACCEPTED',output=str(output),sha256=sha(output),
        quality_accepted=False,formal_eligible=False,source_type='actual pretrained PVD DDPM complete point output; local PCA patch representation',
        input_point_cloud=str(args.point_cloud.resolve()),input_point_cloud_sha256=sha(args.point_cloud),
        transform_contract=str(args.transform_contract.resolve()),transform_contract_sha256=sha(args.transform_contract),
        producer_contract=contract,coordinate_contract=coordinate,
        completion_points=len(complete),complete_unique_5mm_points=len(candidate),
        fixed_conditioning_count=200,generated_model_point_count=1848,conditioning_actual_source_max_abs_error_m=conditioning_error,
        complete_bounds_native_m=[low.tolist(),high.tolist()],complete_extent_native_m=(high-low).tolist(),
        complete_points_inside_full_box_fraction=float((np.abs(complete)<=expected/2+1e-6).all(1).mean()),
        fixed_real_anchor_count=len(observed),retained_prior_patch_count=len(generated),all_anchor_count=len(xyz),
        prior_rejected_outside_full_box=int((~inside).sum()),
        fixed_conditioning_points_not_marked_prior=int((~is_generated).sum()),
        prior_rejected_degenerate_local_PCA=int((inside&is_generated&~valid_normal).sum()),
        prior_removed_near_fixed_real=int((inside&valid_normal&is_generated&(distance<.04)).sum()),
        actual_point_positions_changed=False,manual_shape_or_upsampling=False,mesh_constructed=False,
        patch_normal_policy='PCA16 on actual complete points within min(0.25m,max(4*kNNspacing,0.03m)); require4 neighbors and second covariance eigenvalue>1e-7; reject degenerate patches; sign relative fixed actor origin',
        sigma_tangent_policy='min(0.60 * median physical distance to nearest3 other complete points,0.060m,0.45*normal-radius); no artificial positive lower scale',
        sigma_tangent_prior_stats_m=dict(min=float(patch['sigma'][keep].min()),median=float(np.median(patch['sigma'][keep])),max=float(patch['sigma'][keep].max())),
        patch_geometry_limit='tangent patches approximate only local actual completion support; no connectivity, watertightness, mesh area, or closed wheels asserted',
        observed_geometry_fixed=True,observed_anchor_policy='same5cm real voxel centroids and mapping, bytewise-comparable to existing Omni adapter real block',
        measured_color_anchor_count=int(color_known.sum()),inferred_color_anchor_count=int((~color_known).sum()),color_policy=color_policy,
        observed_to_actual_PVD_complete_point_distance_m=stats(obs_distance),distance_semantics='point-to-point nearest distances; NOT point-to-surface distances',
        observed_to_completion_AABB_distance_lower_bound_m=stats(aabb_lower),
        bottom_coverage=dict(raw_box_bottom_m=float(-expected[2]/2),lowest_measured_m=bottom,
            no_measurement_band_m=[float(-expected[2]/2),bottom],prior_patch_count_in_unobserved_lower_band=int(((generated[:,2]<bottom)&(generated[:,2]>=-expected[2]/2)).sum()),
            policy='all completion geometry and neutral color in this missing lower band remain inferred'),
        view_plot=str(plot),GPU_used=False,training_updates=0,wall_seconds=time.monotonic()-start,old_assets_modified=False)
    atomic_json(output.with_name(output.stem+'_receipt.json'),receipt)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--point-cloud',required=True,type=Path)
    parser.add_argument('--transform-contract',required=True,type=Path)
    parser.add_argument('--point-space',choices=['actor_native_m','pvd_normalized'],default='actor_native_m')
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path)
    prepare(parser.parse_args())
