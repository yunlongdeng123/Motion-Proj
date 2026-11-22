"""Original official-road denominators inside the requested footprint, before rejection.

No residual-based point selection, refit, heldout tuning, RGB, asset or GPU writes.
"""
import argparse, json, time, csv
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from diagnose_nonplanar_inference_cpu import load, read, write, pin, stats, design


def labels(q, g, name):
    u0, v0, _, _ = g['bounds_uvxy'].astype(int)
    ix = np.floor(q[:, 0]).astype(int)-u0
    iy = np.floor(q[:, 1]).astype(int)-v0
    mask = g[name]
    inside = (ix >= 0) & (iy >= 0) & (ix < mask.shape[1]) & (iy < mask.shape[0])
    result = np.zeros(len(q), bool)
    result[inside] = mask[iy[inside], ix[inside]]
    return result


def metrics(residual):
    return dict(points=int(len(residual)), signed_plane_minus_official_m=stats(residual),
                absolute_plane_minus_official_m=stats(abs(residual)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campaign-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); C, out = a.campaign_root, a.output
    out.mkdir(parents=True, exist_ok=False); start = time.time()
    R = C/'road_underbody_local_20261123_r10'
    path = R/'audit/local_surface_expanded_candidate_v1/expanded_candidate_geometry.npz'
    oldpath = R/'audit/local_surface_candidate_v2/candidate_geometry.npz'
    g, old = load(path), load(oldpath)
    grid = load(C/'road_shared_appearance_20261122_r7/shared_grid_geometry.npz')
    M, O = grid['world_from_uv_xy'][:2, :2], grid['world_from_uv_xy'][:2, 3]
    inv = np.linalg.inv(M)
    evidence_path = R/'audit/ground_contact_prior_raw_GT_v2/GT_bottom_and_road_evidence.npz'
    evidence = load(evidence_path)
    X, raw = evidence['official_training_road_worldXYZ'], evidence['official_training_road_source_raw']
    point_uv = (X[:, :2]-O)@inv.T
    requested = labels(point_uv, g, 'requested_cell_asset_mask')
    accepted = labels(point_uv, g, 'cell_asset_mask')
    reconstruction = labels(point_uv, g, 'requested_cell_reconstruction_mask')
    zwindow = abs(X[:, 2]+20.817) <= 1.5
    residual = design(X[:, :2], g['fit_XY_centre'])@g['fit_coefficients']-X[:, 2]
    # Keep *all* official ROAD/LANE points spatially inside the pre-screen mask.
    assert not (accepted & ~requested).any()
    masks = dict(requested_before_credibility_screen=requested,
                 accepted_asset=requested & accepted,
                 rejected_requested_asset=requested & ~accepted,
                 reconstruction_before_credibility_screen=reconstruction,
                 blend_ring_before_credibility_screen=requested & ~reconstruction)
    rows=[]; spatial=[]
    for rr in np.unique(raw):
        frame = raw == rr
        domains = {}
        for name, mask in masks.items():
            use=frame & mask
            domains[name] = metrics(residual[use])
            domains[name]['outside_prior_broad_Z_window_count'] = int((use & ~zwindow).sum())
        rows.append(dict(raw=int(rr), domains=domains))
        ids=np.flatnonzero(frame & requested)
        bins=np.floor(point_uv[ids]/100).astype(int)  # fixed 100 texel = about2m bins, origin globalUV0
        for u,v in np.unique(bins,axis=0):
            group=ids[(bins[:,0]==u)&(bins[:,1]==v)]
            spatial.append(dict(raw=int(rr),u0=int(u*100),u1=int((u+1)*100),v0=int(v*100),v1=int((v+1)*100),
                **metrics(residual[group]),accepted_points=int(accepted[group].sum()),
                rejected_points=int((~accepted[group]).sum()),outside_Z_window_count=int((~zwindow[group]).sum())))
    # Point-wise readout retains the complete pre-screen requested domain.
    cell_uv=g['cell_uv_global']; desired=g['requested_cell_asset_mask']; cell_xy=g['cell_world_xyz'][...,:2]
    Q=cell_xy[desired]
    # Broad XY restriction only for efficient nearest search; no residual or Z pruning.
    bbox=np.stack([g['world_xyz'][...,:2].reshape(-1,2).min(0)-3,
                   g['world_xyz'][...,:2].reshape(-1,2).max(0)+3])
    nearby=(X[:,:2]>=bbox[0]).all(-1)&(X[:,:2]<=bbox[1]).all(-1)
    pool=X[nearby]; poolraw=raw[nearby]; poolres=residual[nearby]
    distance, ids=cKDTree(pool[:,:2]).query(Q,k=16)
    median_signed=np.median(poolres[ids],axis=-1)
    median_abs=np.median(abs(poolres[ids]),axis=-1)
    p95_abs=np.quantile(abs(poolres[ids]),.95,axis=-1)
    closest_res=poolres[ids[:,0]]
    frame_count=np.zeros(len(Q),np.int16)
    direction=np.zeros((len(Q),4),bool)
    per_frame=[]
    for rr in np.unique(poolraw):
        sub=poolraw==rr; frameP=pool[sub]; frameR=poolres[sub]
        d, near=cKDTree(frameP[:,:2]).query(Q)
        frame_count += d<=.8
        dx=frameP[near,:2]-Q;quad=(dx[:,0]>=0)*2+(dx[:,1]>=0)
        for k in range(4):direction[:,k]|=(d<=1.)&(quad==k)
        per_frame.append(dict(raw=int(rr),requested_cells=len(Q),nearest_distance_m=stats(d),
            cells_with_point_within0p8m=int((d<=.8).sum()),
            signed_nearest_point_residual_m=stats(frameR[near]),
            signed_nearest_point_residual_within0p8m_m=stats(frameR[near][d<=.8])))
    # Readout using original height bands is diagnostic, not an automatic approval.
    height_agreement=(median_abs<=.04)&(p95_abs<=.10)
    support_agreement=height_agreement&(frame_count>=3)&(direction.sum(-1)>=3)
    on_asset=g['cell_asset_mask'][desired]
    old_at_query=labels(cell_uv[desired],old,'cell_asset_mask')
    new_only=~old_at_query
    partitions={}
    for name,use in dict(all_requested=np.ones(len(Q),bool),accepted_requested=on_asset,
                         rejected_requested=~on_asset,old_asset_part=old_at_query,new_only=new_only,
                         new_only_accepted=new_only&on_asset).items():
        partitions[name]=dict(cells=int(use.sum()),
            closest_official_distance_m=stats(distance[use,0]),
            nearest16_median_signed_residual_m=stats(median_signed[use]),
            nearest16_median_abs_residual_m=stats(median_abs[use]),
            nearest16_p95_abs_residual_m=stats(p95_abs[use]),
            local_height_agreement_cells=int((use&height_agreement).sum()),
            local_height_agreement_fraction=float(height_agreement[use].mean()) if use.any() else None,
            height_and_multiframe_direction_proxy_cells=int((use&support_agreement).sum()),
            height_and_multiframe_direction_proxy_fraction=float(support_agreement[use].mean()) if use.any() else None)
    def canvas(value,fill=np.nan):
        result=np.full(desired.shape,fill,dtype=value.dtype if value.dtype==bool else np.float32)
        result[desired]=value;return result
    np.savez_compressed(out/'actual_scope_point_residuals.npz',
        original_point_worldXYZ=X,original_point_raw=raw,original_point_globalUV=point_uv,
        original_point_requested_spatial_mask=requested,original_point_accepted_asset_mask=accepted,
        original_point_prior_Zwindow_flag=zwindow,original_point_signed_plane_minus_ground_m=residual,
        requested_cell_mask=desired,accepted_cell_mask=g['cell_asset_mask'],
        nearest16_median_signed_residual_m=canvas(median_signed),
        nearest16_median_absolute_residual_m=canvas(median_abs),nearest16_p95_absolute_residual_m=canvas(p95_abs),
        closest_official_distance_m=canvas(distance[:,0]),closest_official_signed_residual_m=canvas(closest_res),
        local_height_agreement_diagnostic=canvas(height_agreement,False),
        height_and_multiframe_direction_proxy=canvas(support_agreement,False),
        distinct_frames_within0p8m=canvas(frame_count),
        directions_from_nearest_perframe_within1m=canvas(direction.sum(-1)),
        new_only_requested_mask=canvas(new_only,False),bounds_uvxy=g['bounds_uvxy'])
    flat=[]
    for r in spatial:
        flat.append(dict(raw=r['raw'],u0=r['u0'],u1=r['u1'],v0=r['v0'],v1=r['v1'],points=r['points'],
            accepted_points=r['accepted_points'],rejected_points=r['rejected_points'],
            signed_median_m=r['signed_plane_minus_official_m'].get('median'),
            signed_mean_m=r['signed_plane_minus_official_m'].get('mean'),
            absolute_median_m=r['absolute_plane_minus_official_m'].get('median'),
            absolute_p95_m=r['absolute_plane_minus_official_m'].get('p95')))
    with (out/'frame_spatial_2m_bins.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    result=dict(schema='R10.actual_requested_scope_official_road_residual_audit.v1',complete=True,
        decision='DIAGNOSTIC_NO_GEOMETRY_PROMOTION',candidate_geometry=pin(path),
        official_evidence=pin(evidence_path),original_official_training_points_total=len(X),
        requested_spatial_original_point_denominator=int(requested.sum()),
        accepted_asset_original_point_count=int(accepted.sum()),
        rejected_requested_asset_original_point_count=int((requested&~accepted).sum()),
        no_point_residual_or_Z_filter_applied=True,all_spatial_original_denominators_preserved=True,
        prior_Z_window_recorded_only=True,source_raw_frames=np.unique(raw).tolist(),
        pointwise_frame_residuals=rows,spatial_bins_global_uv100texel=spatial,
        requested_cell_neighbourhood_readout=partitions,nearest_same_frame_readout=per_frame,
        height_agreement_definition='nearest16 original official points: median absolute<=.04m and p95 absolute<=.10m; fixed original height bands, diagnostic only',
        support_proxy_definition='height agreement plus >=3 distinct frames with a nearest point <=.8m and >=3 quadrants from nearest-per-frame points <=1m; diagnostic lower-bound directional support, not strict measured witness',
        neighbourhood_is_not_independent_measured_validation=True,
        original_heldout_not_read_for_this_spatial_diagnostic=True,
        old_asset_cells_kept=int((old_at_query&on_asset).sum()),
        old_asset_cells_rejected_by_new_screen=int((old_at_query&~on_asset).sum()),
        candidate_geometry_accepted_for_zero_step=False,model_refitted=False,
        GPU_forward=0,backward=0,updates=0,seconds=time.time()-start,
        diagnostics_npz=pin(out/'actual_scope_point_residuals.npz'),spatial_csv=pin(out/'frame_spatial_2m_bins.csv'))
    write(out/'actual_scope_residual_diagnostic.json',result)
    print(json.dumps(dict(complete=True,receipt=pin(out/'actual_scope_residual_diagnostic.json'),
        points_requested=int(requested.sum()),pointwise_frame=[dict(raw=r['raw'],actual=r['domains']['requested_before_credibility_screen']) for r in rows],
        cell_partitions=partitions)),flush=True)


if __name__=='__main__': main()
