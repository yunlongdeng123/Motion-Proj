"""Read existing official-road caches; compare same-XY cross-frame heights.

No TFRecord, image, GT actor box, heldout frame, model fitting, or GPU access.
"""
from pathlib import Path
import argparse,csv,hashlib,json,time
import numpy as np
from scipy.spatial import cKDTree


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)
def stat(x):
    x=np.asarray(x);return dict(count=len(x),median=float(np.median(x)),mean=float(x.mean()),p05=float(np.quantile(x,.05)),p95=float(np.quantile(x,.95)),abs_median=float(np.median(abs(x))),abs_p95=float(np.quantile(abs(x),.95))) if len(x) else dict(count=0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();start=time.time();r=a.root;assert not a.output.exists()
    scope=r/'audit/local_surface_expanded_candidate_v1/actual_scope_residual_v1';cache=r/'audit/ground_contact_prior_raw_GT_v2/GT_bottom_and_road_evidence.npz'
    with np.load(cache) as z:
        xyz=z['official_training_road_worldXYZ'];raw=z['official_training_road_source_raw'];lid=z['official_training_road_lidar_row_id']
    with np.load(scope/'actual_scope_point_residuals.npz') as z:
        assert np.array_equal(xyz,z['original_point_worldXYZ']) and np.array_equal(raw,z['original_point_raw'])
        requested=z['original_point_requested_spatial_mask'];residual=z['original_point_signed_plane_minus_ground_m']
    assert np.array_equal(np.unique(raw),[60,70,80,95,110,125,145,165])
    assert int(requested.sum())==1426
    records=[];pairs=[]
    for target_raw in [60,70]:
        target=np.flatnonzero(requested&(raw==target_raw));t=xyz[target];target_tree=cKDTree(t[:,:2])
        for reference_raw in [80,95,110,125,145,165]:
            reference=np.flatnonzero(raw==reference_raw);v=xyz[reference];tree=cKDTree(v[:,:2]);distance,index=tree.query(t[:,:2]);ok=distance<=.10
            query=np.flatnonzero(ok);ti=target[ok];ri=reference[index[ok]]
            back=target_tree.query(xyz[ri,:2])[1];mutual=back==query
            dz=xyz[ti,2]-xyz[ri,2];dr=residual[ti]-residual[ri]
            records.append(dict(target_raw=target_raw,reference_raw=reference_raw,target_requested_points=len(target),matched_points=len(ti),matched_fraction=float(ok.mean()),
                reference_in_requested_points=int(requested[ri].sum()),XY_distance_m=stat(distance[ok]),target_minus_reference_Z_m=stat(dz),
                target_minus_reference_old_plane_residual_m=stat(dr),mutual_nearest_points=int(mutual.sum()),mutual_target_minus_reference_Z_m=stat(dz[mutual]),
                matching='nearest XY within fixed 0.10m; no Z, residual, RGB, OBB or per-frame offset filtering'))
            for k,(i,j) in enumerate(zip(ti,ri)):
                pairs.append([target_raw,reference_raw,int(i),int(j),int(lid[i]),int(lid[j]),float(distance[ok][k]),float(dz[k]),float(dr[k]),bool(mutual[k])])
    csv_path=scope/'frame_spatial_2m_bins.csv'
    with csv_path.open() as f:bins=list(csv.DictReader(f))
    by={(int(b['raw']),b['u0'],b['v0']):b for b in bins};bucket=[]
    for row in bins:
        target_raw=int(row['raw'])
        if target_raw not in [60,70]:continue
        for reference_raw in [80,95,110,125,145,165]:
            other=by.get((reference_raw,row['u0'],row['v0']))
            if other is None:continue
            bucket.append(dict(target_raw=target_raw,reference_raw=reference_raw,u0=int(row['u0']),v0=int(row['v0']),target_points=int(row['points']),reference_points=int(other['points']),
                target_signed_median_m=float(row['signed_median_m']),reference_signed_median_m=float(other['signed_median_m']),
                target_minus_reference_signed_median_m=float(row['signed_median_m'])-float(other['signed_median_m'])))
    a.output.mkdir(parents=True);columns=['target_raw','reference_raw','target_cache_row','reference_cache_row','target_lidar_row','reference_lidar_row','XY_distance_m','target_minus_reference_Z_m','target_minus_reference_old_plane_residual_m','mutual_nearest']
    values=np.asarray(pairs,np.float64);np.savez_compressed(a.output/'matched_pairs.npz',columns=np.asarray(columns),pairs=values)
    with (a.output/'matched_2m_bins.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(bucket[0]) if bucket else ['target_raw']);w.writeheader();w.writerows(bucket)
    summary=dict(schema='R10.same_XY_cross_frame_height_CPU_diagnostic.v1',complete=True,CPU_only=True,nativeF=0,nativeB=0,updates=0,
        original_road_cache=pin(cache),actual_scope_points=pin(scope/'actual_scope_point_residuals.npz'),input_bins=pin(csv_path),official_points=len(xyz),requested_points=int(requested.sum()),
        target_frames=[60,70],reference_frames=[80,95,110,125,145,165],fixed_XY_nearest_radius_m=.10,no_Z_or_residual_filter=True,no_models_fitted=True,GT_boxes_and_sealed_heldout_not_read=True,
        pairs=records,matched_2m_bins=len(bucket),matched_pairs=pin(a.output/'matched_pairs.npz'),matched_bins=pin(a.output/'matched_2m_bins.csv'),
        interpretation='Matched XY pairs reduce confounding from different road positions. Remaining height disagreement can include measurement geometry, scan/time effects or calibration; it is not a proof of a unique cause or an automatic correction. 2m buckets are a coarser supplementary comparison.',
        geometry_approval=False,seconds=time.time()-start)
    (a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(complete=True,summary=pin(a.output/'summary.json'),pairs=[{k:x[k] for k in ['target_raw','reference_raw','matched_points','target_requested_points','target_minus_reference_Z_m','mutual_nearest_points']} for x in records]),ensure_ascii=False))


if __name__=='__main__':main()
