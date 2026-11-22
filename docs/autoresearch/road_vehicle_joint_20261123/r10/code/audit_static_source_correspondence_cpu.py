"""All-frame cached static-road nearest-XY agreement; no alignment or fitting.

Three fixed XY radii, full point denominators, directed and mutual pairs.
No Z/residual/image filtering; self matches are not validation evidence.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from scipy.spatial import cKDTree


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--points',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();start=time.time();assert not a.output.exists()
    with np.load(a.points) as z:
        X=z['world_xyz'];raw=z['source_raw'];lidar=z['source_lidar_row_id'];actual=z['actual_requested_spatial_mask'];enum=z['official_semantic']
    frames=np.unique(raw);assert np.array_equal(frames,np.arange(90,171,5)) and np.isfinite(X).all() and np.isin(enum,[18,19]).all()
    radii=np.asarray([.02,.05,.10]);domains=['actual_requested','fit_context'];n=len(frames);shape=(2,3,n,n)
    mats={k:np.full(shape,np.nan) for k in ['coverage_fraction','median_signed_Z_m','mean_abs_Z_m','median_abs_Z_m','p95_abs_Z_m','mutual_median_signed_Z_m','mean_delta_X_m','mean_delta_Y_m']}
    counts=np.zeros(shape,np.int32);mutual_counts=np.zeros_like(counts);denominators=np.zeros((2,n),np.int32);records=[];pair_parts=[]
    frameids=[np.flatnonzero(raw==rr) for rr in frames];ref_trees=[cKDTree(X[ids,:2]) for ids in frameids]
    for di,name in enumerate(domains):
        for i,rr in enumerate(frames):
            ti=frameids[i];ti=ti[actual[ti]] if di==0 else ti;denominators[di,i]=len(ti);tree=cKDTree(X[ti,:2])
            for j,ss in enumerate(frames):
                ri=frameids[j];distance,nearest=ref_trees[j].query(X[ti,:2]);matched=ri[nearest];delta=X[ti]-X[matched]
                back=tree.query(X[matched,:2])[1];mutual=back==np.arange(len(ti))
                for riadius,radius in enumerate(radii):
                    m=distance<=radius;v=delta[m,2];mm=m&mutual;counts[di,riadius,i,j]=m.sum();mutual_counts[di,riadius,i,j]=mm.sum()
                    mats['coverage_fraction'][di,riadius,i,j]=float(m.mean()) if len(m) else np.nan
                    row=dict(domain=name,target_raw=int(rr),reference_raw=int(ss),radius_m=float(radius),target_original_points=len(ti),matches=int(m.sum()),mutual_matches=int(mm.sum()),self_pair=bool(i==j),coverage=float(m.mean()) if len(m) else None)
                    if len(v):
                        values=dict(median_signed_Z_m=float(np.median(v)),mean_abs_Z_m=float(abs(v).mean()),median_abs_Z_m=float(np.median(abs(v))),p95_abs_Z_m=float(np.quantile(abs(v),.95)),
                            mean_delta_X_m=float(delta[m,0].mean()),mean_delta_Y_m=float(delta[m,1].mean()))
                        if mm.any():values['mutual_median_signed_Z_m']=float(np.median(delta[mm,2]))
                        for k,value in values.items():mats[k][di,riadius,i,j]=value
                        row.update(values)
                    records.append(row)
                if di==0 and i!=j:
                    m=distance<=.10;pair_parts.append(dict(target_raw=np.full(m.sum(),rr,np.int16),reference_raw=np.full(m.sum(),ss,np.int16),target_cache_row=ti[m],reference_cache_row=matched[m],target_lidar_row=lidar[ti[m]],reference_lidar_row=lidar[matched[m]],
                        delta_worldXYZ=delta[m],XY_distance_m=distance[m],mutual_nearest=mutual[m],target_enum=enum[ti[m]],reference_enum=enum[matched[m]]))
    a.output.mkdir(parents=True);np.savez_compressed(a.output/'agreement_matrices.npz',frames=frames,radii_m=radii,domains=np.asarray(domains),target_denominators=denominators,matched_counts=counts,mutual_counts=mutual_counts,**mats)
    np.savez_compressed(a.output/'actual_requested_pairs_10cm.npz',**{k:np.concatenate([x[k] for x in pair_parts]) for k in pair_parts[0]})
    (a.output/'all_directed_pair_statistics.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    worst={}
    for domain in domains:
        for radius in radii:
            pool=[x for x in records if x['domain']==domain and x['radius_m']==radius and not x['self_pair'] and x['matches']>=10]
            worst[f'{domain}_{radius:.2f}m']=sorted(pool,key=lambda x:x['p95_abs_Z_m'],reverse=True)[:8]
    try:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
        for j,radius in enumerate(radii):
            z=mats['median_signed_Z_m'][0,j].copy();cov=mats['coverage_fraction'][0,j].copy();np.fill_diagonal(z,np.nan);np.fill_diagonal(cov,np.nan)
            im=axes[0,j].imshow(z*100,vmin=-20,vmax=20,cmap='coolwarm');axes[0,j].set_title(f'XY<={radius*100:g}cm signed Z median (cm)');fig.colorbar(im,ax=axes[0,j],shrink=.7)
            im=axes[1,j].imshow(cov,vmin=0,vmax=1,cmap='viridis');axes[1,j].set_title('matched / all original target points');fig.colorbar(im,ax=axes[1,j],shrink=.7)
            for ax in axes[:,j]:ax.set(xticks=np.arange(n),yticks=np.arange(n),xticklabels=frames,yticklabels=frames,xlabel='reference frame',ylabel='target frame');ax.tick_params(axis='x',rotation=90)
        fig.suptitle('All 17 frames; actual requested domain; diagonal excluded; no Z filtering or alignment');fig.savefig(a.output/'actual_requested_agreement_matrix.png',dpi=160);plt.close(fig)
    except ImportError:pass
    out=dict(schema='R10.fixed_timewindow17_static_source_correspondence_CPU.v1',complete=True,CPU_only=True,nativeF=0,nativeB=0,updates=0,new_models_fitted=0,source_points=pin(a.points),frames=frames.tolist(),
        timewindow_selection='90..170 inclusive every5, predetermined by task window and official TOP availability. All17 kept; temporal window and sample count both changed relative to prior8-frame control.',
        original_context_points=len(X),official_enums=np.unique(enum).tolist(),radii_m=radii.tolist(),domains=domains,target_denominators={name:denominators[i].tolist() for i,name in enumerate(domains)},
        matching='Directed target points to nearest XY reference point in same fixed fit context; <=radius; no Z, residual, RGB, GT-box, or model-agreement selection. Mutual nearest supplement also saved.',
        missing_matches_are_unknown_not_zero_error=True,diagonal_self_matches_not_quality_evidence=True,pose_or_point_alignment_applied=False,frames_removed=False,
        matrices=pin(a.output/'agreement_matrices.npz'),pair_points=pin(a.output/'actual_requested_pairs_10cm.npz'),full_statistics=pin(a.output/'all_directed_pair_statistics.json'),
        largest_p95_pairs_with_at_least10_matches=worst,
        interpretation='Spatial-neighbour agreement is a correspondence-quality diagnostic, not guaranteed identical surfaces, dense physical ground truth, or calibrated probability. The independent original-point LOO and physical gates remain required.',
        geometry_approved=False,seconds=time.time()-start)
    (a.output/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(complete=True,summary=pin(a.output/'summary.json'),frames=len(frames),points=len(X),denominators=out['target_denominators']),ensure_ascii=False))


if __name__=='__main__':main()
