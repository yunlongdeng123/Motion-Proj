"""Replay saved training-fold coefficients on all original points; no refit or heldout."""
import argparse,json
from pathlib import Path
import numpy as np
from diagnose_nonplanar_inference_cpu import load,read,write,pin,stats,design
from diagnose_expanded_actual_scope_cpu import labels


def describe(error):
    s=stats(abs(error))
    return dict(points=len(error),signed_error_m=stats(error),absolute_height_m=s,
        fixed_height_bands_pass=bool(len(error)>=50 and s['median']<=.04 and s['p95']<=.10))


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign-root',type=Path,required=True);a=p.parse_args();C=a.campaign_root
    R=C/'road_underbody_local_20261123_r10';D=R/'audit/expanded_curvature_loo_v1'
    output=D/'training_fold_original_height_audit.json';assert not output.exists()
    selection=read(D/'training_only_CV_selection.json');data=load(D/'training_points.npz');coef=load(D/'all_training_model_coefficients.npz')
    g=load(R/'audit/local_surface_expanded_candidate_v1/expanded_candidate_geometry.npz');grid=load(C/'road_shared_appearance_20261122_r7/shared_grid_geometry.npz')
    X,raw=data['world_xyz'],data['source_raw'];A=design(X[:,:2],data['fit_XY_centre']);frames=selection['train_frames']
    M,O=grid['world_from_uv_xy'][:2,:2],grid['world_from_uv_xy'][:2,3];uv=(X[:,:2]-O)@np.linalg.inv(M).T
    requested=labels(uv,g,'requested_cell_asset_mask');bins=np.floor(uv/100).astype(int)
    model_rows=[];payload={}
    for model in selection['models']:
        name=model['name'];folds=coef[name+'__LOO_coefficients'];full=coef[name+'__coefficients'];loo_error=np.full(len(X),np.nan)
        rows=[];spatial=[];all_loofull=True;all_looactual=True;all_fittedactual=True
        for i,rr in enumerate(frames):
            mask=raw==rr;loo_error[mask]=A[mask]@folds[i]-X[mask,2]
            loo_full=describe(loo_error[mask]);loo_actual=describe(loo_error[mask&requested]);fit_actual=describe((A@full-X[:,2])[mask&requested])
            assert abs(loo_full['absolute_height_m']['mean']-model['fold_mean_absolute_height_error_m'][i])<1e-12
            rows.append(dict(raw=rr,LOO_context=loo_full,LOO_actual_requested=loo_actual,fitted_actual_requested=fit_actual))
            all_loofull&=loo_full['fixed_height_bands_pass'];all_looactual&=loo_actual['fixed_height_bands_pass'];all_fittedactual&=fit_actual['fixed_height_bands_pass']
            for u,v in np.unique(bins[mask&requested],axis=0):
                use=mask&requested&(bins[:,0]==u)&(bins[:,1]==v)
                spatial.append(dict(raw=rr,u0=int(100*u),u1=int(100*(u+1)),v0=int(100*v),v1=int(100*(v+1)),
                    LOO=describe(loo_error[use]),fitted=describe((A@full-X[:,2])[use])))
        assert np.isfinite(loo_error).all()
        payload[name+'__LOO_error_m']=loo_error;payload[name+'__fitted_error_m']=A@full-X[:,2]
        model_rows.append(dict(name=name,frames=rows,actual_requested_spatial_bins_2m=spatial,
            all8_LOO_context_height_pass=bool(all_loofull),all8_LOO_actual_requested_height_pass=bool(all_looactual),
            all8_fitted_actual_requested_height_pass=bool(all_fittedactual)))
    np.savez_compressed(D/'training_original_point_predictions.npz',world_xyz=X,source_raw=raw,actual_requested_mask=requested,**payload)
    result=dict(schema='R10.expanded_original_point_LOO_height_audit.v1',complete=True,
        decision='DIAGNOSTIC_NO_AUTOMATIC_GEOMETRY_APPROVAL',selection=pin(D/'training_only_CV_selection.json'),
        original_points=pin(D/'training_points.npz'),original_context_point_denominator=len(X),
        original_actual_requested_point_denominator=int(requested.sum()),all_original_point_denominators_preserved=True,
        fixed_height_bands=dict(min_points_per_frame=50,absolute_median_max_m=.04,absolute_p95_max_m=.10),
        points_in_small_spatial_bins_below50_are_insufficient_not_silently_dropped=True,
        models=model_rows,raw_predictions=pin(D/'training_original_point_predictions.npz'),
        no_refit=True,no_heldout_read=True,GPU_forward=0,backward=0,updates=0)
    write(output,result)
    print(json.dumps(dict(receipt=pin(output),summary=[dict(name=x['name'],LOO_context_all8=x['all8_LOO_context_height_pass'],
        LOO_actual_all8=x['all8_LOO_actual_requested_height_pass'],fitted_actual_all8=x['all8_fitted_actual_requested_height_pass'],
        raw60=x['frames'][0],raw70=x['frames'][1]) for x in model_rows])),flush=True)


if __name__=='__main__':main()
