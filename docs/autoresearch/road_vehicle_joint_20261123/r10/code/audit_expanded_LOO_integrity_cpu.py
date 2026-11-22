"""Independent cached fold/medoid/prediction/one-SE audit; no model fitting."""
from pathlib import Path
import argparse,ast,hashlib,json,time
import numpy as np
from scipy.spatial import cKDTree


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)
def bank(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def design(xy,centre):
    x,y=(xy-centre).T;return np.c_[np.ones(len(x)),x,y,x*x,x*y,y*y]
def medoid_ids(X,ids,origin):
    groups={}
    for i in ids:groups.setdefault(tuple(np.floor((X[i,:2]-origin)/.1).astype(int)),[]).append(int(i))
    selected=[]
    for cell in sorted(groups):
        rows=np.asarray(groups[cell]);q=X[rows];centre=np.median(q,axis=0);selected.append(rows[np.argmin(np.linalg.norm(q-centre,axis=1))])
    return np.asarray(selected,np.int64)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();start=time.time();r=a.root;d=r/'audit/expanded_curvature_loo_v1';assert not a.output.exists()
    contract=read(d/'curvature_selection_contract.json');table=read(d/'training_only_CV_selection.json');receipt=read(d/'training_data_receipt.json')
    cache=bank(d/'training_points.npz');coeff=bank(d/'all_training_model_coefficients.npz');pred=bank(d/'training_original_point_predictions.npz')
    X,raw=cache['world_xyz'],cache['source_raw'];frames=np.asarray(contract['train_frames']);assert np.array_equal(np.unique(raw),frames)
    g=bank(contract['parent_expansion_scope']['path']);xy=g['cell_world_xyz'][g['requested_cell_asset_mask'],:2];origin=xy.min(0)-.4;centre=(xy.min(0)+xy.max(0))/2
    evidence=Path(contract['input_official_road_evidence']['path'])
    with np.load(evidence) as z:
        allX=z['official_training_road_worldXYZ'];allraw=z['official_training_road_source_raw'];allids=z['official_training_road_lidar_row_id']
    finite=np.isfinite(allX).all(1);distance=np.full(len(allX),np.inf);distance[finite]=cKDTree(xy).query(allX[finite,:2])[0]
    keep=finite&np.isin(allraw,frames)&(distance<=.4)
    assert np.array_equal(keep,cache['all_original_evidence_context_mask']) and np.array_equal(np.flatnonzero(keep),cache['original_evidence_selected_indices'])
    assert np.array_equal(X,allX[keep]) and np.array_equal(raw,allraw[keep]) and np.array_equal(cache['source_lidar_row_id'],allids[keep])
    assert np.array_equal(cache['fit_XY_centre'],centre) and np.array_equal(cache['medoid_origin'],origin)
    ids=medoid_ids(X,np.arange(len(X)),origin);assert np.array_equal(ids,coeff['medoid_indices'])
    assert np.array_equal(pred['world_xyz'],X) and np.array_equal(pred['source_raw'],raw)
    A=design(X[:,:2],centre);results=[];folds=[]
    for j,rr in enumerate(frames):
        tr=np.flatnonzero(raw!=rr);test=raw==rr;mi=medoid_ids(X,tr,origin);row=receipt['folds'][j]
        assert not (raw[mi]==rr).any() and row['original_eval_points']==int(test.sum()) and row['training_medoids']==len(mi)
        folds.append(dict(leftout_raw=int(rr),original_eval_points=int(test.sum()),training_medoids=len(mi),heldout_frame_not_in_fit_medoids=True))
    for model in table['models']:
        name=model['name'];full=A@coeff[name+'__coefficients']-X[:,2];loo=np.zeros(len(X));vals=[];height=[]
        for j,rr in enumerate(frames):
            m=raw==rr;loo[m]=A[m]@coeff[name+'__LOO_coefficients'][j]-X[m,2];vals.append(float(abs(loo[m]).mean()))
            q=m&pred['actual_requested_mask'];e=abs(loo[q]);height.append(dict(raw=int(rr),actual_requested_original_points=int(q.sum()),median_abs_m=float(np.median(e)),p95_abs_m=float(np.quantile(e,.95)),old_height_band_pass=bool(np.median(e)<=.04 and np.quantile(e,.95)<=.10)))
        assert np.max(abs(full-pred[name+'__fitted_error_m']))<1e-12 and np.max(abs(loo-pred[name+'__LOO_error_m']))<1e-12
        vals=np.asarray(vals);mean=float(vals.mean());se=float(vals.std(ddof=1)/np.sqrt(len(vals)))
        assert np.max(abs(vals-model['fold_mean_absolute_height_error_m']))<1e-12 and abs(mean-model['mean_fold_MAE_m'])<1e-12 and abs(se-model['SE_fold_MAE_m'])<1e-12
        results.append(dict(name=name,kind=model['kind'],lambda_m4=model['lambda_m4'],mean=mean,SE=se,actual_requested_height=height))
    best=min(results,key=lambda z:z['mean']);bound=best['mean']+best['SE'];chosen=max([z for z in results if z['mean']<=bound+1e-15],key=lambda z:(z['kind']=='plane',z['lambda_m4']))
    assert best['name']==table['best_mean_model'] and chosen['name']==table['one_SE_preferred_model'] and abs(bound-table['best_mean_plus_SE_limit_m'])<1e-12
    code=r/'code/select_expanded_curvature_loo_cpu.py';tree=ast.parse(code.read_text());imports=[n for n in ast.walk(tree) if isinstance(n,ast.ImportFrom) and n.module=='select_curvature_ridge_loo_cpu']
    assert any(any(z.name=='fit_model' for z in x.names) for x in imports)
    assert not any(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='fit_model' for n in ast.walk(tree))
    family=Path(contract['unchanged_family_source']['path']);assert sha(family)==contract['unchanged_family_source']['sha256']
    report=dict(schema='R10.expanded_LOO_independent_CPU_integrity.v1',complete=True,decision='PIPELINE_INTEGRITY_PASS_PHYSICAL_GEOMETRY_NOT_ACCEPTED',CPU_only=True,nativeF=0,nativeB=0,updates=0,new_models_fitted=0,
        inputs=[pin(d/n) for n in ['training_points.npz','all_training_model_coefficients.npz','training_original_point_predictions.npz','training_only_CV_selection.json','training_data_receipt.json','curvature_selection_contract.json']],
        code=pin(code),original_fit_family=pin(family),official_original_points=len(allX),actual_XY_context_selected_points=len(X),medoids=len(ids),
        finite_and_XY_only_selection_independently_reproduced=True,point_IDs_world_values_exact=True,medoids_independently_recomputed=True,all_original_leftout_points_untrimmed=True,
        fold_denominators=folds,models=results,best_mean_model=best['name'],oneSE_bound_m=bound,oneSE_preferred=chosen['name'],
        all_nine_models_fail_some_actual_requested_height_band=all(not all(x['old_height_band_pass'] for x in z['actual_requested_height']) for z in results),
        heldout_clouds_images_semantics_not_read_by_this_checker=True,GT_boxes_not_used_for_fit_or_verification=True,no_OS_open_tracing_claim=True,
        interpretation='Selection/fold implementation integrity is separate from geometry acceptance. Saved original-point predictions, fixed denominators and one-SE selection were independently replayed; no heldout generalization or pose-bias conclusion is inferred.',
        geometry_approved=False,seconds=time.time()-start)
    a.output.mkdir(parents=True);(a.output/'receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(complete=True,receipt=pin(a.output/'receipt.json'),points=len(X),medoids=len(ids),best=best['name'],oneSE=chosen['name'],all_nine_actual_scope_fail=report['all_nine_models_fail_some_actual_requested_height_band']),ensure_ascii=False))


if __name__=='__main__':main()
