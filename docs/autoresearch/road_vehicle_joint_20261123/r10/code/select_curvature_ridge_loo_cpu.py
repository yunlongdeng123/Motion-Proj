"""Train-frame LOO curvature selection; heldout read only after the selection is saved."""
import argparse,json,time
from pathlib import Path
import numpy as np
from diagnose_nonplanar_inference_cpu import load,read,write,pin,stats,design,normals,medoids
from qualify_local_surface_geometry_cpu import measured_normals


LAMBDAS=[0.,.001,.01,.1,1.,10.,100.,1000.]


def fit_model(A,z,kind,lam,mass=None):
    mass=np.ones(len(z)) if mass is None else np.asarray(mass,np.float64)
    take=mass>0;A=A[take];z=z[take];mass=mass[take]
    used=A[:,:3] if kind=='plane' else A
    if np.linalg.matrix_rank(used)<used.shape[1]:return None
    d=np.array([0.,0.,0.,4.,2.,4.])
    penalty=np.zeros((3,3)) if kind=='plane' else np.diag(d)*lam
    weight=mass.copy();b=np.zeros(used.shape[1]);N=mass.sum()
    for step in range(9):
        # IRLS for mean Huber.03m data term plus lambda*(4bxx²+2bxy²+4byy²).
        lhs=used.T@(weight[:,None]*used)/N+penalty
        rhs=used.T@(weight*z)/N
        b=np.linalg.solve(lhs,rhs)
        weight=mass*np.minimum(1.,.03/np.maximum(abs(used@b-z),1e-12))
    return np.r_[b,np.zeros(3)] if kind=='plane' else b


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    C,out=a.campaign_root,a.output;out.mkdir(parents=True,exist_ok=False);start=time.time()
    R=C/'road_underbody_local_20261123_r10';ev=R/'audit/official_ground_geometry_v1'
    old=read(R/'audit/local_surface_geometry_v1/geometry_qualification_contract.json')
    source=load(ev/'local_original_point_evidence.npz');train=source['ground']&np.isin(source['raw'],old['train_raw'])
    X=source['world_xyz'];raw=source['raw'];trains=np.unique(raw[train]);origin=np.asarray(old['XY_fit_context_bbox'])[0]
    centre=np.mean(np.asarray(old['XY_fit_context_bbox']),axis=0)
    models=[dict(name='plane',kind='plane',lambda_m4=0.)]+[dict(name=f'quadratic_lambda_{v:g}',kind='quadratic',lambda_m4=v) for v in LAMBDAS]
    contract=dict(schema='R10.training_LOO_curvature_ridge_contract.v1',complete=True,written_before_CV=True,
        input=pin(ev/'official_ground_point_receipt.json'),train_frames=trains.tolist(),sealed_heldout_frames=old['heldout_raw'],models=models,
        profile='z=b0+bx*x+by*y+bxx*x²+bxy*x*y+byy*y² around original local XY centre; nonconstant height, not whole-scene constant-Z',
        loss='mean Huber.03m IRLS8 on measured0.1m cell medoids plus lambda_m4*(4*bxx²+2*bxy²+4*byy²); affine slope/intercept not penalized',
        CV='leave one official training frame out; recompute medoid selection on remaining original points; evaluate ALL original official-road/lane points of left-out frame untrimmed; equal per-frame mean absolute error',
        selection='lowest mean fold MAE; one-standard-error rule then prefer plane, else largest lambda; also retain best and one-SE quadratic independently',
        uncertainty='512 training-frame cluster bootstraps at fixed selected lambda and full-training actual medoids, seed20261123; conditional sensitivity, not calibrated probability',
        heldout_policy='selection receipt written before loading heldout point NPZ; heldout not tuning lambda, domains or thresholds',
        activation='none; inferred candidate and physical prior require separate root review; old strict/profile unchanged',GPU_forward=0,updates=0)
    write(out/'curvature_selection_contract.json',contract)
    rows=[];fold_coef={}
    for model in models:
        scores=[];coeff=[]
        for rr in trains:
            ids=medoids(X,np.flatnonzero(train&(raw!=rr)),origin)
            b=fit_model(design(X[ids,:2],centre),X[ids,2],model['kind'],model['lambda_m4']);assert b is not None
            q=train&(raw==rr);error=design(X[q,:2],centre)@b-X[q,2];scores.append(float(abs(error).mean()));coeff.append(b)
        score=np.asarray(scores)
        rows.append(dict(**model,fold_mean_absolute_height_error_m=scores,mean_fold_MAE_m=float(score.mean()),SE_fold_MAE_m=float(score.std(ddof=1)/np.sqrt(len(score)))))
        fold_coef[model['name']]=np.asarray(coeff)
    best=min(rows,key=lambda x:x['mean_fold_MAE_m']);limit=best['mean_fold_MAE_m']+best['SE_fold_MAE_m']
    eligible=[x for x in rows if x['mean_fold_MAE_m']<=limit+1e-15]
    strongest=lambda x:(x['kind']=='plane',x['lambda_m4'])
    preferred=max(eligible,key=strongest)
    quad=[x for x in rows if x['kind']=='quadratic'];best_quad=min(quad,key=lambda x:x['mean_fold_MAE_m'])
    quad_limit=best_quad['mean_fold_MAE_m']+best_quad['SE_fold_MAE_m']
    preferred_quad=max([x for x in quad if x['mean_fold_MAE_m']<=quad_limit+1e-15],key=lambda x:x['lambda_m4'])
    selection=dict(schema='R10.training_only_curvature_selection.v1',complete=True,models=rows,train_frames=trains.tolist(),
        best_mean_model=best['name'],best_mean_plus_SE_limit_m=limit,one_SE_preferred_model=preferred['name'],
        best_mean_quadratic=best_quad['name'],one_SE_quadratic=preferred_quad['name'],
        heldout_read_before_selection=False,selected_from_training_only=True,
        production_geometry_approved=False,contract=pin(out/'curvature_selection_contract.json'))
    write(out/'training_only_CV_selection.json',selection)
    print(json.dumps(dict(stage='selection_saved_before_heldout_read',**selection)),flush=True)
    # The independent heldout point data is deliberately first opened here.
    held=load(R/'audit/local_surface_geometry_v1/strict_measurements.npz')
    s=load(R/'audit/local_surface_geometry_v1/local_surface_geometry.npz');Q=s['world_xyz'][...,:2].reshape(-1,2);AQ=design(Q,centre);shape=s['geometry_valid'].shape
    full_ids=medoids(X,np.flatnonzero(train),origin);A=design(X[full_ids,:2],centre)
    finalists=[next(x for x in models if x['name']==name) for name in dict.fromkeys(['plane',preferred_quad['name'],preferred['name']])]
    results=[];payload={}
    for model in finalists:
        name=model['name'];b=fit_model(A,X[full_ids,2],model['kind'],model['lambda_m4']);height=AQ@b
        rng=np.random.default_rng(20261123);boots=[]
        for _ in range(512):
            counts=np.bincount(rng.integers(0,len(trains),len(trains)),minlength=len(trains));mass=np.array([counts[np.flatnonzero(trains==r)[0]] for r in raw[full_ids]])
            bb=fit_model(A,X[full_ids,2],model['kind'],model['lambda_m4'],mass)
            if bb is not None:boots.append(bb)
        boots=np.asarray(boots);sd=np.zeros(len(Q));lo=sd.copy();hi=sd.copy()
        for q in range(0,len(Q),4096):
            z=AQ[q:q+4096]@boots.T;sd[q:q+4096]=z.std(-1,ddof=1);lo[q:q+4096]=np.quantile(z,.025,axis=-1);hi[q:q+4096]=np.quantile(z,.975,axis=-1)
        checks=[];allcheck=True;normal_count=0
        for r in old['heldout_raw']:
            v=held['world_xyz'][held['source_raw']==r];dz=design(v[:,:2],centre)@b-v[:,2];angles,ok,nn=measured_normals(v,centre,b);hs=stats(abs(dz));ns=stats(angles);normal_count+=len(angles)
            passed=len(dz)>=50 and hs['median']<=.04 and hs['p95']<=.10 and len(angles)>=20 and ns['median']<=10 and ns['p95']<=20;allcheck&=passed
            checks.append(dict(raw=r,signed_height_error_m=stats(dz),absolute_height_error_m=hs,normal_error_deg=ns,threshold_checks_pass=bool(passed),points_trimmed=False))
        allcheck&=normal_count>=100
        prefix=name+'__';payload.update({prefix+'coefficients':b,prefix+'profile_world_xyz':np.c_[Q,height].reshape(*shape,3),
            prefix+'profile_normal_world':normals(Q,centre,b).reshape(*shape,3),prefix+'bootstrap_coefficients':boots,
            prefix+'bootstrap_height_p025_m':lo.reshape(shape),prefix+'bootstrap_height_p975_m':hi.reshape(shape),prefix+'bootstrap_SD_m':sd.reshape(shape),
            prefix+'leave_one_training_frame_out_coefficients':fold_coef[name]})
        results.append(dict(**model,fit_coefficients=b.tolist(),training_medoids=len(full_ids),bootstrap_valid=len(boots),
            bootstrap_SD_in_parent_missing_m=stats(sd[~s['original_geometry_supported'].ravel()]),
            bootstrap_95pct_width_in_parent_missing_m=stats((hi-lo)[~s['original_geometry_supported'].ravel()]),
            fixed_original_heldout_checks_pass=bool(allcheck),heldout=checks,
            production_geometry_approved=False))
    np.savez_compressed(out/'plane_and_selected_ridge_DIAGNOSTIC.npz',uv_global=s['uv_global'],fit_XY_centre=centre,
        original_parent_geometry_supported=s['original_geometry_supported'],direct_witness_mask=s['geometry_valid'],
        renderable_mask=np.zeros(shape,bool),bounds_uvxy=s['bounds_uvxy'],spacing_m=s['spacing_m'],**payload)
    result=dict(schema='R10.LOO_selected_plane_quadratic_diagnostic.v1',complete=True,decision='PENDING_PHYSICAL_PRIOR_AND_ROOT_REVIEW_NOT_RENDERABLE',
        training_selection=pin(out/'training_only_CV_selection.json'),selection=selection,finalists=results,
        candidate=pin(out/'plane_and_selected_ridge_DIAGNOSTIC.npz'),heldout_never_used_to_select=True,
        physical_box_contact_prior_fitted=False,original_geometry_edited=False,new_renderable_texels=0,GPU_forward=0,updates=0,elapsed_seconds=time.time()-start)
    write(out/'curvature_ridge_diagnostic.json',result);print(json.dumps(dict(complete=True,receipt=pin(out/'curvature_ridge_diagnostic.json'),finalists=results)),flush=True)


if __name__=='__main__':main()
