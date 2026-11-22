"""Expanded official-ground training-only LOO selection; no geometry activation.

The optional evaluation phase first opens heldout clouds after selection is pinned.
It never changes prior geometry, cameras, vehicles, appearance or native assets.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion
from PIL import Image
from diagnose_nonplanar_inference_cpu import load, read, write, pin, stats, design, normals, medoids
from select_curvature_ridge_loo_cpu import fit_model, LAMBDAS
from qualify_local_surface_geometry_cpu import brown, measured_normals

TRAIN = [60, 70, 80, 95, 110, 125, 145, 165]
HELDOUT = [134, 179, 198]
CONTEXT_M = .4


def scope(C):
    R=C/'road_underbody_local_20261123_r10'
    gp=R/'audit/local_surface_expanded_candidate_v1/expanded_candidate_geometry.npz'
    g=load(gp)
    xy=g['cell_world_xyz'][g['requested_cell_asset_mask'],:2]
    return R,gp,g,xy,cKDTree(xy)


def select(C,out):
    out.mkdir(parents=True,exist_ok=False); start=time.time()
    R,gp,g,xy,tree=scope(C)
    evidence_path=R/'audit/ground_contact_prior_raw_GT_v2/GT_bottom_and_road_evidence.npz'
    models=[dict(name='plane',kind='plane',lambda_m4=0.)]+[dict(name=f'quadratic_lambda_{v:g}',kind='quadratic',lambda_m4=v) for v in LAMBDAS]
    origin=xy.min(0)-CONTEXT_M; centre=(xy.min(0)+xy.max(0))/2
    contract=dict(schema='R10.expanded_training_LOO_curvature_contract.v1',complete=True,written_before_training_read_and_fit=True,
        unchanged_family_source=pin(R/'code/select_curvature_ridge_loo_cpu.py'),
        parent_expansion_scope=pin(gp),input_official_road_evidence=pin(evidence_path),train_frames=TRAIN,sealed_heldout_frames=HELDOUT,
        requested_mask_key='requested_cell_asset_mask',requested_before_old_credibility_screen=True,
        fit_context='XY Euclidean distance <=.4m from pre-screen requested cell centres, whose footprint already includes .2m blend ring',
        context_m=CONTEXT_M,medoid_origin_XY_m=origin.tolist(),fit_centre_XY_m=centre.tolist(),
        point_filter='Original official ROAD18/LANE19; finite worldXYZ and declared XY context only. No Z, residual, OBB or 2D semantic pruning.',
        models=models,loss='Unchanged mean Huber.03m IRLS8, actual0.1m medoids; lambda*(4*bxx^2+2*bxy^2+4*byy^2), affine terms unpenalized',
        CV='Leave one of8 training frames out; medoids reselected from remaining points; ALL original held-out-training-frame points untrimmed MAE, equal frame weight',
        selection='Minimum mean fold MAE; oneSE bound then prefer plane, else largest lambda; retain best quadratic separately',
        bootstrap='512 frame-cluster draws seed20261123, fixed selected lambda; medoid rows weighted by sampled frame counts, same prior implementation',
        heldout='No heldout input read in select phase. Evaluate phase uses finite originalLiDAR within identical XY context, original OBB margin and original-photo road0 erode2; no Z/residual trim.',
        fixed_heldout_thresholds=dict(min_points_per_view=50,absolute_height_median_max_m=.04,absolute_height_p95_max_m=.10,
            min_normals_per_view=20,min_normals_total=100,normal_median_max_deg=10.,normal_p95_max_deg=20.),
        all_geometry_is_model_inferred=True,activation=False,GPU_forward=0,updates=0)
    write(out/'curvature_selection_contract.json',contract)
    e=load(evidence_path); X=e['official_training_road_worldXYZ']; raw=e['official_training_road_source_raw']; rowid=e['official_training_road_lidar_row_id']
    finite=np.isfinite(X).all(-1); distance=np.full(len(X),np.inf);distance[finite]=tree.query(X[finite,:2])[0]
    keep=finite & np.isin(raw,TRAIN) & (distance<=CONTEXT_M)
    X,raw,rowid,distance=X[keep],raw[keep],rowid[keep],distance[keep]
    assert np.array_equal(np.unique(raw),TRAIN)
    np.savez_compressed(out/'training_points.npz',world_xyz=X,source_raw=raw,source_lidar_row_id=rowid,
        distance_to_requested_footprint_m=distance,fit_XY_centre=centre,medoid_origin=origin,
        original_evidence_selected_indices=np.flatnonzero(keep),all_original_evidence_context_mask=keep)
    ids=medoids(X,np.arange(len(X)),origin)
    folds=[]
    for rr in TRAIN:
        train=raw!=rr; ii=medoids(X,np.flatnonzero(train),origin); test=~train
        folds.append((ii,test))
    data=dict(schema='R10.expanded_official_ground_training_data.v1',complete=True,cache=pin(out/'training_points.npz'),
        points=len(X),medoids=len(ids),per_frame=[dict(raw=int(rr),original_points=int((raw==rr).sum()),
            selected_full_fit_medoids=int((raw[ids]==rr).sum())) for rr in TRAIN],
        folds=[dict(left_out_raw=int(rr),original_eval_points=int(test.sum()),training_original_points=int((~test).sum()),training_medoids=len(ii))
               for rr,(ii,test) in zip(TRAIN,folds)],no_Z_or_residual_point_filter=True,
        no_heldout_inputs_read=True,contract=pin(out/'curvature_selection_contract.json'))
    write(out/'training_data_receipt.json',data)
    print(json.dumps(dict(stage='training_data_ready',**data)),flush=True)
    scores=[]; payload={}
    for model in models:
        values=[]; coefficients=[]
        for ii,test in folds:
            bb=fit_model(design(X[ii,:2],centre),X[ii,2],model['kind'],model['lambda_m4']);assert bb is not None
            err=design(X[test,:2],centre)@bb-X[test,2]
            values.append(float(abs(err).mean()));coefficients.append(bb)
        values=np.asarray(values)
        full=fit_model(design(X[ids,:2],centre),X[ids,2],model['kind'],model['lambda_m4']);assert full is not None
        pred=design(X[:,:2],centre)@full-X[:,2]
        scores.append(dict(**model,fold_mean_absolute_height_error_m=values.tolist(),
            mean_fold_MAE_m=float(values.mean()),SE_fold_MAE_m=float(values.std(ddof=1)/np.sqrt(len(values))),
            fitted_all_original_training_points=[dict(raw=rr,signed_error_m=stats(pred[raw==rr]),absolute_error_m=stats(abs(pred[raw==rr]))) for rr in TRAIN]))
        payload[model['name']+'__coefficients']=full
        payload[model['name']+'__LOO_coefficients']=np.asarray(coefficients)
    best=min(scores,key=lambda x:x['mean_fold_MAE_m']);limit=best['mean_fold_MAE_m']+best['SE_fold_MAE_m']
    preferred=max([x for x in scores if x['mean_fold_MAE_m']<=limit+1e-15],key=lambda x:(x['kind']=='plane',x['lambda_m4']))
    quadratics=[x for x in scores if x['kind']=='quadratic'];bq=min(quadratics,key=lambda x:x['mean_fold_MAE_m'])
    qlimit=bq['mean_fold_MAE_m']+bq['SE_fold_MAE_m'];pq=max([x for x in quadratics if x['mean_fold_MAE_m']<=qlimit+1e-15],key=lambda x:x['lambda_m4'])
    np.savez_compressed(out/'all_training_model_coefficients.npz',fit_XY_centre=centre,medoid_indices=ids,**payload)
    selected=dict(schema='R10.expanded_training_only_curvature_selection.v1',complete=True,models=scores,train_frames=TRAIN,
        best_mean_model=best['name'],best_mean_plus_SE_limit_m=limit,one_SE_preferred_model=preferred['name'],
        best_mean_quadratic=bq['name'],one_SE_quadratic=pq['name'],heldout_read_before_selection=False,
        selected_from_training_only=True,model_coefficients=pin(out/'all_training_model_coefficients.npz'),
        training_data=pin(out/'training_data_receipt.json'),contract=pin(out/'curvature_selection_contract.json'),
        geometry_approved=False,elapsed_seconds=time.time()-start)
    write(out/'training_only_CV_selection.json',selected)
    print(json.dumps(dict(stage='selection_saved_before_any_heldout_input',selection=pin(out/'training_only_CV_selection.json'),
        best_mean_model=best['name'],best_mean_MAE_m=best['mean_fold_MAE_m'],preferred=preferred['name'],
        table=[dict(name=x['name'],MAE_m=x['mean_fold_MAE_m'],SE_m=x['SE_fold_MAE_m']) for x in scores])),flush=True)


def heldout_points(C,R,tree):
    bank=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs')
    points=[]; raws=[]; ids=[]; records=[]; pins=[]
    for rr in HELDOUT:
        op=bank/f'observations/{rr:03d}.json';obs=read(op);lp=bank/obs['lidar_path'];cloud=np.fromfile(lp,dtype='<f4').reshape(-1,6)
        ego=np.asarray(obs['ego_world']).reshape(4,4);X=cloud[:,:3].astype(float)@ego[:3,:3].T+ego[:3,3]
        finite=np.isfinite(X).all(-1);d=np.full(len(X),np.inf);d[finite]=tree.query(X[finite,:2])[0]
        spatial=finite&(d<=CONTEXT_M);rowids=np.flatnonzero(spatial);X=X[spatial]
        boxed=np.zeros(len(X),bool)
        for box in obs['boxes']:
            if 'world_from_actor' not in box:continue
            pose=np.asarray(box['world_from_actor']).reshape(4,4);half=.5*np.asarray(box['dimensions'])+[.15,.15,.08]
            boxed|=(abs((X-pose[:3,3])@pose[:3,:3])<=half).all(-1)
        cp=C/f'road_multisource_20261122_r1/sources/qualified_observations/{rr:03d}.npz';E=load(cp)['world_to_camera']
        sp=bank.parent/f'semantics/{rr:03d}.png';label=np.asarray(Image.open(sp));road=binary_erosion(label==0,iterations=2)
        uv,depth=brown(X,E,np.asarray(obs['camera_intrinsic_Brown']))
        ix=np.rint(np.nan_to_num(uv[:,0])).astype(int);iy=np.rint(np.nan_to_num(uv[:,1])).astype(int)
        inside=np.isfinite(uv).all(-1)&(depth>.1)&(ix>=0)&(iy>=0)&(ix<road.shape[1])&(iy<road.shape[0])
        photo=np.zeros(len(X),bool);photo[inside]=road[iy[inside],ix[inside]];keep=~boxed&photo
        rec=dict(raw=rr,raw_cloud_points=len(cloud),finite_XY_context_points=len(X),
            independent_inside_OBB_points=int(boxed.sum()),independent_photo_road0_erode2_points=int(photo.sum()),
            strict_retained_points=int(keep.sum()),no_Z_or_model_residual_filter=True,semantic_is_original_photo_proxy=True)
        records.append(rec);points.append(X[keep]);raws.append(np.full(keep.sum(),rr,np.int16));ids.append(rowids[keep])
        pins.extend([pin(op),pin(lp),pin(cp),pin(sp)])
    return np.concatenate(points),np.concatenate(raws),np.concatenate(ids),records,pins


def evaluate(C,out):
    start=time.time(); selectionpath=out/'training_only_CV_selection.json';selection=read(selectionpath);selection_pin=pin(selectionpath)
    assert selection['heldout_read_before_selection'] is False
    assert not (out/'expanded_geometry_DIAGNOSTIC.npz').exists()
    R,gp,g,xy,tree=scope(C);train=load(out/'training_points.npz');X,raw=train['world_xyz'],train['source_raw'];centre=train['fit_XY_centre']
    coeff=load(out/'all_training_model_coefficients.npz');full_ids=coeff['medoid_indices'];A=design(X[full_ids,:2],centre)
    # Only now are any heldout cloud / obs / proxy files opened.
    H,hr,hid,records,inputs=heldout_points(C,R,tree)
    np.savez_compressed(out/'expanded_heldout_measurements.npz',world_xyz=H,source_raw=hr,source_lidar_row_id=hid)
    write(out/'expanded_heldout_input_receipt.json',dict(complete=True,first_open_after_selection=selection_pin,
        frame_records=records,actual_readback_inputs=inputs,no_OS_open_tracing_claim=True))
    names=list(dict.fromkeys(['plane',selection['best_mean_quadratic'],selection['one_SE_quadratic'],selection['one_SE_preferred_model']]))
    Q=g['world_xyz'][...,:2].reshape(-1,2);AQ=design(Q,centre);shape=g['candidate_mask'].shape
    payload={}; results=[];frame_index=np.searchsorted(np.asarray(TRAIN),raw[full_ids])
    for name in names:
        model=next(x for x in selection['models'] if x['name']==name);b=coeff[name+'__coefficients']
        rng=np.random.default_rng(20261123);boots=[]
        for _ in range(512):
            counts=np.bincount(rng.integers(0,len(TRAIN),len(TRAIN)),minlength=len(TRAIN));mass=counts[frame_index]
            bb=fit_model(A,X[full_ids,2],model['kind'],model['lambda_m4'],mass)
            if bb is not None:boots.append(bb)
        boots=np.asarray(boots);assert len(boots)==512
        cov=np.cov(boots,rowvar=False);sd=np.sqrt(np.maximum(np.einsum('ni,ij,nj->n',AQ,cov,AQ),0))
        checks=[];passed=True;normal_count=0
        for rr in HELDOUT:
            V=H[hr==rr];dz=design(V[:,:2],centre)@b-V[:,2];ang,_,_=measured_normals(V,centre,b);hs,ns=stats(abs(dz)),stats(ang);normal_count+=len(ang)
            ok=len(V)>=50 and hs['median']<=.04 and hs['p95']<=.10 and len(ang)>=20 and ns['median']<=10 and ns['p95']<=20
            passed &= ok
            checks.append(dict(raw=rr,original_points=len(V),signed_height_error_m=stats(dz),absolute_height_error_m=hs,normal_error_deg=ns,
                fixed_thresholds_pass=bool(ok),points_residual_trimmed=False))
        passed &= normal_count>=100
        prefix=name+'__';payload.update({prefix+'coefficients':b,prefix+'world_xyz':np.c_[Q,AQ@b].reshape(*shape,3),
            prefix+'normal_world':normals(Q,centre,b).reshape(*shape,3),prefix+'bootstrap_coefficients':boots,
            prefix+'conditional_bootstrap_SD_m':sd.reshape(shape)})
        results.append(dict(name=name,coefficients=b.tolist(),training_medoids=len(full_ids),conditional_bootstrap_SD_context_m=stats(sd),
            fixed_expanded_heldout_pass=bool(passed),heldout=checks))
        print(json.dumps(dict(stage='model_evaluation',model=results[-1])),flush=True)
    assert pin(selectionpath)['sha256']==selection_pin['sha256']
    np.savez_compressed(out/'expanded_geometry_DIAGNOSTIC.npz',uv_global=g['uv_global'],bounds_uvxy=g['bounds_uvxy'],
        requested_cell_asset_mask=g['requested_cell_asset_mask'],requested_cell_reconstruction_mask=g['requested_cell_reconstruction_mask'],
        spacing_m=g['spacing_m'],fit_XY_centre=centre,renderable_mask=np.zeros(shape,bool),**payload)
    result=dict(schema='R10.expanded_LOO_geometry_diagnostic.v1',complete=True,decision='MODEL_SELECTION_COMPLETE_GEOMETRY_APPROVAL_PENDING',
        selection=selection_pin,training_data=pin(out/'training_points.npz'),heldout_data=pin(out/'expanded_heldout_measurements.npz'),
        selected_name=selection['one_SE_preferred_model'],models=results,candidate=pin(out/'expanded_geometry_DIAGNOSTIC.npz'),
        no_heldout_used_for_model_selection=True,no_Z_or_residual_point_filter=True,all_geometry_is_inferred=True,
        old_geometry_A20_camera_appearance_unchanged=True,GPU_forward=0,backward=0,updates=0,elapsed_seconds=time.time()-start)
    write(out/'expanded_curvature_diagnostic.json',result)
    print(json.dumps(dict(complete=True,receipt=pin(out/'expanded_curvature_diagnostic.json'))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['select','evaluate']);p.add_argument('--campaign-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();(select if a.phase=='select' else evaluate)(a.campaign_root,a.output)
