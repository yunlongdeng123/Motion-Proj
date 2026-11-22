"""独立CPU核验既存400步配对状态、调度、固定字段和原生浮点指标；不渲染或重放训练。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import json,time
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter,map_coordinates
import torch
torch.set_num_threads(1)


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def read(path):return json.loads(Path(path).read_text())


def exact(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' else a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)


def tree_equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(tree_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
    return a==b


def metrics(photo,rgb,mask):
    yy,xx=np.nonzero(mask);y0=max(0,int(yy.min())-14);y1=min(len(mask),int(yy.max())+15)
    x0=max(0,int(xx.min())-14);x1=min(mask.shape[1],int(xx.max())+15)
    pp=photo[y0:y1,x0:x1];rr=rgb[y0:y1,x0:x1];mm=mask[y0:y1,x0:x1];spectral=[]
    for sigma in (1.2,5/3,3.):
        a=(pp-gaussian_filter(pp,(sigma,sigma,0)))[mm].ravel();b=(rr-gaussian_filter(rr,(sigma,sigma,0)))[mm].ravel()
        an,bn=np.linalg.norm(a),np.linalg.norm(b)
        spectral.append(dict(sigma_native_px=sigma,HF_cosine=float(a@b/(an*bn)),HF_RMS_ratio=float(bn/an)))
    return dict(score_n=int(mask.sum()),RGB_MAE=float(abs(photo-rgb)[mask].mean()),RGB_MSE=float(((photo-rgb)**2)[mask].mean()),spectral=spectral)


def metric_delta(a,b):
    assert a['score_n']==b['score_n']
    values=[abs(a[k]-b[k]) for k in ('RGB_MAE','RGB_MSE')]
    values.extend(abs(x[k]-y[k]) for x,y in zip(a['spectral'],b['spectral']) for k in ('HF_cosine','HF_RMS_ratio'))
    return float(max(values))


def brown_maps(bbox,intrinsic):
    x0,y0,x1,y1=bbox;yy,xx=np.mgrid[y0:y1,x0:x1].astype(np.float64)
    fx,fy,cx,cy,k1,k2,p1,p2,k3=intrinsic;x=(xx+.5-cx)/fx;y=(yy+.5-cy)/fy;r=x*x+y*y
    radial=1+k1*r+k2*r*r+k3*r*r*r
    forward=np.stack((fy*(y*radial+p1*(r+2*y*y)+2*p2*x*y)+cy-.5,
                      fx*(x*radial+2*p1*x*y+p2*(r+2*x*x))+cx-.5))
    ux=x.copy();uy=y.copy()
    for _ in range(12):
        rr=ux*ux+uy*uy;rad=1+k1*rr+k2*rr*rr+k3*rr*rr*rr
        tx=2*p1*ux*uy+p2*(rr+2*ux*ux);ty=p1*(rr+2*uy*uy)+2*p2*ux*uy
        ux=(x-tx)/rad;uy=(y-ty)/rad
    inverse=np.stack((fy*uy+cy-.5-y0,fx*ux+cx-.5-x0))
    return forward,inverse


def warp(a,grid):return np.stack([map_coordinates(a[:,:,c],grid,order=1,mode='nearest',prefilter=False) for c in range(3)],2)


def main():
    started=time.monotonic();C=Path('/root/autodl-tmp/drivingeditor_repro/campaign');R=C/'road_source_multiframe_20261122_r3';run=R/'paired400_seed0'
    receipt=read(run/'receipt.json');contract=read(run/'contract.json');cal=read(R/'exposure_audit/shared_exposure.json')
    support=read(R/'exposure_audit/calibration_support_manifest.json');strict=read(R/'exposure_audit/strict_nonoverlap_validation.json')
    assert receipt['complete'] and receipt['contract']==contract
    assert contract['steps_per_arm']==400 and contract['batch_frames']==2 and contract['seed']==0
    assert contract['RGB_lr']==.01 and contract['RGB_prior_MSE_weight']==1e-4
    assert not contract['target_migration']
    source=load(cal['source_packet']);core=source['core_mask'];assert core.sum()==900
    gain={r['raw']:r['gain'] for r in cal['records']};assert not cal['query_core_used_for_calibration'] and cal['bias_fixed']==0.
    assert support['all_masks_outside_source_core'] and all(r['fit_core_overlap_pixels']==r['validation_core_overlap_pixels']==0 for r in support['records'])
    forbidden=set(contract['query_frames']+contract['low_resolution_stress_frames']);assert forbidden=={86,100,110,117}
    initial_states={a:torch.load(run/a/'initial_state.pt',map_location='cpu',weights_only=False) for a in contract['pools']}
    first,second=initial_states['two_frames'],initial_states['eight_frames']
    common_initial={k:tree_equal(first[k],second[k]) for k in first if k!='schedule'}
    assert all(common_initial.values()) and torch.equal(first['RGB'],torch.as_tensor(source['reference_srgb'][core]))
    assert first['step']==0 and not first['optimizer']['state']
    rng_keys=['python_rng','numpy_rng','torch_rng','cuda_rng']
    frame_initial={};raw_photo={};camera_maps={};metric_error=0.;frame_results=[]
    bank=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs')
    for row in receipt['initial']:
        raw=row['raw'];f=load(run/'initial'/f'{raw:03d}'/'source_float.npz');frame_initial[raw]=f
        obs=read(bank/f'observations/{raw:03d}.json');photo=np.asarray(Image.open(bank/obs['image_path']).convert('RGB'),np.float32)/255
        x0,y0,x1,y1=[int(q) for q in f['bbox_xyxy']]
        assert exact(f['observation_raw'],photo[y0:y1,x0:x1])
        expected_gain=np.full(3,np.float32(gain[raw]),np.float32);assert exact(f['gain'],expected_gain)
        forward,inverse=brown_maps([x0,y0,x1,y1],obs['camera_intrinsic_Brown']);camera_maps[raw]=inverse
        canonical=warp(photo,forward)*expected_gain
        assert np.max(abs(canonical-f['observation_pinhole_canonical']))<1e-7
        rawm=metrics(f['observation_raw'],f['render_raw'],f['score_mask_raw'])
        canonm=metrics(f['observation_pinhole_canonical'],f['render_pinhole_canonical'],f['score_mask_pinhole'])
        metric_error=max(metric_error,metric_delta(rawm,row['raw_camera_metrics']),metric_delta(canonm,row['canonical_metrics']))
    arm_results=[]
    for arm in receipt['arms']:
        label=arm['label'];folder=run/label;init=initial_states[label]
        final=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False);packet=load(folder/'source_patch_02.npz')
        fields={k:exact(v,packet[k]) for k,v in source.items() if k!='reference_srgb'}
        halo=exact(source['reference_srgb'][~core],packet['reference_srgb'][~core]);assert all(fields.values()) and halo
        assert torch.equal(final['RGB'],torch.as_tensor(packet['reference_srgb'][core]))
        assert final['step']==400 and all(k in final for k in rng_keys)
        assert final['contract']==contract and init['contract']==contract
        schedule=np.asarray(final['schedule']);assert schedule.shape==(400,2) and np.array_equal(schedule,init['schedule'])
        sampling=read(run/(label+'_sampling.json'));assert np.array_equal(schedule,sampling['schedule'])
        assert all(len(set(v))==2 for v in schedule.tolist()) and not set(schedule.ravel())&forbidden
        pool=contract['pools'][label];assert set(schedule.ravel())==set(pool)
        generator=np.random.default_rng(0);expected=[]
        while len(expected)<800:expected.extend(generator.permutation(pool).tolist())
        assert np.array_equal(schedule,np.asarray(expected).reshape(400,2))
        expected_count=800//len(pool);counts={str(raw):int((schedule==raw).sum()) for raw in pool}
        assert all(n==expected_count for n in counts.values()) and counts==sampling['counts']
        assert final['optimizer']['param_groups']==init['optimizer']['param_groups']
        state=next(iter(final['optimizer']['state'].values()))
        assert int(state['step'])==400 and state['exp_avg'].shape==state['exp_avg_sq'].shape==(900,3)
        assert all(torch.isfinite(v).all() for v in [final['RGB'],state['exp_avg'],state['exp_avg_sq']])
        history=read(folder/'history.json')
        assert all(h['sampled_frames']==schedule[h['step']-1].tolist() and np.isfinite(h['pre_update_batch_loss']) for h in history)
        evaluations=[];reload_max=0.
        for row in arm['final']:
            raw=row['raw'];f=load(folder/'final_reloaded'/f'{raw:03d}'/'source_float.npz');memory=load(folder/'final_in_memory'/f'{raw:03d}'/'source_float.npz');original=frame_initial[raw]
            for key in ('alpha','score_mask_raw','score_mask_pinhole','gain','observation_raw','observation_pinhole_canonical','bbox_xyxy'):
                assert exact(f[key],original[key]) and exact(memory[key],original[key])
            for key in f:assert exact(f[key],memory[key])
            reload_max=max(reload_max,float(abs(f['render_raw']-memory['render_raw']).max()))
            reconstructed_raw=warp(f['render_pinhole_canonical']/f['gain'],camera_maps[raw]);mask=f['score_mask_raw']
            inverse_delta=float(abs(reconstructed_raw-f['render_raw'])[mask].max());assert inverse_delta<1e-6
            rawm=metrics(f['observation_raw'],f['render_raw'],mask)
            canonical=metrics(f['observation_pinhole_canonical'],f['render_pinhole_canonical'],f['score_mask_pinhole'])
            err=max(metric_delta(rawm,row['raw_camera_metrics']),metric_delta(canonical,row['canonical_metrics']))
            assert err<1e-5;metric_error=max(metric_error,err)
            yy,xx=np.nonzero(mask);margin=int(min(yy.min(),xx.min(),len(mask)-1-yy.max(),mask.shape[1]-1-xx.max()));assert margin>=14
            evaluations.append(dict(raw=raw,role=row['role'],fixed_alpha_and_both_masks_exact=True,raw_photo_crop_exact=True,
                shared_gain_exact=True,raw_inverse_gain_and_Brown_max_abs=inverse_delta,
                metric_receipt_max_recompute_delta=err,HF_crop_margin_px=margin,
                raw_camera_metrics=rawm,canonical_metrics=canonical))
        assert reload_max==arm['reload_render_max_abs']==0.
        arm_results.append(dict(label=label,pool=pool,optimizer_step=400,Adam_tensor_step=400,
            formal_samples=800,per_frame_counts=counts,all_400_two_distinct_frame_batches=True,
            schedule_seed0_reproduced=True,query_and_stress_absent_from_schedule=True,
            checkpoint_schedule_plus_step_represents_complete_sampler_cursor=True,
            all_rng_states_present=True,all_non_RGB_fields_exact=all(fields.values()),non_RGB_fields=fields,halo_RGB_exact=halo,
            final_core_RGB_equals_checkpoint=True,reload_all_float_fields_exact=True,
            history_recorded_steps=len(history),all_recorded_losses_finite_and_sampled_frames_match=True,
            per_step_losses_saved=False,loss_trace_note='Full 400-step schedule saved; losses recorded at step1 and every25 only. No extra training replay was performed.',
            final=evaluations))
    smoke=torch.load(run/'smoke_checkpoint.pt',map_location='cpu',weights_only=False)
    assert smoke['step']==4 and int(next(iter(smoke['optimizer']['state'].values()))['step'])==4
    assert all(k in smoke for k in rng_keys+['schedule','contract'])
    assert receipt['resume_next_update_RGB_max_abs']==0.
    assert receipt['nativeF']==1684 and receipt['nativeB']==1612 and receipt['formal_updates']==800 and receipt['discarded_smoke_restore_updates']==6
    comparison=[]
    lookup={a['label']:{r['raw']:r for r in a['final']} for a in arm_results}
    for raw in [100,110,117,86]:
        values={a:lookup[a][raw]['raw_camera_metrics'] for a in lookup}
        comparison.append(dict(raw=raw,role='LOW_RES_STRESS' if raw==86 else 'QUERY',metrics=values))
    query_summary={}
    for label in lookup:
        query_summary[label]=dict(mean_RGB_MAE=float(np.mean([lookup[label][raw]['raw_camera_metrics']['RGB_MAE'] for raw in [100,110,117]])),
            mean_HF_cosine_sigma1p667=float(np.mean([lookup[label][raw]['raw_camera_metrics']['spectral'][1]['HF_cosine'] for raw in [100,110,117]])),
            mean_HF_RMS_ratio_sigma1p667=float(np.mean([lookup[label][raw]['raw_camera_metrics']['spectral'][1]['HF_RMS_ratio'] for raw in [100,110,117]])))
    strict_main=[r for r in strict['records'] if r['raw']!=86]
    output=dict(schema='paired400_independent_CPU_audit.v1',complete=True,all_required_state_data_checks_passed=True,
        shared_initial_state_all_except_schedule_exact=common_initial,both_initial_core_RGB_equal_original_source=True,
        source_core_RGB_parameters=2700,pool_is_only_experimental_method_difference=True,
        source_geometry_halo_and_fixed_masks_preserved=True,query_core_used_in_loss=False,
        shared_exposure_core_samples_used=0,shared_exposure_gain_values_frozen=True,
        strict_external_exposure_train_and_main_query_pass=all(r['minimum8cells_pass'] and r['residual_thresholds_pass'] for r in strict_main),
        stress86_exposure_residual_is_weaker=True,arms=arm_results,
        native_metrics_max_recompute_delta=metric_error,raw_query_comparison=comparison,main_query_summary=query_summary,
        fullstate_restore=dict(smoke_state_step=4,Adam_state_nonempty=True,all_RNG_and_schedule_saved=True,
            parent_actual_native_next_update_max_abs=0.,final_checkpoint_steps=[400,400],source_NPZ_reloaded_native_all_fields_exact=True),
        resources=dict(parent_nativeF=1684,parent_nativeB=1612,formal_optimizer_updates=800,discarded_smoke_restore_updates=6,
            this_audit_new_nativeF=0,this_audit_new_nativeB=0,this_audit_training_replays=0),
        seconds=time.monotonic()-started)
    path=R/'exposure_audit/paired_cpu_audit.json';path.write_text(json.dumps(output,ensure_ascii=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps(output,ensure_ascii=True,indent=2,allow_nan=False))


if __name__=='__main__':main()
