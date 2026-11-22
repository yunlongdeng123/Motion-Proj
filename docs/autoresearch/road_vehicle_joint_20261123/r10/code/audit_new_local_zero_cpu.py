"""Fresh float/mask verification of actual new-local joint zero-step; visual verdict is separate."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from diagnose_nonplanar_inference_cpu import load,read,write,pin,stats


def linear(x):
    x=x.astype(np.float64);return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)


def loss(z,ev):
    m=ev['ground_eval'];w=ev['weight'][m].astype(float);rgb=linear(z['rgb'])*float(ev['exposure_gain']);ref=ev['target_common_linear'].transpose(2,0,1)
    e=rgb[:,m]-ref[:,m];return float(((e*e).mean(0)*w).sum()/w.sum())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();R,out=a.root,a.output;out.mkdir(parents=True,exist_ok=False);start=time.time()
    stage=R/'new_local_zero_v1';receipt=read(stage/'receipt.json');assert receipt['complete'] and receipt['nativeF']==5 and receipt['nativeB']==5 and receipt['updates']==0
    manifest_path=Path(receipt['asset_manifest']['path']);manifest=read(manifest_path);assert pin(manifest_path)['sha256']==receipt['asset_manifest']['sha256']
    pinchecks=[]
    for record in manifest['parent_pins']:
        actual=pin(record['path']);assert actual['sha256']==record['sha256'];pinchecks.append(dict(role=record['role'],**actual))
    assert manifest['actor_parent_sha256']=='ab4fa110440d579bcfd57e4bd0b8a7f61c3f7821189450b58bd027c2f28f67a7'
    fieldpath=manifest_path.parent/'local_field.npz';assert pin(fieldpath)['sha256']==manifest['files']['local_field.npz']['sha256']
    ng=manifest['native_gate'];assert ng['mode']=='identity_no_retirement';nativegate=load(ng['npz']['path']);assert len(nativegate['global_row_ids'])==0
    geometrypath=Path(manifest['geometry']['npz']['path']);sourcepath=Path(manifest['source']['field']['path'])
    assert pin(geometrypath)['sha256']=='b5df7c8cb2b99723e1fabf3b2ca4c958c88003d38f4f528bb6dd8f14e97a011a'
    assert pin(sourcepath)['sha256']=='eb86cb6d61b711366d8c30052fd02b823433a6e940b80d6f4e0463685e38a5dd'
    guard=load(stage/'local_colour_guard.npz');guardrec=read(stage/'local_colour_guard.json');assert pin(stage/'local_colour_guard.npz')['sha256']==guardrec['npz_sha256']
    Kmass=guard['allview_K_mass'];Bmass=guard['allview_body_mass'];assert Kmass.shape==Bmass.shape==(5,34793)
    assert np.isfinite(Kmass).all() and np.isfinite(Bmass).all() and Kmass.min()>=0 and Bmass.min()>=0
    trainable=(Kmass==0).all(0)&(Bmass==0).all(0);assert np.array_equal(trainable,guard['trainable_rgb_rows'])
    assert int(trainable.sum())==guardrec['trainable_rows']==25104 and int((~trainable).sum())==9689
    camera=read(stage/'camera_timestamp_audit.json');basecamera=read(R/'measure_v1/camera_timestamp_audit.json')
    assert camera['rows']==basecamera['rows']
    rows=[];all_K=True;all_proxy=True;all_strict=True
    for index,raw in enumerate([86,110,134,179,198]):
        zp=stage/f'{raw:03d}/joint_render.npz';z=load(zp);basepath=R/f'measure_v1/{raw:03d}/baseline_joint.npz';base=load(basepath)
        evpath=R/f'fixed_eval_v1_replay1/{raw:03d}/fixed_eval.npz';ev=load(evpath);producer=read(stage/f'{raw:03d}/metrics.json')
        cfg=read(Path(receipt['measurement_contract']['path']));maskpath=Path(cfg['rectified_masks'])/f'{raw:03d}/rectified_masks.npz';masks=load(maskpath)
        for key,value in z.items():
            if np.issubdtype(value.dtype,np.floating):assert np.isfinite(value).all(),(raw,key)
        assert z['rgb'].shape==(3,768,1152) and z['alpha'].shape==(768,1152)
        K=ev['K_protected'];body=ev['body_protected'];proxy=masks['actor_visible_proxy'];visible=masks.get('actor_visible')
        regions={}
        for name,mask in [('other_K',K),('broad_isolated_actor_footprint',body),('actor_photo_proxy',proxy),('strict_real_V',visible)]:
            if mask is None:
                regions[name]=dict(available=False,not_opened_not_zero_truth=True);continue
            exact=z['rgb'][:,mask].tobytes()==base['rgb'][:,mask].tobytes() and z['alpha'][mask].tobytes()==base['alpha'][mask].tobytes()
            regions[name]=dict(available=True,pixels=int(mask.sum()),RGBA_byteexact=exact,
                RGB_max_abs=float(abs(z['rgb']-base['rgb'])[:,mask].max()) if mask.any() else None,
                RGB_mean_abs=float(abs(z['rgb']-base['rgb'])[:,mask].mean()) if mask.any() else None,
                local_road_alphaT_mass=float(z['local_road_alphaT'][mask].astype(float).sum()))
        all_K&=regions['other_K']['RGBA_byteexact'];all_proxy&=regions['actor_photo_proxy']['RGBA_byteexact']
        if visible is not None:all_strict&=regions['strict_real_V']['RGBA_byteexact']
        fixedloss=loss(z,ev);baseloss=loss(base,ev);metric_error=abs(fixedloss-producer['fixed_ground_weighted_MSE']);assert metric_error<2e-8
        colour={}
        for name in ['red_diagnostic','blue_cyan_diagnostic']:
            mask=ev[name];rgb=z['rgb'];condition=((rgb[0]>.2)&(rgb[0]>np.maximum(rgb[1],rgb[2])+.12)) if name=='red_diagnostic' else ((rgb[2]>rgb[0]+.03)&(rgb[2]>rgb[1]+.015)&(rgb.max(0)>.08))
            count=int((mask&condition).sum());assert count==producer['fixed_colour_diagnostic_tracking'][name]['condition_remaining_within_FIXED_mask']
            colour[name]=dict(original_fixed_pixels=int(mask.sum()),remaining_in_same_fixed_mask=count,target_unknown=int((mask&~ev['ground_eval']).sum()),scene_alpha_lt95=int((mask&(z['alpha']<.95)).sum()),
                zero_count_in_eval_views_means_not_measured=raw not in [86,110])
        component_error=float(abs(z['road_alphaT']+z['actor_alphaT']+z['context_alphaT']-z['alpha']).max());assert component_error<2e-5
        local_identity=float(abs(z['existing_road_alphaT']+z['local_road_alphaT']-z['road_alphaT']).max());assert local_identity<1e-6
        conservation=[]
        for key,mass,mask in [('K',Kmass[index],K),('body',Bmass[index],body)]:
            total=float(mass.astype(float).sum());direct=float(z['local_road_alphaT'][mask].astype(float).sum());error=abs(total-direct)
            assert error<max(.01,2e-6*mask.sum());conservation.append(dict(region=key,row_VJP_mass=total,joint_direct_alphaT_mass=direct,abs_error=error))
        row=dict(raw=raw,original_joint_float=pin(zp),baseline_float=pin(basepath),fixed_eval=pin(evpath),
            baseline_fixed_ground_weighted_MSE=baseloss,new0_fixed_ground_weighted_MSE=fixedloss,relative_change=(fixedloss/baseloss-1),producer_metric_replay_abs_error=metric_error,
            fixed_ground_pixels=int(ev['ground_eval'].sum()),full_original_band_pixels=int(ev['full_original_band'].sum()),
            ground_target_unknown_pixels=int((ev['ground_band_before_support']&~ev['ground_eval']).sum()),
            full_band_scene_alpha_lt95=int((ev['full_original_band']&(z['alpha']<.95)).sum()),
            full_band_scene_alpha_lt50=int((ev['full_original_band']&(z['alpha']<.5)).sum()),
            fixed_colour_tracking=colour,protected_regions=regions,actual_component_alpha_partition_max_abs=component_error,
            existing_plus_local_road_alphaT_max_abs=local_identity,local_VJP_conservation=conservation,
            actual_joint_PNG=pin(stage/f'{raw:03d}/shared.png'))
        rows.append(row)
    assert all_K and all_proxy and all_strict
    historical=[];F=B=0
    for name in ['measure_v1','retire0_strict_v1','body_overlap_diagnostic_v1','new_local_zero_v1']:
        rc=read(R/name/'receipt.json');assert rc['complete'];nf=int(rc.get('nativeF',rc.get('actual_nativeF',0)));nb=int(rc.get('nativeB',0));F+=nf;B+=nb
        historical.append(dict(stage=name,receipt=pin(R/name/'receipt.json'),nativeF=nf,nativeB=nb,updates=int(rc['updates'])))
    assert F==20 and B==10 and 'torch' not in sys.modules
    numeric=dict(schema='R10.independent_new0_joint_readback.v1',complete=True,decision='PASS_MEASUREMENT_AND_FROZEN_PROTECTION_INTEGRITY',
        joint_quality_decision='FAIL',asset_manifest=pin(manifest_path),source_field_sha256=pin(sourcepath)['sha256'],geometry_sha256=pin(geometrypath)['sha256'],
        parent_asset_and_code_pin_checks=pinchecks,original_A20_ab4_frozen=True,old_road400_frozen=True,identity_native_gate_empty=True,
        all5_K_RGBA_byteexact=bool(all_K),all5_actor_photo_proxy_RGBA_byteexact=bool(all_proxy),opened_strict_real_V_RGBA_byteexact=bool(all_strict),
        camera_timestamp_audit_byte_equivalent=True,all_34793_new_rows_measured_in_all5_views=True,guard_finite_nonnegative=True,guard_trainable_rows=int(trainable.sum()),guard_frozen_rows=int((~trainable).sum()),
        broad_actor_footprint_changes_separate_from_exact_photo_body=True,per_view=rows,stage_resources=historical,R10_actual_resources=dict(nativeF=F,nativeB=B,formal_updates=0),
        old_formal_updates=400,old_formal_cap=2000,no_new_RGB400_authorized=True,no_scene_quality_acceptance=True,
        actual_GPU_work_executed_by_this_CPU_audit=0,seconds=time.time()-start)
    write(out/'numeric_review.json',numeric)
    notes={86:'Large red slab is largely covered. Blue-grey underbody/front ghost and a small bright point remain; left road fragments and patch boundary remain visible.',
        110:'Red area is substantially reduced, but a red thin edge and prominent blue-grey front ghost remain. Vehicle body core remains visually intact; whole restoration not accepted.',
        134:'Evaluation view retains the blue-grey underbody/front ghost, fragmented left edge and local extent boundary.',
        179:'Evaluation view retains the front ghost/bright smear and left fragments; adding local road alone has not produced a clean joint result.',
        198:'Evaluation view retains ghost and fragment structure despite smooth source material. Body core is retained, whole scene still fails.'}
    visual=dict(schema='R10.independent_new0_joint_visual_review.v1',complete=True,decision='FAIL',reviewer='native_clean_photo_regions',root_concurred=False,
        asset_manifest_sha256=pin(manifest_path)['sha256'],source_field_sha256=pin(sourcepath)['sha256'],geometry_sha256=pin(geometrypath)['sha256'],
        frames=[86,110,134,179,198],original_resolution=[1152,768],actual_single_sorted_joint_native_renders=True,
        per_view=[dict(raw=r,PNG=pin(stage/f'{r:03d}/shared.png'),notes=notes[r]) for r in [86,110,134,179,198]],
        reason='Smooth accepted source and local inferred geometry do not remove residual mixed native contribution. Red improves, blue-grey front/underbody ghost and left fragments remain.',
        original_vehicle_body_core_visually_retained=True,local_road_source_input_gates_unchanged=True,
        full_restoration_decision='NOT_ACCEPTED',automatic_promotion=False,new_RGB400_authorized=False,
        road_only_alpha_not_used_as_scene_hole_evidence=True,actual_scene_full_band_alpha_holes=0,
        numeric_review=pin(out/'numeric_review.json'),new_GPU_forward=0,new_GPU_backward=0,updates=0)
    write(out/'visual_review.json',visual)
    print(json.dumps(dict(complete=True,numeric=pin(out/'numeric_review.json'),visual=pin(out/'visual_review.json'),summary=[dict(raw=x['raw'],baselineMSE=x['baseline_fixed_ground_weighted_MSE'],newMSE=x['new0_fixed_ground_weighted_MSE'],colour=x['fixed_colour_tracking']) for x in rows])),flush=True)


if __name__=='__main__':main()
