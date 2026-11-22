"""Actual saved-native zero-step readback. Visual scene verdict is separately recorded."""
import argparse,json,time
from pathlib import Path
import numpy as np
from diagnose_nonplanar_inference_cpu import load,read,write,pin
from audit_new_local_zero_cpu import loss

GEOMETRY='0d62e87abddd3017a40b06d80c485d331c0feb5c13a433be7f50f6933cda269b'
SOURCE='45b4b9d752f1584019e6687008de489de071a81907ab2c3c1b61b20964f64d32'
RAWS=[86,110,134,179,198]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();R,out=a.root,a.output;out.mkdir(parents=True,exist_ok=False);start=time.time()
    stage=R/'timewindow17_expanded_zero_v1';rp=stage/'receipt.json';rec=read(rp);assert rec['complete'] and rec['nativeF']==rec['nativeB']==5 and rec['updates']==0
    mp=Path(rec['asset_manifest']['path']);manifest=read(mp);assert pin(mp)['sha256']==rec['asset_manifest']['sha256']
    assert manifest['actor_parent_sha256']=='ab4fa110440d579bcfd57e4bd0b8a7f61c3f7821189450b58bd027c2f28f67a7'
    assert pin(manifest['geometry']['npz']['path'])['sha256']==GEOMETRY and pin(manifest['source']['field']['path'])['sha256']==SOURCE
    gate=manifest['native_gate'];assert gate['mode']=='identity_no_retirement';gatearray=load(gate['npz']['path']);assert len(gatearray['global_row_ids'])==0
    guard=load(stage/'local_colour_guard.npz');gr=read(stage/'local_colour_guard.json');assert pin(stage/'local_colour_guard.npz')['sha256']==gr['npz_sha256']
    Kmass,Bmass=guard['allview_K_mass'],guard['allview_body_mass'];assert Kmass.shape==Bmass.shape==(5,175302)
    assert np.isfinite(Kmass).all() and np.isfinite(Bmass).all() and Kmass.min()>=0 and Bmass.min()>=0
    trainable=(Kmass==0).all(0)&(Bmass==0).all(0);assert np.array_equal(trainable,guard['trainable_rgb_rows'])
    assert read(stage/'camera_timestamp_audit.json')['rows']==read(R/'measure_v1/camera_timestamp_audit.json')['rows']
    cfg=read(Path(rec['measurement_contract']['path']));rows=[];allK=True;strict_delta_pass=True
    for index,raw in enumerate(RAWS):
        zp=stage/f'{raw:03d}/joint_render.npz';z=load(zp);bp=R/f'measure_v1/{raw:03d}/baseline_joint.npz';base=load(bp)
        ep=R/f'fixed_eval_v1_replay1/{raw:03d}/fixed_eval.npz';ev=load(ep);prod=read(stage/f'{raw:03d}/metrics.json');m=load(Path(cfg['rectified_masks'])/f'{raw:03d}/rectified_masks.npz')
        for key,val in z.items():
            if np.issubdtype(val.dtype,np.floating):assert np.isfinite(val).all(),(raw,key)
        assert z['rgb'].shape==(3,768,1152) and z['alpha'].shape==(768,1152)
        actual=loss(z,ev);original=loss(base,ev);error=abs(actual-prod['fixed_ground_weighted_MSE']);assert error<2e-8
        regions={}
        for name,mask in [('K',ev['K_protected']),('broad_body',ev['body_protected']),('actor_proxy',m['actor_visible_proxy']),('strict_real_V',m.get('actor_visible'))]:
            if mask is None:
                regions[name]=dict(available=False,not_opened_not_zero_truth=True);continue
            delta=z['rgb'][:,mask]-base['rgb'][:,mask];ad=z['alpha'][mask]-base['alpha'][mask]
            regions[name]=dict(available=True,pixels=int(mask.sum()),RGB_byteexact=z['rgb'][:,mask].tobytes()==base['rgb'][:,mask].tobytes(),
                alpha_byteexact=z['alpha'][mask].tobytes()==base['alpha'][mask].tobytes(),RGB_max_abs=float(abs(delta).max()) if mask.any() else None,
                RGB_mean_abs=float(abs(delta).mean()) if mask.any() else None,alpha_max_abs=float(abs(ad).max()) if mask.any() else None,
                local_road_alphaT=float(z['local_road_alphaT'][mask].astype(float).sum()))
        allK &= regions['K']['RGB_byteexact'] and regions['K']['alpha_byteexact']
        strict=prod['body_comparison'].get('strict_real_V')
        if strict:
            change=float(strict['new_local_photo_L1']-strict['baseline_photo_L1']);strict_delta_pass &= change<=.001
            regions['strict_real_V']['saved_photo_L1_delta']=change;regions['strict_real_V']['nonworsening_0p001_pass']=change<=.001
        rgb=z['rgb'];colours={}
        for name in ['red_diagnostic','blue_cyan_diagnostic']:
            mask=ev[name];condition=((rgb[0]>.2)&(rgb[0]>np.maximum(rgb[1],rgb[2])+.12)) if name=='red_diagnostic' else ((rgb[2]>rgb[0]+.03)&(rgb[2]>rgb[1]+.015)&(rgb.max(0)>.08))
            count=int((mask&condition).sum());assert count==prod['fixed_colour_diagnostic_tracking'][name]['condition_remaining_within_FIXED_mask']
            colours[name]=dict(measured=raw in [86,110],original_mask_pixels=int(mask.sum()),remaining_in_original_mask=count,
                original_target_unknown=int((mask&~ev['ground_eval']).sum()),empty_eval_mask_not_zero_residual_truth=raw not in [86,110])
        ce=float(abs(z['road_alphaT']+z['actor_alphaT']+z['context_alphaT']-z['alpha']).max());assert ce<2e-5
        le=float(abs(z['existing_road_alphaT']+z['local_road_alphaT']-z['road_alphaT']).max());assert le<1e-6
        conservation=[]
        for name,mass,mask in [('K',Kmass[index],ev['K_protected']),('body',Bmass[index],ev['body_protected'])]:
            total=float(mass.astype(float).sum());direct=float(z['local_road_alphaT'][mask].astype(float).sum());e=abs(total-direct);assert e<max(.01,2e-6*mask.sum())
            conservation.append(dict(region=name,row_VJP_sum=total,joint_alphaT_sum=direct,abs_error=e))
        band=ev['full_original_band'];holes=band&(z['alpha']<.95);newholes=holes&(base['alpha']>=.95)
        outside=~band;delta=abs(z['rgb']-base['rgb']).max(0)
        rows.append(dict(raw=raw,actual_joint_float=pin(zp),original_baseline_float=pin(bp),fixed_eval=pin(ep),
            original_MSE=original,actual_MSE=actual,MSE_producer_replay_abs_error=error,original_ground_eval_pixels=int(ev['ground_eval'].sum()),
            original_full_band_pixels=int(band.sum()),full_band_alpha_lt95=int(holes.sum()),full_band_alpha_lt50=int((band&(z['alpha']<.5)).sum()),
            new_alpha95_holes_vs_original=int(newholes.sum()),new_alpha95_holes_in_unknown_target=int((newholes&~ev['ground_eval']).sum()),
            outside_fullband_change_vs_original=dict(changed_pixels_above1e_6=int((outside&(delta>1e-6)).sum()),RGB_max_abs=float(delta[outside].max())),
            protections=regions,fixed_colour_diagnostics=colours,alpha_partition_error=ce,old_plus_local_road_alphaT_error=le,VJP_conservation=conservation,
            shared_PNG=pin(stage/f'{raw:03d}/shared.png')))
    numeric=dict(schema='R10.timewindow17_expanded_zero_independent_readback.v1',complete=True,
        decision='PASS_MEASUREMENT_INTEGRITY' if allK and strict_delta_pass else 'FAIL_PROTECTION',joint_quality_decision='FAIL',
        actual_stage_receipt=pin(rp),asset_manifest=pin(mp),source_field_sha256=SOURCE,geometry_sha256=GEOMETRY,
        original_A20_ab4_frozen=True,old_road400_frozen=True,all_original_native_retained=True,native_gate_identity=True,
        all5_K_RGBA_byteexact=bool(allK),opened_real_V_photo_L1_delta_le0p001=bool(strict_delta_pass),
        physical_body_parameter_freeze_distinct_from_composited_RGBA_invariance=True,
        frozen_camera_timestamp_rows_exact=True,all175302_rows_colour_guard_finite_nonnegative=True,guard_trainable_rows=int(trainable.sum()),guard_frozen_rows=int((~trainable).sum()),
        rows=rows,new_gaussian_updates=0,old_formal_updates=400,old_formal_cap=2000,
        actual_stage_resources=dict(nativeF=5,nativeB=5,updates=0),this_CPU_audit_GPU_calls=0,seconds=time.time()-start)
    write(out/'numeric_review.json',numeric)
    notes={86:'Original vehicle body core remains. Red/blue underbody/front smear and a bright cyan point remain; visible left fragments and local-road transition prevent quality acceptance.',
        110:'Road region is more complete, but prominent blue-grey ghost, thin red edge and cyan/white tip remain below/in front of the vehicle. Body core remains intact.',
        134:'Front/underbody blue-grey ghost persists. Flat local material does not replace the mixed foreground contribution; edge fragments remain.',
        179:'Persistent front ghost and bright tip remain with visible left-side fragments. Original vehicle shape remains identifiable.',
        198:'Residual ghost and small fragments remain despite complete shared material input. Final local restoration remains unaccepted.'}
    visual=dict(schema='R10.timewindow17_expanded_zero_independent_visual.v1',complete=True,decision='FAIL',reviewer='native_clean_photo_regions',root_concurred=False,
        asset_manifest_sha256=pin(mp)['sha256'],source_field_sha256=SOURCE,geometry_sha256=GEOMETRY,frames=RAWS,original_resolution=[1152,768],
        rows=[dict(raw=rr,shared_PNG=pin(stage/f'{rr:03d}/shared.png'),notes=notes[rr]) for rr in RAWS],
        original_vehicle_body_core_visually_retained=True,source_and_geometry_gates_not_repeated_or_revoked=True,
        whole_joint_quality_decision='NOT_ACCEPTED',scene_holes_not_inferred_from_road_only_alpha=True,
        fullband_alpha95_holes=[dict(raw=x['raw'],pixels=x['full_band_alpha_lt95']) for x in rows],
        source_attribution_limit='Original native context, old road400 and original A20 remain present. New jointRGB alone does not identify a residual kernel group or justify deleting all native/context.',
        next_diagnostic='Actual native+oldroad footprint-contained counterfactual; mixed-kernel opacity replacement requires protected outside/K/V and scene-alpha constraints',
        numeric_review=pin(out/'numeric_review.json'),automatic_quality_promotion=False,this_CPU_audit_GPU_calls=0)
    write(out/'visual_review.json',visual)
    print(json.dumps(dict(complete=True,numeric=pin(out/'numeric_review.json'),visual=pin(out/'visual_review.json'),rows=[dict(raw=x['raw'],MSE=x['actual_MSE'],holes=x['full_band_alpha_lt95'],colour=x['fixed_colour_diagnostics'],protections=x['protections']) for x in rows])),flush=True)


if __name__=='__main__':main()
