"""Read-only independent counterfactual review, complete fixed masks and outside spill."""
import argparse,json,time
from pathlib import Path
import numpy as np
from diagnose_nonplanar_inference_cpu import load,read,write,pin,stats
from audit_new_local_zero_cpu import loss


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();R,out=a.root,a.output;out.mkdir(parents=True,exist_ok=False);start=time.time()
    stage=R/'new_local_background_zero_v1';rc=read(stage/'receipt.json');assert rc['complete'] and (rc['nativeF'],rc['nativeB'],rc['updates'])==(5,0,0)
    scope=read(rc['diagnostic_scope']['path']);candidate=load(scope['diagnostic_candidates']['path']);gate=load(rc['diagnostic_gate']['path']);ids=gate['global_row_ids'];assert len(ids)==22816 and np.array_equal(ids,candidate['global_row_ids']) and not gate['gate'].any()
    cfg=read(R/'local_measure_contract_v2.json');basecontract=read(cfg['baseline_contract']);permanent=np.asarray(read(basecontract['context_plan'])['preserved_native_context_global_row_ids'],np.int64)
    assert not np.isin(ids,permanent).any();assert (candidate['owners_by_view']<=0).all() and not candidate['allview_K_mass'].any()
    assert (candidate['allview_ground_mass'].sum(1)>0).all()
    assert read(stage/'camera_timestamp_audit.json')==read(R/'new_local_zero_v1/camera_timestamp_audit.json')
    manifest=read(rc['asset_manifest']['path']);assert manifest['native_gate']['mode']=='identity_no_retirement'
    assert len(load(manifest['native_gate']['npz']['path'])['global_row_ids'])==0
    persistent=[]
    for item in manifest['parent_pins']:
        actual=pin(item['path']);assert actual['sha256']==item['sha256'];persistent.append(dict(role=item['role'],**actual))
    rows=[]
    for j,raw in enumerate([86,110,134,179,198]):
        z=load(stage/f'{raw:03d}/joint_render.npz');prior=load(R/f'new_local_zero_v1/{raw:03d}/joint_render.npz');old=load(R/f'measure_v1/{raw:03d}/baseline_joint.npz')
        evpath=R/f'fixed_eval_v1_replay1/{raw:03d}/fixed_eval.npz';ev=load(evpath);masks=load(Path(cfg['rectified_masks'])/f'{raw:03d}/rectified_masks.npz');producer=read(stage/f'{raw:03d}/metrics.json')
        packet=load(R/f'measure_v1/{raw:03d}/all_context_rows.npz');where=np.searchsorted(packet['global_row_ids'],ids);assert np.array_equal(packet['global_row_ids'][where],ids)
        assert packet['native_context'][where].all() and (packet['owner_ids'][where]<=0).all()
        assert not packet['ROI_alphaT_mass'][where,0].any()
        assert np.array_equal(packet['ROI_alphaT_mass'][where,1],candidate['allview_body_mass'][:,j])
        assert np.array_equal(packet['ROI_alphaT_mass'][where,2],candidate['allview_ground_mass'][:,j])
        for key,value in z.items():
            if np.issubdtype(value.dtype,np.floating):assert np.isfinite(value).all(),(raw,key)
        K=ev['K_protected'];assert z['rgb'][:,K].tobytes()==prior['rgb'][:,K].tobytes() and z['alpha'][K].tobytes()==prior['alpha'][K].tobytes()
        band=ev['full_original_band'];known=ev['ground_eval'];H=band&(z['alpha']<.95)&(prior['alpha']>=.95)
        U=H&~known;shown=load(R/f'local_shared_surface_review_v1/{raw:03d}/surface_review.npz')['shown']
        holes=dict(full_band_new_alpha95=int(H.sum()),unknown_original_target_new_alpha95=int(U.sum()),
            known_original_target_new_alpha95=int((H&known).sum()),
            geometric_asset_material_scope_new_alpha95=int((H&shown).sum()),outside_asset_material_scope_new_alpha95=int((H&~shown).sum()),
            actual_new_local_alphaT_at_holes=stats(z['local_road_alphaT'][H]),
            full_band_alpha_lt50=int((band&(z['alpha']<.5)).sum()))
        assert holes['full_band_new_alpha95']==producer['full_original_band_new_alpha95_holes_vs_newzero']
        assert holes['unknown_original_target_new_alpha95']==producer['unknown_original_target_new_alpha95_holes_vs_newzero']
        assert holes['known_original_target_new_alpha95']+holes['unknown_original_target_new_alpha95']==int(H.sum())
        delta=abs(z['rgb']-prior['rgb']);outside=~band;spill=dict(outside_band_pixels=int(outside.sum()),changed_pixels_above1e_6=int((outside&(delta.max(0)>1e-6)).sum()),RGB_max_abs=float(delta[:,outside].max()),RGB_mean_abs=float(delta[:,outside].mean()))
        assert spill['changed_pixels_above1e_6']==producer['outside_full_band']['changed_pixels_above1e_6']
        f=loss(z,ev);f0=loss(prior,ev);assert abs(f-producer['fixed_ground_weighted_MSE'])<2e-8
        body={}
        for name,m in [('other_K',K),('broad_isolated_body',ev['body_protected']),('actor_photo_proxy',masks['actor_visible_proxy']),('strict_real_V',masks.get('actor_visible'))]:
            if m is None:body[name]=dict(available=False,not_opened_not_zero_truth=True);continue
            valid=m&masks['RGB_valid'];photo=load(Path(cfg['camera_cache'])/f'{raw:03d}/view_cache.npz')['photo_pinhole_srgb'].transpose(2,0,1)
            body[name]=dict(available=True,pixels=int(m.sum()),RGBA_byteexact=z['rgb'][:,m].tobytes()==prior['rgb'][:,m].tobytes() and z['alpha'][m].tobytes()==prior['alpha'][m].tobytes(),
                RGB_max_abs=float(delta[:,m].max()) if m.any() else None,RGB_mean_abs=float(delta[:,m].mean()) if m.any() else None,
                alpha_max_abs=float(abs(z['alpha']-prior['alpha'])[m].max()) if m.any() else None,
                original_photo_L1=float(abs(old['rgb']-photo)[:,valid].mean()) if valid.any() else None,
                add_only_new0_photo_L1=float(abs(prior['rgb']-photo)[:,valid].mean()) if valid.any() else None,
                hard_zero_photo_L1=float(abs(z['rgb']-photo)[:,valid].mean()) if valid.any() else None,
                new_alpha95_holes=int((H&m).sum()))
        colour={}
        for name in ['red_diagnostic','blue_cyan_diagnostic']:
            m=ev[name];rgb=z['rgb'];cond=((rgb[0]>.2)&(rgb[0]>np.maximum(rgb[1],rgb[2])+.12)) if name=='red_diagnostic' else ((rgb[2]>rgb[0]+.03)&(rgb[2]>rgb[1]+.015)&(rgb.max(0)>.08))
            n=int((m&cond).sum());assert n==producer['fixed_colour_diagnostic_tracking'][name]['condition_remaining_within_FIXED_mask']
            colour[name]=dict(fixed_original_pixels=int(m.sum()),remaining=n,original_target_unknown=int((m&~known).sum()),diagnostic_not_defined=raw not in [86,110])
        partition=float(abs(z['road_alphaT']+z['actor_alphaT']+z['context_alphaT']-z['alpha']).max());assert partition<2e-5
        rows.append(dict(raw=raw,fixed_eval=pin(evpath),full_band_pixels=int(band.sum()),fixed_known_ground_pixels=int(known.sum()),
            add_only_MSE=f0,hard_zero_MSE=f,RGB_target_unknown_never_scored_as_zero=True,
            new_alpha_holes=holes,outside_band_spill=spill,body_and_K=body,fixed_colour_tracking=colour,alphaT_partition_max_abs=partition,
            all22816_present_native_unowned_K0_and_oldguard0_verified=True,actual_joint_image=pin(stage/f'{raw:03d}/shared.png')))
    result=dict(schema='R10.independent_hard_background_counterfactual_review.v1',complete=True,decision='FAIL',measurement_integrity='PASS',visual_decision='FAIL',
        stage_receipt=pin(stage/'receipt.json'),policy_scope=pin(rc['diagnostic_scope']['path']),temporary_rows=22816,
        temporary_opacity_only_no_persistent_gate_change=True,persistent_parent_checks=persistent,
        all5_other_K_RGBA_byteexact=True,strict_V_and_actor_proxy_not_RGBA_byteexact=True,per_view=rows,
        independent_original_resolution_visual_review=dict(reviewer='native_clean_photo_regions',frames=[86,110,134,179,198],root_concurred=False,
            notes='All five actual1152 native joint PNGs inspected. Some dark/blue ghost mass reduces, but cyan/white front tail remains. Left/outer ground becomes visibly patchier/coarser; body core is largely retained, but mean photo-L1 cannot certify all original joint body contributions. Not an acceptable replacement.'),
        causal_interpretation='Selecting a row with any positive contribution to a full ground band does not contain that Gaussian kernel footprint in the edit region. Setting the whole row to zero changes broad outside pixels and removes useful coverage. Colour loss alone cannot identify a safe retirement.',
        suggested_next_step='Measure actual whole-frame, outside-editROI, car and K support per native row. Whole-kernel actions only for outside0/protected0 rows. Sparse bounded opacity, if attempted, needs complete-band unknown alpha coverage and outside/car preservation constraints; mixed-footprint kernels remain until separately reconstructed/split. Do not optimize all22816 unconstrained or spend the remaining1600 steps merely to exhaust the cap.',
        new_representation_split_is_not_automatically_native_render_exact=True,
        production_gate_adopted=False,new_formal_fit_authorized=False,old_formal_updates=400,old_formal_cap=2000,R10_cumulative_actual=dict(nativeF=25,nativeB=10,formal_updates=0),
        this_CPU_audit_GPU_forward=0,this_CPU_audit_GPU_backward=0,updates=0,seconds=time.time()-start)
    write(out/'independent_review.json',result);print(json.dumps(dict(complete=True,review=pin(out/'independent_review.json'),summary=[dict(raw=x['raw'],holes=x['new_alpha_holes'],outside=x['outside_band_spill'],colour=x['fixed_colour_tracking']) for x in rows])),flush=True)


if __name__=='__main__':main()
