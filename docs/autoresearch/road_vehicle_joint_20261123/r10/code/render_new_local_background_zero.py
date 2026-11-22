"""Temporary unowned-background opacity zero plus the accepted local road.

This diagnostic never modifies a persistent asset or adopts a production gate.
Selection is the fixed non-colour ground-band rule, not RGB residual masks.
"""
from pathlib import Path
import argparse,time
import numpy as np
from local_gaussian_asset import ACTOR_SHA,bank,read,write,sha,load_local_asset
from local_joint_adapter import LocalJointAdapter
from render_new_local_zero import pin,npy,png

class DiagnosticBackgroundAdapter(LocalJointAdapter):
    def __init__(self,asset_directory,output,gate,proof):
        super().__init__(asset_directory,output,enable_gpu=True)
        assert self.asset.manifest['native_gate']['mode']=='identity_no_retirement' and not len(self.gate_ids)
        self.diagnostic_ids=self.torch.as_tensor(gate['global_row_ids'],device='cuda',dtype=self.torch.long)
        self.diagnostic_proof=proof;self.applied=[]
    def _native_gate(self,raw,packet):
        packet=super()._native_gate(raw,packet);torch=self.torch
        _,group,owners,gids,offset=self.base.context[raw]
        positions=torch.searchsorted(self.diagnostic_ids,gids);safe=positions.clamp_max(len(self.diagnostic_ids)-1)
        selected=(positions<len(self.diagnostic_ids))&(self.diagnostic_ids[safe]==gids)&(group==0)
        assert int(selected.sum())==len(self.diagnostic_ids),'A diagnostic row is missing from a fixed view'
        assert (owners[selected]<=0).all() and not (owners[selected]==21).any()
        assert not torch.isin(gids[selected],self.base.preserve).any()
        # Only the prepared packet is copied. Source tensors and all persistent
        # opacity values retain their initial bytes.
        ids=torch.nonzero(selected).reshape(-1)
        result=dict(packet);opacity=packet['opacities'].clone();opacity[ids]=0;result['opacities']=opacity
        self.applied.append(dict(raw=raw,temporary_zero_rows=int(selected.sum()),native_positive_owners_untouched=True))
        return result

def qualify(a,asset,cfg):
    scope=read(a.policy/'scope.json');gate=bank(a.policy/'diagnostic_gate0.npz');candidate=bank(a.policy/'diagnostic_native_candidates.npz')
    assert scope['complete'] and scope['decision']=='SCOPE_VALID_DIAGNOSTIC_ONLY'
    assert scope['K_exact_zero'] is True and scope['old_permanent_guard_preserved'] is True
    assert scope['policy']=='ground_band_all_unowned_Kfree_oldguardfree_body_overlap_allowed'
    assert scope['production_asset_ready'] is False and scope['optimization_allowed'] is False
    assert scope['baseline_contract_sha256']==asset.manifest['parent_contract']['sha256']==sha(cfg['baseline_contract'])
    assert scope['actual_attribution_receipt_sha256']==sha(a.measurement/'receipt.json')
    assert scope['diagnostic_gate']['sha256']==sha(a.policy/'diagnostic_gate0.npz')
    assert scope['diagnostic_candidates']['sha256']==sha(a.policy/'diagnostic_native_candidates.npz')
    ids=gate['global_row_ids'];values=gate['gate'];assert ids.dtype==np.int64 and values.dtype==np.float32
    assert ids.ndim==1 and len(ids)>0 and np.array_equal(ids,np.unique(ids)) and np.array_equal(ids,candidate['global_row_ids'])
    assert values.shape==ids.shape and (values==0).all();assert candidate['frames'].tolist()==cfg['frames']
    assert candidate['owners_by_view'].shape==(len(ids),len(cfg['frames'])) and (candidate['owners_by_view']<=0).all()
    assert (candidate['allview_K_mass']==0).all() and (candidate['allview_ground_mass'].sum(1)>0).all()
    assert scope['selection_uses_RGB_or_red_blue_masks'] is False and scope['selection_uses_source_target_residual'] is False
    assert scope['body_overlap_allowed_only_for_counterfactual'] is True and scope['canonical_native21_and_other_positive_owners_fixed'] is True
    measurement=read(a.measurement/'receipt.json');assert measurement['complete'] and measurement['actual_nativeF']==5 and measurement['nativeB']==5
    return scope,gate,candidate

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['asset','measurement-contract','measurement','policy','zero','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--baseline-measurement',type=Path)
    p.add_argument('--targets',type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args()
    a.baseline_measurement=a.baseline_measurement or a.measurement
    asset=load_local_asset(a.asset,verify_parent_pins=True);cfg=read(a.measurement_contract)
    assert cfg['accepted_actor_sha256']==asset.manifest['actor_parent_sha256']==ACTOR_SHA
    assert cfg['frames']==asset.manifest['frames'] and asset.manifest['native_gate']['mode']=='identity_no_retirement'
    scope,gate,candidate=qualify(a,asset,cfg);zero_receipt=read(a.zero/'receipt.json')
    assert zero_receipt['complete'] and zero_receipt['asset_manifest']['sha256']==sha(a.asset/'manifest.json')
    if a.targets:
        target_receipt=read(a.targets/'receipt.json');assert target_receipt['complete'] and target_receipt['zero_receipt']['sha256']==sha(a.zero/'receipt.json')
    baseline_receipt=read(a.baseline_measurement/'receipt.json')
    assert baseline_receipt['complete'] and baseline_receipt['baseline_contract_sha256']==asset.manifest['parent_contract']['sha256']
    # Open every actual input before GPU construction. A newer attribution
    # directory is not implicitly an old RGB-baseline directory.
    for raw in cfg['frames']:
        for file in [a.baseline_measurement/f'{raw:03d}/baseline_joint.npz',a.zero/f'{raw:03d}/joint_render.npz']:
            assert file.is_file(),str(file)
            with np.load(file,allow_pickle=False) as z:
                assert {'rgb','alpha'}.issubset(z.files) and z['rgb'].shape==(3,768,1152) and z['alpha'].shape==(768,1152)
        evpath=Path(cfg['fixed_eval'])/f'{raw:03d}/fixed_eval.npz';assert evpath.is_file()
        if a.targets:
            file=a.targets/f'{raw:03d}/target.npz';assert file.is_file()
            with np.load(file,allow_pickle=False) as z:assert z['support'].shape==(768,1152) and z['target_common_linear'].shape==(768,1152,3)
    if not a.execute:print(dict(prepared=True,nativeF=0,nativeB=0,updates=0,expectedF=5,diagnostic_rows=len(gate['global_row_ids']),persistent_native_gate_remains_identity=True,all_five_baseline_zero_targets_paths_verified=True));return
    import sys
    sys.path.insert(0,asset.manifest['r8_code'])
    from local_row_control import metrics
    from fit_new_local_rgb import weighted_mse
    assert not a.output.exists();a.output.mkdir(parents=True);started=time.time();rt=None;error=None;rows=[];attempts=0
    try:
        rt=DiagnosticBackgroundAdapter(a.asset,a.output/'native',gate,scope)
        with rt.torch.no_grad():
            for raw in cfg['frames']:
                ev=bank(Path(cfg['fixed_eval'])/f'{raw:03d}/fixed_eval.npz');ref=bank(a.zero/f'{raw:03d}/joint_render.npz')
                original=bank(a.baseline_measurement/f'{raw:03d}/baseline_joint.npz')
                masks=bank(Path(cfg['rectified_masks'])/f'{raw:03d}/rectified_masks.npz');cache=bank(Path(cfg['camera_cache'])/f'{raw:03d}/view_cache.npz')
                attempts+=1;out=rt.render(raw);rgb=npy(out['rgb']);alpha=npy(out['alpha']).squeeze();K=ev['K_protected']
                assert rgb[:,K].tobytes()==ref['rgb'][:,K].tobytes() and alpha[K].tobytes()==ref['alpha'][K].tobytes(),'Protected other K changed'
                folder=a.output/f'{raw:03d}';folder.mkdir();local=npy(out['local_road_alpha']);road=npy(out['road_alpha'])
                np.savez_compressed(folder/'joint_render.npz',rgb=rgb,alpha=alpha,
                    depth_expected_m=npy(out['depth_sum_m']).squeeze()/np.maximum(alpha,1e-8),road_alphaT=road,local_road_alphaT=local,
                    existing_road_alphaT=road-local,actor_alphaT=npy(out['actor_alpha']),context_alphaT=npy(out['context_alpha']),unified_actor_alphaT=npy(out['unified_actor_alpha']))
                png(folder/'shared.png',rgb);png(folder/'alpha.png',alpha);png(folder/'local_road_alphaT.png',local)
                png(folder/'RGB_change_vs_new_local_zero.png',np.clip(abs(rgb-ref['rgb'])*4,0,1))
                record=metrics(rgb,alpha,ref,ev);photo=cache['photo_pinhole_srgb'].transpose(2,0,1);valid=masks['RGB_valid']
                domains=dict(accepted_broad_body=ev['body_protected'],actor_proxy=masks['actor_visible_proxy'],other_K=K)
                if 'actor_visible' in masks:domains['strict_real_V']=masks['actor_visible']
                body={}
                for name,d in domains.items():
                    paired=d&valid;body[name]=dict(full_pixels=int(d.sum()),photo_valid_pixels=int(paired.sum()),
                        new_local_zero_photo_L1=float(abs(ref['rgb']-photo)[:,paired].mean()) if paired.any() else None,
                        original_baseline_photo_L1=float(abs(original['rgb']-photo)[:,paired].mean()) if paired.any() else None,
                        background_zero_photo_L1=float(abs(rgb-photo)[:,paired].mean()) if paired.any() else None,
                        RGB_max_delta_new_zero=float(abs(rgb-ref['rgb'])[:,d].max()) if d.any() else None,
                        RGB_mean_delta_new_zero=float(abs(rgb-ref['rgb'])[:,d].mean()) if d.any() else None,
                        alpha_max_delta_new_zero=float(abs(alpha-ref['alpha'])[d].max()) if d.any() else None,
                        RGBA_byteexact_new_zero=bool(rgb[:,d].tobytes()==ref['rgb'][:,d].tobytes() and alpha[d].tobytes()==ref['alpha'][d].tobytes()))
                tracking={}
                for name in ['red_diagnostic','blue_cyan_diagnostic']:
                    fixed=ev[name]
                    condition=((rgb[0]>.2)&(rgb[0]>np.maximum(rgb[1],rgb[2])+.12)) if name=='red_diagnostic' else ((rgb[2]>rgb[0]+.03)&(rgb[2]>rgb[1]+.015)&(rgb.max(0)>.08))
                    tracking[name]=dict(fixed_original_pixels=int(fixed.sum()),condition_remaining_within_FIXED_mask=int((fixed&condition).sum()),
                        scene_alpha_below95=int((fixed&(alpha<.95)).sum()),unknown_fixed_target=int((fixed&~ev['ground_eval']).sum()),denominator_never_reselected=True)
                if a.targets:
                    t=bank(a.targets/f'{raw:03d}/target.npz');record['new_surface_fixed_target_MSE']=weighted_mse(rgb,t['target_common_linear'],t['weight'],t['support'],float(t['exposure_gain']))
                    record['new_surface_fixed_target_pixels']=int(t['support'].sum());record['new_alpha_used_for_target_selection']=False
                record.update(raw=raw,body_comparison=body,fixed_colour_diagnostic_tracking=tracking,
                    original_vehicle_SHA=ACTOR_SHA,all_original_vehicle_parameters_frozen=True,other_K_RGBA_byteexact=True,
                    local_source_and_old_road400_frozen=True,native21_and_other_positive_owners_untouched=True,
                    selection_independent_of_RGB=True,new_local_alphaT_total=float(local.astype(np.float64).sum()),
                    full_original_band_new_alpha95_holes_vs_newzero=int((ev['full_original_band']&(alpha<.95)&(ref['alpha']>=.95)).sum()),
                    unknown_original_target_new_alpha95_holes_vs_newzero=int((ev['full_original_band']&~ev['ground_eval']&(alpha<.95)&(ref['alpha']>=.95)).sum()),
                    strict_V_photo_L1_nonworsening_001=(body['strict_real_V']['background_zero_photo_L1']<=body['strict_real_V']['original_baseline_photo_L1']+.001) if 'strict_real_V' in body else None)
                write(folder/'metrics.json',record);rows.append(record);print(dict(raw=raw,nativeF=rt.calls,nativeB=0,fixed_MSE=record['fixed_ground_weighted_MSE']),flush=True)
        rt.assert_fixed();write(a.output/'camera_timestamp_audit.json',dict(rows=rt.camera_audit));write(a.output/'temporary_gate_trace.json',rt.applied)
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        write(a.output/'receipt.json',dict(schema='R10.new_local_unowned_background_zero_diagnostic.v1',complete=error is None,error=error,
            native_F_attempts=attempts,nativeF=rt.calls if rt else 0,nativeB=0,updates=0,new_local_optimizer_updates=0,
            asset_manifest=pin(a.asset/'manifest.json'),diagnostic_scope=pin(a.policy/'scope.json'),diagnostic_gate=pin(a.policy/'diagnostic_gate0.npz'),
            measurement_receipt=pin(a.measurement/'receipt.json'),baseline_measurement_receipt=pin(a.baseline_measurement/'receipt.json'),new_local_zero_receipt=pin(a.zero/'receipt.json'),rows=rows,
            all_input_paths_verified_before_GPU=True,
            original_vehicle_SHA=ACTOR_SHA,all_original_vehicle_parameters_frozen=True,old_road400_frozen=True,local_asset_unchanged=True,
            all_native_source_parameters_unchanged=True,temporary_native_opacity_gate_applied=True,all_native_retained=False,
            native21_and_other_positive_owners_untouched=True,persistent_native_gate_remains_identity=True,actual_single_depth_sort=True,
            body_proxy_may_change_and_is_reported_separately=True,strict_V_not_assumed_available_for_heldout_views=True,
            existing_representation_formal_updates=400,old_representation_budget=2000,new_diffusion_calls=0,new_semantic_calls=0,
            diagnostic_only=True,whole_scene_quality_accepted=False,default_promotion=False,production_gate_adopted=False,seconds=time.time()-started))

if __name__=='__main__':main()
