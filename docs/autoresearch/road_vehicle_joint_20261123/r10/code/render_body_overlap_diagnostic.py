"""A separately registered zero-update counterfactual, not a selected asset.

Only certified K-free/permanent-guard-free unowned red contributors are gated.
Accepted vehicle/native21 assets stay exact; body occlusion changes are exposed.
"""
from pathlib import Path
import argparse,time
import numpy as np
from r10_common import contracts,read,load,write,sha,source_gate,npy,LocalGateAdapter
from local_row_control import evaluation,metrics,png,pin


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',type=Path,required=True);p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--measurement',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    cfg,base=contracts(a.contract);source=source_gate(cfg,strict=a.execute);evroot,_=evaluation(cfg)
    scope=read(a.policy/'scope.json');gate=load(a.policy/'diagnostic_gate0.npz')
    assert scope['complete'] and scope['decision']=='SCOPE_VALID_DIAGNOSTIC_ONLY' and scope['all_five_K_exact_zero']
    assert scope['baseline_contract_sha256']==sha(cfg['baseline_contract']) and scope['source_measurement']['sha256']==sha(a.measurement/'receipt.json')
    assert scope['diagnostic_gate']['sha256']==sha(a.policy/'diagnostic_gate0.npz') and (gate['gate']==0).all()
    if not a.execute:print(dict(prepared=True,candidate_rows=len(gate['global_row_ids']),expectedF=5,nativeF=0,nativeB=0,updates=0,body_occlusion_may_change=True));return
    if not len(gate['global_row_ids']):print(dict(not_run=True,reason='empty diagnostic set',nativeF=0));return
    import torch
    a.output.mkdir(parents=True,exist_ok=False);rt=None;error=None;rows=[];started=time.time()
    try:
        rt=LocalGateAdapter(base,a.output/'native',gate['global_row_ids'],cfg['deleted_target_owner_ids'])
        inverse=np.linalg.inv(rt.transform)
        with torch.no_grad():
            for raw in cfg['frames']:
                ev=load(evroot/f'{raw:03d}/fixed_eval.npz');ref=load(a.measurement/f'{raw:03d}/baseline_joint.npz')
                masks=load(Path(cfg['rectified_masks'])/f'{raw:03d}/rectified_masks.npz');cache=load(Path(cfg['camera_cache'])/f'{raw:03d}/view_cache.npz')
                original=load(Path(cfg['original_R8_baseline_contract']).parent/f'baseline_A0_five_views/{raw:03d}/joint_render.npz')
                out=rt.render(raw);rgb=npy(out['rgb']);alpha=npy(out['alpha']).squeeze();K=ev['K_protected']
                assert rgb[:,K].tobytes()==ref['rgb'][:,K].tobytes() and alpha[K].tobytes()==ref['alpha'][K].tobytes(),'Other protected K changed'
                folder=a.output/f'{raw:03d}';folder.mkdir()
                np.savez_compressed(folder/'joint_render.npz',rgb=rgb,alpha=alpha,
                    depth_expected_m=npy(out['depth_sum_m']).squeeze()/np.maximum(alpha,1e-8),road_alphaT=npy(out['road_alpha']),actor_alphaT=npy(out['actor_alpha']),
                    context_alphaT=npy(out['context_alpha']),unified_actor_alphaT=npy(out['unified_actor_alpha']))
                png(folder/'shared.png',rgb);png(folder/'alpha.png',alpha);png(folder/'baseline_phase0.png',ref['rgb']);png(folder/'original_A0_joint.png',original['rgb'])
                png(folder/'RGB_change_display.png',np.clip(abs(rgb-ref['rgb'])*4,0,1))
                record=metrics(rgb,alpha,ref,ev);photo=cache['photo_pinhole_srgb'].transpose(2,0,1);valid=masks['RGB_valid']
                domains=dict(accepted_body=ev['body_protected'],actor_proxy=masks['actor_visible_proxy'],other_K=K)
                if 'actor_visible' in masks:domains['strict_real_V']=masks['actor_visible']
                body_rows={}
                for name,d in domains.items():
                    paired=d&valid
                    body_rows[name]=dict(full_pixels=int(d.sum()),photo_valid_pixels=int(paired.sum()),
                        phase0_photo_L1=float(abs(ref['rgb']-photo)[:,paired].mean()) if paired.any() else None,
                        counterfactual_photo_L1=float(abs(rgb-photo)[:,paired].mean()) if paired.any() else None,
                        original_A0_photo_L1=float(abs(original['rgb']-photo)[:,paired].mean()) if paired.any() else None,
                        RGB_max_delta_phase0=float(abs(rgb-ref['rgb'])[:,d].max()) if d.any() else None,
                        RGB_mean_delta_phase0=float(abs(rgb-ref['rgb'])[:,d].mean()) if d.any() else None,
                        alpha_max_delta_phase0=float(abs(alpha-ref['alpha'])[d].max()) if d.any() else None,
                        RGB_mean_delta_original_A0=float(abs(rgb-original['rgb'])[:,d].mean()) if d.any() else None)
                diagnostics={}
                for name in ['red_diagnostic','blue_cyan_diagnostic']:
                    fixed=ev[name]
                    if name=='red_diagnostic':condition=(rgb[0]>.2)&(rgb[0]>np.maximum(rgb[1],rgb[2])+.12)
                    else:condition=(rgb[2]>rgb[0]+.03)&(rgb[2]>rgb[1]+.015)&(rgb.max(0)>.08)
                    diagnostics[name]=dict(fixed_original_pixels=int(fixed.sum()),condition_remaining_within_FIXED_mask=int((fixed&condition).sum()),
                        scene_alpha_below95_within_FIXED_mask=int((fixed&(alpha<.95)).sum()),target_unknown_within_FIXED_mask=int((fixed&~ev['ground_eval']).sum()),
                        denominator_never_reselected=True,not_a_quality_PASS=True)
                record.update(raw=raw,policy='body_overlap_diagnostic',other_K_RGBA_exact=True,body_comparison=body_rows,fixed_colour_diagnostic_tracking=diagnostics,
                    canonical_asset_SHA=cfg['accepted_actor_sha256'],canonical_and_native21_parameters_unchanged=True)
                # Locator geometry is the original prepared native rows, not a clean-road initializer.
                context,group,owner,gids,_=rt.context[raw];take=(group==0)&torch.isin(gids,torch.as_tensor(gate['global_row_ids'],device='cuda'))
                assert int(take.sum())==len(gate['global_row_ids']);means=npy(context['means3D'][take])
                world=means.astype(np.float64)@inverse[:3,:3].T+inverse[:3,3]
                np.savez_compressed(folder/'retired_prepared_geometry.npz',global_row_ids=npy(gids[take]),owner_ids=npy(owner[take]),
                    means_model=means,means_world_locator_only=world,scales_model=npy(context['scales'][take]),rotations_model=npy(context['rotations'][take]),
                    opacity_before=npy(context['opacities'][take]),gate=np.zeros(int(take.sum()),np.float32),
                    model_from_world=rt.transform,clean_road_geometry_certificate=np.asarray(False))
                write(folder/'metrics.json',record);rows.append(record)
        rt.assert_fixed();write(a.output/'camera_timestamp_audit.json',dict(rows=rt.camera_audit))
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        write(a.output/'receipt.json',dict(schema='R10.body_overlap_zero_step_counterfactual.v1',complete=error is None,error=error,
            nativeF=rt.calls if rt else 0,nativeB=0,updates=0,seconds=time.time()-started,policy_scope=pin(a.policy/'scope.json'),
            source_field_sha256=cfg['source']['sha256'],accepted_actor_sha256=cfg['accepted_actor_sha256'],candidate_rows=len(gate['global_row_ids']),rows=rows,
            body_occlusion_is_allowed_to_change_but_accepted_assets_are_fixed=True,other_K_and_old_guard_not_relaxed=True,
            diagnostic_only=True,new_geometry=False,formal_updates_consumed=0,quality_accepted=False,production_asset_ready=False,optimization_allowed=False))


if __name__=='__main__':main()
