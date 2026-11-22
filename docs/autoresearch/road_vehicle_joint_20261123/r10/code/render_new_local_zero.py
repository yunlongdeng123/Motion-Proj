"""Root-scheduled actual local-road/original-vehicle joint zero-step control.

One native depth sort per view. Optional feature VJP measures the appended
local rows' actual K/body contributions in those same five forwards.
"""
from pathlib import Path
import argparse,time
import numpy as np
from local_gaussian_asset import ACTOR_SHA,load_local_asset,sha,read,write,bank
from local_joint_adapter import LocalJointAdapter,readiness

def pin(p):return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)
def npy(t):return t.detach().cpu().numpy()
def png(p,x):
    from PIL import Image
    x=np.asarray(x)
    if x.ndim==3 and x.shape[0] in (1,3):x=x.transpose(1,2,0)
    if x.ndim==3 and x.shape[-1]==1:x=x[...,0]
    Image.fromarray(np.rint(np.clip(x,0,1)*255).astype(np.uint8)).save(p)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['asset','measurement-contract','baseline-measurement','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--measure-row-guard',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
    asset=load_local_asset(a.asset,verify_parent_pins=True);cfg=read(a.measurement_contract)
    import sys
    sys.path.insert(0,asset.manifest['r8_code'])
    from joint_rgb_common import srgb_to_linear_numpy
    from local_row_control import metrics
    assert cfg['accepted_actor_sha256']==asset.manifest['actor_parent_sha256']==ACTOR_SHA
    assert cfg['frames']==asset.manifest['frames'] and cfg['existing_formal_updates']==400
    assert asset.manifest['native_gate'].get('mode')=='identity_no_retirement','This first comparison retains all native context'
    assert len(asset.native_gate['global_row_ids'])==0
    assert sha(cfg['baseline_contract'])==asset.manifest['parent_contract']['sha256']
    evroot=Path(cfg['fixed_eval']);evreceipt=read(evroot/'receipt.json');assert evreceipt['complete']
    for row in evreceipt['rows']:assert sha(row['fixed_eval']['path'])==row['fixed_eval']['sha256']
    baseline_receipt=read(a.baseline_measurement/'receipt.json');assert baseline_receipt['complete']
    assert baseline_receipt['baseline_contract_sha256']==asset.manifest['parent_contract']['sha256']
    if not a.execute:
        print(dict(prepared=True,asset_readiness=readiness(a.asset),expected_nativeF=len(cfg['frames']),expected_nativeB=len(cfg['frames']) if a.measure_row_guard else 0,updates=0,all_old_native_retained=True,original_vehicle_frozen=True));return
    import torch
    assert not a.output.exists();a.output.mkdir(parents=True);started=time.time();rt=None;B=0;attempts=0;error=None;rows=[];km=[];bm=[];preserved=[]
    initial_local={k:v.copy() for k,v in asset.field.items()};actor_before=sha(read(asset.manifest['parent_contract']['path'])['actors'][0]['asset'])
    try:
        rt=LocalJointAdapter(a.asset,a.output/'native',enable_gpu=True)
        for raw in cfg['frames']:
            ev=bank(evroot/f'{raw:03d}/fixed_eval.npz');ref=bank(a.baseline_measurement/f'{raw:03d}/baseline_joint.npz')
            masks=bank(Path(cfg['rectified_masks'])/f'{raw:03d}/rectified_masks.npz');cache=bank(Path(cfg['camera_cache'])/f'{raw:03d}/view_cache.npz')
            packet=rt.packet(raw);local_start=len(packet['means3D'])-len(asset.field['xyz']);assert local_start>0
            if a.measure_row_guard:
                columns=[rt.local_channel+1,rt.local_channel+2];assert columns[-1]<24
                feature=packet['features'].detach().clone();feature[:,columns]=1;feature.requires_grad_(True);packet=dict(packet,features=feature)
            cam=rt.cameras[raw];attempts+=1
            out=rt.scene._bindings().raster(cam,rt.scene.pipe,0,packet,rt.scene._background);rt.base.calls+=1
            rt.base.trace.append(dict(call=rt.base.calls,raw=raw,component='shared',new_local_asset=True,single_joint_native_raster=True,row_guard=a.measure_row_guard))
            env=rt.scene.environment(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
            rgb=npy(out['rgb']+(1-out['alpha'])*env);alpha=npy(out['alpha']).squeeze()
            if a.measure_row_guard:
                regions=[ev['K_protected'],ev['body_protected']]
                loss=sum((out['extra'][c-2]*torch.as_tensor(m,device='cuda')).sum() for c,m in zip(columns,regions))
                grad=torch.autograd.grad(loss,feature)[0][:,columns];B+=1;assert float(grad.min())>=0
                mass=npy(grad[local_start:]);assert mass.shape==(len(asset.field['xyz']),2)
                conserved=[]
                for j,(c,m) in enumerate(zip(columns,regions)):
                    one_error=float(abs(npy(out['extra'][c-2])-alpha).max());row_sum=float(npy(grad[:,j]).astype(np.float64).sum());direct=float(alpha[m].astype(np.float64).sum())
                    assert one_error<3e-6 and abs(row_sum-direct)<max(.01,2e-6*m.sum())
                    conserved.append(dict(region=['K','body'][j],row_mass=row_sum,direct_alpha_mass=direct,abs_error=abs(row_sum-direct),constant_one_alpha_max_abs=one_error))
                km.append(mass[:,0]);bm.append(mass[:,1]);del grad,loss,feature
            else:conserved=[]
            folder=a.output/f'{raw:03d}';folder.mkdir()
            local=npy(out['extra'][rt.local_channel-2]);road=npy(out['extra'][0])
            np.savez_compressed(folder/'joint_render.npz',rgb=rgb,alpha=alpha,
                depth_expected_m=npy(out['depth_sum']).squeeze()/rt.meter_to_model/np.maximum(alpha,1e-8),
                road_alphaT=road,local_road_alphaT=local,existing_road_alphaT=road-local,
                actor_alphaT=npy(out['extra'][1]),context_alphaT=npy(out['extra'][2]),unified_actor_alphaT=npy(out['extra'][3:3+len(rt.actors)]))
            png(folder/'shared.png',rgb);png(folder/'alpha.png',alpha);png(folder/'local_road_alphaT.png',local);png(folder/'baseline_phase0.png',ref['rgb'])
            png(folder/'RGB_change_display.png',np.clip(abs(rgb-ref['rgb'])*4,0,1))
            record=metrics(rgb,alpha,ref,ev);photo=cache['photo_pinhole_srgb'].transpose(2,0,1);valid=masks['RGB_valid']
            domains=dict(accepted_body=ev['body_protected'],actor_proxy=masks['actor_visible_proxy'],other_K=ev['K_protected'])
            if 'actor_visible' in masks:domains['strict_real_V']=masks['actor_visible']
            body={}
            for name,d in domains.items():
                paired=d&valid;body[name]=dict(full_pixels=int(d.sum()),photo_valid_pixels=int(paired.sum()),
                    baseline_photo_L1=float(abs(ref['rgb']-photo)[:,paired].mean()) if paired.any() else None,
                    new_local_photo_L1=float(abs(rgb-photo)[:,paired].mean()) if paired.any() else None,
                    RGB_max_delta=float(abs(rgb-ref['rgb'])[:,d].max()) if d.any() else None,
                    RGB_mean_delta=float(abs(rgb-ref['rgb'])[:,d].mean()) if d.any() else None,
                    alpha_max_delta=float(abs(alpha-ref['alpha'])[d].max()) if d.any() else None,
                    new_local_alphaT_mass=float(local[d].astype(np.float64).sum()))
            tracking={}
            for name in ['red_diagnostic','blue_cyan_diagnostic']:
                fixed=ev[name]
                condition=((rgb[0]>.2)&(rgb[0]>np.maximum(rgb[1],rgb[2])+.12)) if name=='red_diagnostic' else ((rgb[2]>rgb[0]+.03)&(rgb[2]>rgb[1]+.015)&(rgb.max(0)>.08))
                tracking[name]=dict(fixed_original_pixels=int(fixed.sum()),condition_remaining_within_FIXED_mask=int((fixed&condition).sum()),
                    scene_alpha_below95=int((fixed&(alpha<.95)).sum()),unknown_fixed_target=int((fixed&~ev['ground_eval']).sum()),new_local_alphaT_mass=float(local[fixed].astype(np.float64).sum()),denominator_never_reselected=True)
            record.update(raw=raw,body_comparison=body,fixed_colour_diagnostic_tracking=tracking,feature_VJP_conservation=conserved,
                all_native_retained=True,old_road400_frozen=True,original_vehicle_SHA=ACTOR_SHA,
                protected_K_RGBA_byteexact=bool(rgb[:,ev['K_protected']].tobytes()==ref['rgb'][:,ev['K_protected']].tobytes() and alpha[ev['K_protected']].tobytes()==ref['alpha'][ev['K_protected']].tobytes()),
                new_local_alphaT_total=float(local.astype(np.float64).sum()))
            write(folder/'metrics.json',record);rows.append(record);print(dict(raw=raw,F=rt.calls,B=B,fixed_loss=record['fixed_ground_weighted_MSE']),flush=True)
            del out,packet
        rt.assert_fixed();assert actor_before==sha(read(asset.manifest['parent_contract']['path'])['actors'][0]['asset'])==ACTOR_SHA
        for k,v in initial_local.items():assert np.array_equal(v,asset.field[k])
        if a.measure_row_guard:
            K=np.stack(km);body=np.stack(bm);trainable=(K==0).all(0)&(body==0).all(0)
            np.savez_compressed(a.output/'local_colour_guard.npz',frames=np.asarray(cfg['frames']),allview_K_mass=K,allview_body_mass=body,trainable_rgb_rows=trainable)
            write(a.output/'local_colour_guard.json',dict(schema='R10.actual_new_local_RGB_guard.v1',complete=True,decision='PASS',
                field_sha256=asset.manifest['files']['local_field.npz']['sha256'],npz_sha256=sha(a.output/'local_colour_guard.npz'),
                all_five_K_body_verified=True,trainable_rows=int(trainable.sum()),frozen_rows=int((~trainable).sum()),
                all_new_field_K_body_zero=bool((K==0).all() and (body==0).all()),quality_acceptance=False,
                interpretation='PASS certifies actual row measurement and a colour-gradient mask; it does not accept joint imagery or authorize opacity/scale changes'))
        write(a.output/'camera_timestamp_audit.json',dict(rows=rt.camera_audit))
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        write(a.output/'receipt.json',dict(schema='R10.new_local_joint_zero_step.v1',complete=error is None,error=error,
            asset_manifest=pin(a.asset/'manifest.json'),baseline_measurement_receipt=pin(a.baseline_measurement/'receipt.json'),fixed_eval_receipt=pin(evroot/'receipt.json'),
            measurement_contract=pin(a.measurement_contract),native_F_attempts=attempts,nativeF=rt.calls if rt else 0,nativeB=B,
            updates=0,existing_representation_formal_updates=400,old_representation_budget=2000,new_local_optimizer_updates=0,
            original_vehicle_SHA=ACTOR_SHA,all_original_vehicle_parameters_frozen=True,all_native_retained=True,old_road400_frozen=True,
            actual_single_depth_sort=True,new_diffusion_calls=0,new_semantic_calls=0,rows=rows,
            whole_scene_quality_accepted=False,default_promotion=False,seconds=time.time()-started))

if __name__=='__main__':main()
