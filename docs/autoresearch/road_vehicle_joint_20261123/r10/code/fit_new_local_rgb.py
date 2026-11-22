"""Bounded RGB-only fitting of a new local road, with original parents frozen.

Targets are fixed before training from the accepted direct surface projection.
No alpha-based good-pixel selection; unknown material/geometry stays excluded
and counted. CPU preparation and actual CUDA execution are separate actions.
"""
from pathlib import Path
import argparse,json,random,sys,time
import numpy as np
from local_gaussian_asset import ACTOR_SHA,C0,read,write,sha,bank,load_local_asset,sample_linear
from local_joint_adapter import LocalJointAdapter

def pin(p):return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)
def ar(t):return t.detach().cpu().numpy()
def png(path,value):
    from PIL import Image
    value=np.asarray(value)
    if value.ndim==3 and value.shape[0] in (1,3):value=value.transpose(1,2,0)
    if value.ndim==3 and value.shape[-1]==1:value=value[...,0]
    Image.fromarray(np.rint(np.clip(value,0,1)*255).astype(np.uint8)).save(path)
def linear(x):return np.where(x<=.04045,x/12.92,((np.maximum(x,0)+.055)/1.055)**2.4).astype(np.float32)
def field_sample(a,uv,valid):
    shape=uv.shape[:-1];v,m=sample_linear(a.reshape(*a.shape[:2],-1) if a.ndim==3 else np.repeat(a[...,None],3,axis=-1),uv.reshape(-1,2),valid)
    return v.reshape(*shape,-1),m.reshape(shape)
def weighted_mse(rgb,target,weight,support,gain):
    w=weight[support];error=linear(rgb)*gain-target.transpose(2,0,1)
    return float(((error[:,support]**2).mean(0)*w).sum()/w.sum()) if w.sum() else None

def prepare_targets(a,asset,cfg):
    assert a.surface_review and a.output and not a.output.exists();a.output.mkdir(parents=True)
    source=asset.manifest['source'];field=bank(source['field']['path']);fieldsha=sha(source['field']['path']);assert fieldsha==source['field']['sha256']
    review=read(a.surface_review/'receipt.json');assert review['complete'] and not review['native_Gaussian_render'] and review['final_atlas_sha256']==fieldsha
    geometry_sha=asset.manifest['geometry']['npz']['sha256'];assert review['geometry_sha256']==geometry_sha
    permission=field['permission_mask']&(field['source_kind']>0)&np.isfinite(field['atlas_linear']).all(-1)
    real=field['real_locks'].astype(np.float32);tau=np.clip(field['effective_teacher_blend'],0,1)
    weights=np.stack([real,(1-real)*(1-tau),(1-real)*tau],-1)@np.array([1.,.5,.35],np.float32)
    rows=[];evroot=Path(cfg['fixed_eval'])
    for raw in cfg['frames']:
        ev=bank(evroot/f'{raw:03d}/fixed_eval.npz');surface=bank(a.surface_review/f'{raw:03d}/surface_review.npz')
        uv=surface['atlas_uv_global'];target,legal=field_sample(field['atlas_linear'],uv,permission)
        w,_=field_sample(weights,uv,permission);weight=w[...,0]
        # Fixed full original ground band before old geometry support. The
        # accepted model supplies a separately labelled geometry hypothesis.
        support=ev['ground_band_before_support']&surface['projection_valid']&surface['shown']&legal
        assert not (support&(ev['K_protected']|ev['body_protected'])).any()
        gain=float(ev['exposure_gain']);folder=a.output/f'{raw:03d}';folder.mkdir()
        np.savez_compressed(folder/'target.npz',target_common_linear=target,weight=np.where(support,weight,0).astype(np.float32),
            support=support,whole_original_band=ev['full_original_band'],ground_band_before_support=ev['ground_band_before_support'],
            unsupported_or_unknown_ground=ev['ground_band_before_support']&~support,exposure_gain=np.asarray(gain),atlas_uv_global=uv)
        old=bank(a.baseline_measurement/f'{raw:03d}/baseline_joint.npz');new=bank(a.zero/f'{raw:03d}/joint_render.npz')
        row=dict(raw=raw,full_original_band_pixels=int(ev['full_original_band'].sum()),ground_before_support_pixels=int(ev['ground_band_before_support'].sum()),
            shared_surface_target_pixels=int(support.sum()),unknown_ground_target_pixels=int((ev['ground_band_before_support']&~support).sum()),
            old_original_fixed_ground_pixels=int(ev['ground_eval'].sum()),new_alpha_used_for_selection=False,geometry_is_model_inferred=True,
            original_vehicle_road400_new_target_MSE=weighted_mse(old['rgb'],target,weight,support,gain),
            new_local_zero_new_target_MSE=weighted_mse(new['rgb'],target,weight,support,gain),target=pin(folder/'target.npz'))
        write(folder/'scope.json',row);rows.append(row)
    write(a.output/'receipt.json',dict(schema='R10.fixed_new_local_surface_supervision.v1',complete=True,CPU_only=True,
        field_sha256=fieldsha,geometry_sha256=geometry_sha,direct_surface_receipt=pin(a.surface_review/'receipt.json'),
        original_fixed_eval_receipt=pin(evroot/'receipt.json'),zero_receipt=pin(a.zero/'receipt.json'),frames=cfg['frames'],rows=rows,
        training_frames=[86,110],evaluation_only_frames=[134,179,198],new_alpha_used_for_selection=False,
        input_geometry_not_measured_ground_truth=True,original_fixed_eval_retained=True,unknown_RGB_not_zero_ground_truth=True,
        new_diffusion_calls=0,GPU_forward=0,GPU_backward=0,updates=0))
    print(dict(complete=True,prepared_fixed_targets=True,counts={r['raw']:r['shared_surface_target_pixels'] for r in rows}))

def arrays(rt,raw):
    j=rt.render(raw);alpha=ar(j['alpha']).squeeze();road=ar(j['road_alpha']);local=ar(j['local_road_alpha'])
    return dict(rgb=ar(j['rgb']),alpha=alpha,depth_expected_m=ar(j['depth_sum_m']).squeeze()/np.maximum(alpha,1e-8),
        road_alphaT=road,local_road_alphaT=local,existing_road_alphaT=road-local,actor_alphaT=ar(j['actor_alpha']),
        context_alphaT=ar(j['context_alpha']),unified_actor_alphaT=ar(j['unified_actor_alpha']))

def evaluate(rt,a,cfg,directory,step,curve,targets):
    from local_row_control import metrics
    folder=Path(directory);folder.mkdir(parents=True,exist_ok=False);rows=[]
    with rt.torch.no_grad():
        for raw in cfg['frames']:
            out=arrays(rt,raw);ref=bank(a.baseline_measurement/f'{raw:03d}/baseline_joint.npz');zero=bank(a.zero/f'{raw:03d}/joint_render.npz')
            ev=bank(Path(cfg['fixed_eval'])/f'{raw:03d}/fixed_eval.npz');t=targets[raw];protect=ev['K_protected']|ev['body_protected']
            assert out['alpha'].tobytes()==zero['alpha'].tobytes(),'Fixed geometry alpha changed'
            assert out['rgb'][:,protect].tobytes()==zero['rgb'][:,protect].tobytes(),'K/body colour changed despite guard'
            current=folder/f'{raw:03d}';current.mkdir();np.savez_compressed(current/'joint_render.npz',**out)
            for name,k in [('shared','rgb'),('alpha','alpha'),('local_road_alphaT','local_road_alphaT')]:png(current/(name+'.png'),out[k])
            record=metrics(out['rgb'],out['alpha'],ref,ev)
            record.update(raw=raw,new_surface_fixed_target_MSE=weighted_mse(out['rgb'],t['target_common_linear'],t['weight'],t['support'],float(t['exposure_gain'])),
                new_surface_fixed_target_pixels=int(t['support'].sum()),original_fixed_eval_domain_unchanged=True,new_alpha_used_for_target_selection=False,
                original_vehicle_SHA=ACTOR_SHA,K_body_RGBA_exact_vs_new_zero=True)
            # Track fixed diagnostics without reselecting their denominator.
            diag={}
            for name in ['red_diagnostic','blue_cyan_diagnostic']:
                rgb=out['rgb'];m=ev[name];c=((rgb[0]>.2)&(rgb[0]>np.maximum(rgb[1],rgb[2])+.12)) if name=='red_diagnostic' else ((rgb[2]>rgb[0]+.03)&(rgb[2]>rgb[1]+.015)&(rgb.max(0)>.08))
                diag[name]=dict(fixed_original_pixels=int(m.sum()),condition_remaining_within_FIXED_mask=int((m&c).sum()),denominator_never_reselected=True)
            record['fixed_colour_diagnostic_tracking']=diag
            write(current/'metrics.json',record);rows.append(record)
            curve.append(dict(step=step,raw=raw,original_fixed_ground_MSE=record['fixed_ground_weighted_MSE'],new_surface_fixed_target_MSE=record['new_surface_fixed_target_MSE'],new_surface_fixed_target_pixels=record['new_surface_fixed_target_pixels']))
    write(folder/'metrics.json',dict(step=step,rows=rows));return rows

def fit(a,asset,cfg):
    import torch
    sys.path.insert(0,asset.manifest['r8_code'])
    from fit_joint_rgb import BalancedViews,snapshot,restore,tree_equal
    from joint_replay_state_cpu import compare_states
    target_receipt=read(a.targets/'receipt.json');zero_receipt=read(a.zero/'receipt.json');guard=read(a.zero/'local_colour_guard.json')
    assert target_receipt['complete'] and zero_receipt['complete'] and guard['complete'] and guard['decision']=='PASS'
    assert target_receipt['field_sha256']==asset.manifest['source']['field']['sha256']
    assert zero_receipt['asset_manifest']['sha256']==sha(a.asset/'manifest.json')
    assert not a.output.exists();a.output.mkdir(parents=True);started=time.time();rt=None;updates=0;backward=0;error=None;records=[];curve=[];replay=None;rows=[]
    random.seed(20261123);np.random.seed(20261123);torch.manual_seed(20261123);torch.cuda.manual_seed_all(20261123)
    for r in target_receipt['rows']:assert sha(r['target']['path'])==r['target']['sha256']
    targets={raw:bank(a.targets/f'{raw:03d}/target.npz') for raw in cfg['frames']}
    try:
        rt=LocalJointAdapter(a.asset,a.output/'native',enable_gpu=True);rt.enable_local_rgb(a.zero/'local_colour_guard.npz',a.zero/'local_colour_guard.json')
        rt.fit_cfg=dict(source=dict(sha256=target_receipt['field_sha256']));rt.guard_sha256=guard['npz_sha256'];rt.trainable=[rt.local.dc]
        initial=rt.local.dc.detach().clone();sampler=BalancedViews([86,110],20261123)
        # Adam parameter is SH DC; convert the old road's 0.01 shader-RGB
        # step scale to DC units, instead of silently shrinking by C0.
        optimizer=torch.optim.Adam([rt.local.dc],lr=.01/C0)
        ts={raw:dict(target=torch.as_tensor(v['target_common_linear'].transpose(2,0,1),device='cuda'),weight=torch.as_tensor(v['weight'],device='cuda'),support=torch.as_tensor(v['support'],device='cuda'),gain=float(v['exposure_gain'])) for raw,v in targets.items()}
        write(a.output/'effective_fit_contract.json',dict(formal_steps=400,smoke_logical_steps=6,train_frames=[86,110],evaluation_only_frames=[134,179,198],seed=20261123,
            learning_rate_shader_RGB=.01,optimizer_learning_rate_DC=.01/C0,RGB_prior=.0001,only_new_local_DC_trainable=True,
            fixed_full_ground_band_before_support=True,new_alpha_used_for_target_selection=False,source=pin(a.targets/'receipt.json'),guard=pin(a.zero/'local_colour_guard.json'),
            old_formal_updates_remain400=True,engineering_steps_separate=True))
        evaluate(rt,a,cfg,a.output/'initial_readout',0,curve,targets)
        def train(step,is_replay=False):
            nonlocal updates,backward
            raw=sampler.next();optimizer.zero_grad(set_to_none=True);out=rt.render(raw);t=ts[raw]
            x=out['rgb'].clamp_min(0);pred=torch.where(x<=.04045,x/12.92,((x+.055)/1.055).pow(2.4))*t['gain']
            m=t['support'];w=t['weight'][m];assert m.any() and float(w.sum())>0
            material=((pred[:,m]-t['target'][:,m]).square().mean(0)*w).sum()/w.sum()
            prior=((rt.local.dc-initial)*C0).square().mean();loss=material+.0001*prior;assert torch.isfinite(loss)
            loss.backward();backward+=1;optimizer.step();updates+=1
            with torch.no_grad():rt.local.dc.clamp_(-.5/C0,.5/C0)
            record=dict(step=step,raw=raw,replay=is_replay,loss=float(loss.detach()),fixed_supervision_MSE=float(material.detach()),prior=float(prior.detach()),fixed_supervision_pixels=int(m.sum()))
            records.append(record)
            if step%25==0 or is_replay or a.mode=='smoke':print(record,flush=True)
            return record
        steps=6 if a.mode=='smoke' else 400
        for step in range(1,steps+1):
            row=train(step)
            if a.mode=='smoke' and step==4:torch.save(snapshot(rt,optimizer,sampler,step),a.output/'checkpoint_step000004.pt')
            if a.mode=='smoke' and step==5:
                expected=snapshot(rt,optimizer,sampler,step);state=torch.load(a.output/'checkpoint_step000004.pt',map_location='cpu',weights_only=False)
                restore(rt,optimizer,sampler,state);restored=compare_states(state,snapshot(rt,optimizer,sampler,4),a.output/'checkpoint_restore',strict=True);assert restored['decision']=='PASS'
                replayed=train(step,True);actual=snapshot(rt,optimizer,sampler,step);checked=compare_states(expected,actual,a.output/'cuda_replay',eps_multiplier=4)
                replay=dict(complete=True,decision=checked['decision'],restore_complete_state_bitexact=True,replay_same_frame=row['raw']==replayed['raw'],
                    replay_loss_abs=abs(row['loss']-replayed['loss']),replay_full_state_bitexact=tree_equal(expected,actual),all_state_leaves_checked=True,
                    floating_rule='abs(replay-ref)<=4*eps(dtype)*(1+abs(ref))',counters_RNG_exact=True,loss_bound=1e-7,extra_replay_updates=1)
                write(a.output/'restore_replay.json',replay);assert checked['decision']=='PASS' and replay['replay_same_frame'] and replay['replay_loss_abs']<=1e-7
            if a.mode=='fit' and step in [100,200]:evaluate(rt,a,cfg,a.output/f'eval_step{step:06d}',step,curve,targets)
            if step%100==0 or step==steps:torch.save(snapshot(rt,optimizer,sampler,step),a.output/f'checkpoint_step{step:06d}.pt')
        rows=evaluate(rt,a,cfg,a.output/'final_readout',steps,curve,targets);rt.assert_fixed()
        rt.export(a.output/'asset',dict(actual_optimizer_updates=updates,logical_steps=steps,mode=a.mode,formal_updates=400 if a.mode=='fit' else 0,
            smoke_updates=updates if a.mode=='smoke' else 0,only_new_local_RGB=True,source_target_receipt_sha256=sha(a.targets/'receipt.json'),guard_npz_sha256=guard['npz_sha256']))
        write(a.output/'losses.json',records);write(a.output/'fixed_eval_curve.json',curve)
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        write(a.output/'receipt.json',dict(schema='R10.actual_new_local_RGB_fit.v1',complete=error is None,error=error,mode=a.mode,
            asset_manifest=pin(a.asset/'manifest.json'),fixed_targets=pin(a.targets/'receipt.json'),nativeF=rt.calls if rt else 0,nativeB=backward,
            actual_optimizer_updates=updates,new_local_optimizer_updates=updates if a.mode=='fit' else 0,engineering_smoke_updates=updates if a.mode=='smoke' else 0,
            original_vehicle_SHA=ACTOR_SHA,all_original_vehicle_parameters_frozen=True,all_native_retained=True,old_road400_frozen=True,
            existing_representation_formal_updates=400,old_representation_budget=2000,actual_single_depth_sort=True,new_local_centres_normals_scales_opacity_fixed=True,
            train_frames=[86,110],evaluation_only_frames=[134,179,198],new_alpha_used_for_target_selection=False,restore_replay=replay,
            rows=rows,new_diffusion_calls=0,new_semantic_calls=0,fresh_process_verify='PENDING',whole_scene_quality_accepted=False,
            peak_GPU_bytes=int(torch.cuda.max_memory_allocated()) if torch.cuda.is_initialized() else 0,seconds=time.time()-started))

def verify(a,asset,cfg):
    assert a.expected and not a.output.exists();a.output.mkdir(parents=True);rt=None;error=None;rows=[]
    try:
        rt=LocalJointAdapter(a.asset,a.output/'native',enable_gpu=True)
        with rt.torch.no_grad():
            for raw in cfg['frames']:
                got=arrays(rt,raw);ref=bank(a.expected/f'{raw:03d}/joint_render.npz');assert got.keys()==ref.keys();values={}
                folder=a.output/f'{raw:03d}';folder.mkdir();np.savez_compressed(folder/'joint_render.npz',**got)
                for k,v in got.items():
                    values[k]=float(abs(v-ref[k]).max());assert v.dtype==ref[k].dtype and v.shape==ref[k].shape and v.tobytes()==ref[k].tobytes(),(raw,k,values[k])
                rows.append(dict(raw=raw,all_float_arrays_byteexact=True,max_abs=values))
        rt.assert_fixed()
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc));raise
    finally:write(a.output/'receipt.json',dict(schema='R10.fresh_local_joint_asset_native_reload.v1',complete=error is None,error=error,
        actual_fresh_loader=True,training_targets_loaded=False,asset_manifest=pin(a.asset/'manifest.json'),nativeF=rt.calls if rt else 0,nativeB=0,updates=0,
        original_vehicle_SHA=ACTOR_SHA,all_native_retained=True,old_road400_frozen=True,rows=rows))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['prepare-targets','smoke','fit','verify'],required=True)
    for k in ['asset','measurement-contract','output']:p.add_argument('--'+k,type=Path,required=True)
    for k in ['surface-review','targets','baseline-measurement','zero','expected']:p.add_argument('--'+k,type=Path)
    p.add_argument('--execute',action='store_true');a=p.parse_args();asset=load_local_asset(a.asset,verify_parent_pins=True);cfg=read(a.measurement_contract)
    assert asset.manifest['actor_parent_sha256']==cfg['accepted_actor_sha256']==ACTOR_SHA and cfg['frames']==asset.manifest['frames']
    sys.path.insert(0,asset.manifest['r8_code'])
    if a.mode=='prepare-targets':assert a.baseline_measurement and a.zero;prepare_targets(a,asset,cfg);return
    if not a.execute:print(dict(prepared=True,mode=a.mode,GPU_forward=0,GPU_backward=0,updates=0,formal_steps=400,new_asset_updates_separate_from_old400=True));return
    if a.mode=='verify':verify(a,asset,cfg)
    else:assert a.targets and a.zero and a.baseline_measurement;fit(a,asset,cfg)

if __name__=='__main__':main()
