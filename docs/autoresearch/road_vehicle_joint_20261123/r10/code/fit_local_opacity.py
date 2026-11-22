"""Minimal root-scheduled opacity attenuation: prepare / smoke / fit / reload.

One actual joint raster and opacity backward per update. Local asset, approved
vehicle, native positive owners and all geometry/colours remain immutable.
The fresh sidecar loader does not read any targets or training contract.
"""
from pathlib import Path
import argparse,random,sys,time
import numpy as np
from local_gaussian_asset import bank,read,write,sha
from local_opacity_gate import OpacityGateAdapter,load_opacity_sidecar,pin
from fit_new_local_rgb import arrays,png


def check_inputs(path):
    cfg=read(path);assert cfg['complete'] and cfg['formal_steps']==400 and cfg['learning_rate']==.001
    assert cfg['train_frames']==[86,110] and cfg['evaluation_only_frames']==[134,179,198]
    for spec in [*cfg['input_pins'],cfg['eligible'],*[row['target'] for row in cfg['rows']]]:
        assert sha(spec['path'])==spec['sha256'],('Changed frozen input',spec['path'])
    assert sha(cfg['source']['field'])==cfg['source']['sha256']
    return cfg


def checks(value,target,cfg,*,zero=False):
    rgb=value['rgb'];alpha=value['alpha'];out=target['outside_allowed_edit'];protect=target['K']|target['car']
    result=dict(finite=bool(all(np.isfinite(v).all() for v in value.values())),
        K_car_RGB_exact=rgb[:,protect].tobytes()==target['parentzero_rgb'][:,protect].tobytes(),
        K_car_alpha_exact=alpha[protect].tobytes()==target['parentzero_alpha'][protect].tobytes(),
        outside_RGB_max=float(abs(rgb-target['original_rgb'])[:,out].max(initial=0)),
        outside_alpha_max=float(abs(alpha-target['original_alpha'])[out].max(initial=0)),
        new_full_band_alpha95_holes=int((target['alpha_floor_mask']&(alpha<.95)).sum()),
        unknown_inside_alpha95_holes=int((target['alpha_floor_mask']&target['unknown_inside']&(alpha<.95)).sum()),
        outside_changed_RGB_pixels_above1e6=int((out&(abs(rgb-target['original_rgb']).max(0)>1e-6)).sum()))
    passed=result['finite'] and result['K_car_RGB_exact'] and result['K_car_alpha_exact']
    passed&=result['outside_RGB_max']<=cfg['outside_RGB_max_abs_bound']
    passed&=result['outside_alpha_max']<=cfg['outside_alpha_max_abs_bound']
    passed&=result['new_full_band_alpha95_holes']==0
    if zero:
        result['step0_full_RGBA_exact_parentzero']=rgb.tobytes()==target['parentzero_rgb'].tobytes() and alpha.tobytes()==target['parentzero_alpha'].tobytes()
        passed&=result['step0_full_RGBA_exact_parentzero']
    result['decision']='PASS' if passed else 'FAIL'
    return result


def linear(torch,x):
    z=x.clamp_min(0)
    return torch.where(z<=.04045,z/12.92,((z+.055)/1.055).pow(2.4))


def losses(rt,raw,joint):
    t=rt.targets[raw];torch=rt.torch;rgb=joint['rgb'];alpha=joint['alpha'].squeeze();w=rt.fit_cfg['weights']
    def ms(value,mask):return value[...,mask].square().mean() if mask.any() else value.sum()*0
    def mx(value,mask):return value[...,mask].abs().max().square() if mask.any() else value.sum()*0
    diff=rgb-t['original_rgb'];adiff=alpha-t['original_alpha'];outside=t['outside_allowed_edit'];protect=t['K']|t['car']
    parts=dict(inside_joint_canonical_linear_MSE=ms(linear(torch,rgb)*t['gain']-t['target_common_linear'],t['support']),
        outside_original_sRGB_MSE=ms(diff,outside),outside_original_sRGB_max_squared=mx(diff,outside),
        outside_original_alpha_MSE=ms(adiff,outside),outside_original_alpha_max_squared=mx(adiff,outside),
        K_car_original_RGB_alpha_MSE=ms(diff,protect)+ms(adiff,protect),
        full_band_original_alpha95_floor_hinge_squared=ms(torch.relu(.95-alpha),t['alpha_floor_mask']),
        opacity_attenuation_L1_prior=(1-rt.opacity_parameter).mean())
    total=sum(w[k]*v for k,v in parts.items());assert torch.isfinite(total)
    return total,{k:float(v.detach()) for k,v in parts.items()}


def evaluate(rt,cfg,targets,directory,step):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);rows=[]
    from local_row_control import metrics
    from fit_new_local_rgb import linear as np_linear
    for raw in cfg['frames']:
        with rt.torch.no_grad():value=arrays(rt,raw)
        folder=directory/f'{raw:03d}';folder.mkdir();t=targets[raw]
        np.savez_compressed(folder/'joint_render.npz',**value)
        for name,key in [('shared','rgb'),('alpha','alpha'),('local_road_alphaT','local_road_alphaT')]:png(folder/(name+'.png'),value[key])
        guard=checks(value,t,cfg,zero=step==0)
        ref=dict(rgb=t['original_rgb'],alpha=t['original_alpha'])
        ev=bank(Path(cfg['fixed_eval'])/f'{raw:03d}/fixed_eval.npz')
        rec=metrics(value['rgb'],value['alpha'],ref,ev)
        support=t['support'];residual=np_linear(value['rgb'])*float(t['gain'])-t['target_common_linear']
        rec.update(raw=raw,step=step,guard=guard,fixed_inside_target_pixels=int(support.sum()),
            fixed_inside_joint_MSE=float(np.square(residual[:,support]).mean()),
            target_denominator_not_reselected=True,strict_real_V_opened=raw in cfg['train_frames'])
        car=t['car'];photo=t['photo_srgb']
        rec['car_photo_L1_change']=float(abs(value['rgb']-photo)[:,car].mean()-abs(t['original_rgb']-photo)[:,car].mean()) if car.any() else None
        write(folder/'metrics.json',rec);rows.append(rec)
    result=dict(complete=True,step=step,rows=rows,decision='PASS' if all(x['guard']['decision']=='PASS' for x in rows) else 'FAIL',
        train_fixed_joint_mean_MSE=float(np.mean([x['fixed_inside_joint_MSE'] for x in rows if x['raw'] in cfg['train_frames']])),
        evaluation_only_frames_not_used_for_parameter_selection=True,whole_scene_quality_accepted=False)
    write(directory/'metrics.json',result);write(directory/'camera_timestamp_audit.json',dict(rows=rt.camera_audit))
    assert result['decision']=='PASS',('Actual joint protection failed; preserve run',result)
    return result


def train_one(rt,optimizer,sampler,step,records,*,replay=False):
    raw=sampler.next();optimizer.zero_grad(set_to_none=True);joint=rt.render(raw)
    # Guard the same actual joint forward; never substitute a layer subtraction.
    guard=checks({'rgb':joint['rgb'].detach().cpu().numpy(),'alpha':joint['alpha'].detach().cpu().numpy().squeeze()},rt.targets_numpy[raw],rt.fit_cfg)
    assert guard['decision']=='PASS',('Pre-update actual joint guard',raw,guard)
    loss,parts=losses(rt,raw,joint);loss.backward();rt.backward_calls+=1
    grad=rt.opacity_parameter.grad
    assert grad is not None and rt.torch.isfinite(grad).all(),'Native opacity backward missing/nonfinite'
    maximum=float(grad.abs().max())
    # The analytic prior also differentiates g. Remove it for this audit so
    # a missing native-opacity adjoint cannot pass on the prior alone.
    native_maximum=float((grad+rt.fit_cfg['weights']['opacity_attenuation_L1_prior']/grad.numel()).abs().max())
    assert native_maximum>1e-12,'Native data opacity gradient is missing/identically zero after removing the analytic prior'
    optimizer.step()
    with rt.torch.no_grad():rt.opacity_parameter.clamp_(0,1)
    rt.actual_updates+=1;rt.assert_opacity_only()
    row=dict(step=step,raw=raw,replay=replay,loss=float(loss.detach()),opacity_gradient_abs_max=maximum,native_data_opacity_gradient_abs_max=native_maximum,
        gate_min=float(rt.opacity_parameter.min().detach()),gate_max=float(rt.opacity_parameter.max().detach()),**parts)
    records.append(row);print(row,flush=True);return row


def fit(a,cfg):
    import torch
    sys.path.insert(0,cfg['r8_code'])
    from fit_joint_rgb import BalancedViews,snapshot,restore,tree_equal
    from joint_replay_state_cpu import compare_states
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'contract.json',cfg);write(a.output/'root_approval.json',read(a.approval))
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    rt=None;error=None;records=[];history=[];started=time.time();replay=None;stop=None;completed=0
    targets={r['raw']:bank(r['target']['path']) for r in cfg['rows']}
    eligible=bank(cfg['eligible']['path'])
    try:
        rt=OpacityGateAdapter(cfg['asset'],a.output/'native',eligible['global_row_ids'],eligible['group_kind'],enable_gpu=True)
        rt.fit_cfg=cfg;rt.guard_sha256=cfg['eligible']['sha256'];rt.backward_calls=0;rt.actual_updates=0;rt.targets_numpy=targets
        rt.targets={raw:{k:torch.as_tensor(v,device='cuda') for k,v in t.items()} for raw,t in targets.items()}
        sampler=BalancedViews(cfg['train_frames'],cfg['seed'])
        optimizer=torch.optim.Adam(rt.trainable,lr=cfg['learning_rate'],betas=tuple(cfg['betas']),eps=cfg['epsilon'])
        initial=evaluate(rt,cfg,targets,a.output/'step000000',0);history.append(initial)
        assert torch.equal(rt.opacity_parameter,torch.ones_like(rt.opacity_parameter))
        steps=cfg['smoke_steps'] if a.mode=='smoke' else cfg['formal_steps']
        for step in range(1,steps+1):
            row=train_one(rt,optimizer,sampler,step,records);completed=step
            if a.mode=='smoke' and step==4:torch.save(snapshot(rt,optimizer,sampler,step),a.output/'checkpoint_step000004.pt')
            if a.mode=='smoke' and step==5:
                expected=snapshot(rt,optimizer,sampler,step);state=torch.load(a.output/'checkpoint_step000004.pt',map_location='cpu',weights_only=False)
                restore(rt,optimizer,sampler,state)
                strict=compare_states(state,snapshot(rt,optimizer,sampler,4),a.output/'checkpoint_restore',strict=True)
                assert strict['decision']=='PASS'
                replayed=train_one(rt,optimizer,sampler,step,records,replay=True)
                actual=snapshot(rt,optimizer,sampler,step);rule=cfg['replay_validation']
                compared=compare_states(expected,actual,a.output/'cuda_replay',eps_multiplier=rule['eps_multiplier'])
                replay=dict(complete=True,restore_all_state_byteexact=True,replay_same_raw=row['raw']==replayed['raw'],
                    loss_abs=abs(row['loss']-replayed['loss']),decision=compared['decision'],
                    replay_bitexact=tree_equal(expected,actual),all_optimizer_and_RNG_leaves_checked=True,
                    extra_updates=1,extraF=1,extraB=1)
                write(a.output/'restore_replay.json',replay)
                assert replay['decision']=='PASS' and replay['replay_same_raw'] and replay['loss_abs']<=rule['loss_abs_bound']
            if step in cfg['checkpoint_evaluation_steps'] or step==steps:
                torch.save(snapshot(rt,optimizer,sampler,step),a.output/f'checkpoint_step{step:06d}.pt')
                evaluated=evaluate(rt,cfg,targets,a.output/f'step{step:06d}',step);history.append(evaluated)
                if a.mode=='fit' and step==100:
                    gain=1-evaluated['train_fixed_joint_mean_MSE']/max(initial['train_fixed_joint_mean_MSE'],1e-20)
                    if gain<cfg['early_stop']['step100_train_mean_MSE_improvement_below']:stop='fixed_train_target_improvement_below_2_percent_at100'
                if a.mode=='fit' and step==200:
                    previous=next(x for x in history if x['step']==100)
                    gain=1-evaluated['train_fixed_joint_mean_MSE']/max(previous['train_fixed_joint_mean_MSE'],1e-20)
                    if gain<cfg['early_stop']['step200_relative_to100_train_mean_MSE_improvement_below']:stop='fixed_train_target_improvement_below_1_percent_since100'
                if (a.output/'STOP').exists():stop='root_STOP_file_at_checkpoint'
                if stop:break
        final=history[-1];assert final['step']==completed
        record=dict(mode=a.mode,logical_steps=completed,actual_optimizer_updates=rt.actual_updates,
            parent_formal_updates=400,cumulative_formal_updates=400+(completed if a.mode=='fit' else 0),
            eligible_rows_sha256=cfg['eligible']['sha256'],method_contract_sha256=sha(a.contract),
            gate_only=True,new_diffusion_calls=0)
        rt.export_sidecar(a.output/'asset',record)
        write(a.output/'losses.json',records)
        improvement=1-final['train_fixed_joint_mean_MSE']/max(initial['train_fixed_joint_mean_MSE'],1e-20)
        write(a.output/'receipt.json',dict(schema='R10.actual_global_opacity_fit.v1',complete=True,**record,
            early_stop=stop,source_field_sha256=cfg['source']['sha256'],restore_replay=replay,
            numerical_training_improvement=improvement,
            numerical_effect_gate_proposal_met=improvement>=cfg['formal_effect_proposal']['train_fixed_inside_mean_MSE_minimum_improvement'],
            all_five_actual_protection_pass=True,fresh_reload='PENDING',root_actual_joint_visual_review='PENDING',
            whole_scene_quality_accepted=False,final_readout=str(a.output/f'step{completed:06d}')))
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        if rt is not None:
            np.savez_compressed(a.output/'last_gate_diagnostic.npz',global_row_ids=rt.opacity_ids.cpu().numpy(),gate=rt.opacity_parameter.detach().cpu().numpy())
        write(a.output/'resources.json',dict(complete=error is None,error=error,mode=a.mode,
            nativeF=rt.calls if rt else 0,nativeB=rt.backward_calls if rt else 0,actual_optimizer_updates=rt.actual_updates if rt else 0,
            logical_steps_completed=completed,formal_updates=completed if a.mode=='fit' else 0,
            smoke_updates=(rt.actual_updates if rt else 0) if a.mode=='smoke' else 0,
            seconds=time.time()-started,peak_GPU_bytes=int(torch.cuda.max_memory_allocated()) if torch.cuda.is_initialized() else 0,
            new_diffusion_calls=0,source_sha256=cfg['source']['sha256']))


def verify(a):
    assert a.sidecar and a.expected
    a.output.mkdir(parents=True,exist_ok=False);rt=None;rows=[];error=None
    try:
        rt=load_opacity_sidecar(a.sidecar,a.output/'native',enable_gpu=True)
        for raw in rt.frames:
            with rt.torch.no_grad():value=arrays(rt,raw)
            ref=bank(a.expected/f'{raw:03d}/joint_render.npz');assert set(value)==set(ref)
            folder=a.output/f'{raw:03d}';folder.mkdir();np.savez_compressed(folder/'joint_render.npz',**value)
            exact={k:v.dtype==ref[k].dtype and v.shape==ref[k].shape and v.tobytes()==ref[k].tobytes() for k,v in value.items()}
            rows.append(dict(raw=raw,all_float_keys_byteexact=all(exact.values()),keys=exact))
            assert all(exact.values()),('Fresh target-free loader parity failed',raw,exact)
        rt.assert_opacity_only()
        write(a.output/'camera_timestamp_audit.json',dict(rows=rt.camera_audit))
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        write(a.output/'receipt.json',dict(complete=error is None,error=error,decision='PASS' if error is None else 'FAIL',
            target_free_actual_native_reload=True,fit_contract_or_targets_read=False,sidecar_manifest=pin(a.sidecar/'manifest.json'),
            nativeF=rt.calls if rt else 0,nativeB=0,updates=0,rows=rows))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['prepare','smoke','fit','verify'],default='prepare')
    p.add_argument('--contract',type=Path);p.add_argument('--approval',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sidecar',type=Path);p.add_argument('--expected',type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.mode=='verify':
        if a.execute:verify(a)
        else:print(dict(prepared=True,target_free_reload=True,nativeF=0,nativeB=0))
        return
    assert a.contract;cfg=check_inputs(a.contract)
    if not a.execute or a.mode=='prepare':
        print(dict(prepared=True,contract=pin(a.contract),selected_rows=cfg['selected_rows'],nativeF=0,nativeB=0,updates=0));return
    assert a.approval,'Root reviews the exact bound contract before GPU smoke'
    approval=read(a.approval)
    assert approval['complete'] and approval['contract_sha256']==sha(a.contract)
    assert approval['smoke_authorized'] if a.mode=='smoke' else approval['formal_authorized']
    if a.mode=='fit':
        probe=read(approval['smoke_fresh_reload']['path'])
        assert sha(approval['smoke_fresh_reload']['path'])==approval['smoke_fresh_reload']['sha256']
        assert probe['complete'] and probe['decision']=='PASS'
    fit(a,cfg)


if __name__=='__main__':main()
