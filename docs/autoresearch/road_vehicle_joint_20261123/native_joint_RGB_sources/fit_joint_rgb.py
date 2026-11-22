"""Explicitly gated native joint RGB fitting, lifecycle smoke, and fresh verify.

Default action is CPU prepare. Root schedules every --execute CUDA action.
Smoke: six logical updates plus one separately counted checkpoint replay.
Formal: 400 fresh updates, balanced shuffled 086/110; other frames evaluate only.
"""
from pathlib import Path
import argparse,copy,random,shutil,time
import numpy as np
from joint_rgb_common import configuration,source_gate,read,load,sha,write,linear_to_srgb_numpy


def guard_gate(cfg):
    p=Path(cfg['colour_guard']);r=read(p/'receipt.json');z=load(p/'colour_guard.npz')
    assert r['complete'] and r['updates']==0 and r['source_RGBA_exact'] is True
    assert r['source_sha256']==cfg['source']['sha256'] and r['guard_sha256']==sha(p/'colour_guard.npz')
    assert np.array_equal(z['frames'],cfg['frames']) and np.isfinite(z['road_K_mass']).all() and (z['road_K_mass']>=0).all()
    assert np.array_equal(z['road_Kpositive_union'],(z['road_K_mass']>0).any(0))
    return dict(path=str(p),receipt_sha256=sha(p/'receipt.json'),guard_sha256=r['guard_sha256'],all_five_views=True)


class BalancedViews:
    def __init__(self,frames,seed):self.frames=list(frames);self.rng=random.Random(seed);self.block=[]
    def next(self):
        if not self.block:self.block=self.frames.copy();self.rng.shuffle(self.block)
        return self.block.pop()
    def state(self):return dict(block=self.block.copy(),rng=self.rng.getstate())
    def restore(self,state):self.block=state['block'].copy();self.rng.setstate(state['rng'])


def cpu(value):
    import torch
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(cpu(v) for v in value)
    return copy.deepcopy(value)


def snapshot(rt,optimizer,sampler,logical_step):
    torch=rt.torch
    return dict(parameters=cpu(rt.trainable),optimizer=cpu(optimizer.state_dict()),sampler=sampler.state(),logical_step=logical_step,
        python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
        source_sha256=rt.fit_cfg['source']['sha256'],guard_sha256=rt.guard_sha256)


def tree_equal(a,b):
    import torch
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(tree_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
    return a==b


def restore(rt,optimizer,sampler,state):
    torch=rt.torch
    assert state['source_sha256']==rt.fit_cfg['source']['sha256'] and state['guard_sha256']==rt.guard_sha256
    with torch.no_grad():
        for p,value in zip(rt.trainable,state['parameters']):p.copy_(value.to(p.device))
    optimizer.load_state_dict(state['optimizer']);sampler.restore(state['sampler'])
    random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state_all(state['cuda_rng'])
    assert tree_equal(snapshot(rt,optimizer,sampler,state['logical_step']),state),'Checkpoint state restore is not bitexact'


def png(path,value):
    from PIL import Image
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    value=np.asarray(value)
    if value.ndim==3 and value.shape[0] in (1,3,4):value=value.transpose(1,2,0)
    if value.ndim==3 and value.shape[-1]==1:value=value[...,0]
    Image.fromarray(np.rint(np.clip(value,0,1)*255).astype(np.uint8)).save(path)


def render_arrays(rt,raw):
    with rt.torch.no_grad():j=rt.render(raw);r=rt.render(raw,'road');a=rt.render(raw,'actor')
    ar=lambda v:v.detach().cpu().numpy()
    alpha=ar(j['alpha']).squeeze();ra=ar(r['alpha']).squeeze();aa=ar(a['alpha']).squeeze()
    joint=dict(rgb=ar(j['rgb']),alpha=alpha,depth_expected_m=ar(j['depth_sum_m']).squeeze()/np.maximum(alpha,1e-8),
        road_alphaT=ar(j['road_alpha']),actor_alphaT=ar(j['actor_alpha']),context_alphaT=ar(j['context_alpha']))
    if 'unified_actor_alpha' in j:joint['unified_actor_alphaT']=ar(j['unified_actor_alpha'])
    component=dict(road_rgb=ar(r['rgb']),road_alpha=ra,actor_rgb=ar(a['rgb']),actor_alpha=aa,
        road_depth_sum_m=ar(r['depth_sum_m']).squeeze(),actor_depth_sum_m=ar(a['depth_sum_m']).squeeze(),
        road_depth_expected_m=ar(r['depth_sum_m']).squeeze()/np.maximum(ra,1e-8),
        actor_depth_expected_m=ar(a['depth_sum_m']).squeeze()/np.maximum(aa,1e-8))
    return joint,component


def evaluate(rt,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);records=[]
    from joint_rgb_common import srgb_to_linear_numpy
    for raw in rt.frames:
        folder=directory/f'{raw:03d}';folder.mkdir();j,c=render_arrays(rt,raw);t=rt.targets[raw]
        prior=load(Path(rt.fit_cfg['baseline_render'])/f'{raw:03d}/joint_render.npz');K=t['K'].cpu().numpy()
        ke=float(abs(j['rgb'][:,K]-prior['rgb'][:,K]).max()) if K.any() else 0.
        ae=float(abs(j['alpha']-prior['alpha']).max());assert ke==0 and ae==0,'Frozen K RGB or scene alpha changed'
        np.savez_compressed(folder/'joint_render.npz',**j);np.savez_compressed(folder/'component_render.npz',**c)
        for name,value in [('shared',j['rgb']),('road',c['road_rgb']),('actor',c['actor_rgb']),('alpha',j['alpha']),
                           ('road_alpha',c['road_alpha']),('actor_alpha',c['actor_alpha'])]:png(folder/(name+'.png'),value)
        target=t['target_numpy'];support=t['support_numpy'];weight=t['weight'].cpu().numpy()
        png(folder/'source_material.png',linear_to_srgb_numpy(target/t['gain']));png(folder/'source_support.png',support)
        np.savez_compressed(folder/'source_target.npz',canonical_linear=target,support=support,weight=weight,
            source_weights=t['source_weight'],exposure_gain=np.asarray(t['gain']))
        material=support&(c['road_alpha']>.95);jm=support&(j['road_alphaT']>.95)
        straight=c['road_rgb']/np.maximum(c['road_alpha'],1e-6)
        pred=srgb_to_linear_numpy(j['rgb'])*t['gain'];m_pred=srgb_to_linear_numpy(straight)*t['gain']
        V=t['V'].cpu().numpy();proxy=t['proxy'].cpu().numpy();photo=t['photo'].cpu().numpy()
        mae=lambda a,m:float(abs(a)[:,m].mean()) if m.any() else None
        rec=dict(raw=raw,road_material_pixels=int(material.sum()),joint_road_pixels=int(jm.sum()),
            road_material_common_linear_MAE=mae(m_pred-target.transpose(2,0,1),material),
            joint_road_common_linear_MAE=mae(pred-target.transpose(2,0,1),jm),
            joint_actor_trueV_pixels=int(V.sum()),joint_actor_trueV_sRGB_MAE=mae(j['rgb']-photo,V),
            joint_actor_proxy_pixels=int(proxy.sum()),joint_actor_proxy_sRGB_MAE=mae(j['rgb']-photo,proxy),
            K_RGB_max_abs=ke,scene_alpha_max_abs=ae,raw_output_exposure_not_modified=True)
        write(folder/'metrics.json',rec);records.append(rec)
    write(directory/'metrics.json',dict(rows=records,source_sha256=rt.fit_cfg['source']['sha256'],geometry_alpha_and_K_exact=True))
    write(directory/'camera_timestamp_audit.json',dict(rows=rt.camera_audit));return records


def train_one(rt,optimizer,sampler,logical_step,records,replay=False):
    raw=sampler.next();optimizer.zero_grad(set_to_none=True)
    joint=rt.render(raw);road=rt.render(raw,'road');loss,metrics=rt.losses(raw,joint,road)
    loss.backward();rt.backward_calls+=2;optimizer.step()
    with rt.torch.no_grad():rt.road_parameter.clamp_(0,1)
    record=dict(step=logical_step,raw=raw,replay=replay,loss=float(loss.detach()),**metrics);records.append(record)
    print(record,flush=True);return record


def run_fit(args,cfg,base,gate,guard):
    import torch
    from joint_rgb_runtime import JointColourRuntime
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    args.output.mkdir(parents=True,exist_ok=False);write(args.output/'accepted_source_gate.json',gate)
    write(args.output/'effective_baseline_contract.json',base);write(args.output/'fit_contract.json',cfg)
    rt=None;updates=0;records=[];error=None;started=time.time();replay_receipt=None
    try:
        rt=JointColourRuntime(cfg,base,args.output/'native');sampler=BalancedViews(cfg['train_frames'],cfg['seed'])
        groups=[dict(params=[rt.road_parameter],lr=cfg['learning_rate_road']),dict(params=rt.actor_parameters,lr=cfg['learning_rate_actor_DC'])]
        optimizer=torch.optim.Adam(groups)
        write(args.output/'baseline_refs.json',dict(directory=cfg['baseline_render'],zero_update_RGBA_proved_by_colour_guard=True,guard=guard))
        evaluate(rt,args.output/'atlas_assignment_0')
        steps=cfg['smoke_steps'] if args.mode=='smoke' else cfg['formal_steps']
        for step in range(1,steps+1):
            row=train_one(rt,optimizer,sampler,step,records);updates+=1
            if args.mode=='smoke' and step==4:
                torch.save(snapshot(rt,optimizer,sampler,step),args.output/'checkpoint_step000004.pt')
            if args.mode=='smoke' and step==5:
                expected=snapshot(rt,optimizer,sampler,step);state=torch.load(args.output/'checkpoint_step000004.pt',map_location='cpu',weights_only=False)
                restore(rt,optimizer,sampler,state)
                from joint_replay_state_cpu import compare_states
                restored=compare_states(state,snapshot(rt,optimizer,sampler,state['logical_step']),args.output/'checkpoint_restore',strict=True)
                assert restored['decision']=='PASS'
                replayed=train_one(rt,optimizer,sampler,step,records,replay=True);updates+=1
                actual=snapshot(rt,optimizer,sampler,step);bitexact=tree_equal(expected,actual)
                maxdiff=max(float((x-y).abs().max()) if x.numel() else 0. for x,y in zip(expected['parameters'],actual['parameters']))
                rule=cfg['replay_validation'];assert rule['eps_multiplier']==4 and rule['loss_abs_bound']==1e-7
                checked=compare_states(expected,actual,args.output/'cuda_replay',eps_multiplier=rule['eps_multiplier'])
                replay_receipt=dict(restore_complete_state_bitexact=True,replay_parameters_optimizer_RNG_bitexact=bitexact,
                    replay_same_raw=row['raw']==replayed['raw'],replay_loss_max_abs=abs(row['loss']-replayed['loss']),parameter_max_abs=maxdiff,
                    replay_numeric_decision=checked['decision'],all_float_state_checked=True,counters_RNG_exact=True,
                    numerical_contract=rule,CUDA_float_atomic_order_not_claimed_bitexact=True,
                    main_smoke_logical_steps=6,extra_replay_updates=1,extra_replay_F=2,extra_replay_B=2)
                write(args.output/'restore_replay.json',replay_receipt)
                assert checked['decision']=='PASS' and row['raw']==replayed['raw'] and abs(row['loss']-replayed['loss'])<=rule['loss_abs_bound'],'Complete native replay exceeded fixed numerical contract; preserve this run'
            if step%100==0 or step==steps:torch.save(snapshot(rt,optimizer,sampler,step),args.output/f'checkpoint_step{step:06d}.pt')
        evaluate(rt,args.output/'final_readout');rt.assert_colour_only();rt.export_asset(args.output/'asset',steps,updates)
        write(args.output/'losses.json',records)
        write(args.output/'receipt.json',dict(schema='R8.actual_joint_RGB_fit.v1',complete=True,mode=args.mode,logical_steps=steps,
            actual_optimizer_updates=updates,formal_steps=400,formal_run_fresh_initialization=True,source_sha256=cfg['source']['sha256'],
            guard_sha256=rt.guard_sha256,geometry_sigma_opacity_normal_tracks_fixed=True,native_context_and_old_guards_fixed=True,
            actor_only_DC=True,road_only_RGB=True,train_frames=cfg['train_frames'],evaluation_only_frames=cfg['evaluation_only_frames'],
            no_sealed_evaluation_supervision=True,new_diffusion_calls=0,restore_replay=replay_receipt,
            fresh_process_verify_required=True,fresh_process_verify='PENDING',whole_scene_restoration_accepted=False))
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        write(args.output/'resources.json',dict(complete=error is None,error=error,mode=args.mode,
            nativeF=rt.calls if rt else 0,nativeB=rt.backward_calls if rt else 0,actual_optimizer_updates=updates,
            seconds=time.time()-started,new_diffusion_calls=0,source_sha256=cfg['source']['sha256'],
            peak_GPU_bytes=int(torch.cuda.max_memory_allocated()) if torch.cuda.is_initialized() else 0))


def run_verify(args,cfg,gate):
    from joint_identity_adapter import JointIdentityAdapter
    contract=read(args.asset/'joint_contract.json');proof=read(args.asset/'proof.json')
    assert proof['source_sha256']==cfg['source']['sha256'] and proof['joint_contract_sha256']==sha(args.asset/'joint_contract.json')
    args.output.mkdir(parents=True,exist_ok=False);rt=None;rows=[];error=None
    try:
        rt=JointIdentityAdapter(contract,args.output/'native')
        for raw in cfg['frames']:
            j,c=render_arrays(rt,raw);expected=args.expected/f'{raw:03d}'
            reloaded=args.output/'reloaded'/f'{raw:03d}';reloaded.mkdir(parents=True)
            copied=args.output/'expected'/f'{raw:03d}';copied.mkdir(parents=True)
            values={}
            for filename,value in [('joint_render.npz',j),('component_render.npz',c)]:
                np.savez_compressed(reloaded/filename,**value);shutil.copy2(expected/filename,copied/filename);ref=load(expected/filename)
                assert set(ref)==set(value),('Fresh loader omitted/added a float readout',raw,filename,set(ref)^set(value))
                for key,array in value.items():
                    error_abs=float(abs(array-ref[key]).max());values[filename+':'+key]=error_abs
                    assert array.dtype==ref[key].dtype and array.tobytes()==ref[key].tobytes(),(raw,key,error_abs)
            baseline=load(Path(cfg['baseline_render'])/f'{raw:03d}/joint_render.npz')
            K=load(Path(cfg['rectified_masks'])/f'{raw:03d}/rectified_masks.npz')['protected']
            assert j['rgb'][:,K].tobytes()==baseline['rgb'][:,K].tobytes() and j['alpha'].tobytes()==baseline['alpha'].tobytes()
            rows.append(dict(raw=raw,all_float_arrays_byteexact=True,max_abs=values,K_RGB_exact=True,alpha_exact=True))
        rt.assert_fixed();write(args.output/'camera_timestamp_audit.json',dict(rows=rt.camera_audit))
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        write(args.output/'receipt.json',dict(schema='R8.fresh_process_joint_asset_native_reload.v1',complete=error is None,error=error,
            nativeF=rt.calls if rt else 0,nativeB=0,updates=0,source_gate=gate,source_field_sha256=cfg['source']['sha256'],asset_proof_sha256=sha(args.asset/'proof.json'),rows=rows,
            actual_fresh_loader=True,expected_float_buffers_preserved=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['prepare','smoke','fit','verify'],default='prepare');p.add_argument('--execute',action='store_true')
    p.add_argument('--asset',type=Path);p.add_argument('--expected',type=Path);a=p.parse_args()
    cfg,base=configuration(a.contract);gate=source_gate(cfg,strict=a.execute)
    if not a.execute or a.mode=='prepare':
        print(dict(prepared=True,requested_mode=a.mode,source_gate=gate,actual_nativeF=0,actual_nativeB=0,updates=0,
            execution_requires_explicit_flag=True,formal_steps=400,train_frames=cfg['train_frames'],evaluation_only_frames=cfg['evaluation_only_frames']));return
    guard=guard_gate(cfg)
    if a.mode=='verify':
        assert a.asset and a.expected;run_verify(a,cfg,gate)
    else:run_fit(a,cfg,base,gate,guard)


if __name__=='__main__':main()
