"""matched-I0显式native生命周期/唯一800步控制；默认inspect不运行GPU。"""
from pathlib import Path
import argparse,copy,json,shutil,time
import numpy as np
import torch
from matched_i0_training import P,G,NativeMatchedI0State,MatchedI0,cpu,equal,digest


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def tensor_maxima(a,b,path=''):
    if torch.is_tensor(a):return {path:float((a.double()-b.double()).abs().max()) if a.numel() else 0.}
    if isinstance(a,dict):
        out={}
        if a.keys()!=b.keys():raise ValueError('replay state keys不同:'+path)
        for k in a:out.update(tensor_maxima(a[k],b[k],path+'/'+str(k)))
        return out
    if isinstance(a,(tuple,list)):
        out={}
        if len(a)!=len(b):raise ValueError('replay state list不同:'+path)
        for i,(x,y)in enumerate(zip(a,b)):out.update(tensor_maxima(x,y,path+'/'+str(i)))
        return out
    if isinstance(a,np.ndarray):
        if not np.array_equal(a,b):raise ValueError('replay non-tensorstate不同:'+path)
    elif a!=b:raise ValueError('replay metadata不同:'+path)
    return {}


def evaluate(state,output):
    from PIL import Image
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    colors=state.model().detach();rows=[]
    if not hasattr(state,'joint_baseline'):state.joint_baseline={}
    with torch.no_grad():
        for raw in(86,110,134,179,198):
            road=state.ctx.render_road(raw,colors);joint=state.ctx.render_joint(raw,colors)
            base=state.ctx.baseline(raw)
            if raw not in state.joint_baseline:state.joint_baseline[raw]=state.ctx.render_joint(raw,state.ctx.base_rgb)['rgb'].detach().clone()
            alpha_error=float((road['alpha']-base['alpha']).abs().max())
            K_numpy=np.asarray(state.ctx.K_masks[raw],dtype=bool)
            K=torch.as_tensor(K_numpy,device=colors.device)
            Kmax=float((joint['rgb'][:,K]-state.joint_baseline[raw][:,K]).abs().max())
            if alpha_error!=0 or Kmax!=0:raise RuntimeError('matchedI0 alpha/K硬门不通过')
            metrics={}
            if raw in state.targets:
                t=state.targets[raw]
                for k in('A','C','S'):
                    difference=(road['rgb']-t['fixed_alpha'][None]*t['target_'+k]).abs()[:,t[k]]
                    metrics[k]=float(difference.mean())
            np.savez_compressed(output/f'{raw:03d}_native.npz',road_rgb=road['rgb'].cpu().numpy(),
                road_alpha=road['alpha'][0].cpu().numpy(),joint_rgb=joint['rgb'].cpu().numpy(),K=K_numpy)
            Image.fromarray(np.round(road['rgb'].cpu().numpy().transpose(1,2,0).clip(0,1)*255).astype(np.uint8)).save(output/f'{raw:03d}_road.png')
            rows.append(dict(raw=raw,role='train'if raw in(86,110)else'diagnostic_not_blind',alpha_max=alpha_error,K_pixels=int(K.sum()),joint_K_max=Kmax,metrics=metrics))
    value=dict(GPU_executed=True,rows=rows,all_alpha_exact=True,all_joint_K_exact=True,
        fixed_denominators=dict(S3131=3131,fullS3187=3187,old2222=2222,new909=909,remaining56=56,K77266=77266),
        independent_delta_parameters=int(state.model.delta.numel()),initial_CPU_cache_error=state.initial_CPU_cache_error)
    write(output/'evaluation.json',value);return value


def export(state,output,record):
    output=Path(output);colors=state.model().detach();world=np.zeros(len(state.ctx.road['active']),bool)
    world[state.ctx.active_world_rows]=state.model.eligible.cpu().numpy()
    state.ctx.save_asset(colors,world,output,training_record=dict(record,derived_ci=True,
        initialization='same serializedzero filteredA0 query',effective_learned_RGB_not_observed_prior=True))
    torch.save(state.snapshot(),output/'matched_I0_state.pt')
    # 独立从已保存model状态重建ci，不把bakedroad当初始颜色来源。
    saved=torch.load(output/'matched_I0_state.pt',map_location='cpu',weights_only=False)
    model=MatchedI0(saved['model']['c0'],saved['model']['eligible']);model.load_state_dict(saved['model'])
    error=float((model().cpu()-colors.cpu()).abs().max())
    if error!=0:raise RuntimeError('serialized I0 model重建ci不精确')
    receipt=state.ctx.validate_public_reload(colors,output,output.parent/'public_reload.json')
    write(output/'control_model_reload.json',dict(saved_delta_c0_requery_exact=True,ci_max=error,
        public_loader=receipt,zero_extra_optimizer_updates=True))
    state.ctx.assert_frozen();return receipt


def smoke(state,output):
    output.mkdir(parents=True,exist_ok=False);begin=time.monotonic()
    records=[state.step(),state.step()]
    if records[0]['gradient_nonzero']<=0:raise RuntimeError('初始nativephoto到独立delta梯度为0')
    source=state.snapshot();torch.save(source,output/'checkpoint_step2.pt');immutable=copy.deepcopy(source)
    state.restore(source);exact=equal(source,state.snapshot())
    if not exact:raise RuntimeError('I0完整state恢复不精确')
    records.append(state.step());a=state.snapshot()
    if not equal(source,immutable):raise RuntimeError('restore污染了A2原snapshot(Adam step alias)')
    state.restore(source);records.append(state.step());b=state.snapshot()
    if not equal(source,immutable):raise RuntimeError('第二重放污染了A2 snapshot')
    RNG_exact=equal(a['RNG'],b['RNG'])
    if not RNG_exact:raise RuntimeError('nextstep replay Torch/CUDA/NumPy/Python RNG不同')
    maxima=tensor_maxima(a,b)
    parameter_error=max(v for k,v in maxima.items()if k.startswith('/model/'))
    if parameter_error>1e-6:raise RuntimeError('I0 nextstep参数差>1e-6')
    state.restore(a);torch.save(state.snapshot(),output/'checkpoint_selected_step3.pt')
    evaluation=evaluate(state,output/'rendered')
    reload=export(state,output/'asset',dict(role='matchedI0_smoke',physical_engineering_updates=4,logical_updates=3,formal_updates=0))
    result=dict(schema='matchedI0_native_lifecycle.v1',passed=True,GPU_executed=True,
        engineering_physical_updates=4,selected_logical_updates=3,formal_updates=0,
        source_binding=state.binding,
        restore_exact=exact,snapshot_immutable_after_both_nextsteps=True,nextstate_bitwise=equal(a,b),
        replay_maxima=maxima,replay_model_max=parameter_error,full_RNG_sampler_metadata_exact=RNG_exact,
        evaluation=evaluation,public_reload=reload,initial_CPU_cache_error=state.initial_CPU_cache_error,
        native_forward_calls_in_context=state.ctx.native_calls,additional_public_clone_forwards=5,
        native_backward_calls=state.backward_calls,elapsed_seconds=time.monotonic()-begin,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        records=records,fresh_formal_entry='newconstructor +purposeformal +zero delta, never smoke learned state',
        production_promoted=False,human_verdict=None)
    write(output/'completed.json',result);return result


def formal(state,output,smoke_path,resumed_from=None):
    prerequisite=json.loads(Path(smoke_path).read_text())
    if not prerequisite.get('passed')or not prerequisite['restore_exact']or not prerequisite['snapshot_immutable_after_both_nextsteps']:
        raise ValueError('未通过真实I0生命周期')
    if prerequisite.get('source_binding')!=state.binding:raise ValueError('正式输入/代码不同于已验真实smoke')
    output.mkdir(parents=True,exist_ok=False);begin=time.monotonic()
    if resumed_from is None and(state.cursor!=0 or float(state.model.delta.detach().abs().max())!=0):raise RuntimeError('formal首次必须freshci0/δ0')
    initial_cursor=state.cursor
    before=evaluate(state,output/'rendered_before');torch.save(state.snapshot(),output/f'checkpoint_entry_step{initial_cursor:03d}.pt')
    for _ in range(800-initial_cursor):
        row=state.step()
        if state.cursor==1 or state.cursor%50==0:print(json.dumps(row),flush=True)
        if state.cursor%100==0:
            torch.save(state.snapshot(),output/f'checkpoint_step{state.cursor:03d}.pt')
            write(output/'progress.json',dict(updates=state.cursor,elapsed_seconds=time.monotonic()-begin,last=row))
    if state.cursor!=800 or len(state.trace)!=800:raise RuntimeError('正式预算不是800')
    after=evaluate(state,output/'rendered');reload=export(state,output/'asset',dict(role='matchedI0_formal',optimizer_updates=800,purpose='formal'))
    write(output/'sampler_trace.json',state.trace)
    delta=state.model.delta.detach();effective=state.model.free_colors().detach()-state.model.c0[state.model.ids]
    result=dict(schema='matchedI0_formal_result.v1',completed=True,GPU_executed=True,optimizer_updates=800,
        updates_this_process=800-initial_cursor,resumed_from=str(resumed_from)if resumed_from else None,
        source_initialization='same filteredA0 trueworld field zero state',scientific_rows=84266,
        independent_scalar_parameters=int(delta.numel()),same_sampler_seed101=True,
        raw_delta_bound=[-1,1],effective_row_delta_squared=float(effective.square().mean()),
        raw_delta_abs_max=float(delta.abs().max()),raw_bound_hits=int((delta.abs()>=1).sum()),
        regularization='effectiveRGB delta squared*.001; differs from sharedfield own L/H regularizers',
        native_forward_calls_in_context=state.ctx.native_calls,additional_public_clone_forwards=5,
        native_backward_calls=state.backward_calls,elapsed_seconds=time.monotonic()-begin,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        evaluation_before=before,evaluation_after=after,public_reload=reload,
        array_template=str(output/'rendered/{raw:03d}_native.npz'),
        target_roles='Areal,C/Spseudo, sameS3131/q1; not blind due199BUILD',
        automatic_extension=False,production_promoted=False,human_verdict=None)
    write(output/'completed.json',result);write(P/'matched_I0/results.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['inspect','smoke','formal'],default='inspect')
    parser.add_argument('--execute',action='store_true');parser.add_argument('--output',type=Path)
    parser.add_argument('--contract',type=Path,default=P/'matched_I0/execution_contract.json')
    parser.add_argument('--lease-file',type=Path,default=P/'gpu_lease.json');parser.add_argument('--lease-owner',default='stage_qa')
    parser.add_argument('--smoke-receipt',type=Path,default=P/'matched_I0/smoke_r1/completed.json')
    parser.add_argument('--resume',type=Path);args=parser.parse_args()
    if args.mode=='inspect':
        print(json.dumps(dict(native_default=False,formal_budget=800,engineering_smoke_updates=4,CPU_only_prepared=True)));return
    if not args.execute:raise SystemExit('必须--execute和root匹配单GPU lease；默认不运行')
    if not args.contract.exists():raise SystemExit('缺root登记matchedI0合同，禁止native')
    purpose='formal'if args.mode=='formal'else'smoke'
    state=NativeMatchedI0State(args.contract,lease_file=args.lease_file,lease_owner=args.lease_owner,purpose=purpose)
    state.binding[str(Path(__file__))]=digest(__file__)
    if args.resume:
        if purpose!='formal':raise ValueError('显式resume仅用于已注册formal断点，smoke另目录保留')
        state.restore(torch.load(args.resume,map_location='cpu',weights_only=False))
    output=args.output or P/'matched_I0'/('formal_r1'if purpose=='formal'else'smoke_r1')
    sources=output.with_name(output.name+'_sources');sources.mkdir(parents=True,exist_ok=False)
    for name in['matched_i0_training.py','run_matched_i0.py']:
        shutil.copyfile(Path(__file__).with_name(name),sources/name)
    write(sources/'binding.json',state.binding)
    try:
        result=smoke(state,output)if purpose=='smoke'else formal(state,output,args.smoke_receipt,args.resume)
        print(json.dumps(dict(output=str(output),done=True,updates=result.get('optimizer_updates',0))),flush=True)
    finally:state.ctx.close()


if __name__=='__main__':main()
