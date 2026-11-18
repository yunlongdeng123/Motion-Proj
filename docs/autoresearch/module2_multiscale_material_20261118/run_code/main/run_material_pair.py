"""fresh A0：共用A200后fork L800与ABC800，唯一1400物理正式更新。"""
from pathlib import Path
import argparse,copy,json,sys,time
import numpy as np
import torch
from train_world_material import TrainingState,P,G,digest
from field_native_io import save_field_asset,load_field_context


def evaluate(state,output,colors=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(G/'code'))
    from road_capacity_native import load_npz,write_json
    frequency=load_npz(G/'check3/frozen_frequency_masks.npz');rows=[]
    with torch.no_grad():
        if colors is None:colors=state.colors()
        for raw in state.ctx.frames:
            r,j=state.ctx.render_road(raw,colors),state.ctx.render_joint(raw,colors)
            rgb,alpha,joint=r['rgb'].cpu().numpy(),r['alpha'][0].cpu().numpy(),j['rgb'].cpu().numpy()
            parent=load_npz(Path(state.contract['coverage_parent']).parent/f'{raw:03d}_native.npz')
            K=state.ctx.K_masks[raw];diff=np.abs(joint-parent['joint_rgb']).max(0)
            if np.any(diff[K]!=0) or not np.array_equal(alpha,parent['road_alpha']):raise RuntimeError('正式field破坏K/覆盖冻结')
            mv=frequency[f'{raw:03d}_vertical'];mh=frequency[f'{raw:03d}_horizontal']
            row=dict(raw=raw,role='train' if raw in state.targets else 'diagnostic',K_pixels=int(K.sum()),
                K_RGB_max=float(diff[K].max()),alpha_exact_parent=True,
                vertical_E=float(np.abs(rgb[:,2:]-2*rgb[:,1:-1]+rgb[:,:-2])[:,mv].mean()),
                horizontal_E=float(np.abs(rgb[:,:,2:]-2*rgb[:,:,1:-1]+rgb[:,:,:-2])[:,mh].mean()))
            if raw in state.targets:
                t=state.targets[raw];mask={k:t[k].cpu().numpy() for k in ['A','C','S','fullS']}
                C=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_capacity_cpu_20261117_r1')
                previous=load_npz(C/f'supervision_v4_candidate_r1/{raw:03d}.npz')
                mask['S_old']=previous['S_target_shadow_photo'];mask['S_new']=mask['S']&~mask['S_old']
                mask['S_thin56']=mask['fullS']&~mask['S']
                errors={}
                for name,m in mask.items():
                    source='S' if name in ['S','fullS','S_old','S_new','S_thin56'] else name
                    target=t['target_'+source].cpu().numpy();error=np.abs(rgb-alpha[None]*target)
                    errors[name]=dict(pixels=int(m.sum()),premult_sRGB_MAE=float(error[:,m].mean()) if m.any() else None,
                        alpha_mean=float(alpha[m].mean()) if m.any() else None)
                row['teacher_metrics']=errors
            rows.append(row)
            np.savez_compressed(output/f'{raw:03d}_native.npz',road_rgb=rgb,road_alpha=alpha,joint_rgb=joint,K=K)
    write_json(output/'evaluation.json',dict(rows=rows,shared_coverage=True,full_S=3187,supported_S=3131,
        old_supported_S=2222,new_supported_S=909,low_S=56,teacher_roles='A measured; C/S frozen pseudo',
        shader_color_space='native premult sRGB'))
    return rows


def reload_check(state,asset,output):
    loaded=load_field_context(asset,lease_file=P/'gpu_lease.json',lease_owner=state.ctx.lease_owner)
    rows=[];colors=state.colors().detach()
    with torch.no_grad():
        for raw in state.ctx.frames:
            expected=state.ctx.render_road(raw,colors);actual=loaded.render_road(raw);cache=loaded.base.render_road(raw)
            error=float((expected['rgb']-actual['rgb']).abs().max());cache_error=float((actual['rgb']-cache['rgb']).abs().max())
            if max(error,cache_error)>1e-6:raise RuntimeError('权威field重载native不一致')
            rows.append(dict(raw=raw,field_native_RGB_max=error,baked_cache_RGB_max=cache_error,
                RGB8_max=int((torch.round(expected['rgb']*255)-torch.round(actual['rgb']*255)).abs().max())))
    receipt=dict(query=loaded.query_receipt,rows=rows,additional_loaded_context_native_calls=loaded.base.native_calls,
        expected_calls_already_in_training_context=len(rows))
    Path(output).write_text(json.dumps(receipt,indent=2)+'\n');loaded.close();return receipt


def run(output,owner,resume=None):
    launch=json.loads((P/'formal_launch_contract.json').read_text())
    if launch.get('formal_authorized') is not True or launch.get('physical_updates')!=1400:
        raise ValueError('正式唯一1400更新尚未获root登记')
    smoke_path=Path(launch['smoke_receipt']);smoke=json.loads(smoke_path.read_text())
    minimum=float(launch.get('minimum_smoke_seconds',600))
    if (smoke.get('completed') is not True or smoke['elapsed_seconds']<minimum or
            not smoke['all_K_checks_exact'] or not smoke['checks']):raise ValueError('真实成功生命周期/时长/检查未通过')
    names={row['test'] for row in smoke['checks']}
    required={'next_A_replay','next_C_replay','authoritative_field_reload','final_all34197300_worldquad_rebuild_and_native'}
    if not required<=names:raise ValueError('成功smoke缺完整恢复/field生命周期检查')
    for row in smoke['checks']:
        if 'full_model_optimizer_RNG_sampler_exact' in row and row['full_model_optimizer_RNG_sampler_exact'] is not True:
            raise ValueError('smoke完整恢复未过')
        if row.get('maximum_state_difference',0)>1e-6:raise ValueError('smoke nextstep重放未过')
        if row['test']=='authoritative_field_reload':
            if row['ci_max']>1e-6 or any(max(v['field_native_RGB_max'],v['baked_cache_RGB_max'])>1e-6 for v in row['rows']):
                raise ValueError('smoke权威field/native重载未过')
    output=Path(output);output.mkdir(parents=True,exist_ok=resume is not None)
    state=TrainingState(P/'cpu_exact_r1/field_initial.pt',P/'method_scope_r1/scope.npz',P/'stage0_native_r1',P/'execution_contract.json',owner,purpose='formal')
    if torch.count_nonzero(state.model.low) or torch.count_nonzero(state.model.high):raise ValueError('formal必须fresh zero delta，不能smoke暖启动')
    state.binding[str(Path(__file__))]=digest(__file__)
    start=time.monotonic();physical=0;logs=[];reloads={}
    if resume is None:
        (output/'frozen_contract.json').write_text(json.dumps(dict(binding=state.binding,method=state.contract,smoke_receipt=str(smoke_path),
            physical_updates=1400,each_method800=True,common_prefix200=True,mask_statistics=state.mask_statistics),indent=2)+'\n')
        evaluate(state,output/'center_A0',state.colors(center_only=True));evaluate(state,output/'filtered_A0',state.colors())
    else:
        frozen=json.loads((output/'frozen_contract.json').read_text())
        if frozen['binding']!=state.binding:raise ValueError('formal恢复源/算法不一致')
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        if saved.get('purpose')!='formal':raise ValueError('不能从smoke恢复formal')
        state.arm=saved['arm'];state.restore(saved)
        physical=state.cursor+(600 if state.arm=='ABC' else 0)
    def update(stage):
        nonlocal physical
        row=state.step(stage);physical+=1;row['physical_update']=physical;logs.append(row)
        if physical%10==0:print(json.dumps(dict(physical_update=physical,arm=state.arm,cursor=state.cursor,stage=stage,loss=row['loss'],seconds=time.monotonic()-start)),flush=True)
        with (output/'updates.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        if physical%100==0:state.save_checkpoint(output/f'physical_{physical:04d}.pt')
    def save_stage(name,reload=False):
        if (output/name/'stage_complete.json').exists():return
        path=output/name
        if path.exists():path.rename(path.with_name(name+'.partial_preserved_'+str(time.time_ns())))
        path.mkdir();state.save_checkpoint(path/'checkpoint.pt')
        colors=save_field_asset(state,path/'asset',dict(purpose='formal',stage=name,method_cursor=state.cursor,physical_updates=physical))
        evaluate(state,path/'rendered',colors)
        high=state.model.effective_high_statistics(state.model.high_output_mask)
        stats={k:v.detach().cpu().tolist() for k,v in high.items()}
        stats.update(low_tanh_saturation=int((torch.tanh(state.model.low).abs()>.99).sum()),
            high_tanh_saturation=int((torch.tanh(state.model.high).abs()>.99).sum()))
        (path/'field_statistics.json').write_text(json.dumps(stats,indent=2)+'\n')
        if reload:reloads[name]=reload_check(state,path/'asset',path/'field_public_reload.json')
        state.assert_frozen()
        (path/'stage_complete.json').write_text(json.dumps(dict(completed=True,cursor=state.cursor,physical_update=physical))+'\n')
    if state.arm=='common':
        while state.cursor<200:update('A')
        save_stage('A200')
    A=torch.load(output/'A200/checkpoint.pt',map_location='cpu',weights_only=False)
    low_A=A['params']['low'].clone();adam_A=copy.deepcopy(A['optimizer']['state'].get(0,{}))
    if state.arm in ['common','L_control']:
        state.arm='L_control'
        while state.cursor<800:update('L')
        save_stage('L800',reload=True)
        state.restore(A,allow_common_fork=True);state.arm='ABC'
    L_trace=torch.load(output/'L800/checkpoint.pt',map_location='cpu',weights_only=False)['trace']
    while state.cursor<500:update('B')
    from run_material_lifecycle import compare
    if state.cursor==500:
        if not torch.equal(state.model.low.cpu(),low_A):raise RuntimeError('B300未保持L参数exact')
        if compare(adam_A,state.snapshot()['optimizer']['state'].get(0,{}))!=0:raise RuntimeError('B300未保持AdamL exact')
        save_stage('B500')
    B=torch.load(output/'B500/checkpoint.pt',map_location='cpu',weights_only=False)
    while state.cursor<800:update('C')
    save_stage('ABC800',reload=True)
    if physical!=1400 or state.trace!=L_trace:raise RuntimeError('共享前缀/未来samplertrace或正式预算不符')
    drift=dict(C_low_parameter_max=float((state.model.low.detach().cpu()-B['params']['low']).abs().max()),
        C_high_parameter_max=float((state.model.high.detach().cpu()-B['params']['high']).abs().max()),B_low_and_AdamL_frozen_exact=True)
    all_events=[json.loads(line) for line in (output/'updates.jsonl').read_text().splitlines() if line.strip()]
    result=dict(schema='multiscale_material_pair_formal.v1',GPU_executed=True,formal_physical_updates=physical,
        actual_optimizer_updates_all_attempts=len(all_events),engineering_replay_updates=max(0,len(all_events)-physical),
        A_common=200,L_only_total=800,ABC_total=800,seed=101,trace_each=state.trace,
        steps_per_frame={str(raw):state.trace.count(raw) for raw in [86,110]},future_trace_exact_shared=True,
        native_calls_training_context=state.ctx.native_calls,additional_public_reload_calls=sum(v['additional_loaded_context_native_calls'] for v in reloads.values()),
        reloads=reloads,drift=drift,seconds=time.monotonic()-start,geometry_alpha_scale_active_frozen=True,
        production_promotion=False,human_verdict=None)
    (output/'updates.json').write_text(json.dumps(logs,indent=2)+'\n');(output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    state.ctx.close();print(json.dumps(dict(completed=True,formal_physical_updates=physical,seconds=result['seconds'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--lease-owner',required=True)
    p.add_argument('--resume',type=Path)
    a=p.parse_args();run(a.output,a.lease_owner,a.resume)
