"""真实约600s生命周期smoke；工程更新独立计账，禁止作为formal初值。"""
from pathlib import Path
import argparse,copy,json,sys,time
import numpy as np
import torch
from train_world_material import TrainingState,P,cpu
from field_native_io import save_field_asset,load_field_context


def compare(a,b):
    if torch.is_tensor(a):return float((a.double()-b.double()).abs().max()) if a.numel() else 0.
    if isinstance(a,np.ndarray):return 0. if np.array_equal(a,b) else float('inf')
    if isinstance(a,dict):return max([compare(a[k],b[k]) for k in a]+[0.]) if a.keys()==b.keys() else float('inf')
    if isinstance(a,(list,tuple)):return max([compare(x,y) for x,y in zip(a,b)]+[0.]) if len(a)==len(b) else float('inf')
    return 0. if a==b else float('inf')


def evaluate(state,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);rows=[]
    with torch.no_grad():
        colors=state.colors()
        for raw in state.ctx.frames:
            r=state.ctx.render_road(raw,colors);j=state.ctx.render_joint(raw,colors)
            with np.load(Path(state.contract['coverage_parent']).parent/f'{raw:03d}_native.npz') as f:
                alpha=f['road_alpha'];joint=f['joint_rgb']
            difference=np.abs(j['rgb'].cpu().numpy()-joint).max(0);K=state.ctx.K_masks[raw]
            if np.any(difference[K]!=0) or not np.array_equal(r['alpha'][0].cpu().numpy(),alpha):
                raise RuntimeError('smoke field颜色变化破坏固定覆盖或jointK')
            np.savez_compressed(output/f'{raw:03d}.npz',rgb=r['rgb'].cpu().numpy(),alpha=alpha)
            rows.append(dict(raw=raw,K_pixels=int(K.sum()),K_RGB_max=float(difference[K].max()),alpha_exact=True))
    (output/'receipt.json').write_text(json.dumps(dict(rows=rows,K_pixel_frames=sum(r['K_pixels'] for r in rows)),indent=2)+'\n')
    return rows


def run(output,owner,seconds=600,max_updates=360,resume_initial=None):
    if seconds!=600 or max_updates!=360:raise ValueError('smoke时长/工程更新上界由root锁定，不扫描')
    numeric=json.loads((P/'cpu_numeric_r1/receipt.json').read_text())
    if numeric['numeric_gate_pass'] is not True:raise ValueError('数值资格未过，禁止优化smoke')
    json.loads((P/'lifecycle_contract.json').read_text())
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    s=TrainingState(P/'cpu_exact_r1/field_initial.pt',P/'method_scope_r1/scope.npz',P/'stage0_native_r1',P/'execution_contract.json',owner,purpose='smoke')
    (output/'loss_spec.json').write_text(json.dumps(dict(space='native premult sRGB',q='binary1 inside frozen A/C/S masks',
        source_roles=dict(A='observed',C='pseudo hole',S='pseudo targetshadow'),S_photo=3131,S_full=3187,lowS=56,
        weights=dict(A=1,C=.5,S=1),regularizers=s.contract['optimizer']['regularizers']),indent=2)+'\n')
    start=time.monotonic();physical=0;history=[];checks=[];timings=[]
    def update(stage,kind='engineering'):
        nonlocal physical
        torch.cuda.synchronize();t=time.monotonic();row=s.step(stage);torch.cuda.synchronize()
        physical+=1;row.update(physical_update=physical,kind=kind,seconds=time.monotonic()-t)
        timings.append(row['seconds']);history.append(row)
        if physical%10==0:print(json.dumps(dict(physical_updates=physical,elapsed=time.monotonic()-start,stage=stage,last_seconds=row['seconds'])),flush=True)
    def restore_exact(saved,label):
        s.arm=saved['arm'];s.restore(saved)
        discrepancy=compare(saved,s.snapshot())
        if discrepancy!=0:raise RuntimeError('完整恢复非exact: '+label+str(discrepancy))
        checks.append(dict(test=label,full_model_optimizer_RNG_sampler_exact=True))
    if resume_initial is None:
        update('A');update('A')
    else:
        saved=torch.load(resume_initial,map_location='cpu',weights_only=False)
        if saved['cursor']!=2 or saved['purpose']!='smoke':raise ValueError('工程恢复仅允许原smoke磁盘A2')
        differences={key for key in set(saved['binding'])|set(s.binding) if saved['binding'].get(key)!=s.binding.get(key)}
        allowed={str(Path(__file__).with_name('train_world_material.py'))}
        if not differences<=allowed:raise ValueError('restore-only工程迁移遇到非预期输入变化: '+str(differences))
        migration=dict(source=str(resume_initial),changed_binding_keys=sorted(differences),
            change='deepcopy Adam restore prevents CPU step alias; forward/loss/fields/params/optimizer/RNG/trace unchanged')
        saved['binding']=s.binding.copy();s.restore(saved)
        (output/'restore_compatibility_migration.json').write_text(json.dumps(migration,indent=2)+'\n')
    A=copy.deepcopy(s.snapshot());s.save_checkpoint(output/'A2.pt')
    evaluate(s,output/'evaluate_A2')
    restore_exact(A,'A2_restore')
    update('A','duplicate_next_path1');next1=copy.deepcopy(s.snapshot())
    restore_exact(A,'A2_restore_for_replay');update('A','duplicate_next_path2');next2=copy.deepcopy(s.snapshot())
    replay=compare(next1,next2);checks.append(dict(test='next_A_replay',maximum_state_difference=replay,bitexact=replay==0))
    torch.save(dict(path1=next1,path2=next2),output/'next_A_replay_states.pt')
    (output/'next_A_replay.json').write_text(json.dumps(dict(maximum_state_difference=replay),indent=2)+'\n')
    if replay>1e-6:raise RuntimeError('nextstep重放差异过大')
    restore_exact(A,'A2_for_B_entry');low=cpu(s.model.low);adamL=cpu(s.optimizer.state_dict()['state'].get(0,{}))
    update('B');update('B');B=copy.deepcopy(s.snapshot());s.save_checkpoint(output/'B4.pt')
    if compare(low,cpu(s.model.low))!=0 or compare(adamL,cpu(s.optimizer.state_dict()['state'].get(0,{})))!=0:
        raise RuntimeError('B阶段未精确冻结L/AdamL')
    restore_exact(B,'B4_restore');update('C','next_C_path1');C1=copy.deepcopy(s.snapshot())
    restore_exact(B,'B4_restore_for_C_replay');update('C','next_C_path2');C2=copy.deepcopy(s.snapshot())
    replayC=compare(C1,C2);checks.append(dict(test='next_C_replay',maximum_state_difference=replayC,bitexact=replayC==0))
    torch.save(dict(path1=C1,path2=C2),output/'next_C_replay_states.pt')
    if replayC>1e-6:raise RuntimeError('C下一步重放差异过大')
    final_short=copy.deepcopy(s.snapshot());restore_exact(A,'A2_for_L_fork');s.arm='L_control';update('L','control_fork_entry')
    restore_exact(final_short,'C5_after_control_entry');s.save_checkpoint(output/'C5.pt')
    evaluate(s,output/'evaluate_C5')
    # 9次真实工程更新；证明权威field重载而非只重载派生rowRGB。
    colors=save_field_asset(s,output/'field_reload_asset',dict(role='smoke9 engineering updates; never formal initialization'))
    loaded=load_field_context(output/'field_reload_asset',lease_file=P/'gpu_lease.json',lease_owner=owner)
    requery_error=float((loaded.colors-colors).abs().max());rows=[]
    with torch.no_grad():
        for raw in s.ctx.frames:
            expected=s.ctx.render_road(raw,colors);actual=loaded.render_road(raw);cached=loaded.base.render_road(raw)
            error=float((actual['rgb']-expected['rgb']).abs().max());cache_error=float((actual['rgb']-cached['rgb']).abs().max())
            if max(error,cache_error)>1e-6:raise RuntimeError('sharedfield重query→native/derivedcache不一致')
            rows.append(dict(raw=raw,field_native_RGB_max=error,baked_cache_RGB_max=cache_error,
                RGB8_max=int((torch.round(actual['rgb']*255)-torch.round(expected['rgb']*255)).abs().max())))
    checks.append(dict(test='authoritative_field_reload',query=loaded.query_receipt,ci_max=requery_error,rows=rows))
    additional_public_calls=loaded.base.native_calls
    loaded.close();s.assert_frozen()
    # 真实短程继续A/B/C轮换，反复阶段保存/评估，wallclock约600s；参数不进入formal。
    cycle=0
    while time.monotonic()-start<seconds and physical<max_updates:
        stage=('A','B','C')[cycle%3]
        update(stage)
        if cycle%30==29:
            checkpoint=copy.deepcopy(s.snapshot());s.save_checkpoint(output/f'continuation_{physical:03d}.pt')
            restore_exact(checkpoint,'continuation_'+str(physical));evaluate(s,output/f'evaluate_{physical:03d}')
        cycle+=1
    s.save_checkpoint(output/'final_smoke_state.pt');s.assert_frozen()
    (output/'updates.json').write_text(json.dumps(history,indent=2)+'\n')
    result=dict(schema='multiscale_real_lifecycle_smoke.v1',GPU_executed=True,optimizer_updates=physical,
        formal_updates=0,elapsed_seconds=time.monotonic()-start,target_seconds=600,maximum_engineering_updates=360,
        initial_lifecycle_updates=9 if resume_initial is None else 7,
        additional_continuation_updates=physical-(9 if resume_initial is None else 7),checks=checks,
        prior_failed_attempt_updates=0 if resume_initial is None else 4,
        step_seconds_mean=float(np.mean(timings)),step_seconds_max=float(np.max(timings)),
        estimated1400_core_seconds=float(np.mean(timings))*1400,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        native_calls_training_context=s.ctx.native_calls,additional_public_loader_native_calls=additional_public_calls,
        total_native_calls=s.ctx.native_calls+additional_public_calls,field_reload_path_calls=15,
        field_reload_expected_calls_already_in_training_context=5,formal_must_start_fresh_zero_fields=True,
        all_geometry_and_coverage_frozen=True,all_K_checks_exact=True)
    (output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');s.ctx.close();print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--lease-owner',required=True)
    p.add_argument('--resume-initial',type=Path)
    a=p.parse_args();run(a.output,a.lease_owner,resume_initial=a.resume_initial)
