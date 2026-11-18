"""新V4 A/C/S真实教师：固定覆盖，I0/W4同条件原生颜色训练。"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
import numpy as np
from road_capacity_native import (load_context,load_npz,write_json,assert_gpu_lease,
    GPU_CAMPAIGN,CPU_CAMPAIGN,REPO,CANONICAL_ASSET,FRAMES)

def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):value.update(block)
    return value.hexdigest()

def differences(a,b,path=''):
    import torch
    if torch.is_tensor(a):return [] if torch.is_tensor(b) and torch.equal(a.cpu(),b.cpu()) else [path]
    if isinstance(a,np.ndarray):return [] if isinstance(b,np.ndarray) and np.array_equal(a,b) else [path]
    if isinstance(a,dict):
        if not isinstance(b,dict) or a.keys()!=b.keys():return [path]
        return [item for key in a for item in differences(a[key],b[key],path+'/'+str(key))]
    if isinstance(a,(list,tuple)):
        if type(a)!=type(b) or len(a)!=len(b):return [path]
        return [item for i,(x,y) in enumerate(zip(a,b)) for item in differences(x,y,path+'/'+str(i))]
    return [] if a==b else [path]

def color_core():
    path=REPO/'scripts/worldsim_v82/drivingeditor_dynamic_edit/road_world_texture_capacity.py'
    spec=importlib.util.spec_from_file_location('capacity_color_core',path);module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module;spec.loader.exec_module(module);return module

def prepare_targets(directory,device):
    import torch
    values={}
    for raw in (86,110):
        data=load_npz(Path(directory)/f'{raw:03d}.npz')
        keys=('target_A_srgb','target_C_srgb','target_S_srgb','A_mask','C_hole_photo','S_target_shadow_photo',
            'q_A','q_C_hole','q_S','outside','K','coverage_A','coverage_C','coverage_S','fixed_initial_alpha')
        values[raw]={key:torch.as_tensor(data[key].copy(),device=device) for key in keys}
        if bool(((values[raw]['A_mask']|values[raw]['C_hole_photo']|values[raw]['S_target_shadow_photo'])&values[raw]['K']).any()):
            raise ValueError('target photo与K重叠')
        expected=~(values[raw]['coverage_A']|values[raw]['coverage_C']|values[raw]['coverage_S']|values[raw]['K'])
        if not torch.equal(expected,values[raw]['outside']):raise ValueError('outside不是完整授权coverage补集')
    return values

def rgb_l1(pred,target,mask,q=None):
    weight=mask.to(pred.dtype)
    if q is not None:weight=weight*q
    return ((pred-target).abs()*weight[None]).sum()/(3*weight.sum().clamp_min(1))

def edge_l1(pred,target,mask,q):
    weight=mask*q;mx=mask[:,1:]&mask[:,:-1];my=mask[1:]&mask[:-1]
    wx=weight[:,1:].minimum(weight[:,:-1])*mx;wy=weight[1:].minimum(weight[:-1])*my
    dx=(pred[:,:,1:]-pred[:,:,:-1])-(target[:,:,1:]-target[:,:,:-1])
    dy=(pred[:,1:]-pred[:,:-1])-(target[:,1:]-target[:,:-1])
    return (dx.abs().mul(wx[None]).sum()+dy.abs().mul(wy[None]).sum())/(3*(wx.sum()+wy.sum()).clamp_min(1))

def objective(ctx,model,targets,raw,weights):
    out=ctx.render_road(raw,model());target=targets[raw];alpha=ctx.baseline(raw)['alpha'].detach()
    a=rgb_l1(out['rgb'],alpha*target['target_A_srgb'],target['A_mask'],target['q_A'])
    c=rgb_l1(out['rgb'],alpha*target['target_C_srgb'],target['C_hole_photo'],target['q_C_hole'])
    s=rgb_l1(out['rgb'],alpha*target['target_S_srgb'],target['S_target_shadow_photo'],target['q_S'])
    ea=edge_l1(out['rgb'],alpha*target['target_A_srgb'],target['A_mask'],target['q_A'])
    ec=edge_l1(out['rgb'],alpha*target['target_C_srgb'],target['C_hole_photo'],target['q_C_hole'])
    outside=rgb_l1(out['rgb'],ctx.baseline(raw)['rgb'],target['outside'])
    retained=rgb_l1(out['rgb'],ctx.baseline(raw)['rgb'],target['K'])
    delta=(model()-model.base_srgb)[model.eligible]
    prior=delta.square().mean()
    total=(weights['A_photo']*a+weights['C_photo']*c+weights['S_photo']*s+weights['edge_A']*ea+weights['edge_C']*ec
        +weights['outside_RGB']*outside+weights['retained_RGB']*retained+weights['effective_row_delta_squared']*prior)
    return total,dict(A=a,C=c,S=s,edge_A=ea,edge_C=ec,outside=outside,retained=retained,prior=prior),out

def metric_values(values):return {key:float(value.detach()) for key,value in values.items()}

def evaluate(ctx,model,targets,weights,output=None):
    import torch
    frequency=load_npz(GPU_CAMPAIGN/'check3/frozen_frequency_masks.npz')
    rows=[]
    if output:Path(output).mkdir(parents=True,exist_ok=False)
    with torch.no_grad():
        colors=model()
        for raw in FRAMES:
            if raw in targets:value,metrics,out=objective(ctx,model,targets,raw,weights)
            else:out=ctx.render_road(raw,colors);metrics={}
            rgb=out['rgb'];alpha=out['alpha'][0]
            mv=torch.as_tensor(frequency[f'{raw:03d}_vertical'],device=rgb.device)
            mh=torch.as_tensor(frequency[f'{raw:03d}_horizontal'],device=rgb.device)
            md=torch.as_tensor(frequency[f'{raw:03d}_domain'],device=rgb.device)
            vertical=(rgb[:,2:]-2*rgb[:,1:-1]+rgb[:,:-2]).abs()[:,mv].double().mean()
            horizontal=(rgb[:,:,2:]-2*rgb[:,:,1:-1]+rgb[:,:,:-2]).abs()[:,mh].double().mean()
            delta=(rgb-ctx.baseline(raw)['rgb']).square()[:,md].double().mean()
            full=load_npz(CPU_CAMPAIGN/f'ideal_targets_r2/constant/{raw:03d}.npz',('mask',))['mask']
            full=torch.as_tensor(full,device=rgb.device,dtype=torch.bool)
            row=dict(raw=raw,role='train' if raw in targets else 'diagnostic',fixed_road_pixels=int(full.sum()),
                support_gt_point9=int((full&(alpha>.9)).sum()),gap_lt_point05=int((full&(alpha<.05)).sum()),
                thin_point05_to_point9=int((full&(alpha>=.05)&(alpha<=.9)).sum()),
                vertical_E=float(vertical),horizontal_E=float(horizontal),RGB_delta_energy=float(delta),
                frequency_domain_pixels=int(md.sum()),metrics=metric_values(metrics))
            rows.append(row)
            if output:np.savez_compressed(Path(output)/f'{raw:03d}.npz',rgb_srgb=rgb.cpu().numpy(),alpha=alpha.cpu().numpy())
    return dict(rows=rows,full_road_pixels=sum(row['fixed_road_pixels'] for row in rows),
        frequency_domain_pixels=sum(row['frequency_domain_pixels'] for row in rows),
        support_gt_point9=sum(row['support_gt_point9'] for row in rows),scope='fixed coverage, 2 train/3 diagnose, native premult sRGB')

def verify_bindings(args,protocol):
    execution=json.loads((GPU_CAMPAIGN/'execution_contract.json').read_text())
    if execution.get('gpu_authorized') is not True or digest(CPU_CAMPAIGN/'protocol_cpu_prepared.json')!=execution['parent_protocol_sha256']:
        raise ValueError('GPU授权或父协议内容不匹配')
    frozen=json.loads((args.targets/'freeze.json').read_text())
    for item in frozen['source_files']+frozen['code_files']+frozen['target_files']:
        if digest(item['path'])!=item['sha256']:raise ValueError('固定监督/source变化:'+item['path'])
    files=[Path(__file__),Path(__file__).with_name('road_capacity_native.py'),CPU_CAMPAIGN/'protocol_cpu_prepared.json',GPU_CAMPAIGN/'execution_contract.json',
        args.scope,args.scope.with_suffix('.json'),args.targets/'freeze.json',args.targets/'manifest.json',
        *[args.targets/f'{raw:03d}.npz' for raw in (86,110)],GPU_CAMPAIGN/'check3/frozen_frequency_masks.npz',
        CANONICAL_ASSET/'road.npz',CANONICAL_ASSET/'provenance.npz',CANONICAL_ASSET/'persistent/scene.json',CANONICAL_ASSET/'persistent/proof.json',
        REPO/'scripts/worldsim_v82/drivingeditor_dynamic_edit/road_world_texture_basis.py',
        REPO/'scripts/worldsim_v82/drivingeditor_dynamic_edit/road_world_texture_capacity.py']
    return dict(schema='native_real_color_pair_contract.v1',arm=args.arm,formal=args.mode=='formal',steps=args.steps,
        seed=101,optimizer=protocol['check3']['optimizer'],loss=protocol['check3']['objective'],delta_node_bounds=protocol['check3']['delta_node_bounds'],
        TV=0,qualification_version='V4_A_C_S',source_role='qualified_common_actual_alphaT',
        input_binding={str(path.resolve()):digest(path) for path in files},scope_source=str(args.scope),
        no_coverage_or_geometry_updates=True)

def train(args):
    assert_gpu_lease(args.lease_file,args.lease_owner)
    import torch
    if args.mode=='formal' and args.steps!=200:raise ValueError('正式每arm只能200步')
    if args.steps<1:raise ValueError('steps必须为正')
    stop=args.steps if args.stop_step is None else args.stop_step
    if not 0<=stop<=args.steps:raise ValueError('stop-step越预算')
    protocol=json.loads((CPU_CAMPAIGN/'protocol_cpu_prepared.json').read_text());contract=verify_bindings(args,protocol)
    out=args.output;out.mkdir(parents=True,exist_ok=False);write_json(out/'frozen_contract.json',contract)
    torch.set_num_threads(2);torch.manual_seed(101);torch.cuda.manual_seed_all(101);np.random.seed(101);random.seed(101)
    ctx=load_context(lease_file=args.lease_file,lease_owner=args.lease_owner)
    scope=load_npz(args.scope);description=json.loads(args.scope.with_suffix('.json').read_text())
    if str(scope['scope_role'])!='qualified_common_actual_alphaT_v4_ACS' or not description['scope_ready_for_common_pair']:
        raise ValueError('不是新V4 native共同资格')
    if not np.array_equal(scope['active_world_rows'],ctx.world_row_ids) or not np.array_equal(scope['base_srgb'],ctx.base_rgb.cpu().numpy()):raise ValueError('资格与canonical native行/c0不一致')
    eligible=scope['eligible_color']
    expected=scope['physical_legal']&~scope['conservative_K_guard']&~scope['near_guard']&(scope['A_seen']|scope['S_seen']|(scope['C_seen']&~scope['known']))
    if not np.array_equal(eligible,expected):raise ValueError('新A/C/S资格公式不一致')
    core=color_core();plan=core.make_color_basis_plan(ctx.world_xyz,ctx.base_rgb.cpu().numpy(),eligible,scope['conservative_K_guard']|scope['near_guard'],
        spacing_m=.04,world_row_ids=ctx.world_row_ids,scope_source=str(args.scope),source_role='qualified_common_actual_alphaT')
    if not np.array_equal(plan.eligible,eligible):raise ValueError('共享basis额外改变共同scope')
    model=core.RoadColorParameterization(plan,'independent_ci' if args.arm=='I0' else 'shared_world').to(ctx.base_rgb.device)
    if not torch.equal(model(),ctx.base_rgb):raise RuntimeError('共同零修正初始化不exact')
    plan.save(out/'basis_plan.npz');targets=prepare_targets(args.targets,ctx.base_rgb.device);weights=protocol['check3']['objective']
    binding=[]
    for raw in FRAMES:
        base=ctx.baseline(raw);cached=load_npz(CPU_CAMPAIGN/f'ideal_targets_r2/geometry/{raw:03d}.npz',('baseline_forward_rgb_srgb__not_teacher','fixed_alpha'))
        cached_rgb=torch.as_tensor(cached['baseline_forward_rgb_srgb__not_teacher'],device=ctx.base_rgb.device)
        cached_alpha=torch.as_tensor(cached['fixed_alpha'],device=ctx.base_rgb.device)
        error=dict(raw=raw,RGB_max=float((base['rgb']-cached_rgb).abs().max()),alpha_max=float((base['alpha'][0]-cached_alpha).abs().max()))
        if max(error['RGB_max'],error['alpha_max'])>1e-6:raise RuntimeError('当前native与冻结c0缓存偏差:'+str(error))
        if raw in targets and float((base['alpha'][0]-targets[raw]['fixed_initial_alpha']).abs().max())>1e-6:raise RuntimeError('candidate alpha与native不一致')
        binding.append(error)
    write_json(out/'initial_native_binding.json',dict(rows=binding,frozen_frequency_mask_reselected=False,full_frequency_pixels=37404))
    optimizer=torch.optim.Adam([model.delta_values],**{key:protocol['check3']['optimizer'][key] for key in ('lr','betas','eps','weight_decay')})
    sampler=random.Random(101);queue=[];trace=[];start=0
    def snapshot(step):
        return dict(schema='native_real_pair_complete_state.v1',step=step,model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(optimizer.state_dict()),
            sampler=sampler.getstate(),queue=queue.copy(),trace=trace.copy(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            numpy_rng=np.random.get_state(),python_rng=random.getstate(),contract=contract,formal_run=args.mode=='formal',arm=args.arm)
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu');start=saved['step']
        if saved['contract']!=contract or not 0<=start<=stop or (args.mode=='formal' and not saved['formal_run']):raise ValueError('恢复合同/预算/正式角色不一致')
        original=model.state_dict()
        for key,value in saved['model'].items():
            if value.shape!=original[key].shape or not bool(torch.isfinite(value).all()):raise ValueError('恢复tensor shape/finite无效')
            if key!='delta_values' and not torch.equal(value.cpu(),original[key].cpu()):raise ValueError('恢复试图改变固定model buffer')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);sampler.setstate(saved['sampler'])
        queue=saved['queue'].copy();trace=saved['trace'].copy()
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
        mismatch=differences(saved,snapshot(start))
        if mismatch:raise RuntimeError('完整恢复不exact:'+str(mismatch))
        write_json(out/'restore.json',dict(full_state_restored_exact=True,step=start,differences=mismatch))
    else:torch.save(snapshot(0),out/'checkpoint000000.pt')
    write_json(out/'evaluation_before.json',evaluate(ctx,model,targets,weights))
    gradients=[]
    for raw in (86,110):
        value,metrics,_=objective(ctx,model,targets,raw,weights)
        gradient=torch.autograd.grad(value,model.delta_values,retain_graph=True)[0]
        s_gradient=torch.autograd.grad(metrics['S'],model.delta_values)[0]
        if not bool(torch.isfinite(gradient).all()) or float(gradient.abs().max())==0 or float(s_gradient.abs().max())==0:raise RuntimeError('native实际颜色/S梯度无效')
        gradients.append(dict(raw=raw,total_gradient_max=float(gradient.abs().max()),S_gradient_max=float(s_gradient.abs().max()),finite=True))
    write_json(out/'gradient_probe.json',dict(actual_native_raster=True,rows=gradients,opacity_scale_geometry_parameters=0))
    records=[];begin=time.monotonic()
    for step in range(start+1,stop+1):
        if not queue:queue=[86,110];sampler.shuffle(queue)
        raw=queue.pop();optimizer.zero_grad(set_to_none=True);value,metrics,_=objective(ctx,model,targets,raw,weights)
        if not bool(torch.isfinite(value)):raise FloatingPointError('loss非finite')
        value.backward()
        if not bool(torch.isfinite(model.delta_values.grad).all()):raise FloatingPointError('gradient非finite')
        optimizer.step();model.project();trace.append(raw)
        record=dict(step=step,raw=raw,loss=float(value.detach()),**metric_values(metrics));records.append(record)
        if step==start+1 or step%25==0:print(json.dumps(record),flush=True)
        if step%50==0 or step==stop:torch.save(snapshot(step),out/f'checkpoint{step:06d}.pt')
    ctx.assert_frozen();colors=model().detach();ctx.audit_K(colors,out/'joint_K.json')
    write_json(out/'evaluation_after.json',evaluate(ctx,model,targets,weights,out/'rendered'))
    np.savez_compressed(out/'learned_correction.npz',delta_values=model.delta_values.detach().cpu().numpy(),mode=np.asarray(model.mode),
        effective_rgb=colors.cpu().numpy(),active_world_rows=ctx.world_row_ids)
    write_json(out/'parameterization.json',dict(arm=args.arm,mode=model.mode,GPU_executed=True,source_role=plan.source_role,qualification_version='V4_A_C_S',
        delta_node_bounds=[-1,1],world_spacing_m=.04,trainable_scalars=model.delta_values.numel(),world_TV=0,
        raw_field_and_frozen_row_override_separate=True,immutable_prior_ledger_preserved=True))
    asset=ctx.save_asset(colors,scope['world_eligible_color'],out/'asset',training_record=dict(arm=args.arm,mode=args.mode,step=stop,
        formal_updates=stop if args.mode=='formal' else 0,actual_updates_in_process=stop-start,scope_source=str(args.scope),qualification_version='V4_A_C_S'))
    ctx.validate_public_reload(colors,asset,out/'public_reload.json')
    saved_road=load_npz(asset/'road.npz',('reference_srgb',));saved_prior=load_npz(asset/'provenance.npz')
    only_known=scope['S_only_known']&eligible
    shadow_changed=np.any(saved_road['reference_srgb'][ctx.world_row_ids[only_known]]!=ctx.road['reference_srgb'][ctx.world_row_ids[only_known]],axis=1)
    immutable_keys=('known','observed_reference_srgb','observed_source_raw','source_raw','source_world_row')
    ledger={key:np.array_equal(saved_prior[key],ctx.provenance[key],equal_nan=True) for key in immutable_keys if key in ctx.provenance}
    if not all(ledger.values()):raise RuntimeError('原观测prior ledger被改')
    write_json(out/'shadow_known_persistence.json',dict(S_only_known_eligible=int(only_known.sum()),effective_RGB_changed_rows=int(shadow_changed.sum()),
        changed_effective_RGB_public_reload_verified=True,prior_ledger_fields_exact=ledger,source_known_is_prior_not_learned_measurement=True))
    write_json(out/'training.json',records)
    write_json(out/'completed.json',dict(completed=stop==args.steps,arm=args.arm,formal=args.mode=='formal',step=stop,
        actual_updates=stop-start,formal_budget=200,seconds=time.monotonic()-begin,checkpoint=str(out/f'checkpoint{stop:06d}.pt'),
        GPU_executed=True,coverage_geometry_active_frozen=True))
    print(json.dumps(dict(completed=True,arm=args.arm,step=stop,output=str(out))),flush=True);ctx.close()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arm',choices=('I0','W4'),required=True)
    parser.add_argument('--mode',choices=('smoke','formal'),required=True);parser.add_argument('--steps',type=int,required=True)
    parser.add_argument('--stop-step',type=int);parser.add_argument('--resume',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scope',type=Path,default=GPU_CAMPAIGN/'check1/native_r1/qualified_scope.npz')
    parser.add_argument('--targets',type=Path,default=CPU_CAMPAIGN/'supervision_v4_candidate_r1')
    parser.add_argument('--lease-file',type=Path,default=GPU_CAMPAIGN/'gpu_lease.json');parser.add_argument('--lease-owner',default='road_asset')
    train(parser.parse_args())

if __name__=='__main__':main()
