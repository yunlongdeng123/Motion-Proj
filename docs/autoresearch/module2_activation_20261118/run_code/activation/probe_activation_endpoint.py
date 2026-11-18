"""残余73真实贡献行的有界覆盖端点；不训练、不保存默认修复资产。"""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
from road_activation_builder import CAMPAIGN, PREVIOUS, REPO, FRAMES, read_npz, write_json, coverage_record, evaluation_masks

R8_ASSET=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_asset_20261116_r1/asset_r8_guarded')


def run(asset, visibility, native_source, output, lease_file, lease_owner):
    sys.path.insert(0,str(PREVIOUS/'code'))
    sys.path.insert(0,str(REPO/'scripts/worldsim_v82'))
    from road_capacity_native import assert_gpu_lease,load_context
    assert_gpu_lease(lease_file,lease_owner)
    import torch
    from drivingeditor_dynamic_edit.road_direct_teacher_training import conservative_retained_guard
    torch.set_num_threads(2)
    output,visibility,native_source=map(Path,(output,visibility,native_source))
    contract=json.loads((CAMPAIGN/'stage2_endpoint_contract.json').read_text())
    if Path(contract['input_asset']).resolve()!=Path(asset).resolve() or contract['optimizer_updates']!=0:
        raise ValueError('端点合同不匹配')
    previous=json.loads((visibility/'results.json').read_text())
    used=45+previous['native_calls']
    planned_calls=35  # 最大足迹5 + 三状态×五帧(road+joint)30；baseline复用实际A1。
    if used+planned_calls>contract['budget']['campaign_native_call_ceiling']:
        raise RuntimeError('计划超出本轮 native 调用预算')
    output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    ctx=load_context(asset_dir=asset,lease_file=lease_file,lease_owner=lease_owner)
    scope=read_npz(visibility/'endpoint_candidate_scope.npz')
    if not np.array_equal(scope['active_world_rows'],ctx.active_world_rows):
        raise ValueError('真实贡献 scope active 顺序不匹配')
    r8=read_npz(R8_ASSET/'road.npz')
    for key in ['model_xyz','normal_world']:
        if not np.array_equal(r8[key],ctx.road[key]):
            raise ValueError('原R8几何与当前A1不一致: '+key)
    absolute_upper=r8['gaussian_sigma_m'][:,:2]*1.15
    candidate=scope['world_eligible'][ctx.active_world_rows]
    current=ctx.road['gaussian_sigma_m'][ctx.active_world_rows,:2]
    if np.any(current[candidate]>absolute_upper[ctx.active_world_rows][candidate]+1e-8):
        raise ValueError('当前下界超过原R8绝对sigma上界')
    ratio=np.ones(len(candidate),np.float64)
    ratio[candidate]=np.maximum(1,np.minimum(*(absolute_upper[ctx.active_world_rows][candidate]/current[candidate]).T))
    maximum_guard,K_masks,K_record=conservative_retained_guard(ctx.scene,ctx.provider,len(candidate),ratio,ctx.manifest.data['frames'])
    calls=5
    eligible=candidate&~maximum_guard
    ratio[~eligible]=1
    eligible_ids=np.flatnonzero(eligible)
    device=ctx.base_rgb.device
    ids_tensor=torch.as_tensor(eligible_ids,device=device)
    ratio_tensor=torch.as_tensor(ratio,device=device,dtype=ctx.base_rgb.dtype)
    def packet(raw,opacity,scale):
        value=ctx.base_packet(raw)
        if opacity:
            op=value['opacities'].clone()
            op[ids_tensor]=torch.maximum(op[ids_tensor],op.new_tensor(1-1e-5))
            value=dict(value,opacities=op)
        if scale:
            sigma=value['scales'].clone()
            sigma[:,:2]*=ratio_tensor[:,None]
            value=dict(value,scales=sigma)
        return value
    world_scope=np.zeros(len(ctx.road['active']),bool)
    world_scope[ctx.active_world_rows]=eligible
    world_max_guard=np.zeros(len(world_scope),bool)
    world_max_guard[ctx.active_world_rows]=maximum_guard
    np.savez_compressed(output/'maximum_endpoint_scope.npz',world_eligible=world_scope,
        world_maximum_K_guard=world_max_guard,active_world_rows=ctx.active_world_rows,
        active_scale_factor=ratio,absolute_sigma_upper_m=absolute_upper,
        source_remaining73_mass=scope['actual_remaining73_mass'])
    write_json(output/'maximum_footprint_K.json',K_record)
    states=[('opacity_max',True,False),('tangent_max',False,True),('combined_max',True,True)]
    results=[]
    for raw in FRAMES:
        saved=read_npz(native_source/f'{raw:03d}_native.npz')
        masks=evaluation_masks(raw)
        if raw in (86,110):
            masks['remaining_thin']=saved['fixed_low_S']&(saved['road_alpha']<=.9)
        baseline_regions={name:coverage_record(saved['road_alpha'],mask) for name,mask in masks.items()}
        frame=dict(raw=raw,baseline=baseline_regions,states={})
        arrays={name:value for name,value in masks.items()}
        arrays['baseline_road_alpha']=saved['road_alpha']
        for label,opacity,scale in states:
            with torch.no_grad():
                rows=packet(raw,opacity,scale)
                camera=ctx.camera(raw)
                road=ctx.scene._bindings().raster(camera,ctx.scene.pipe,0,rows,ctx.scene._background)
                calls+=1
                alpha=road['alpha'][0].cpu().numpy()
                values,owners,camera,_=ctx.scene._rows(raw,(10,),(),False)
                values=dict(values,mask=values['mask']&~ctx.scene._contains(owners,(10,)))
                offset=len(values['means3D'])-len(ctx.active_world_rows)
                if not torch.equal(values['means3D'][offset:],rows['means3D']):
                    raise RuntimeError('endpoint joint road tail mismatch')
                for key in ['opacities','scales']:
                    value=values[key].clone();value[offset:]=rows[key];values=dict(values,**{key:value})
                features=values['features'].new_zeros((len(values['means3D']),24))
                features[:,:2]=values['features'][:,:2]
                ns=len(ctx.scene.static.get_xyz)
                features[:ns,2]=1;features[ns:offset,3]=1
                new=ctx.provenance['activation_new'][ctx.active_world_rows]
                features[offset+torch.as_tensor(np.flatnonzero(~new),device=device),4]=1
                features[offset+torch.as_tensor(np.flatnonzero(new),device=device),5]=1
                joint=ctx.scene._bindings().raster(camera,ctx.scene.pipe,0,dict(values,features=features),ctx.scene._background)
                calls+=1
                joint_rgb=joint['rgb']+(1-joint['alpha'])*ctx.scene.environment(camera.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
                difference=np.abs(joint_rgb.cpu().numpy()-saved['joint_rgb']).max(0)
                K=K_masks[raw]
                if np.any(difference[K]!=0):
                    raise RuntimeError('最大端点 full joint K 不为 exact0: '+str((raw,label,float(difference[K].max()))))
                layers={name:joint['extra'][i].cpu().numpy() for i,name in enumerate(['static_alphaT','dynamic_alphaT','old_road_alphaT','new_road_alphaT'])}
            arrays[label+'_road_alpha']=alpha
            for name,value in layers.items():arrays[label+'_joint_'+name]=value
            frame['states'][label]=dict(regions={name:dict(coverage=coverage_record(alpha,mask),
                baseline_thin_to_supported=int((mask&(saved['road_alpha']<=.9)&(alpha>.9)).sum()),
                previously_supported_lost=int((mask&(saved['road_alpha']>.9)&(alpha<=.9)).sum()),
                joint_group_mass={group:float(value[mask].sum(dtype=np.float64)) for group,value in layers.items()}) for name,mask in masks.items()},
                K_pixels=int(K.sum()),joint_K_RGB_max=float(difference[K].max()))
        np.savez_compressed(output/f'{raw:03d}_endpoint.npz',**arrays)
        results.append(frame)
        print(json.dumps(dict(stage='isolated_maximum_endpoint_complete',raw=raw,native_calls=calls)),flush=True)
    if calls!=planned_calls:
        raise AssertionError('真实调用计数与预登记不符: '+str(calls))
    ctx.assert_frozen();ctx.close()
    result=dict(schema='road_activation_bounded_endpoint.v1',GPU_executed=True,optimizer_updates=0,
        source_asset=str(asset),source_visibility=str(visibility),source_original_R8=str(R8_ASSET),
        original_candidate_rows=int(candidate.sum()),maximum_K_rejected_candidate_rows=int((candidate&maximum_guard).sum()),
        effective_opacity_rows=int(eligible.sum()),effective_scale_growth_rows=int((eligible&(ratio>1.000001)).sum()),
        maximum_tangent_factor=float(ratio.max()),opacity_upper=1-1e-5,
        sigma_definition='absolute originalR8 sigma times1.15; no compound enlargement of currentS2',
        rows=results,native_calls=calls,campaign_native_calls=used+calls,seconds=time.monotonic()-start,
        all_five_frame_joint_K_exact=True,stored_scene_or_asset_modified=False,
        default_asset_written=False,endpoint_is_control_not_proposed_final_method=True)
    write_json(output/'results.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset',type=Path,default=CAMPAIGN/'native_r1/asset_A1_guarded')
    p.add_argument('--visibility',type=Path,default=CAMPAIGN/'visibility_r2')
    p.add_argument('--native-source',type=Path,default=CAMPAIGN/'native_r1')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lease-file',type=Path,default=CAMPAIGN/'gpu_lease.json')
    p.add_argument('--lease-owner',required=True)
    a=p.parse_args();r=run(a.asset,a.visibility,a.native_source,a.output,a.lease_file,a.lease_owner)
    print(json.dumps(dict(completed=True,campaign_native_calls=r['campaign_native_calls'],endpoint_rows=r['effective_opacity_rows'])),flush=True)
