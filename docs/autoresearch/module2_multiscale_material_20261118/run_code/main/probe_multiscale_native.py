"""新覆盖父资产的native前置资格；当前S3131梯度，0优化，不依赖field控制器。"""
from pathlib import Path
import argparse,json,sys
import numpy as np

P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
G=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_capacity_gpu_20261118_r1')
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_capacity_cpu_20261117_r1')
ASSET=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_activation_20261118_r1/B32_beam95_native_r1/asset_guarded')
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')

def run(output,owner):
    sys.path.insert(0,str(G/'code'));sys.path.insert(0,str(REPO/'scripts/worldsim_v82'))
    from road_capacity_native import load_context,assert_gpu_lease,write_json,load_npz
    assert_gpu_lease(P/'gpu_lease.json',owner)
    import torch
    from drivingeditor_dynamic_edit.road_direct_teacher_training import conservative_retained_guard
    from drivingeditor_dynamic_edit.native_row_contribution import attribute_prepared_rows
    output=Path(output);output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    ctx=load_context(asset_dir=ASSET,lease_file=P/'gpu_lease.json',lease_owner=owner)
    guard,Km,Kr=conservative_retained_guard(ctx.scene,ctx.provider,len(ctx.active_world_rows),np.ones(len(ctx.active_world_rows)),ctx.manifest.data['frames'])
    if sum(int(m.sum()) for m in Km.values())!=77266:raise RuntimeError('完整K分母不符')
    n=len(ctx.base_rgb);masses={k:np.zeros(n,np.float64) for k in ['A','C','S']};stats=[];calls=5
    for raw in [86,110]:
        d=load_npz(C/f'supervision_v4_candidate_r1/{raw:03d}.npz')
        with torch.no_grad():baseline=ctx.render_road(raw)
        calls+=1;alpha=baseline['alpha'][0].cpu().numpy()
        masks=dict(A=d['coverage_A']&(alpha>.9),C=d['coverage_C']&(alpha>.9),S=d['S_target_shadow']&(alpha>.9))
        packet=ctx.base_packet(raw);owners=torch.zeros(n,device=ctx.base_rgb.device,dtype=torch.long)
        att=attribute_prepared_rows(ctx.scene,packet,owners,ctx.camera(raw),masks);calls+=1
        for j,key in enumerate(att.region_names):masses[key]=np.maximum(masses[key],att.alpha_mass[:,j])
        rgb=ctx.base_rgb.detach().clone().requires_grad_(True);render=ctx.render_road(raw,rgb);calls+=1
        losses={};gradients={}
        for key,targetkey in [('A','target_A_srgb'),('C','target_C_srgb'),('S','target_S_srgb')]:
            mask=torch.as_tensor(masks[key],device=rgb.device);target=torch.as_tensor(d[targetkey],device=rgb.device)
            loss=(render['rgb']-render['alpha'].detach()*target).abs()[:,mask].mean()
            grad=torch.autograd.grad(loss,rgb,retain_graph=True)[0]
            losses[key]=float(loss);gradients[key]=dict(nonzero=int((grad!=0).sum()),finite=bool(torch.isfinite(grad).all()))
        np.savez_compressed(output/f'{raw:03d}_targets.npz',**masks,fullS=d['S_target_shadow'],fixed_alpha=alpha,
            target_A=d['target_A_srgb'],target_C=d['target_C_srgb'],target_S=d['target_S_srgb'])
        stats.append(dict(raw=raw,pixels={k:int(v.sum()) for k,v in masks.items()},fullS=int(d['S_target_shadow'].sum()),
            lowS=int((d['S_target_shadow']&(alpha<=.9)).sum()),losses=losses,color_leaf_gradient=gradients))
    p=ctx.provenance;ids=ctx.active_world_rows
    available=p['effective_material_available'][ids]
    eligible=(masses['A']>0)|(masses['C']>0)|(masses['S']>0)
    eligible &= available & ~guard & ~p['guarded_world_rows'][ids] & (ctx.road['minimum_model_depth'][ids]<=0)
    np.savez_compressed(output/'scope.npz',active_world_rows=ids,eligible=eligible,conservative_K_guard=guard,
        material_available=available,actual_A_mass=masses['A'],actual_C_mass=masses['C'],actual_S_mass=masses['S'])
    if sum(r['pixels']['S'] for r in stats)!=3131:raise RuntimeError('新S3131固定监督分母不符')
    write_json(output/'K_footprint.json',Kr);ctx.assert_frozen();ctx.close()
    result=dict(schema='multiscale_native_stage0_packet.v1',GPU_executed=True,optimizer_updates=0,native_calls=calls,
        S_photo=3131,S_full=3187,low_S=56,rows=stats,qualified_color_rows=int(eligible.sum()),
        field_gradient_checked=False,color_leaf_probe_only=True,field_numeric_qualification_pending=True)
    write_json(output/'receipt.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--lease-owner',required=True)
    a=p.parse_args();run(a.output,a.lease_owner)
