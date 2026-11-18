"""只读已审fixed-Σ beam候选；新packet保护后保存已存在known节点激活控制。"""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
from road_activation_builder import CAMPAIGN,PREVIOUS,REPO,FRAMES,read_npz,write_json,build_activation_asset,evaluation_masks,coverage_record


def run(parent,native_parent,proposal,review,output,lease_file,lease_owner,prior_calls):
    sys.path.insert(0,str(PREVIOUS/'code'));sys.path.insert(0,str(REPO/'scripts/worldsim_v82'))
    from road_capacity_native import assert_gpu_lease,load_context
    assert_gpu_lease(lease_file,lease_owner)
    review_path=Path(review)
    reviewed=json.loads(review_path.read_text())
    if reviewed.get('status')!='PASS':raise ValueError('fixed-Σ beam source/material CPU审查尚未通过')
    if prior_calls+25>200:raise RuntimeError('N累计native调用超过200前须复核')
    contract=json.loads((CAMPAIGN/'stage3_completion_contract.json').read_text())
    if not contract:raise ValueError('缺已登记completion控制合同')
    import torch
    from PIL import Image
    from drivingeditor_dynamic_edit.road_direct_teacher_training import conservative_retained_guard
    torch.set_num_threads(2)
    parent,native_parent,output=map(Path,(parent,native_parent,output));start=time.monotonic()
    previous=json.loads((native_parent/'results.json').read_text())
    if Path(previous['final_asset']).resolve()!=parent.resolve():raise ValueError('baseline native arrays不属于指定parent资产')
    output.mkdir(parents=True,exist_ok=False)
    prepared=output/'asset_prepared'
    build_activation_asset(parent,proposal,prepared,initialize_observed=True,pin=True)
    ctx=load_context(asset_dir=prepared,lease_file=lease_file,lease_owner=lease_owner)
    guard,kmasks,kr=conservative_retained_guard(ctx.scene,ctx.provider,len(ctx.active_world_rows),np.ones(len(ctx.active_world_rows)),ctx.manifest.data['frames'])
    n=len(ctx.road['active']);worldK=np.zeros(n,bool);worldK[ctx.active_world_rows]=guard;near=np.zeros(n,bool)
    for raw in FRAMES:
        packet,camera=ctx.base_packet(raw),ctx.camera(raw)
        z=(packet['means3D']@camera.world_view_transform[:3,2]+camera.world_view_transform[3,2]).cpu().numpy()*ctx.loader['unit_scale']
        near[ctx.active_world_rows[(z>0)&(z<=.2)]]=True
    ctx.close();calls=5
    np.savez_compressed(output/'new_packet_guards.npz',world_K_guard=worldK,world_near_guard=near)
    write_json(output/'new_packet_K.json',kr)
    final=output/'asset_guarded'
    built=build_activation_asset(parent,proposal,final,guard_npz=output/'new_packet_guards.npz',initialize_observed=True,pin=True)
    ctx=load_context(asset_dir=final,lease_file=lease_file,lease_owner=lease_owner)
    rows=[]
    with torch.no_grad():
        for raw in FRAMES:
            before=read_npz(native_parent/f'{raw:03d}_native.npz');domain=evaluation_masks(raw)
            if raw in (86,110):
                A1=read_npz(CAMPAIGN/f'native_r1/{raw:03d}_native.npz')
                domain['A1_remaining73']=A1['fixed_low_S']&(A1['road_alpha']<=.9)
                domain['parent_remaining']=A1['fixed_low_S']&(before['road_alpha']<=.9)
            road,joint=ctx.render_road(raw),ctx.render_joint(raw)
            rgb,alpha,jrgb=road['rgb'].cpu().numpy(),road['alpha'][0].cpu().numpy(),joint['rgb'].cpu().numpy()
            diff=np.abs(jrgb-before['joint_rgb']).max(0);K=kmasks[raw]
            if (diff[K]!=0).any():raise RuntimeError('known beam newpacket joint K不exact')
            regions={name:dict(before=coverage_record(before['road_alpha'],mask),after=coverage_record(alpha,mask),
                newly_supported=int((mask&(before['road_alpha']<=.9)&(alpha>.9)).sum()),
                lost_support=int((mask&(before['road_alpha']>.9)&(alpha<=.9)).sum())) for name,mask in domain.items()}
            rows.append(dict(raw=raw,regions=regions,K_pixels=int(K.sum()),joint_K_RGB_max=float(diff[K].max())))
            np.savez_compressed(output/f'{raw:03d}_native.npz',road_rgb=rgb,road_alpha=alpha,joint_rgb=jrgb,
                parent_road_alpha=before['road_alpha'],K=K,**domain)
            for suffix,value in [('road',rgb),('joint',jrgb)]:
                Image.fromarray(np.clip(np.rint(value.transpose(1,2,0)*255),0,255).astype(np.uint8)).save(output/f'{raw:03d}_{suffix}.webp',lossless=True)
    reload=ctx.validate_public_reload(ctx.base_rgb,final,output/'public_reload.json')
    calls+=ctx.native_calls+5;ctx.assert_frozen();ctx.close()
    if calls!=25:raise AssertionError('known-beam实际调用计数异常')
    result=dict(schema='road_activation_known_fixed_sigma_beam.v1',GPU_executed=True,optimizer_updates=0,
        source_parent=str(parent),source_proposal=str(proposal),source_CPU_review=str(review_path),
        final_asset=str(final),activation=built,rows=rows,public_reload=reload,native_calls=calls,
        prior_campaign_calls=prior_calls,campaign_native_calls=prior_calls+calls,seconds=time.monotonic()-start,
        all_five_frame_joint_K_exact=True,all_existing_geometry_sigma_opacity_and_oldactiveRGB_exact=True,
        original_A1_remaining73_domain_preserved=True,production_promotion=False)
    write_json(output/'results.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--native-parent',type=Path,required=True)
    p.add_argument('--proposal',type=Path,required=True);p.add_argument('--review',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--lease-file',type=Path,default=CAMPAIGN/'gpu_lease.json')
    p.add_argument('--lease-owner',required=True);p.add_argument('--prior-calls',type=int,required=True)
    a=p.parse_args();r=run(a.parent,a.native_parent,a.proposal,a.review,a.output,a.lease_file,a.lease_owner,a.prior_calls)
    print(json.dumps(dict(completed=True,new_active=r['activation']['new_active_rows'],campaign_native_calls=r['campaign_native_calls'])),flush=True)
