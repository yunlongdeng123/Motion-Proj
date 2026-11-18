"""已冻结覆盖资产与已冻结I0原oldactive颜色的单次零更新组合。"""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
from road_activation_builder import CAMPAIGN,PREVIOUS,PARENT,REPO,FRAMES,read_npz,write_json,sha256,pin_asset,evaluation_masks,coverage_record

N_ASSET=CAMPAIGN/'B32_beam95_native_r1/asset_guarded'
I0_ASSET=PREVIOUS/'check3/I0/formal_r1/asset'


def plan(output,parent=N_ASSET,I0=I0_ASSET):
    output,parent,I0=map(Path,(output,parent,I0))
    output.mkdir(parents=True,exist_ok=False)
    base=read_npz(PARENT/'road.npz');road=read_npz(parent/'road.npz');learned=read_npz(I0/'road.npz')
    with np.load(I0/'provenance.npz') as f:
        allowed=f['capacity_color_eligible'].copy();c0=f['capacity_source_prior_reference_srgb'].copy()
    prior=read_npz(parent/'provenance.npz')
    checks={}
    checks['I0_nonRGB_exact_oldS2']=all(np.array_equal(learned[k],base[k]) for k in base if k!='reference_srgb')
    checks['N_geometry_sigma_opacity_exact_oldS2']=all(np.array_equal(road[k],base[k]) for k in base if k not in ['active','reference_srgb'])
    checks['N_oldactiveRGB_exact_oldS2']=np.array_equal(road['reference_srgb'][base['active']],base['reference_srgb'][base['active']])
    checks['I0_c0_exact_oldS2']=np.array_equal(c0,base['reference_srgb'])
    checks['I0_allowed_only_original_active']=not np.any(allowed&~base['active'])
    changed=(learned['reference_srgb']!=base['reference_srgb']).any(1)
    checks['I0_changed_only_allowed']=not np.any(changed&~allowed)
    new=road['active']&~base['active'];checks['new892_no_I0_scope_overlap']=not np.any(new&allowed)
    scenes=[json.loads((a/'persistent/scene.json').read_text()) for a in (parent,I0)]
    for d in scenes:
        for key in ['road','static_gates']:d.pop(key,None)
    checks['same_native_scene_cameras_ownership']=scenes[0]==scenes[1]
    if not all(checks.values()):raise ValueError('组合输入冲突: '+str(checks))
    combined=road['reference_srgb'].copy();combined[allowed]=learned['reference_srgb'][allowed]
    applied=(combined!=road['reference_srgb']).any(1)
    currentK=read_npz(CAMPAIGN/'B32_beam95_native_r1/new_packet_guards.npz')['world_K_guard']
    checks['I0_effective_change_outside_current_fullpacket_K']=not np.any(applied&currentK)
    if not all(checks.values()):raise ValueError('组合初始K资格冲突')
    np.savez_compressed(output/'composition_plan.npz',allowed_original_active=allowed,applied_learned_override=applied,
        candidate_reference_srgb=combined,I0_source_c0=c0,new_activation_rows=new,
        N_new_reference_srgb=road['reference_srgb'][new])
    receipt=dict(schema='road_activation_frozen_I0_composition_plan.v1',CPU_only=True,GPU_executed=False,
        parent_asset=str(parent),parent_road_sha256=sha256(parent/'road.npz'),I0_asset=str(I0),
        I0_road_sha256=sha256(I0/'road.npz'),oldS2_asset=str(PARENT),checks=checks,
        allowed_rows=int(allowed.sum()),learned_override_rows=int(applied.sum()),new_activation_rows=int(new.sum()),
        new_observed_rows=int((new&prior['known']&(prior['source']==1)).sum()),new_frozen_teacher_rows=int((new&prior['B32_new']).sum()),
        prior_ledger_policy='N source/known/unknown/observed_reference and source metadata remain byte-exact; learned I0 RGB is a separate effective override',
        material_role='existing I0 direct perGaussian fitted RGB; no new observation or current world atlas claim',
        optimizer_updates=0,new_SD_or_TRELLIS_calls=0,native_qualification_pending=True,production_promotion=False)
    write_json(output/'feasibility.json',receipt);return receipt


def materialize(plan_dir,output,*,guard=None,pin=True):
    plan_dir,output=map(Path,(plan_dir,output))
    if output.exists():raise FileExistsError('组合资产须保存到新目录')
    info=json.loads((plan_dir/'feasibility.json').read_text());parent=Path(info['parent_asset']);I0=Path(info['I0_asset'])
    if sha256(parent/'road.npz')!=info['parent_road_sha256'] or sha256(I0/'road.npz')!=info['I0_road_sha256']:
        raise ValueError('组合冻结输入发生变化')
    r,p,d=read_npz(parent/'road.npz'),read_npz(parent/'provenance.npz'),read_npz(plan_dir/'composition_plan.npz')
    out=dict(r);out['reference_srgb']=d['candidate_reference_srgb'].copy()
    blocked=np.zeros(len(r['active']),bool)
    if guard is not None:
        g=read_npz(guard);blocked=g['world_K_guard']|g['world_near_guard']
        out['reference_srgb'][blocked]=r['reference_srgb'][blocked]
    applied=(out['reference_srgb']!=r['reference_srgb']).any(1)
    if np.any(applied&~d['allowed_original_active']):raise AssertionError('组合改动I0资格之外RGB')
    if not np.array_equal(out['reference_srgb'][d['new_activation_rows']],r['reference_srgb'][d['new_activation_rows']]):
        raise AssertionError('组合修改892新行材质')
    q=dict(p)
    q.update(I0_learned_override_eligible=d['allowed_original_active'],I0_learned_override_applied=applied,
        I0_source_c0=d['I0_source_c0'],I0_override_parent_rgb=r['reference_srgb'],
        I0_effective_rgb=out['reference_srgb'],I0_fresh_guarded_override=blocked&d['applied_learned_override'],
        effective_row_rgb=out['reference_srgb'],current_shader_mode=np.asarray('direct_perGaussian_prior_plus_frozen_I0_override'),
        effective_material_source_role=np.asarray('base source provenance retained; I0_learned_override_applied identifies changed effectiveRGB'))
    output.mkdir(parents=True)
    np.savez_compressed(output/'road.npz',**out);np.savez_compressed(output/'provenance.npz',**q)
    record=dict(schema='road_activation_frozen_I0_composite.v1',parent_asset=str(parent),I0_asset=str(I0),
        composition_plan=str(plan_dir),only_road_field_changed='reference_srgb',
        allowed_original_active=int(d['allowed_original_active'].sum()),applied_override_rows=int(applied.sum()),
        fresh_K_or_near_guarded_override_rows=int((blocked&d['applied_learned_override']).sum()),
        all892_new_rows_RGB_exact_N=True,all_nonRGB_road_fields_exact=True,original_observed_prior_ledger_exact=True,
        learned_effective_override_separate_from_prior=True,optimizer_updates=0,production_promotion=False,
        known_I0_quality_failure_retained=True,native_qualification_pending=guard is None)
    write_json(output/'receipt.json',record)
    if pin:pin_asset(parent,output,record)
    return record


def native(plan_dir,output,review,lease_file,lease_owner):
    sys.path.insert(0,str(PREVIOUS/'code'));sys.path.insert(0,str(REPO/'scripts/worldsim_v82'))
    from road_capacity_native import assert_gpu_lease,load_context
    assert_gpu_lease(lease_file,lease_owner)
    contract=json.loads((CAMPAIGN/'stage4_composition_contract.json').read_text())
    source_review=json.loads(Path(review).read_text())
    if not contract or source_review.get('status')!='PASS':raise ValueError('组合合同或来源CPU审查未就绪')
    import torch
    from PIL import Image
    from drivingeditor_dynamic_edit.road_direct_teacher_training import conservative_retained_guard
    torch.set_num_threads(2);start=time.monotonic();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    prepared=output/'asset_prepared';materialize(plan_dir,prepared)
    ctx=load_context(asset_dir=prepared,lease_file=lease_file,lease_owner=lease_owner)
    K,kmasks,kr=conservative_retained_guard(ctx.scene,ctx.provider,len(ctx.active_world_rows),np.ones(len(ctx.active_world_rows)),ctx.manifest.data['frames'])
    n=len(ctx.road['active']);worldK=np.zeros(n,bool);worldK[ctx.active_world_rows]=K;near=np.zeros(n,bool)
    for raw in FRAMES:
        packet,camera=ctx.base_packet(raw),ctx.camera(raw)
        z=(packet['means3D']@camera.world_view_transform[:3,2]+camera.world_view_transform[3,2]).cpu().numpy()*ctx.loader['unit_scale']
        near[ctx.active_world_rows[(z>0)&(z<=.2)]]=True
    ctx.close();calls=5
    np.savez_compressed(output/'fresh_guards.npz',world_K_guard=worldK,world_near_guard=near);write_json(output/'fresh_K.json',kr)
    final=output/'asset_guarded';record=materialize(plan_dir,final,guard=output/'fresh_guards.npz')
    ctx=load_context(asset_dir=final,lease_file=lease_file,lease_owner=lease_owner)
    rows=[]
    with torch.no_grad():
        for raw in FRAMES:
            before=read_npz(CAMPAIGN/f'B32_beam95_native_r1/{raw:03d}_native.npz')
            r,j=ctx.render_road(raw),ctx.render_joint(raw)
            rgb,alpha,jrgb=r['rgb'].cpu().numpy(),r['alpha'][0].cpu().numpy(),j['rgb'].cpu().numpy()
            if not np.array_equal(alpha,before['road_alpha']):raise RuntimeError('颜色组合却改变coverage alpha')
            difference=np.abs(jrgb-before['joint_rgb']).max(0);mask=kmasks[raw]
            if (difference[mask]!=0).any():raise RuntimeError('组合实际joint K未精确保持')
            domain=evaluation_masks(raw)
            rows.append(dict(raw=raw,coverage_all_exact=True,K_pixels=int(mask.sum()),joint_K_RGB_max=float(difference[mask].max()),
                regions={name:coverage_record(alpha,m) for name,m in domain.items()}))
            np.savez_compressed(output/f'{raw:03d}_native.npz',road_rgb=rgb,road_alpha=alpha,joint_rgb=jrgb,K=mask,**domain)
            for suffix,value in [('road',rgb),('joint',jrgb)]:
                Image.fromarray(np.clip(np.rint(value.transpose(1,2,0)*255),0,255).astype(np.uint8)).save(output/f'{raw:03d}_{suffix}.webp',lossless=True)
    reload=ctx.validate_public_reload(ctx.base_rgb,final,output/'public_reload.json');calls+=ctx.native_calls+5;ctx.assert_frozen();ctx.close()
    if calls!=25:raise AssertionError('组合实际调用不等于25')
    result=dict(schema='road_activation_frozen_I0_composition_native.v1',GPU_executed=True,optimizer_updates=0,
        final_asset=str(final),composition=record,rows=rows,public_reload=reload,native_calls=calls,prior_campaign_calls=169,
        campaign_native_calls=194,seconds=time.monotonic()-start,all_alpha_exact_N=True,all_joint_K_exact=True,production_promotion=False)
    write_json(output/'results.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['plan','native'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--plan',type=Path,default=CAMPAIGN/'stage4_cpu_r1')
    p.add_argument('--review',type=Path);p.add_argument('--lease-file',type=Path,default=CAMPAIGN/'gpu_lease.json');p.add_argument('--lease-owner')
    a=p.parse_args();r=plan(a.output) if a.command=='plan' else native(a.plan,a.output,a.review,a.lease_file,a.lease_owner)
    print(json.dumps(r,ensure_ascii=False,allow_nan=False),flush=True)
