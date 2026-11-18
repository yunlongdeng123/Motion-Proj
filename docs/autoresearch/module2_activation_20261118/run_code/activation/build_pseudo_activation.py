"""仅消费已登记B32 frozen2D pseudo许可；原观测ledger与既有活动行精确冻结。"""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
from road_activation_builder import CAMPAIGN,PREVIOUS,CPU_CAMPAIGN,REPO,FRAMES,read_npz,write_json,pin_asset,sha256,coverage_record,evaluation_masks


def build(parent, initialization, output, *, guard=None, pin=False):
    parent,initialization,output=map(Path,(parent,initialization,output))
    if output.exists():raise FileExistsError('保留旧现场，输出须为新目录')
    r,p,d=read_npz(parent/'road.npz'),read_npz(parent/'provenance.npz'),read_npz(initialization)
    ids=d['proposed_world_rows'];n=len(r['active'])
    if ids.shape!=(32,) or ids.dtype.kind not in 'iu' or len(np.unique(ids))!=32:
        raise ValueError('本控制仅允许独立登记的32个世界节点')
    if (ids<0).any() or (ids>=n).any() or r['active'][ids].any():raise ValueError('B32须为父A1 inactive节点')
    if p['known'][ids].any() or (~p['unknown'][ids]).any() or (p['source'][ids]!=0).any():
        raise ValueError('B32原ledger必须保持unknown/source0而不是observed')
    if (~p['surface_supported'][ids]).any() or p['guarded_world_rows'][ids].any() or (r['minimum_model_depth'][ids]>0).any():
        raise ValueError('B32不在既有可信表面或触及旧guard/near')
    if not np.array_equal(d['source_world_xyz'],p['world_xyz'][ids]) or not np.array_equal(d['parent_effective_rgb_sparse'],r['reference_srgb'][ids]):
        raise ValueError('B32初始化与父A1世界点/颜色绑定不符')
    if (d['pseudo_source_role']!=3).any() or (d['pseudo_source_permission_q']!=1).any() or not np.isfinite(d['pseudo_source_uv_native']).all():
        raise ValueError('B32缺完整frozen2D nativefootprint许可')
    rgb=d['proposed_effective_rgb_sparse']
    if rgb.shape!=(32,3) or not np.isfinite(rgb).all() or ((rgb<0)|(rgb>1)).any():raise ValueError('非法pseudo RGB')
    g={} if guard is None else read_npz(guard)
    K=g.get('world_K_guard',np.zeros(n,bool));near=g.get('world_near_guard',np.zeros(n,bool))
    if K.shape!=(n,) or near.shape!=(n,):raise ValueError('新packet guard须完整world域')
    requested=np.zeros(n,bool);requested[ids]=True
    accepted=requested&~K&~near
    local=accepted[ids]
    out=dict(r);out['active']=r['active']|accepted;out['reference_srgb']=r['reference_srgb'].copy()
    out['reference_srgb'][ids[local]]=rgb[local]
    for key in r:
        if key not in ['active','reference_srgb'] and not np.array_equal(r[key],out[key]):raise AssertionError('B32未授权字段变化')
        if out[key].shape and out[key].shape[0]==n and not np.array_equal(out[key][r['active']],r[key][r['active']]):
            raise AssertionError('B32改变父A1活动节点: '+key)
    q=dict(p)
    source=p.get('effective_material_source',p['source']).copy();source[accepted]=3
    available=p.get('effective_material_available',(~p['unknown']&(p['source']>0))).copy();available[accepted]=True
    unknown=p.get('effective_material_unknown',p['unknown']).copy();unknown[accepted]=False
    legal=p.get('effective_coverage_legal',p['legal_write']).copy();legal[accepted]=True
    source_raw=p.get('effective_material_source_raw',p['source_raw']).copy();source_raw[ids[local]]=d['pseudo_source_raw'][local]
    source_uv=p.get('effective_material_source_uv_native',p['source_uv_native']).astype(np.float64,copy=True);source_uv[ids[local]]=d['pseudo_source_uv_native'][local]
    q.update(B32_requested=requested,B32_new=accepted,B32_K_guard=K,B32_near_guard=near,
        B32_parent_rgb=r['reference_srgb'],B32_effective_rgb=out['reference_srgb'],
        effective_row_rgb=out['reference_srgb'],effective_active=out['active'],
        effective_coverage_requested=p['effective_coverage_requested']|accepted,
        effective_material_source=source,effective_material_available=available,effective_material_unknown=unknown,
        effective_coverage_legal=legal,effective_material_source_raw=source_raw,effective_material_source_uv_native=source_uv,
        B32_native_permission_q=np.where(accepted,1.,0.).astype(np.float32))
    output.mkdir(parents=True)
    np.savez_compressed(output/'road.npz',**out);np.savez_compressed(output/'provenance.npz',**q)
    delta=dict(parent_world_row_id=ids,activation_new=local,material_initialization_kind=np.full(32,3,np.uint8),
        pseudo_source_raw=d['pseudo_source_raw'],pseudo_source_uv_native=d['pseudo_source_uv_native'],
        pseudo_source_permission_q=d['pseudo_source_permission_q'],K_guard=K[ids],near_guard=near[ids])
    for key in ['active','model_xyz','normal_world','gaussian_sigma_m','road_opacity','reference_srgb']:
        delta['parent_'+key]=r[key][ids];delta['effective_'+key]=out[key][ids]
    np.savez_compressed(output/'activation_delta.npz',**delta)
    record=dict(schema='road_activation_B32_frozen_teacher.v1',parent_asset=str(parent),initialization=str(initialization),
        initialization_sha256=sha256(initialization),proposed_rows=32,new_active_rows=int(accepted.sum()),
        K_rejected=int((requested&K).sum()),near_rejected=int((requested&near).sum()),
        only_road_fields_changed=['active','reference_srgb'],existing_active_all_fields_exact=True,
        all_geometry_sigma_opacity_exact=True,prior_source_known_unknown_observed_ledger_exact=True,
        effective_material_role='frozen2D target-shadow pseudo; original source0/unknown ledger stays unchanged',
        effective_qualification_fields=['effective_material_source','effective_material_available','effective_material_unknown','effective_coverage_legal','effective_coverage_requested'],
        prior_A1_activation_fields_role='historical A1 stage record, current B32 effect recorded separately',
        permission_q_role='binary nativefootprint validity, not calibrated teacher correctness',
        optimizer_updates=0,new_SD_or_TRELLIS_calls=0,production_promotion=False,native_qualified=guard is not None)
    write_json(output/'receipt.json',record)
    if pin:pin_asset(parent,output,record)
    return record


def native(initialization,parent,native_parent,output,lease_file,lease_owner):
    sys.path.insert(0,str(PREVIOUS/'code'));sys.path.insert(0,str(REPO/'scripts/worldsim_v82'))
    from road_capacity_native import assert_gpu_lease,load_context
    assert_gpu_lease(lease_file,lease_owner)
    contract=json.loads((CAMPAIGN/'stage3_completion_contract.json').read_text())
    if not contract:raise ValueError('缺少已登记B32控制合同')
    import torch
    from PIL import Image
    from drivingeditor_dynamic_edit.road_direct_teacher_training import conservative_retained_guard
    torch.set_num_threads(2);start=time.monotonic();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    prepared=output/'asset_prepared';build(parent,initialization,prepared,pin=True)
    ctx=load_context(asset_dir=prepared,lease_file=lease_file,lease_owner=lease_owner)
    K,masks,K_receipt=conservative_retained_guard(ctx.scene,ctx.provider,len(ctx.active_world_rows),np.ones(len(ctx.active_world_rows)),ctx.manifest.data['frames'])
    n=len(ctx.road['active']);worldK=np.zeros(n,bool);worldK[ctx.active_world_rows]=K
    worldNear=np.zeros(n,bool)
    for raw in FRAMES:
        packet,camera=ctx.base_packet(raw),ctx.camera(raw)
        z=(packet['means3D']@camera.world_view_transform[:3,2]+camera.world_view_transform[3,2]).cpu().numpy()*ctx.loader['unit_scale']
        worldNear[ctx.active_world_rows[(z>0)&(z<=.2)]]=True
    np.savez_compressed(output/'new_packet_guards.npz',world_K_guard=worldK,world_near_guard=worldNear)
    write_json(output/'new_packet_K.json',K_receipt);ctx.close();calls=5
    final=output/'asset_B32_guarded';record=build(parent,initialization,final,guard=output/'new_packet_guards.npz',pin=True)
    ctx=load_context(asset_dir=final,lease_file=lease_file,lease_owner=lease_owner)
    ids=read_npz(initialization)['proposed_world_rows'];results=[]
    for raw in FRAMES:
        before=read_npz(Path(native_parent)/f'{raw:03d}_native.npz')
        domain=evaluation_masks(raw)
        if raw in (86,110):
            domain['A1_remaining73']=before['fixed_low_S']&(before['road_alpha']<=.9)
            teacher=read_npz(CPU_CAMPAIGN/f'supervision_v4_candidate_r1/{raw:03d}.npz')
            related=(np.isin(teacher['triangle_world_rows'],ids)&(teacher['triangle_weights']>1e-7)).any(-1)
            domain['B32_related_remaining15']=domain['A1_remaining73']&related
        with torch.no_grad():r,j=ctx.render_road(raw),ctx.render_joint(raw)
        rgb,alpha,joint=r['rgb'].cpu().numpy(),r['alpha'][0].cpu().numpy(),j['rgb'].cpu().numpy()
        difference=np.abs(joint-before['joint_rgb']).max(0);Kmask=ctx.K_masks[raw]
        if (difference[Kmask]!=0).any():raise RuntimeError('B32 newpacket fulljoint K notexact')
        regions={name:dict(before=coverage_record(before['road_alpha'],mask),after=coverage_record(alpha,mask),
            newly_supported=int((mask&(before['road_alpha']<=.9)&(alpha>.9)).sum()),
            lost_support=int((mask&(before['road_alpha']>.9)&(alpha<=.9)).sum())) for name,mask in domain.items()}
        results.append(dict(raw=raw,regions=regions,K_pixels=int(Kmask.sum()),joint_K_RGB_max=float(difference[Kmask].max())))
        np.savez_compressed(output/f'{raw:03d}_native.npz',road_rgb=rgb,road_alpha=alpha,joint_rgb=joint,
            A1_road_alpha=before['road_alpha'],K=Kmask,**domain)
        for suffix,value in [('road',rgb),('joint',joint)]:
            Image.fromarray(np.clip(np.rint(value.transpose(1,2,0)*255),0,255).astype(np.uint8)).save(output/f'{raw:03d}_{suffix}.webp',lossless=True)
    reload=ctx.validate_public_reload(ctx.base_rgb,final,output/'public_reload.json')
    calls+=ctx.native_calls+5;ctx.assert_frozen();ctx.close()
    result=dict(schema='road_activation_B32_native.v1',GPU_executed=True,optimizer_updates=0,rows=results,
        source_parent_A1=str(parent),source_initialization=str(initialization),final_asset=str(final),activation=record,
        native_calls=calls,prior_campaign_calls=91,campaign_native_calls=91+calls,seconds=time.monotonic()-start,
        public_reload=reload,all_five_frame_joint_K_exact=True,production_promotion=False)
    write_json(output/'results.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['build','native'])
    p.add_argument('--initialization',type=Path,default=CAMPAIGN/'supervision_B32_r1/activation_pseudo_init.npz')
    p.add_argument('--parent',type=Path,default=CAMPAIGN/'native_r1/asset_A1_guarded')
    p.add_argument('--native-parent',type=Path,default=CAMPAIGN/'native_r1')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--guard',type=Path);p.add_argument('--pin',action='store_true')
    p.add_argument('--lease-file',type=Path,default=CAMPAIGN/'gpu_lease.json');p.add_argument('--lease-owner')
    a=p.parse_args()
    r=build(a.parent,a.initialization,a.output,guard=a.guard,pin=a.pin) if a.command=='build' else native(a.initialization,a.parent,a.native_parent,a.output,a.lease_file,a.lease_owner)
    print(json.dumps(r,ensure_ascii=False,allow_nan=False),flush=True)
