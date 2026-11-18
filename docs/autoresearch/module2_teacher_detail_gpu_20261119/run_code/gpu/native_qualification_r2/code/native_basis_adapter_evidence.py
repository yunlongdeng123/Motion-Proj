"""未来smoke专用24-feature权重导出器；默认CPU inspect，当前合同禁止native。

导入不加载Torch/native。不存在formal模式，不调度后台任务。
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np

Q=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1')


class NativeExecutionBlocked(RuntimeError):pass


def scope_signature(active_world_rows,eligible):
    ids=np.asarray(active_world_rows,np.int64);mask=np.asarray(eligible)
    if mask.dtype!=bool or mask.shape!=ids.shape:raise ValueError('显式eligible须匹配active world rows')
    return hashlib.sha256(ids[mask].astype('<i8').tobytes()).hexdigest()


def require_future_smoke(authorization,*,raw=None,scope_sha=None):
    if authorization is None:raise NativeExecutionBlocked('CPU准备阶段没有native/GPU授权')
    a=json.loads(Path(authorization).read_text())if isinstance(authorization,(str,Path))else dict(authorization)
    if a.get('schema')!='pixel_native_basis_smoke.v1'or a.get('phase')!='native_smoke'or a.get('native_allowed')is not True or a.get('gpu_authorized')is not True or a.get('formal_allowed')is not False:
        raise NativeExecutionBlocked('须由root在用户开GPU后登记独立smoke合同；本轮不执行native')
    if raw is not None and raw not in a.get('frames',[]):raise NativeExecutionBlocked('frame未获smoke授权')
    if scope_sha is not None and a.get('scope_world_rows_sha256')!=scope_sha:raise NativeExecutionBlocked('授权scope与实际eligible身份不同')
    if not isinstance(a.get('max_native_forward_calls'),int)or a['max_native_forward_calls']<=0:raise NativeExecutionBlocked('缺显式调用预算')
    return a


def frozen_baseline_override(source_road_rgb,full_rgb,road_flag,eligible,candidate):
    """纯tensor资格检查；候选只能改eligible，道路readonly与其他分支逐值保留。"""
    import torch
    fixed=candidate.detach().to(device=full_rgb.device,dtype=full_rgb.dtype)
    if fixed.shape!=source_road_rgb.shape:raise ValueError('road baseline override shape不同')
    if not bool(torch.isfinite(fixed).all())or bool(((fixed<0)|(fixed>1)).any()):raise ValueError('road baseline须为finite[0,1]')
    local=eligible[road_flag]
    if not torch.equal(fixed[~local],source_road_rgb[~local]):raise ValueError('baseline override试图改readonly road ci')
    output=full_rgb.clone();indices=torch.nonzero(road_flag,as_tuple=False)[:,0]
    output[indices[local]]=fixed[local]
    return output


def _one_pass(ctx,raw,rows,full_eligible,road_flag,mode,baseline_colors=None):
    import torch
    camera=ctx.camera(raw);scene=ctx.scene
    rows={k:v.detach()if torch.is_tensor(v)else v for k,v in rows.items()}
    if rows['colors_precomp']is None or rows['shs']is not None:raise ValueError('必须用已求值direct RGB，不允许再走SH clamp')
    c=rows['colors_precomp'];e=torch.as_tensor(full_eligible,device=c.device,dtype=torch.bool)
    if e.shape!=(len(c),)or road_flag.shape!=e.shape:raise ValueError('native row scope/topology不符')
    if baseline_colors is not None:
        c=frozen_baseline_override(ctx.base_rgb,c,road_flag,e,baseline_colors);rows=dict(rows,colors_precomp=c)
    features=c.new_zeros((len(c),24));features[:,:2]=rows['features'][:,:2]
    features[:,2:5]=c*e[:,None]          # extra0:3 eligible RGB
    features[:,5]=e.to(c.dtype)          # extra3 W_e: actual same-sort alpha*T
    features[:,6:9]=c*(~e)[:,None]       # extra4:7 direct readonly RGB
    features[:,9]=rows['features'][:,0]*e # extra7 eligible depthmass_model
    features[:,10]=rows['features'][:,0].square()*e
    features[:,11]=road_flag.to(c.dtype) # extra9 whole-road alphaT, diagnostic only
    features[:,12]=(~e).to(c.dtype)      # extra10 readonly unit weight
    ctx.native_calls+=1
    out=scene._bindings().raster(camera,scene.pipe,0,dict(rows,features=features),scene._background)
    if out['extra'].shape[0]!=22:raise RuntimeError('不是已登记24-feature ABI')
    env=torch.zeros_like(out['rgb'])
    if mode=='joint'and scene.environment is not None:
        env=(1-out['alpha'])*scene.environment(camera.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
    extra=out['extra'];baseline=out['rgb']+env;eligible_rgb=extra[:3]
    direct_readonly=extra[4:7]+env
    rgb_error=float((direct_readonly+eligible_rgb-baseline).abs().max())
    alpha_error=float((extra[3:4]+extra[10:11]-out['alpha']).abs().max())
    if max(rgb_error,alpha_error)>5e-6:raise RuntimeError('same-pass RGB/alpha分解守恒未过')
    K=torch.as_tensor(ctx.K_masks[raw],device=c.device,dtype=torch.bool)
    Kweight=float(extra[3][K].abs().max())
    # 不在这里将W[K]乘0，也不copy父joint图。scope错误必须返回调用者修复。
    # Joint K hard gate runs in owner runner immediately after raw arrays are persisted.
    arrays=dict(baseline_rgb=baseline,eligible_rgb=eligible_rgb,eligible_weight=extra[3],
        readonly_rgb=baseline-eligible_rgb,direct_readonly_rgb=direct_readonly,alpha=out['alpha'][0],
        entire_road_weight=extra[9],readonly_weight=extra[10],eligible_depth_mass_model=extra[7],
        eligible_depth2_mass_model2=extra[8],K=K)
    return {k:v.detach().cpu().numpy()for k,v in arrays.items()},dict(mode=mode,K_eligible_weight_max=Kweight,
        RGB_conservation_max=rgb_error,alpha_conservation_max=alpha_error,native_forward_calls=1)


def export_native_basis(ctx,raw,eligible_active,output,*,authorization=None,delete_actor_ids=(10,),baseline_colors=None):
    """授权后的未来实际API；本轮在require_future_smoke处阻断，未调用ctx。

    每个view/edit重新取全场native rows并排序。road/joint各一次，scope只选road子集。
    """
    auth=require_future_smoke(authorization) # 在任何Torch/native import、ctx访问之前
    if baseline_colors is not None and auth.get('allow_baseline_color_override')is not True:raise NativeExecutionBlocked('默认回退原ci；baseline override仅可由smoke显式登记')
    signature=scope_signature(ctx.active_world_rows,eligible_active)
    require_future_smoke(auth,raw=int(raw),scope_sha=signature)
    if tuple(delete_actor_ids)!=tuple(auth.get('delete_actor_ids',[10])):raise NativeExecutionBlocked('编辑请求未绑定')
    ctx.require_lease();ctx.assert_frozen()
    if ctx.native_calls+2>auth['max_native_forward_calls']:raise NativeExecutionBlocked('导出将超smoke native调用预算')
    import torch
    if not torch.equal(ctx.scene._background,torch.zeros_like(ctx.scene._background)):raise ValueError('当前导出合同要求黑底native，环境单独留readonly')
    output=Path(output)
    if output.exists():raise FileExistsError('保留旧导出，使用新目录')
    road=ctx.base_packet(raw);n=len(ctx.active_world_rows);mask=np.asarray(eligible_active,bool)
    if baseline_colors is not None and baseline_colors.shape!=ctx.base_rgb.shape:raise ValueError('共同base颜色须为Nactive×3')
    rd_flag=torch.ones(n,device=road['means3D'].device,dtype=torch.bool)
    road_arrays,road_receipt=_one_pass(ctx,raw,road,mask,rd_flag,'road_only',baseline_colors)
    scene=ctx.scene;rows,owners,camera,_=scene._rows(raw,(),(),False)
    rows=dict(rows,mask=rows['mask']&~scene._contains(owners,tuple(delete_actor_ids)))
    start=len(rows['means3D'])-n
    if start<0 or not torch.equal(rows['means3D'][start:],road['means3D']):raise ValueError('当前scene road尾部与active身份不一致，不能猜offset')
    e=np.zeros(len(rows['means3D']),bool);e[start:]=mask
    rflag=torch.zeros(len(e),device=road['means3D'].device,dtype=torch.bool);rflag[start:]=True
    joint_arrays,joint_receipt=_one_pass(ctx,raw,rows,e,rflag,'joint',baseline_colors)
    ctx.assert_frozen();output.mkdir(parents=True)
    np.savez_compressed(output/'road_basis.npz',**road_arrays)
    np.savez_compressed(output/'joint_basis.npz',**joint_arrays)
    np.savez_compressed(output/'row_scope.npz',active_world_rows=ctx.active_world_rows,eligible_active=mask)
    meta=dict(schema='native_eligible_pixel_material_basis.v1',renderer_mode='eligible_native_weights_pixel_world',raw=int(raw),
        source_asset=str(ctx.asset_dir),scope_world_rows_sha256=signature,eligible_rows=int(mask.sum()),
        readonly_road_rows=int((~mask).sum()),delete_actor_ids=list(delete_actor_ids),
        baseline_color_override_used=baseline_colors is not None,
        fallback_color_source='ctx original saved road ci'if baseline_colors is None else'explicit frozen eligible-only smoke override; readonly remains original',
        camera=dict(width=camera.image_width,height=camera.image_height,world_view_transform=camera.world_view_transform.detach().cpu().tolist(),
            full_proj_transform=camera.full_proj_transform.detach().cpu().tolist()),
        meter_to_model=float(ctx.road['meter_to_model']),geometry_sigma_opacity_fixed=True,
        road_weight='eligible road-only alphaT; never substitute joint weight',joint_weight='eligible same full-scene depth-sort alphaT',
        pre_output_clamp=False,feature_ABI=24,road=road_receipt,joint=joint_receipt,native_forward_calls=2,
        cache_is_world_asset=False,new_view_or_edit_requires_reexport=True,phase='native_smoke',formal_allowed=False)
    (output/'basis.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n');return meta


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=['inspect','export'],default='inspect')
    parser.add_argument('--authorization',type=Path);args=parser.parse_args()
    if args.mode=='inspect':
        print(json.dumps(dict(schema='pixel_basis_adapter_CPU_preparation.v1',native_default=False,formal_mode_exists=False,
            required_feature_channels=13,native_ABI_channels=24,road_joint_passes_per_view=2,
            future_API='export_native_basis(ctx, raw, eligible_active, output, authorization=smoke_contract)',
            weights_cache_is_asset=False)));return
    require_future_smoke(args.authorization)
    raise NativeExecutionBlocked('CLI不构造GPU context。未来root授权smoke入口显式创建ctx并调用此API；没有自动队列。')


if __name__=='__main__':main()
