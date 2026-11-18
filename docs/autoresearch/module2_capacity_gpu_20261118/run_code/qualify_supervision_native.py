"""Gate1：原生A/C/S贡献、真实梯度、完整S覆盖上界与五帧K保护。

只写新GPU campaign的诊断证据；不优化、不改旧监督/资产/源码。
"""
from pathlib import Path
import argparse,gc,json,os,sys,time
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import numpy as np
from PIL import Image
import torch

REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
sys.path.insert(0,str(REPO/'scripts/worldsim_v82'))
from drivingeditor_dynamic_edit.persistent_road_handoff import load_persistent_handoff
from drivingeditor_dynamic_edit.native_row_contribution import attribute_prepared_rows
from drivingeditor_dynamic_edit.road_direct_teacher_training import DirectRoadParameters,raster,joint_rgb,conservative_retained_guard,masked_rgb_l1

FRAMES=(86,110,134,179,198)


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def write(path,data):Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def say(message):print(json.dumps(message,ensure_ascii=False),flush=True)


def coverage(mask,alpha,road_alphaT=None):
    result=dict(pixels=int(mask.sum()),gap=int((mask&(alpha<.05)).sum()),thin=int((mask&(alpha>=.05)&(alpha<=.9)).sum()),
        support=int((mask&(alpha>.9)).sum()),alpha_mean=float(alpha[mask].mean()) if mask.any() else None)
    if road_alphaT is not None:
        result['joint_road_alphaT']=dict(zero=int((mask&(road_alphaT==0)).sum()),positive_le_point05=int((mask&(road_alphaT>0)&(road_alphaT<=.05)).sum()),
            gt_point05=int((mask&(road_alphaT>.05)).sum()),mean=float(road_alphaT[mask].mean()) if mask.any() else None)
    return result


def prepared(state,key,maximum=False,opacity_only=False,scale_only=False):
    scene=state['scene'];camera=scene.camera_provider(key);packet=state['provider'].rows(key,camera,scene.pipe)
    packet={k:v.detach() if torch.is_tensor(v) else v for k,v in packet.items()}
    if maximum:
        model=state['model'];packet=model.packet(packet,'live')
        if not scale_only:
            op=packet['opacities'].detach().clone();op[model.opacity_ids]=1-1e-5;packet=dict(packet,opacities=op)
        if not opacity_only:
            sc=packet['scales'].detach().clone();sc[model.scale_ids,:2]=model.base_scales[model.scale_ids,:2]*model.scale_upper[:,None]
            packet=dict(packet,scales=sc)
    return {k:v.detach() if torch.is_tensor(v) else v for k,v in packet.items()},camera


def joint_layers(state,key,maximum=False):
    scene=state['scene'];values,owners,camera,_=scene._rows(key,(10,),(),False)
    values=dict(values,mask=values['mask']&~scene._contains(owners,(10,)))
    packet,_=prepared(state,key,maximum=maximum);n=len(packet['means3D']);offset=len(values['means3D'])-n
    if not torch.equal(values['means3D'][offset:],packet['means3D']):raise RuntimeError('road tail topology mismatch')
    for name in ['colors_precomp','opacities','scales']:
        value=values[name].detach().clone();value[offset:]=packet[name];values=dict(values,**{name:value})
    ns=len(scene.static.get_xyz)
    features=values['features'].detach().clone();features[:,2:]=0;features[:ns,2]=1;features[ns:offset,3]=1;features[offset:,4]=1
    out=scene._bindings().raster(camera,scene.pipe,0,dict(values,features=features),scene._background)
    rgb=out['rgb']+(1-out['alpha'])*scene.environment(camera.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
    return dict(rgb=rgb.detach().cpu().numpy(),alpha=out['alpha'][0].detach().cpu().numpy(),
        static_alphaT=out['extra'][0].detach().cpu().numpy(),dynamic_alphaT=out['extra'][1].detach().cpu().numpy(),road_alphaT=out['extra'][2].detach().cpu().numpy())


def grad_record(value):
    if value is None:return dict(connected=False,finite=True,max_abs=0.,nonzero_values=0)
    return dict(connected=True,finite=bool(torch.isfinite(value).all()),max_abs=float(value.abs().max()) if value.numel() else 0.,nonzero_values=int((value!=0).sum()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-manifest',type=Path,required=True);ap.add_argument('--proof',type=Path,required=True)
    ap.add_argument('--asset',type=Path,required=True);ap.add_argument('--supervision',type=Path,required=True);ap.add_argument('--legacy-scopes',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(101);np.random.seed(101);torch.cuda.reset_peak_memory_stats();start=time.time()
    receipt=dict(schema='road_capacity_native_gate1.v1',completed=False,optimizer_updates=0,formal_updates=0,
        source_asset=str(args.asset),supervision=str(args.supervision),scope_role='qualified_common_actual_alphaT_v4_ACS',engineering_attempt=1)
    write(args.output/'receipt.json',receipt);say(dict(stage='loading_native_source',source=str(args.source_manifest)))
    scene,manifest,loader=load_persistent_handoff(args.source_manifest,args.proof)
    if Path(manifest.road).resolve()!=(args.asset/'road.npz').resolve():raise RuntimeError('public source is not canonical oldS2')
    for component in [scene.static,scene.dynamic,scene.environment]:
        for p in component.parameters() if hasattr(component,'parameters') else ():p.requires_grad_(False)
    if not torch.equal(scene._background,torch.zeros_like(scene._background)):raise RuntimeError('road-only premult background must be black')
    provider=next(p for p in scene._providers if p.kind=='road');road=load(args.asset/'road.npz');prior=load(args.asset/'provenance.npz');legacy=load(args.legacy_scopes)
    active=np.flatnonzero(road['active']);n=len(active);world_n=len(road['active'])
    if not np.array_equal(active,legacy['active_world_rows']):raise RuntimeError('legacy active topology mismatch')
    ratio=np.maximum(1,np.minimum(legacy['absolute_sigma_upper_m'][:,0]/road['gaussian_sigma_m'][:,0],legacy['absolute_sigma_upper_m'][:,1]/road['gaussian_sigma_m'][:,1]))
    bbox,K_masks,K_receipt=conservative_retained_guard(scene,provider,n,ratio[active],manifest.data['frames'])
    guard=legacy['conservative_K_guard']|bbox
    if int(sum(mask.sum() for mask in K_masks.values()))!=77266:raise RuntimeError('fixed five-frame K denominator changed')
    physical=(prior['surface_supported']&prior['legal_write']&~prior['unknown']&~prior['guarded_world_rows']&(road['minimum_model_depth']<=0))[active]&~guard
    data={raw:load(args.supervision/f'{raw:03d}.npz') for raw in [86,110]}
    masses={name:np.zeros(n,np.float64) for name in ['A','C','S','A_full','C_full','S_full','S_gap','S_thin']}
    baseline={};packets={};attribution=[];baseline_match=[]
    for raw in [86,110]:
        camera=scene.camera_provider(raw);packet=provider.rows(raw,camera,scene.pipe);packet={k:v.detach() if torch.is_tensor(v) else v for k,v in packet.items()}
        with torch.no_grad():out=scene._bindings().raster(camera,scene.pipe,0,packet,scene._background)
        alpha=out['alpha'][0].detach().cpu().numpy();d=data[raw]
        regions=dict(A=d['A_mask'],C=d['C_hole_photo'],S=d['S_target_shadow_photo'],A_full=d['coverage_A'],C_full=d['coverage_C'],S_full=d['S_target_shadow'],
            S_gap=d['S_target_shadow']&(alpha<.05),S_thin=d['S_target_shadow']&(alpha>=.05)&(alpha<=.9))
        owners=torch.zeros(n,device='cuda',dtype=torch.long)
        att=attribute_prepared_rows(scene,packet,owners,camera,regions)
        for j,name in enumerate(att.region_names):masses[name]=np.maximum(masses[name],att.alpha_mass[:,j])
        np.savez_compressed(args.output/f'{raw:03d}_actual_contribution.npz',active_world_rows=active,alpha_mass=att.alpha_mass,region_names=np.asarray(att.region_names))
        baseline[raw]={k:out[k].detach() for k in ['rgb','alpha']};packets[raw]=packet
        baseline_match.append(dict(raw=raw,road_alpha_max_abs_vs_r3_reference=float(np.abs(alpha-d['fixed_initial_alpha']).max())))
        attribution.append(dict(raw=raw,region_pixels=att.region_pixels,conservation_relative_error=att.conservation_relative_error))
        say(dict(stage='attribution_complete',raw=raw,S_seen=int((att.alpha_mass[:,2]>0).sum()),conservation=att.conservation_relative_error))
    A,C,S=[masses[name]>0 for name in ['A','C','S']];known=prior['known'][active]
    eligible=physical&(A|S|(C&~known));S_only_known=physical&S&~A&~C&known
    opacity_ids=np.flatnonzero(physical);scale_ids=np.flatnonzero(physical&(ratio[active]>1.000001))
    packet=packets[86];model=DirectRoadParameters(packet['colors_precomp'],packet['opacities'],packet['scales'],np.flatnonzero(eligible),opacity_ids,scale_ids,ratio[active][scale_ids])
    state=dict(scene=scene,manifest=manifest,provider=provider,packets=packets,model=model)
    scope=dict(active_world_rows=active,world_row_ids=active,world_xyz=prior['world_xyz'][active],base_srgb=road['reference_srgb'][active],
        A_seen=A,C_seen=C,S_seen=S,known=known,physical_legal=physical,legacy_K_guard=legacy['conservative_K_guard'],native_K_guard=bbox,
        conservative_K_guard=guard,near_guard=road['minimum_model_depth'][active]>0,eligible_color=eligible,color=eligible,
        S_only_known=S_only_known,new_vs_legacy=eligible&~legacy['color'],actual_A_mass=masses['A'],actual_C_mass=masses['C'],actual_S_mass=masses['S'],
        actual_A_coverage_mass=masses['A_full'],actual_C_coverage_mass=masses['C_full'],actual_S_coverage_mass=masses['S_full'],
        maximum_coverage_opacity_rows=physical,maximum_coverage_scale_rows=physical&(ratio[active]>1.000001),absolute_sigma_upper_m=legacy['absolute_sigma_upper_m'],
        scope_role=np.asarray('qualified_common_actual_alphaT_v4_ACS'),scope_source=np.asarray(str((args.output/'receipt.json').resolve())))
    for name in ['A_seen','C_seen','S_seen','known','physical_legal','legacy_K_guard','native_K_guard','conservative_K_guard','near_guard','eligible_color','S_only_known']:
        value=np.zeros(world_n,bool);value[active]=scope[name];scope['world_'+name]=value
    np.savez_compressed(args.output/'qualified_scope.npz',**scope)
    np.savez_compressed(args.output/'K_masks.npz',**{f'{raw:03d}':K_masks[raw] for raw in FRAMES})
    scope_json=dict(schema='qualified_common_actual_alphaT_v4_ACS',source='canonical oldS2 road-only native raster',world_rows=world_n,active_rows=n,
        eligible_color_rows=int(eligible.sum()),legacy_color_rows=int(legacy['color'].sum()),new_vs_legacy_rows=int((eligible&~legacy['color']).sum()),
        A_seen_rows=int(A.sum()),C_seen_rows=int(C.sum()),S_seen_rows=int(S.sum()),S_only_known_rows=int(S_only_known.sum()),
        S_seen_known_eligible_rows=int((S&known&eligible).sum()),K_world_guard_rows=int(guard.sum()),legacy_K_guard_rows=int(legacy['conservative_K_guard'].sum()),
        recomputed_K_guard_equal_legacy=bool(np.array_equal(bbox,legacy['conservative_K_guard'])),
        formula='physical_legal & ~(legacy_K_guard | recomputed_max_tile_K_guard) & (A_seen | S_seen | (C_seen & ~known))',
        max_coverage_probe='all physical/legal/material-available near-unlocked K-unlocked active rows, opacity to1-1e-5 and inherited originalR8 absolute tangent sigma ceiling; read-only upper-bound diagnostic, not formal permission',
        immutable_observed_prior_ledger_changed=False,scope_ready_for_common_pair=True)
    write(args.output/'qualified_scope.json',scope_json);say(dict(stage='qualified_scope_ready',**scope_json))
    gradients=[]
    for raw in [86,110]:
        d=data[raw];out=raster(state,raw,'photo');alpha=out['alpha'][0].detach()
        for role,maskkey,targetkey,qkey in [('A','A_mask','target_A_srgb','q_A'),('C','C_hole_photo','target_C_srgb','q_C_hole'),('S','S_target_shadow_photo','target_S_srgb','q_S')]:
            target=torch.tensor(d[targetkey],device='cuda');mask=torch.tensor(d[maskkey],device='cuda');q=torch.tensor(d[qkey],device='cuda')
            loss=masked_rgb_l1(out['rgb'],alpha[None]*target,mask,q)
            grads=torch.autograd.grad(loss,[model.colors,model.opacity_logits,model.scale_factor],retain_graph=True,allow_unused=True)
            row=dict(raw=raw,route='photo',role=role,loss=float(loss),RGB=grad_record(grads[0]),opacity=grad_record(grads[1]),scale=grad_record(grads[2]))
            if not row['RGB']['finite'] or row['RGB']['nonzero_values']==0 or grads[1] is not None or grads[2] is not None:raise RuntimeError('native photo gradient route failed')
            gradients.append(row)
        del out
        out=raster(state,raw,'coverage');mask=torch.tensor(d['S_target_shadow'],device='cuda');loss=((out['alpha'][0]-1).square()*mask).sum()/mask.sum().clamp_min(1)
        grads=torch.autograd.grad(loss,[model.colors,model.opacity_logits,model.scale_factor],allow_unused=True)
        row=dict(raw=raw,route='coverage_diagnostic',role='full_S',loss=float(loss),RGB=grad_record(grads[0]),opacity=grad_record(grads[1]),scale=grad_record(grads[2]))
        if grads[0] is not None or not row['opacity']['finite'] or not row['scale']['finite']:raise RuntimeError('native coverage gradient route failed')
        gradients.append(row);del out
    write(args.output/'native_gradients.json',dict(rows=gradients,optimizer_updates=0,all_routes_pass=True))
    summaries=[];K_checks=[]
    for raw in FRAMES:
        with torch.no_grad():
            baseline_rgb,_=joint_rgb(state,raw,apply_model=False)
            mask=torch.tensor(K_masks[raw],device='cuda')
            for maximum,color in [(False,None),(False,0.),(False,1.),(True,None),(True,0.),(True,1.)]:
                rgb,_=joint_rgb(state,raw,maximum=maximum,color_stress=color)
                difference=(rgb-baseline_rgb).abs().max(0).values
                row=dict(raw=raw,maximum_coverage=maximum,color_stress=color,K_pixels=int(mask.sum()),K_float_max_abs=float(difference[mask].max()),
                    K_any_float_change_pixels=int((difference[mask]>0).sum()),K_gt_one_byte_pixels=int((difference[mask]>1/255).sum()))
                K_checks.append(row)
                if row['K_float_max_abs']!=0:raise RuntimeError('full K native hard protection failed')
            if raw in [86,110]:
                d=data[raw];layers0=joint_layers(state,raw);layers1=joint_layers(state,raw,maximum=True)
                states={};saved=dict(full_S=d['S_target_shadow'],photo_S=d['S_target_shadow_photo'],**{'baseline_joint_'+k:v for k,v in layers0.items()},**{'maximum_joint_'+k:v for k,v in layers1.items()})
                for label,oponly,sconly in [('baseline',False,False),('opacity_max',True,False),('tangent_scale_max',False,True),('combined_max',False,False)]:
                    packet,camera=prepared(state,raw,maximum=label!='baseline',opacity_only=oponly,scale_only=sconly)
                    out=scene._bindings().raster(camera,scene.pipe,0,packet,scene._background);a=out['alpha'][0].cpu().numpy()
                    saved[label+'_road_only_alpha']=a
                    states[label]=dict(full_S=coverage(d['S_target_shadow'],a,layers0['road_alphaT'] if label=='baseline' else layers1['road_alphaT'] if label=='combined_max' else None),
                        photo_S=coverage(d['S_target_shadow_photo'],a),baseline_gaps_closed=int((d['S_target_shadow']&(d['fixed_initial_alpha']<.05)&(a>=.05)).sum()),
                        low_reference_reaches_support=int((d['S_target_shadow']&(d['fixed_initial_alpha']<=.9)&(a>.9)).sum()))
                np.savez_compressed(args.output/f'{raw:03d}_coverage_probe.npz',**saved)
                summaries.append(dict(raw=raw,states=states,baseline_joint_conservation_max=float(np.abs(layers0['static_alphaT']+layers0['dynamic_alphaT']+layers0['road_alphaT']-layers0['alpha']).max()),
                    maximum_joint_conservation_max=float(np.abs(layers1['static_alphaT']+layers1['dynamic_alphaT']+layers1['road_alphaT']-layers1['alpha']).max())))
        say(dict(stage='joint_K_and_coverage_complete',raw=raw))
    scene.assert_persistent_unchanged()
    write(args.output/'K_protection.json',dict(fixed_five_frame_pixels=77266,all_exact_zero=True,rows=K_checks,maximum_native_tile_guard=K_receipt))
    write(args.output/'full_S_coverage_control.json',dict(rows=summaries,thresholds=dict(gap='alpha<.05',thin='.05<=alpha<=.9',support='alpha>.9'),
        optimizer_updates=0,assets_written=False,maximum_is_inherited_absolute_bound_not_compounded=True,RGB_divided_by_alpha=False,
        control_interpretation='remaining full-S coverage/occlusion limits are not evidence of insufficient 2D generator capacity'))
    receipt.update(completed=True,public_loader=loader,scope=scope_json,attribution=attribution,baseline_match=baseline_match,
        native_gradient_routes_pass=True,K_full_five_frame_exact_zero=True,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),
        seconds=time.time()-start,scene_persistent_unchanged=True,old_supervision_modified=False,old_source_modified=False,
        rendering_operations='actual native diagnostic only',formal_budget_used=0,default_asset_written=False)
    write(args.output/'receipt.json',receipt)
    say(dict(stage='gate1_complete',seconds=receipt['seconds'],eligible_color=scope_json['eligible_color_rows'],peak_bytes=receipt['peak_GPU_allocated_bytes']))
    del state,model,scene,provider,packets,baseline,packet;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
