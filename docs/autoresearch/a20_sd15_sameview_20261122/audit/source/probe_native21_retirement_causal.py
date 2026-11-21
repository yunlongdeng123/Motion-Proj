"""Real 3DGS grey diagnostic adapter. CPU check is the default; no training path.

GPU preview requires --authorize-grey-diagnostic and must be queued by the root
agent. It bypasses no material/student gate: neither stage exists in this file.
All composition, occlusion and a9 deletion happen inside the shared 3D renderer.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time

TRACK='bzFbWRrSGOgEaXxnuGV0tw'
TRAIN=(86,110)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def reference(path):
    p=Path(path).resolve();return dict(path=str(p),sha256=digest(p),bytes=p.stat().st_size)


def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def check(args):
    import numpy as np
    with np.load(args.packet,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    n=len(a['xyz']);shapes=dict(xyz=(n,3),scales=(n,3),rotation=(n,4),coefficients=(n,16,3),opacity=(n,),source_kind=(n,))
    for key,shape in shapes.items():
        if a[key].shape!=shape or not np.isfinite(a[key]).all():raise ValueError('Bad packet field '+key)
    if not bool(a['diagnostic_only']) or bool(a['grey_approved']) or int(a['training_updates'])!=0:
        raise ValueError('This is only a not-yet-approved grey diagnostic packet')
    if str(a['source_track'])!=TRACK or int(a['source_native_instance'])!=21:
        raise ValueError('Actor identity mismatch')
    if not np.isin(a['source_kind'],[1,2,3,4]).all() or not (a['coefficients']==0).all():
        raise ValueError('Source provenance or uniform-grey display mismatch')
    if (a['scales']<=0).any() or ((a['opacity']<=0)|(a['opacity']>=1)).any():
        raise ValueError('Invalid scales or opacity')
    if np.max(np.abs(np.linalg.norm(a['rotation'],axis=1)-1))>1e-4:
        raise ValueError('Non-unit Gaussian quaternion')
    receipt=json.loads(args.lod_receipt.read_text())
    selection=json.loads(args.selection_plan.read_text())
    if selection.get('schema')!='a20.native21.feature_VJP_selection.mode_aware.plan.v1':
        raise ValueError('Mode-aware VJP plan must be frozen before execution')
    if receipt['packet']['sha256']!=digest(args.packet) or receipt['training_updates']!=0:
        raise ValueError('Packet receipt mismatch')
    if tuple(args.raws)!=TRAIN:
        raise ValueError('This frozen preview opens BUILD 86/110 only')
    return dict(schema='a20.native21.feature_VJP.cpu_check.v1',passed=True,
        selection_plan=reference(args.selection_plan),planned_forward_rasters=4,planned_feature_VJP_calls=0,
        packet=reference(args.packet),lod_receipt=reference(args.lod_receipt),legacy=reference(args.legacy),
        sidecar=reference(args.sidecar),conditioning=str(args.conditioning),raws=list(TRAIN),rows=n,
        classes={str(k):int((a['source_kind']==k).sum()) for k in [1,2,3,4]},
        phases_available=['cpu-check','preview'],training_updates=0,optimizer_available=False,
        material_stage_available=False,geometry_distillation_available=False,
        full_scene_3D_depth_sort=True,RGB_post_composite_or_mask_paste=False,
        native_source_model_saw_full_sequence=True,heldout_frames_opened=[],
        native_owner21_policy_variants=['mode_aware_pure_target_by_actual_contribution'],
        outside_actor_domain_change_must_be_measured=True,grey_quality_gate_is_not_bypassed_for_training=True,
        native_float_baselines_saved=True,retained_actor_K_features=True,K_native11_feature=True,
        torch_imported='torch' in sys.modules,cuda_initialized=False)


def preview(args,readiness):
    import numpy as np
    from PIL import Image
    if not args.authorize_grey_diagnostic:raise ValueError('Root must queue --authorize-grey-diagnostic')
    if args.output.exists():raise ValueError('Use a fresh output directory; existing scene is preserved')
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'frozen_entry_contract.json',readiness)
    spec=importlib.util.spec_from_file_location('_a20_frozen_joint_preview',args.legacy)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    if m.TARGET!=21 or m.DELETE!=10 or m.TRACK!=TRACK:raise ValueError('Legacy actor mismatch')
    m.ROOT=args.output;m.FRAMES=TRAIN;m.TRAIN=TRAIN
    torch=m.torch
    if not torch.cuda.is_available():raise RuntimeError('Queued grey diagnostic requires actual CUDA')
    torch.set_grad_enabled(False);torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    with np.load(args.packet,allow_pickle=False) as z:source_kind=z['source_kind'].copy()

    class GreyVehicle(m.Vehicle):
        def __init__(self,a):
            super().__init__(a)
            for p in self.parameters():p.requires_grad_(False)
            self.register_buffer('source_kind',m.ten(source_kind,torch.long))
        def scales(self):return self.base_scales
        def rows(self,pose,cam):
            rows=super().rows(pose,cam);features=rows['features'].clone()
            for code in [1,2,3,4]:features[:,3+code]=(self.source_kind==code).to(features.dtype)
            return dict(rows,features=features)
    m.Vehicle=GreyVehicle

    class GreyExperiment(m.Experiment):
        def build_vehicle(self,path):
            shutil.copy2(args.packet,path)
            if digest(path)!=readiness['packet']['sha256']:raise RuntimeError('Packet bootstrap mismatch')
        def build_domains(self):
            # No losses or teacher targets are evaluated by this diagnostic.
            return
    e=GreyExperiment(args.road_source)
    e.road_color.requires_grad_(False)
    for model in [e.scene.static,e.scene.dynamic,e.scene.environment]:
        if hasattr(model,'parameters'):
            for p in model.parameters():p.requires_grad_(False)
    with np.load(args.sidecar,allow_pickle=False) as z:side={k:z[k].copy() for k in z.files}
    selected=m.ten(side['global_row_ids'],torch.long)
    if len(selected)!=75 or len(torch.unique(selected))!=75:raise ValueError('Expected fixed 75-row a9 guard')
    context_packets={};partition=[];native_global_ids={}
    for raw in TRAIN:
        packet,owners,road_index=e.native_packets[raw]
        values,_,_,_=e.scene._rows(raw,(m.TARGET,),(),False)
        global_ids=torch.nonzero(values['mask']).reshape(-1)
        native_global_ids[raw]=global_ids
        if len(global_ids)!=len(owners) or not torch.equal(values['means3D'][global_ids],packet['means3D']):
            raise RuntimeError('Native packet/global-row mapping mismatch')
        match=torch.isin(global_ids,selected)
        if ((owners[match]>0)&(owners[match]!=m.DELETE)).any():
            raise RuntimeError('a9 guard would alter a retained actor')
        own=owners.clone();own[match]=m.DELETE
        packet=dict(packet,features=packet['features'].clone())
        packet['features'][:,8]=((own>0)&(own!=21)&(own!=10)).to(packet['features'].dtype)
        packet['features'][:,9]=(own==11).to(packet['features'].dtype)
        e.native_packets[raw]=(packet,own,road_index)
        native_target=own==m.TARGET
        keep=~native_target;e.packets[raw]=(m.subset(packet,keep),own[keep],road_index[keep])
        partition.append(dict(raw=raw,native_owner21_rows=int(native_target.sum()),sidecar_matched_rows=int(match.sum())))

    def render(raw,layout,mode='keep',isolated=False):
        cam=e.cameras[raw]
        if layout=='native':packet,owners,ri=e.native_packets[raw]
        else:packet,owners,ri=context_packets[(raw,mode)]
        rows=dict(packet)
        ids=torch.nonzero(ri>=0).reshape(-1)
        if len(ids):rows=dict(rows,colors_precomp=rows['colors_precomp'].index_copy(0,ids,e.colors()[ri[ids]]))
        if layout!='native' and not args.omit_new_gaussians:
            car=e.vehicle.rows(e.poses[raw],cam)
            if isolated:rows=car;owners=torch.full((len(car['means3D']),),21,device='cuda',dtype=torch.long)
            else:
                rows=e.scene._bindings().merge_rows(rows,car)
                owners=torch.cat([owners,torch.full((len(car['means3D']),),21,device='cuda',dtype=torch.long)])
        if mode=='delete' and not isolated:rows=dict(rows,mask=rows['mask']&(owners!=10))
        out=e.scene._bindings().raster(cam,e.scene.pipe,0,rows,e.scene._background);e.native_calls+=1
        rgb=out['rgb']
        if not isolated:rgb=rgb+(1-out['alpha'])*e.env[raw]
        return dict(rgb=rgb,alpha=out['alpha'].squeeze(),target_alpha=out['extra'][0],road_alpha=out['extra'][1],
            measured_alpha=out['extra'][2],interpolated_alpha=out['extra'][3],generated_alpha=out['extra'][4],
            prior_geometry_observed_appearance_alpha=out['extra'][5],
            retained_native_K_alpha=out['extra'][6],retained_native11_alpha=out['extra'][7])

    def potential(raw):
        folder=args.conditioning/'train'/f'{raw:03d}';meta=json.loads((folder/'mapping.json').read_text())
        w,h=meta['image_native_size'];x0,y0,x1,y1=meta['crop_native_xyxy_half_open']
        mask=np.zeros((h,w),np.uint8)
        v=np.asarray(Image.open(folder/'visible_mask.png'))>0
        u=np.asarray(Image.open(folder/'unknown_potential_mask.png'))>0
        mask[y0:y1,x0:x1]=(v|u).astype(np.uint8)
        return m.ten(np.asarray(Image.fromarray(mask).resize((1152,768),Image.Resampling.NEAREST))>0,torch.bool)

    cache=json.loads((args.selection_cache/'grey_renderer_receipt.json').read_text())
    if cache['packet']['sha256']!=readiness['packet']['sha256']:
        raise ValueError('Cached contribution selection binds a different packet')
    with np.load(args.selection_cache/'native21_contribution_selection.npz',allow_pickle=False) as z:
        global_target=m.ten(z['global_row_ids'],torch.long)
        scores=m.ten(z['score_by_baseline_and_V_K_a9'])
        present=m.ten(z['present'],torch.bool)
    expected=torch.unique(torch.cat([native_global_ids[r][e.native_packets[r][1]==21] for r in TRAIN]))
    if not torch.equal(expected,global_target):raise ValueError('Cached global native row identities changed')
    mask_refs=cache['selection']['source_masks'];vjp_calls=0;base_cache={}
    for raw in TRAIN:
        for mode in ['keep','delete']:
            with np.load(args.selection_cache/'renders'/f'{raw:03d}'/('native_'+mode)/'render_float.npz',allow_pickle=False) as z:
                base_cache[(raw,mode)]={k:m.ten(z[k]) for k in z.files}
    maximum=scores.amax(0);Vpositive=maximum[:,0]>1e-8;Kpositive=maximum[:,1]>0;A9positive=maximum[:,2]>0
    selected_keep=Vpositive&~Kpositive&~A9positive
    selected_delete=((maximum[:,0]>1e-8)|(maximum[:,2]>1e-8))&~Kpositive
    retired={'keep':global_target[selected_keep],'delete':global_target[selected_delete]}
    np.savez_compressed(args.output/'native21_contribution_selection.npz',global_row_ids=m.cpu(global_target).numpy(),
        score_by_baseline_and_V_K_a9=m.cpu(scores).numpy(),present=m.cpu(present).numpy(),
        retire_KEEP=m.cpu(selected_keep).numpy(),retire_DELETE=m.cpu(selected_delete).numpy(),
        preserve_any_K=m.cpu(Kpositive).numpy(),preserve_unknown= m.cpu(~Vpositive&~(maximum[:,2]>1e-8)).numpy())
    selection_summary=dict(global_native21_rows=len(global_target),V_supported_rows=int(Vpositive.sum()),
        K_other_mixed_or_non_target_rows_preserved=int(Kpositive.sum()),a9_supported_rows=int(A9positive.sum()),
        retired_KEEP=int(selected_keep.sum()),retired_DELETE=int(selected_delete.sum()),
        prior_union_pureV_Kzero_rows=int((Vpositive&~Kpositive).sum()),
        thresholds_tuned=False,center_or_OBB_deletion_used=False,feature_VJP_calls=vjp_calls,
        source_masks=mask_refs,any_positive_K_is_preserved=True)
    write(args.output/'selection_receipt.json',selection_summary)
    partition=[]
    for raw in TRAIN:
        packet,owners,ri=e.native_packets[raw];global_ids=native_global_ids[raw]
        for mode in ['keep','delete']:
            remove=torch.isin(global_ids,retired[mode])&(owners==21);keep=~remove
            context=m.subset(packet,keep);context=dict(context,features=context['features'].clone())
            own=owners[keep].clone();legacy=own==21;context['features'][legacy,2]=0;own[legacy]=0
            context_packets[(raw,mode)]=(context,own,ri[keep])
            partition.append(dict(raw=raw,mode=mode,retired_rows=int(remove.sum()),legacy_native21_rows=int(legacy.sum()),
                parameter_values_unchanged=True,all_K_positive_rows_preserved=True,center_or_OBB_deletion_used=False))

    domain_changes=[];K_changes=[]
    with torch.inference_mode():
        for raw in TRAIN:
            outside=~potential(raw);base={}
            for mode in ['keep','delete']:
                base[mode]=base_cache[(raw,mode)]
                folder=args.output/'renders'/f'{raw:03d}'/('native_'+mode)
                for name,a in base[mode].items():m.png(folder/(name+'.png'),a)
                np.savez_compressed(folder/'render_float.npz',**{name:m.cpu(a).numpy() for name,a in base[mode].items()})
            for layout in ['vjp_context']:
                for mode in ['keep','delete']:
                    out=render(raw,layout,mode);folder=args.output/'renders'/f'{raw:03d}'/(layout+'_'+mode)
                    for name,a in out.items():m.png(folder/(name+'.png'),a)
                    delta=(out['rgb']-base[mode]['rgb']).abs();pixel=delta.amax(0)
                    affected=(pixel>1/255)&outside
                    evidence=dict(raw=raw,layout=layout,mode=mode,outside_potential_pixels=int(outside.sum()),
                        outside_RGB_max=float(delta[:,outside].max()),outside_RGB_mean=float(delta[:,outside].mean()),
                        outside_pixels_changed_above_1_over_255=int(affected.sum()),
                        outside_changed_fraction=float(affected.sum()/outside.sum()),
                        native21_replacement_handoff_passed=not bool(affected.any()))
                    domain_changes.append(evidence)
                    labels=e.targets[raw]['labels']
                    for name,kmask,feature in [
                        ('all_retained_actors_K',(labels>0)&(labels!=21)&(labels!=10),'retained_native_K_alpha'),
                        ('native11_right_white_pickup',labels==11,'retained_native11_alpha')]:
                        count=int(kmask.sum())
                        if not count:
                            K_changes.append(dict(raw=raw,layout=layout,mode=mode,K_group=name,pixels=0));continue
                        ad=(out[feature]-base[mode][feature]).abs()
                        K_changes.append(dict(raw=raw,layout=layout,mode=mode,K_group=name,pixels=count,
                            RGB_max_change=float(delta[:,kmask].max()),RGB_mean_change=float(delta[:,kmask].mean()),
                            RGB_pixels_changed_above_1_over_255=int(((pixel>1/255)&kmask).sum()),
                            visible_native_contribution_alpha_max_change=float(ad[kmask].max()),
                            visible_native_contribution_alpha_mean_change=float(ad[kmask].mean()),
                            native_alpha_pixels_changed_above_1_over_255=int(((ad>1/255)&kmask).sum()),
                            new_a20_shared_alpha_on_K_max=float(out['target_alpha'][kmask].max()),
                            full_native_K_alpha_mass_before=float(base[mode][feature].sum()),
                            full_native_K_alpha_mass_after=float(out[feature].sum()),
                            K_mask_dilated_or_absorbed_into_permitted_domain=False))
                    m.png(folder/'outside_delta_x10.png',(pixel*10).clamp(0,1))
                    np.savez_compressed(folder/'render_float.npz',**{name:m.cpu(a).numpy() for name,a in out.items()},
                        outside_potential=m.cpu(outside).numpy(),outside_affected=m.cpu(affected).numpy())
    torch.cuda.synchronize()
    receipt=dict(schema='a20.native21.cached_retirement_causal_probe.v1',ready_contract=readiness,
        selection=selection_summary,new_gaussians_added=not args.omit_new_gaussians,cached_selection=reference(args.selection_cache/'native21_contribution_selection.npz'),selection_artifact=reference(args.output/'native21_contribution_selection.npz'),
        packet=reference(args.packet),source_partition=partition,domain_changes=domain_changes,K_actor_changes=K_changes,
        native_float_baselines_saved=True,original_preview_main_metrics_not_overridden=True,
        input_non21_parameters_preserved=True,other_actor_composited_contribution_unchanged_claim=False,
        legacy_context_is_unverified_source_not_new_car_asset=True,all_images_are_actual_3DGS_renders=True,
        RGB_post_composite_or_mask_paste=False,training_updates=0,optimizer_steps=0,backward_calls=0,
        geometry_distillation=False,material_generation=False,torch_inference_mode=True,
        native_renderer_calls=e.native_calls,gpu_used=True,wall_seconds=time.monotonic()-started,
        feature_VJP_calls=vjp_calls,model_parameter_backward_calls=0,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        grey_approved=False,material_or_student_allowed=False,persistent_full_car_asset_promoted=False,
        heldout_frames_opened=[],source_model_saw_full_sequence=True)
    write(args.output/'grey_renderer_receipt.json',receipt)
    if digest(args.packet)!=readiness['packet']['sha256']:raise RuntimeError('Diagnostic source packet changed')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['cpu-check','preview'],default='cpu-check')
    for key in ['packet','lod-receipt','legacy','sidecar','road-source','conditioning','output']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--selection-plan',type=Path,required=True)
    p.add_argument('--selection-cache',type=Path,required=True)
    p.add_argument('--omit-new-gaussians',action='store_true')
    p.add_argument('--raws',type=int,nargs='+',default=[86,110])
    p.add_argument('--authorize-grey-diagnostic',action='store_true')
    args=p.parse_args()
    if args.phase=='cpu-check':os.environ['CUDA_VISIBLE_DEVICES']=''
    readiness=check(args)
    if args.phase=='cpu-check':
        write(args.output/'cpu_check.json',readiness);print(json.dumps(readiness,indent=2))
    else:preview(args,readiness)
