"""多Actor配对编辑与动态分支诊断。所有分层来自模型贡献，不乘GT body mask。"""
from pathlib import Path
import math
import numpy as np
import torch
from .raster import reference_raster, cuda_raster
from .math3d import normalize, quaternion_matrix
from .evaluate import save_png, encode
from .train import write
from .losses import masked_mean


def actor_only(camera, geometry, backend, capacity, actor=None):
    selected = geometry["owner"] >= 0 if actor is None else geometry["owner"] == actor
    g = {name: value[selected] for name, value in geometry.items()}
    features = torch.cat((g["appearance"], torch.ones_like(g["opacity"])), -1)
    if backend == "reference":
        raw = reference_raster(camera, g["xyz"], g["cov"], g["opacity"], features)
    else:
        raw = cuda_raster(camera, g["xyz"], g["cov"], g["opacity"], features, capacity)
    return raw[..., :3], raw[..., 3]


def edit_rotation(axis, angle):
    axis = normalize(torch.as_tensor(axis, dtype=torch.float32))
    q = torch.cat((torch.tensor([math.cos(angle/2)]), axis*math.sin(angle/2)))
    return quaternion_matrix(q).tolist()


def stream_table(targets,motion_targets=None):
    motion_targets=targets if motion_targets is None else motion_targets
    names = ['observed','shadow_off','delete_selected','static_rgb','coverage','shadow_effect','fixed_static',
             'dynamic/isolated_rgb','dynamic/isolated_alpha','dynamic/visible_rgb','dynamic/visible_alpha','dynamic/owner_id']
    for actor in targets:
        prefix=f'actors/a{actor:02d}'
        names += [prefix+'/'+s for s in ['delete','delete_dynamic','isolated_rgb','isolated_alpha','visible_alpha']]
        if actor in motion_targets:
            names += [prefix+'/'+s for s in ['translate','yaw15','yaw30','translate_dynamic','yaw15_dynamic','yaw30_dynamic']]
    return names


@torch.no_grad()
def evaluate_many(model, dataset, output, targets, backend='cuda', videos=True, protocol=None):
    targets=list(dict.fromkeys(int(a) for a in targets))
    if not targets or any(a < 0 or a >= len(dataset.actor_ids) for a in targets):
        raise ValueError('多Actor目标必须为当前manifest有效唯一索引')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    protocol=dict(protocol or {})
    motion_ids=[a['raw_track_id'] for a in protocol.get('primary_actors',[])]
    motion_targets=[a for a in targets if dataset.actor_ids[a] in motion_ids] if motion_ids else targets
    move_axis=np.asarray(protocol.get('translation_axis_world',[1,0,0]),float)
    move_axis/=np.linalg.norm(move_axis)
    yaw_axis=protocol.get('rotation_axis_world',[0,0,1])
    names=stream_table(targets,motion_targets)
    for name in names:(output/name).mkdir(parents=True)
    table=[dict(actor_index=a,raw_track_id=dataset.actor_ids[a],directory=f'actors/a{a:02d}') for a in targets]
    write(output/'actor_table.json',dict(actors=table,all_scene_actor_ids=dataset.actor_ids,protocol=protocol,
          dynamic_isolated='all Actor Gaussians on black, no static scene, no GT masking',
          dynamic_visible='full-scene alpha*T weighted Actor radiance, static occlusion preserved',
          raw_alpha='isolated Actor opacity, not multiplied by body mask',
          deletion_scope='only targets admitted by camera+LiDAR evidence; delete_selected removes those targets together',
          static_scope='background-only branch diagnostic, not a claimed deletion test for all scene tracks'))
    anchor=dataset.frame(protocol.get('fixed_camera_index',12) if len(dataset.rows)>12 else len(dataset.rows)//2)
    fixed_ref=None;records=[]
    palette=torch.tensor([[((a*47+53)%255)/255,((a*89+127)%255)/255,((a*137+191)%255)/255] for a in range(len(dataset.actor_ids))],device=model.field.xyz.device)
    for i in range(len(dataset.rows)):
        frame=dataset.frame(i);phase=math.sin(2*math.pi*i/max(1,len(dataset.rows)-1))
        geometry,p=model.buffers(frame,len(dataset.actor_ids),backend,isolated=False)
        image,_=model.full_image(p,geometry);off,_=model.full_image(p,geometry,shadows=False)
        dynamic,raw_alpha=actor_only(frame.camera,geometry,backend,model.config.max_features)
        images=dict(observed=image,shadow_off=off,coverage=p['alpha'][...,None].expand_as(image),
                    shadow_effect=(off-image).abs().clamp(0,1)*3)
        images.update({'dynamic/isolated_rgb':dynamic,'dynamic/isolated_alpha':raw_alpha[...,None].expand_as(image),
                       'dynamic/visible_rgb':p['dynamic_visible_rgb'],
                       'dynamic/visible_alpha':p['visible_labels'].sum(-1)[...,None].expand_as(image),
                       'dynamic/owner_id':torch.einsum('hwk,kc->hwc',p['visible_labels'],palette)})
        all_delete={a:{'delete':True} for a in range(len(dataset.actor_ids))}
        g,pg=model.buffers(frame,len(dataset.actor_ids),backend,all_delete,isolated=False)
        images['static_rgb'],_=model.full_image(pg,g)
        no_shadow,_=model.full_image(pg,g,shadows=False)
        sg,sp=model.buffers(frame,len(dataset.actor_ids),backend,{a:{'delete':True} for a in targets},isolated=False)
        images['delete_selected'],_=model.full_image(sp,sg)
        fixed=type(frame)(frame.frame_id,frame.index,frame.split,anchor.camera,anchor.rgb,anchor.masks,
                          anchor.mask_known,anchor.road,anchor.depth,frame.poses,frame.valid)
        fg,fp=model.buffers(fixed,len(dataset.actor_ids),backend,all_delete,isolated=False)
        images['fixed_static'],_=model.full_image(fp,fg)
        if fixed_ref is None:fixed_ref=images['fixed_static'].clone()
        actor_records=[]
        for actor in targets:
            prefix=f'actors/a{actor:02d}'
            isolated,alpha=actor_only(frame.camera,geometry,backend,model.config.max_features,actor)
            images[prefix+'/isolated_rgb']=isolated
            images[prefix+'/isolated_alpha']=alpha[...,None].expand_as(image)
            images[prefix+'/visible_alpha']=p['visible_labels'][...,actor,None].expand_as(image)
            cases=[('delete',{'delete':True})]
            if actor in motion_targets:cases += [('translate',{'translation':(move_axis*phase).tolist()}),
                              ('yaw15',{'rotation_matrix':edit_rotation(yaw_axis,math.radians(15)*phase)}),
                              ('yaw30',{'rotation_matrix':edit_rotation(yaw_axis,math.radians(30)*phase)})]
            for name,edit in cases:
                eg,ep=model.buffers(frame,len(dataset.actor_ids),backend,{actor:edit},isolated=False)
                images[prefix+'/'+name],_=model.full_image(ep,eg)
                if name!='delete':
                    images[prefix+'/'+name+'_dynamic']=actor_only(frame.camera,eg,backend,model.config.max_features,actor)[0]
                else:
                    images[prefix+'/delete_dynamic']=actor_only(frame.camera,eg,backend,model.config.max_features)[0]
            mask=frame.masks[...,actor] & frame.mask_known[actor]
            actor_records.append(dict(actor_index=actor,raw_track_id=dataset.actor_ids[actor],pose_valid=bool(frame.valid[actor]),
                  mask_known=bool(frame.mask_known[actor]),body_pixels=int(mask.sum()),
                  body_rgb_mae=float(masked_mean((image-frame.rgb).abs(),mask)) if mask.any() else None,
                  visible_alpha_sum=float(p['visible_labels'][...,actor].sum()),isolated_alpha_sum=float(alpha.sum()),
                  deletion_visible_difference=float((images[prefix+'/delete']-image).abs().mean()),
                  isolated_alpha_outside_body=float(masked_mean(alpha,~mask)) if frame.mask_known[actor] else None))
        for name,value in images.items():save_png(output/name/(frame.frame_id+'.png'),value)
        records.append(dict(frame_id=frame.frame_id,split=frame.split,rgb_mae=float((image-frame.rgb).abs().mean()),
                actors=actor_records,static_shadow_off_max=float((no_shadow-images['static_rgb']).abs().max()),
                fixed_static_max=float((images['fixed_static']-fixed_ref).abs().max())))
        write(output/'metrics.json',dict(frames=records,paired_edit_GT=False,human_verdict=None))
        write(output/'progress.json',dict(stage='multi_actor_evaluation',frame=i+1,total=len(dataset.rows),streams=len(names)))
    encoded=[encode(output/name) for name in names] if videos else []
    evidence=[dict(**entry,any_original_view_evidence=any(r['actors'][j]['visible_alpha_sum']>.01 for r in records),
                   note='no visible deletion evidence is not an editing success') for j,entry in enumerate(table)]
    write(output/'evaluation.json',dict(frames=len(records),streams=names,videos=encoded,targets=table,
             PNGs=len(records)*len(names),actor_evidence=evidence,manual_visual_review_complete=False,human_verdict=None,problem_solved=None))
    return records
