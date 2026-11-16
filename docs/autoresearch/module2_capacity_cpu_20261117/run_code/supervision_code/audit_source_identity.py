"""Read-only source identity and teacher-write audit; CPU only, no inference."""
from pathlib import Path
import argparse,hashlib,json,os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from road_asset_supervision_v4 import verify_shadow_identity,other_shadow_guard,pool_native


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,d):Path(path).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU-only preparation requires CUDA_VISIBLE_DEVICES empty')
    ap=argparse.ArgumentParser();ap.add_argument('--campaign-root',type=Path,default=Path('/root/autodl-tmp/drivingeditor_repro/campaign'))
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();base=args.campaign_root
    args.output.mkdir(parents=True,exist_ok=False)
    old=base/'road_direct_teacher_20261116_r1';maskroot=base/'road_mask_decoupling_20261116_r1';shadowroot=base/'object_shadow_gpu_20261116_r1'
    rows=[];source_files=[]
    for raw in [86,110]:
        sid=f'scene065_raw{raw:03d}';p=maskroot/'prepared'/sid
        paths=dict(observation=p/'observation.png',teacher=maskroot/'predictions'/sid/'B/raw_native.png',
            roles=p/'B/roles_native.npz',common=p/'common_native.npz',teacher_guard=maskroot/'output_guard_probe'/sid/'native_guard_roles.npz',
            field=shadowroot/'runs/postprocess_local_u_r2'/sid/'provider/shadow_field.npz',
            packet=shadowroot/'runs/official_demo_r1'/sid/'provider/paired_instances.npz',
            routing=shadowroot/'evaluation/results_postprocess_r2/routing'/sid/'routing_receipt.json',
            old_supervision=old/'supervision_r3'/f'{raw:03d}.npz')
        observation=np.asarray(Image.open(paths['observation']).convert('RGB'));teacher=np.asarray(Image.open(paths['teacher']).convert('RGB'))
        roles=load(paths['roles']);common=load(paths['common']);guard=load(paths['teacher_guard']);field=load(paths['field']);packet=load(paths['packet']);sup=load(paths['old_supervision'])
        meta=json.loads(str(field['metadata_json'].item()));routing=json.loads(paths['routing'].read_text());policy=routing['public_routing_receipt']['policy']
        verify_shadow_identity(meta,observation,common['target'])
        other,pairreceipt=other_shadow_guard(packet,meta,field['probability'],policy['low'])
        selected=meta['diagnostics']['official_instance_pairs'][meta['diagnostics']['selected_pair_index']]
        member=selected['matched_target_member'];obj=packet['soft_masks'][member];target=common['target'];K=common['K']
        object_indices=np.flatnonzero(packet['classes']==meta['diagnostics']['verified_object_class_id'])
        overlaps=np.sum(packet['soft_masks'][object_indices]*target[None],axis=(1,2),dtype=np.float64)
        unions=np.sum(packet['soft_masks'][object_indices],axis=(1,2),dtype=np.float64)+target.sum()-overlaps
        ious=overlaps/np.maximum(unions,1e-12);rank=np.argsort(-ious,kind='stable');selected_iou=float(ious[np.flatnonzero(object_indices==member)[0]])
        hard=obj>=.5
        interior=(roles['shadow_high']&roles['W_output']&~roles['shadow_uncertain']&~roles['shadow_boundary_guard']
            &field['valid']&(field['uncertainty']<=policy['max_write_uncertainty'])&roles['road_observed']&~roles['physical_unknown']&~roles['K'])
        if meta['diagnostics']['association_uncertainty']>policy['max_write_uncertainty']:interior[:]=False
        safe=interior&~other;any_S=pool_native(safe,sup['K'].shape,'any');full_S=pool_native(safe,sup['K'].shape,'all')
        legal_S=full_S&sup['geometry_valid']&~sup['retained_projection_guard']&~sup['K']
        delta=(teacher.astype(np.float32)-observation.astype(np.float32))/255
        luminosity=np.array([.2126,.7152,.0722],np.float32)
        row=dict(raw=raw,identity_verified=True,selected_object_member=int(member),selected_shadow_member=int(selected['shadow_member']),
            selected_association_id=int(selected['association_id']),selected_object_soft_iou=selected_iou,
            selected_object_iou_rank=int(np.flatnonzero(object_indices[rank]==member)[0])+1,
            all_object_soft_iou_ranking=[dict(member=int(object_indices[i]),iou=float(ious[i])) for i in rank],
            selected_hard_object_pixels=int(hard.sum()),selected_hard_object_intersect_K_pixels=int((hard&K).sum()),
            selected_soft_object_mass_inside_K=float(obj[K].sum(dtype=np.float64)),selected_object_soft_mass=float(obj.sum(dtype=np.float64)),
            original_target_intersect_K_pixels=int((target&K).sum()),
            original_teacher_W_output_equals_guard_W_old=bool(np.array_equal(roles['W_output'],guard['W_old'])),
            original_teacher_W_old_subset_W_guard=bool(not (guard['W_old']&~guard['W_guard']).any()),
            source_W_output_pixels=int(roles['W_output'].sum()),original_teacher_W_guard_pixels=int(guard['W_guard'].sum()),
            S_native_pixels=int(safe.sum()),S_intersect_original_teacher_W_guard=int((safe&guard['W_guard']).sum()),
            S_outside_original_teacher_W_guard=int((safe&~guard['W_guard']).sum()),
            S_renderer_full_pixels=int(full_S.sum()),S_renderer_geometry_pixels=int(legal_S.sum()),
            S_renderer_photo_r3_reference_pixels=int((legal_S&sup['fixed_high_alpha']).sum()),
            S_any_intersect_old_A_mask=int((any_S&sup['A_mask']).sum()),S_any_intersect_old_A_visible=int((any_S&sup['A_visible']).sum()),
            S_full_intersect_old_A_mask=int((full_S&sup['A_mask']).sum()),S_full_intersect_old_A_visible=int((full_S&sup['A_visible']).sum()),
            target_shadow_possible_intersect_A_visible=int((pool_native(roles['shadow_possible'],sup['K'].shape,'any')&sup['A_visible']).sum()),
            other_shadow_intersect_A_visible=int((pool_native(other,sup['K'].shape,'any')&sup['A_visible']).sum()),
            target_other_shadow_interior_overlap_native=int((interior&other).sum()),
            teacher_minus_observed_sRGB_signed_mean_in_S=delta[safe].mean(0).tolist() if safe.any() else None,
            teacher_minus_observed_sRGB_luma_quantiles_in_S=np.quantile(delta[safe]@luminosity,[0,.1,.5,.9,1]).tolist() if safe.any() else None,
            teacher_generation_scope_proven=True,teacher_shadow_quality_verified=False,qS_fixed=False,
            probability_and_association_uncalibrated=True,inherited_policy=policy,pair_routing=pairreceipt)
        overlay=observation.astype(np.float32)
        for mask,color,alpha in [(other,[190,90,240],.25),(safe,[40,230,125],.50)]:overlay[mask]=(1-alpha)*overlay[mask]+alpha*np.asarray(color)
        for mask,color in [(K,[255,75,90]),(hard,[255,190,40]),(target,[40,220,255])]:
            edge=mask&~binary_erosion(mask,iterations=2);overlay[edge]=color
        rgb=np.uint8(np.clip(np.floor(overlay+.5),0,255));Image.fromarray(rgb).save(args.output/f'{raw:03d}_native_overlay.webp',lossless=True,method=4)
        y,x=np.where(target|safe);roi=[max(int(x.min())-32,0),max(int(y.min())-32,0),min(int(x.max())+33,rgb.shape[1]),min(int(y.max())+33,rgb.shape[0])]
        x0,y0,x1,y1=roi
        Image.fromarray(np.concatenate([observation[y0:y1,x0:x1],teacher[y0:y1,x0:x1],rgb[y0:y1,x0:x1]],axis=1)).save(args.output/f'{raw:03d}_observation_teacher_overlay_crop.webp',lossless=True,method=4)
        row['crop_roi_xyxy']=roi
        np.savez_compressed(args.output/f'{raw:03d}_native_routing_evidence.npz',target_core=target,selected_object_soft=obj,
            target_shadow_interior=safe,other_shadow_prediction=other,original_teacher_W_guard=guard['W_guard'])
        rows.append(row);source_files.extend(dict(role=k,path=str(v),sha256=sha(v)) for k,v in paths.items())
        print(json.dumps({k:v for k,v in row.items() if k not in ['pair_routing','all_object_soft_iou_ranking']},ensure_ascii=False),flush=True)
    write(args.output/'source_identity.json',dict(schema='road_capacity_source_identity_audit.v1',rows=rows,source_files=source_files,
        overlay_legend=dict(cyan='original target contour',orange='selected detector object p>=.5 contour; display only',red='retained K contour',green='safe target-shadow candidate interior',purple='other paired shadow possible'),
        GPU_operations=0,training_updates=0,render_operations=0,qS_fixed=False,old_source_or_run_modified=False))


if __name__=='__main__':main()
