"""仅读取builder显式保存的有限pixel-beam候选；不定义邻域、不激活、不改台账。"""
from pathlib import Path
import argparse,importlib.util,json,os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import numpy as np
from prepare_activation_supervision import load,write,corner_values,source_roles,choose_source


def audit(args):
    from PIL import Image
    with np.load(args.candidates,allow_pickle=False) as z:
        if args.rows_key not in z.files:raise KeyError(args.rows_key)
        original_rows=z[args.rows_key].copy()
    if original_rows.ndim!=1 or not np.issubdtype(original_rows.dtype,np.integer):raise ValueError('explicit beam row list must be int[N]')
    ids=np.unique(original_rows.astype(np.int64))
    p=load(args.parent_asset/'provenance.npz',['world_xyz','known','unknown','source','source_raw','source_world_row',
        'source_uv_native','observed_reference_srgb','row_rgb','legal_write','W_world','surface_supported','guarded_world_rows','coverage_requested'])
    r=load(args.parent_asset/'road.npz',['active','reference_srgb','normal_world','minimum_model_depth','gaussian_sigma_m','road_opacity'])
    if (ids<0).any() or (ids>=len(r['active'])).any():raise ValueError('invalid original world ID')
    if args.output.exists():raise FileExistsError('preserve old audit output')
    args.output.mkdir(parents=True)
    world=p['world_xyz'][ids];normal=r['normal_world'][ids]
    physical=p['surface_supported'][ids]&~p['guarded_world_rows'][ids]&(r['minimum_model_depth'][ids]<=0)
    known_rgb=p['observed_reference_srgb'][ids]
    unit_known=np.isfinite(known_rgb).all(1)&(known_rgb>=0).all(1)&(known_rgb<=1).all(1)
    # 历史observed UV缺省NaN不等于观测台账丢失；frame/world谱系单列。
    traceable=(p['source_raw'][ids]>=0)&(p['source_world_row'][ids]>=0)
    observed=p['known'][ids]&~p['unknown'][ids]&unit_known&traceable
    base=args.cpu_campaign.parent
    helper_path=base/'road_material_field_20261112_r1/source/world_feature_field.py'
    spec=importlib.util.spec_from_file_location('frozen_projection',helper_path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    cfg=json.loads((base/'scene065_material_loop_20261114_r1/config.json').read_text())
    evidence=[];frame_stats=[]
    for raw in [86,110]:
        case=f'scene065_raw{raw:03d}'
        prepared=base/'road_mask_decoupling_20261116_r1/prepared'/case
        native_teacher=base/'road_mask_decoupling_20261116_r1/predictions'/case/'B/raw_native.png'
        guard_path=base/'road_mask_decoupling_20261116_r1/output_guard_probe'/case/'native_guard_roles.npz'
        routing_path=args.cpu_campaign/f'supervision_source_identity_r1/{raw:03d}_native_routing_evidence.npz'
        camera=helper.camera_from_observation(json.loads((Path(cfg['bank'])/f'observations/{raw:03d}.json').read_text()))
        uv,depth,inside=helper.project_world_to_camera(world,**{k:camera[k] for k in ['camera_K_native','world_to_camera','image_hw','distortion_Brown']})
        direction=np.linalg.inv(camera['world_to_camera'])[:3,3]-world
        distance=np.linalg.norm(direction,axis=1)
        cosine=np.maximum(np.sum(direction*normal,axis=1)/np.maximum(distance,1e-6),0)
        roles=load(prepared/'B/roles_native.npz',['road_observed','C_mat','visible_support'])
        common=load(prepared/'common_native.npz',['physical','H_B','K'])
        guard=load(guard_path,['W_guard','K'])
        routing=load(routing_path,['target_shadow_interior','other_shadow_prediction','original_teacher_W_guard'])
        if not np.array_equal(guard['W_guard'],routing['original_teacher_W_guard']) or not np.array_equal(guard['K'],common['K']):raise ValueError('source guard binding differs')
        A=roles['road_observed']&roles['C_mat']&roles['visible_support']&~common['H_B']&~common['K']
        d=source_roles(inside,distance,cosine,corner_values(guard['W_guard'],uv),corner_values(guard['K'],uv),
            corner_values(A,uv),corner_values(common['physical'],uv),corner_values(routing['target_shadow_interior'],uv),
            corner_values(routing['other_shadow_prediction'],uv))
        d['authorized']&=physical
        role=np.where(d['A_permission'],1,np.where(d['C_permission'],2,np.where(d['S_permission'],3,0))).astype(np.uint8)
        image=np.asarray(Image.open(native_teacher).convert('RGB'),np.float32)/255
        teacher=helper.bilinear_sample(image,uv,camera['image_hw']);del image
        image=np.asarray(Image.open(prepared/'observation.png').convert('RGB'),np.float32)/255
        A_rgb=helper.bilinear_sample(image,uv,camera['image_hw']);del image
        rgb=np.where((role==1)[:,None],A_rgb,teacher)
        d.update(raw=np.asarray(raw),uv_native=uv,depth=depth,inside=inside,distance_m=distance,cosine=cosine,
            role=role,source_rgb_srgb=rgb,selection_score=cosine/(distance**2+4))
        evidence.append(d)
        np.savez_compressed(args.output/f'{raw:03d}_source_license.npz',world_rows=ids,**d)
        frame_stats.append(dict(raw=raw,source_permission=int(d['authorized'].sum()),fresh_A=int(d['A_permission'].sum()),
            pseudo_C=int(d['C_permission'].sum()),pseudo_S=int(d['S_permission'].sum()),K_any4=int(d['K_any4'].sum()),
            W_guard_all4=int(d['W_guard_all4'].sum()),angle_failed=int((cosine<=.025).sum()),outside_camera=int((~inside).sum())))
    auth=np.stack([d['authorized'] for d in evidence]);roles=np.stack([d['role'] for d in evidence])
    colors=np.stack([d['source_rgb_srgb'] for d in evidence]);scores=np.stack([d['selection_score'] for d in evidence])
    best,fresh,rgb=choose_source(auth,roles,colors,scores)
    row=np.arange(len(ids));raws=np.array([86,110],np.int16)
    selected_raw=np.where(fresh,raws[best],-1).astype(np.int16)
    selected_role=np.where(fresh,roles[best,row],0).astype(np.uint8)
    selected_uv=np.stack([d['uv_native'] for d in evidence])[best,row].copy();selected_uv[~fresh]=np.nan
    candidate_material=np.where(observed[:,None],known_rgb,rgb).astype(np.float32)
    candidate_material[~(observed|fresh)]=np.nan
    eligible=~r['active'][ids]&physical&(observed|fresh)
    both=auth.all(0);delta=np.abs(colors[0]-colors[1])
    np.savez_compressed(args.output/'beam_material_audit.npz',world_rows=ids,parent_active=r['active'][ids],
        physical_world_qualified=physical,immutable_observed_available=observed,fresh_source_permission=fresh,
        candidate_qualified_pending_native=eligible,candidate_effective_rgb_sparse=candidate_material,
        parent_effective_rgb_sparse=r['reference_srgb'][ids],selected_fresh_source_raw=selected_raw,
        selected_fresh_source_role=selected_role,selected_fresh_source_uv_native=selected_uv,
        per_view_authorized=auth,per_view_source_role=roles,per_view_source_rgb_srgb=colors,
        both_sources_valid=both,crossview_abs_RGB_delta=delta,
        old_known=p['known'][ids],old_unknown=p['unknown'][ids],old_source=p['source'][ids],
        old_source_raw=p['source_raw'][ids],old_source_world_row=p['source_world_row'][ids],
        old_source_uv_native=p['source_uv_native'][ids],old_observed_reference_srgb=known_rgb,
        old_legal_write=p['legal_write'][ids],old_coverage_requested=p['coverage_requested'][ids],
        surface_supported=p['surface_supported'][ids],source_guard=p['guarded_world_rows'][ids],near=r['minimum_model_depth'][ids]>0,
        original_sigma_m=r['gaussian_sigma_m'][ids],original_opacity=r['road_opacity'][ids],
        status=np.asarray('READ_ONLY_FINITE_BUILDER_BEAM_MATERIAL_PERMISSION_AUDIT_NOT_ACTIVATED'))
    counts=dict(input_explicit_rows=len(original_rows),unique_beam_world_rows=len(ids),already_active=int(r['active'][ids].sum()),
        surface_supported=int(p['surface_supported'][ids].sum()),near=int((r['minimum_model_depth'][ids]>0).sum()),
        existing_guard=int(p['guarded_world_rows'][ids].sum()),physical_qualified=int(physical.sum()),
        observed_material_available=int(observed.sum()),fresh_source_permission=int(fresh.sum()),
        eligible_observed_init=int((eligible&observed).sum()),eligible_pseudo_or_fresh_init=int((eligible&~observed&fresh).sum()),
        candidate_qualified_pending_native=int(eligible.sum()),material_still_missing=int((~observed&~fresh).sum()),
        old_legal_route_false=int((~p['legal_write'][ids]).sum()),old_request_false=int((~p['coverage_requested'][ids]).sum()),
        both_teachers_valid=int(both.sum()),crossview_RGB_meanabs=float(delta[both].mean()) if both.any() else None,
        crossview_RGB_maxabs=float(delta[both].max()) if both.any() else None)
    write(args.output/'beam_material_audit.json',dict(schema='fixed_pixel_beam_material_permission_audit.v1',
        parent_asset=str(args.parent_asset),explicit_beam_candidates=str(args.candidates),world_rows_key=args.rows_key,
        counts=counts,frames=frame_stats,projection_source=str(helper_path),
        scope='only builder-saved explicit beam IDs; no new radius, topology, activation, native or optimization',
        candidate_material_policy='immutable measured prior first; otherwise independent source permission; source conflict reported, no teacher averaging',
        legacy_requested_policy='recorded only; not a physical existence gate',
        original_ledger_immutable=True,old_assets_modified=False,activation_authorized=False,native_guard_qualified=False,
        crossview_conflict_resolved=False,GPU_operations=0,optimizer_updates=0,human_verdict=None))
    print(json.dumps(counts))


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU only')
    ap=argparse.ArgumentParser();ap.add_argument('--parent-asset',type=Path,required=True)
    ap.add_argument('--cpu-campaign',type=Path,required=True);ap.add_argument('--candidates',type=Path,required=True)
    ap.add_argument('--rows-key',required=True);ap.add_argument('--output',type=Path,required=True)
    audit(ap.parse_args())

if __name__=='__main__':main()
