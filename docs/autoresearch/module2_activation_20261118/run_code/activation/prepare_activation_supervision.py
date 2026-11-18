"""从固定965低S的精确三角锚点生成CPU激活/材质候选；不改旧资产。"""
from pathlib import Path
import argparse,csv,importlib.util,json,os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import numpy as np


def load(path,keys=None):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files if keys is None or k in keys}
def write(path,d):Path(path).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def propose(road,prior,candidate_rows):
    ids=np.unique(np.asarray(candidate_rows,np.int64));n=len(road['active'])
    if ids.ndim!=1 or (ids<0).any() or (ids>=n).any():raise ValueError('world-row IDs out of parent topology')
    # 旧legal/unknown是本阶段已知材质首批资格，不能将其宣称物理永久非法。
    eligible=(~road['active']&prior['surface_supported']&prior['legal_write']&prior['known']&~prior['unknown']
        &~prior['guarded_world_rows']&(road['minimum_model_depth']<=0))
    selected=ids[eligible[ids]];held=ids[~road['active'][ids]&~eligible[ids]];active=ids[road['active'][ids]]
    rgb=prior['observed_reference_srgb'][selected]
    if not np.isfinite(rgb).all() or ((rgb<0)|(rgb>1)).any():raise ValueError('known observed prior RGB is invalid')
    uv=prior['source_uv_native'][selected]
    traceable=(prior['source_raw'][selected]>=0)&(prior['source_world_row'][selected]>=0)&np.isfinite(prior['world_xyz'][selected]).all(1)
    if not traceable.all():raise ValueError('first batch lacks observed-prior frame/world lineage')
    original_uv_missing=~np.isfinite(uv).all(1)
    proposed_active=road['active'].copy();proposed_active[selected]=True
    material_change=np.zeros(n,bool);material_change[selected]=np.any(road['reference_srgb'][selected]!=rgb,axis=1)
    return dict(candidate_world_rows=ids,proposed_world_rows=selected,blocked_world_rows=held,already_active_world_rows=active,
        original_active=road['active'].copy(),proposed_active=proposed_active,
        parent_effective_rgb_sparse=road['reference_srgb'][selected].copy(),observed_reference_rgb_sparse=rgb.copy(),
        proposed_effective_rgb_sparse=rgb.copy(),material_init_world_rows=selected.copy(),
        material_init_required=material_change,material_init_required_sparse=material_change[selected],
        material_source_type=prior['source'][selected].copy(),material_source_raw=prior['source_raw'][selected].copy(),
        material_source_uv_native=uv.copy(),material_source_ledger_row=prior['source_world_row'][selected].copy(),
        material_source_world_xyz=prior['world_xyz'][selected].copy(),material_source_original_uv_missing=original_uv_missing,
        material_source_confidence=prior['confidence'][selected].copy(),
        old_known=prior['known'].copy(),old_unknown=prior['unknown'].copy(),old_world_legal_write=prior['legal_write'].copy(),
        old_coverage_requested=prior['coverage_requested'].copy(),old_surface_supported=prior['surface_supported'].copy(),
        old_source_world_guard=prior['guarded_world_rows'].copy(),old_source_near=road['minimum_model_depth']>0,
        native_newrow_K_status=np.asarray('UNVERIFIED: old active-only K table cannot qualify inactive new rows'),
        held_reason=np.asarray('legacy parent legal/material gate unresolved; not a permanent physical illegality label'),
        world_row_id_semantics=np.asarray('original complete parent road.npz row index; not active-local index'),
        material_source_role=np.asarray('immutable parent observed material prior, frame/world lineage; original pixel UV missing must stay explicit; not raw-pixel GT'))



def corner_values(array, uv):
    """原生像素边界原点、中心0.5；严格四邻域资格独立于RGB插值权重。"""
    h, w = array.shape[:2]
    uv = np.asarray(uv, np.float64)
    low = np.floor(np.nan_to_num(uv) - .5).astype(np.int64)
    x = np.clip(low[:, 0], 0, w - 2)
    y = np.clip(low[:, 1], 0, h - 2)
    return np.stack([array[y, x], array[y, x + 1], array[y + 1, x], array[y + 1, x + 1]], axis=1)


def source_roles(inside, distance, cosine, W, K, A, C, S, other):
    """继承已有视角门；不使用旧requested、legal_write或当前渲染alpha。"""
    ray = inside & (distance < 60) & (cosine > .025)
    safe = ~K.any(1)
    W_all = W.all(1)
    clean_pseudo = W_all & ~other.any(1)
    a = ray & safe & A.all(1)
    c = ray & safe & clean_pseudo & C.all(1)
    s = ray & safe & clean_pseudo & S.all(1)
    return dict(ray_eligible=ray, W_guard_all4=W_all, K_any4=K.any(1), other_shadow_any4=other.any(1),
                A_permission=a, C_permission=c, S_permission=s, authorized=a | c | s)


def choose_source(authorized, roles, rgb, score):
    """真实可见A优先；否则沿用cos/(distance²+4)择一源，不跨view平均。"""
    authorized = np.asarray(authorized, bool)
    roles = np.asarray(roles, np.uint8)
    rgb = np.asarray(rgb)
    score = np.asarray(score)
    a_available = (authorized & (roles == 1)).any(0)
    usable = authorized & (~a_available[None, :] | (roles == 1))
    best = np.argmax(np.where(usable, score, -np.inf), axis=0)
    valid = usable.any(0)
    row = np.arange(authorized.shape[1])
    selected_rgb = rgb[best, row].copy()
    selected_rgb[~valid] = np.nan
    return best, valid, selected_rgb


def review_held(args):
    """只新建许可/有效材质候选；旧观测台账、known与active保持原样。"""
    from PIL import Image
    base = args.cpu_campaign.parent
    N = args.output.parent
    held_path = N / 'supervision_r1/held_legacy_gate_rows.npz'
    held = load(held_path)
    ids, world = held['world_rows'], held['world_xyz']
    if len(ids) != 118:
        raise ValueError('fixed held118 denominator changed')
    if (args.output / 'teacher_permission_review.npz').exists():
        raise FileExistsError('preserve previous permission review')
    args.output.mkdir(parents=True, exist_ok=True)
    road = load(args.parent_asset / 'road.npz', ['active', 'reference_srgb', 'normal_world', 'minimum_model_depth'])
    prior = load(args.parent_asset / 'provenance.npz', ['known', 'unknown', 'legal_write', 'requested', 'old_requested',
        'coverage_requested', 'surface_supported', 'guarded_world_rows', 'W_world', 'source', 'observed_reference_srgb'])
    normal = road['normal_world'][ids]
    world_safe = prior['surface_supported'][ids] & ~prior['guarded_world_rows'][ids] & (road['minimum_model_depth'][ids] <= 0)
    original_helper = base / 'road_material_field_20261112_r1/source/world_feature_field.py'
    spec = importlib.util.spec_from_file_location('frozen_world_projection', original_helper)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    cfg = json.loads((base / 'scene065_material_loop_20261114_r1/config.json').read_text())
    identity = json.loads((args.cpu_campaign / 'supervision_source_identity_r1/source_identity.json').read_text())
    identity_rows = {r['raw']: r for r in identity['rows']}
    frame_evidence, frame_stats, source_files = [], [], []
    for raw in [86, 110]:
        case = f'scene065_raw{raw:03d}'
        prepared = base / 'road_mask_decoupling_20261116_r1/prepared' / case
        observation_path = Path(cfg['bank']) / f'observations/{raw:03d}.json'
        teacher_path = base / 'road_mask_decoupling_20261116_r1/predictions' / case / 'B/raw_native.png'
        guard_path = base / 'road_mask_decoupling_20261116_r1/output_guard_probe' / case / 'native_guard_roles.npz'
        routing_path = args.cpu_campaign / f'supervision_source_identity_r1/{raw:03d}_native_routing_evidence.npz'
        camera = helper.camera_from_observation(json.loads(observation_path.read_text()))
        uv, depth, inside = helper.project_world_to_camera(world, **{k: camera[k] for k in
            ['camera_K_native', 'world_to_camera', 'image_hw', 'distortion_Brown']})
        origin = np.linalg.inv(camera['world_to_camera'])[:3, 3]
        direction = origin - world
        distance = np.linalg.norm(direction, axis=1)
        cosine = np.maximum(np.sum(direction * normal, axis=1) / np.maximum(distance, 1e-6), 0)
        roles = load(prepared / 'B/roles_native.npz', ['road_observed', 'C_mat', 'visible_support'])
        common = load(prepared / 'common_native.npz', ['physical', 'H_B', 'K'])
        guard = load(guard_path, ['W_guard', 'K'])
        routing = load(routing_path, ['target_shadow_interior', 'other_shadow_prediction', 'original_teacher_W_guard'])
        if not identity_rows[raw]['identity_verified'] or not np.array_equal(guard['W_guard'], routing['original_teacher_W_guard']):
            raise ValueError('teacher identity or original write guard binding differs')
        if not np.array_equal(common['K'], guard['K']):
            raise ValueError('original K masks differ')
        A_native = roles['road_observed'] & roles['C_mat'] & roles['visible_support'] & ~common['H_B'] & ~common['K']
        evidence = source_roles(inside, distance, cosine,
            corner_values(guard['W_guard'], uv), corner_values(guard['K'], uv), corner_values(A_native, uv),
            corner_values(common['physical'], uv), corner_values(routing['target_shadow_interior'], uv),
            corner_values(routing['other_shadow_prediction'], uv))
        evidence['authorized'] &= world_safe
        role = np.where(evidence['A_permission'], 1, np.where(evidence['C_permission'], 2,
            np.where(evidence['S_permission'], 3, 0))).astype(np.uint8)
        teacher_image = np.asarray(Image.open(teacher_path).convert('RGB'), np.float32) / 255
        teacher_rgb = helper.bilinear_sample(teacher_image, uv, camera['image_hw'])
        del teacher_image
        observed_image = np.asarray(Image.open(prepared / 'observation.png').convert('RGB'), np.float32) / 255
        observed_rgb = helper.bilinear_sample(observed_image, uv, camera['image_hw'])
        del observed_image
        rgb = np.where((role == 1)[:, None], observed_rgb, teacher_rgb)
        score = cosine / (distance ** 2 + 4)
        evidence.update(raw=np.asarray(raw), uv_native=uv, depth=depth, inside=inside, distance_m=distance,
            cosine=cosine, role=role, source_rgb_srgb=rgb, teacher_rgb_srgb=teacher_rgb,
            actual_observed_rgb_srgb=observed_rgb, selection_score=score, world_safe=world_safe,
            source_permission_q=evidence['authorized'].astype(np.float32))
        frame_evidence.append(evidence)
        np.savez_compressed(args.output / f'{raw:03d}_source_license.npz', world_rows=ids, **evidence)
        frame_stats.append(dict(raw=raw, inside=int(inside.sum()), distance_failed=int((distance >= 60).sum()),
            view_cosine_failed=int((cosine <= .025).sum()),
            ray_eligible=int(evidence['ray_eligible'].sum()), original_W_guard_all4=int(evidence['W_guard_all4'].sum()),
            original_K_any4=int(evidence['K_any4'].sum()), non_target_shadow_any4=int(evidence['other_shadow_any4'].sum()),
            A_permission=int(evidence['A_permission'].sum()), C_permission=int(evidence['C_permission'].sum()),
            S_permission=int(evidence['S_permission'].sum()), authorized=int(evidence['authorized'].sum())))
        source_files.append(dict(raw=raw, original_Brown_observation=str(observation_path), native_teacher=str(teacher_path),
            actual_observation=str(prepared / 'observation.png'), original_teacher_guard=str(guard_path),
            source_identity_and_safe_shadow=str(routing_path), roles=str(prepared / 'B/roles_native.npz'),
            common=str(prepared / 'common_native.npz')))
    auth = np.stack([d['authorized'] for d in frame_evidence])
    role = np.stack([d['role'] for d in frame_evidence])
    rgb = np.stack([d['source_rgb_srgb'] for d in frame_evidence])
    score = np.stack([d['selection_score'] for d in frame_evidence])
    best, valid, effective_rgb = choose_source(auth, role, rgb, score)
    row = np.arange(len(ids))
    raw_array = np.array([86, 110], np.int16)
    selected_raw = np.where(valid, raw_array[best], -1).astype(np.int16)
    selected_role = np.where(valid, role[best, row], 0).astype(np.uint8)
    uv = np.stack([d['uv_native'] for d in frame_evidence])
    selected_uv = uv[best, row].copy(); selected_uv[~valid] = np.nan
    delta = np.abs(rgb[0] - rgb[1])
    both = auth.all(0)
    max_delta = delta.max(1)
    full_permission = np.zeros(len(road['active']), bool); full_permission[ids[valid]] = True
    remains = load(N / 'native_r1/110_remaining_thin.npz', ['triangle_world_rows', 'triangle_weights', 'inactive_all_corners'])
    thin_unknown = remains['inactive_all_corners']
    thin_anchors = np.unique(remains['triangle_world_rows'][thin_unknown][remains['triangle_weights'][thin_unknown] > 1e-7])
    relevant15 = np.isin(ids, thin_anchors)
    result = dict(world_rows=ids, world_xyz=world, parent_effective_rgb_sparse=road['reference_srgb'][ids],
        permission_qualified=valid, permission_world_rows=ids[valid], fresh_permission_world=full_permission,
        proposed_effective_rgb_sparse=effective_rgb[valid].astype(np.float32),
        proposed_source_raw=selected_raw[valid], proposed_source_uv_native=selected_uv[valid],
        proposed_source_role=selected_role[valid], proposed_source_permission_q=np.ones(int(valid.sum()), np.float32),
        candidate_effective_rgb_all118=effective_rgb.astype(np.float32), selected_source_raw=selected_raw,
        selected_source_role=selected_role, selected_source_uv_native=selected_uv,
        per_view_authorized=auth, per_view_role=role, per_view_source_rgb_srgb=rgb,
        per_view_uv_native=uv, per_view_selection_score=score, both_sources_valid=both,
        crossview_abs_RGB_delta=delta, crossview_max_channel_delta=max_delta,
        crossview_over_1code=both & (max_delta > 1/255), crossview_over_2code=both & (max_delta > 2/255),
        crossview_over_4code=both & (max_delta > 4/255), associated_with_remaining15_thin=relevant15,
        old_known=prior['known'][ids], old_unknown=prior['unknown'][ids], old_legal_write=prior['legal_write'][ids],
        old_requested=prior['requested'][ids], old_coverage_requested=prior['coverage_requested'][ids],
        old_generated_W_world=prior['W_world'][ids], old_surface_supported=prior['surface_supported'][ids],
        old_guarded_world_rows=prior['guarded_world_rows'][ids], old_near=road['minimum_model_depth'][ids] > 0,
        old_material_source=prior['source'][ids], old_observed_reference_srgb=prior['observed_reference_srgb'][ids],
        world_row_id_semantics=np.asarray('original full parent road.npz row index; never active-local'),
        status=np.asarray('FRESH_SOURCE_PERMISSION_PLUS_SEPARATE_PSEUDO_EFFECTIVE_RGB_CANDIDATE_NOT_ACTIVATED'),
        native_newrow_K_qualified=np.zeros(len(ids), bool), activation_authorized=np.asarray(False),
        source_roles=np.asarray('0=unqualified;1=fresh_actual_visible_A;2=frozen_teacher_C_pseudo;3=frozen_teacher_target_shadow_S_pseudo'))
    np.savez_compressed(args.output / 'teacher_permission_review.npz', **result)
    with (args.output / 'teacher_permission_rows.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['world_row', 'fresh_permission', 'selected_raw', 'selected_role', 'associated_with_remaining15_thin',
            'both_teachers_valid', 'RGB_meanabs_delta', 'RGB_maxabs_delta', 'view86_cosine', 'view110_cosine',
            'old_legal_write', 'old_unknown', 'old_coverage_requested'])
        for j, wid in enumerate(ids):
            writer.writerow([int(wid), bool(valid[j]), int(selected_raw[j]), int(selected_role[j]), bool(relevant15[j]),
                bool(both[j]), float(delta[j].mean()) if both[j] else '', float(max_delta[j]) if both[j] else '',
                float(frame_evidence[0]['cosine'][j]), float(frame_evidence[1]['cosine'][j]),
                bool(prior['legal_write'][wid]), bool(prior['unknown'][wid]), bool(prior['coverage_requested'][wid])])
    stats = dict(held_world_rows=len(ids), fresh_permission_rows=int(valid.sum()), no_source_view_qualified=int((~valid).sum()),
        both_teachers_valid=int(both.sum()), actual_A_material_rows=int((selected_role == 1).sum()),
        pseudo_C_rows=int((selected_role == 2).sum()), pseudo_target_shadow_rows=int((selected_role == 3).sum()),
        remaining15_thin_associated_world_rows=int(relevant15.sum()), remaining15_thin_associated_fresh_permission_rows=int((relevant15 & valid).sum()),
        remaining_unknown_thin_pixels=int(thin_unknown.sum()),
        selected_source_raw_counts={str(int(k)): int(v) for k,v in zip(*np.unique(selected_raw[valid], return_counts=True))},
        crossview_RGB_meanabs=float(delta[both].mean()), crossview_RGB_maxabs=float(delta[both].max()),
        crossview_max_channel_quantiles=np.quantile(max_delta[both], [0,.25,.5,.75,.9,1]).tolist(),
        crossview_within_fixed_levels={str(k)+'/255': int((both & (max_delta <= k/255)).sum()) for k in [1,2,4]},
        inherited_angle_failed_world_rows=ids[~valid].tolist())
    receipt = dict(schema='activation_source_permission_review.v1', parent_asset=str(args.parent_asset),
        held_source=str(held_path), counts=stats, frames=frame_stats, source_files=source_files,
        projection_source=str(original_helper), original_builder=str(base / 'road_asset_20261116_r1/code/build_scene065_asset.py'),
        historical_gate={'legal_write_line':143, 'formula':'known | ((surface_supported & (base.active | requested)) & ~protected)',
            'generation_pre_gate_line':62, 'formula_generation':'requested & inside & distance<60 & cosine>.025; then native W_guard & ~K at all4 neighbors',
            'conclusion':'old legal_write=False and W_world=False conflate legacy request/activation routing with material availability; neither proves physical illegality'},
        permission_policy='independent original Brown projection; inherited distance<60/cosine>.025; all4 original native W_guard for pseudo; no K/other-shadow; valid supported source world rows',
        material_policy='actual visible A first; otherwise deterministic original cosine/(distance²+4) winner; NEVER average conflicting 2D teachers; explicit per-view differences',
        q_semantics='source_permission_q is binary eligibility only, not calibrated shadow confidence or a chosen training loss weight',
        native_newrow_K_qualified=False, activation_authorized=False, source_conflict_resolved=False,
        old_known_status_changed=False, old_observed_ledger_modified=False, old_assets_modified=False,
        pseudo_is_observed_GT=False, exposure_optimized=False, active_or_geometry_modified=False,
        selected_effective_rgb_role='separate proposed pseudo shader initialization, not atlas/prior/source ledger rewrite',
        limits=['113 native source permissions do not certify new Gaussian screen footprints or K safety',
            '111 dual-view colors differ; conflict table is diagnostic, not a new acceptance threshold',
            '5 inherited angle-failed rows remain unqualified, not permanently physically illegal',
            'only 15 of remaining73 thin pixels are linked to unknown inactive triangles; other58 remain footprint associations',
            'triangle support/association does not prove each pixel causal response'],
        GPU_operations=0, optimizer_updates=0, human_verdict=None)
    write(args.output / 'teacher_permission_review.json', receipt)
    print(json.dumps(dict(output=str(args.output), counts=stats), ensure_ascii=False), flush=True)

def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU only preparation')
    ap=argparse.ArgumentParser();ap.add_argument('--parent-asset',type=Path,required=True);ap.add_argument('--cpu-campaign',type=Path,required=True)
    ap.add_argument('--old-gpu-campaign',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--review-held',action='store_true');args=ap.parse_args()
    if args.review_held:
        review_held(args);return
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'activation_proposal.npz').exists():raise FileExistsError('preserve previous proposal')
    road=load(args.parent_asset/'road.npz',['active','reference_srgb','minimum_model_depth','gaussian_sigma_m','road_opacity'])
    prior=load(args.parent_asset/'provenance.npz',['surface_supported','legal_write','known','unknown','guarded_world_rows','coverage_requested',
        'observed_reference_srgb','source','source_raw','source_uv_native','source_world_row','confidence','world_xyz','coverage_trusted','row_rgb','preserved_legacy_rgb'])
    all_ids=[];frame_rows=[];maps={}
    for raw in [86,110]:
        d=load(args.cpu_campaign/'supervision_v4_candidate_r1'/f'{raw:03d}.npz',['triangle_world_rows','triangle_weights','S_target_shadow','target_S_srgb'])
        low=load(args.old_gpu_campaign/'check1/remaining_floor_cpu_r1'/f'{raw:03d}_source_state_maps.npz',['fixed_low_S'])['fixed_low_S']
        ids=d['triangle_world_rows'][low];weights=d['triangle_weights'][low];used=weights>1e-7
        if (ids[used]<0).any():raise ValueError('fixed low-S point lost mesh hit')
        selected=np.unique(ids[used]);all_ids.append(selected)
        maps[f'{raw:03d}_low_S']=low;maps[f'{raw:03d}_full_S']=d['S_target_shadow'];maps[f'{raw:03d}_triangle_rows']=ids;maps[f'{raw:03d}_triangle_weights']=weights
        frame_rows.append(dict(raw=raw,fixed_low_S_pixels=int(low.sum()),positive_weight_triangle_anchor_rows=len(selected),inactive_rows=int((~road['active'][selected]).sum())))
    if sum(x['fixed_low_S_pixels'] for x in frame_rows)!=965:raise ValueError('fixed low-S denominator changed')
    result=propose(road,prior,np.concatenate(all_ids));ids=result['proposed_world_rows'];held=result['blocked_world_rows']
    np.savez_compressed(args.output/'activation_proposal.npz',**result)
    np.savez_compressed(args.output/'fixed_domains_and_anchor_maps.npz',**maps)
    np.savez_compressed(args.output/'held_legacy_gate_rows.npz',world_rows=held,world_xyz=prior['world_xyz'][held],
        parent_legal_write=prior['legal_write'][held],parent_unknown=prior['unknown'][held],parent_known=prior['known'][held],
        surface_supported=prior['surface_supported'][held],source_world_guard=prior['guarded_world_rows'][held],
        source_near=road['minimum_model_depth'][held]>0,parent_coverage_requested=prior['coverage_requested'][held],
        status=np.asarray('HOLD_PENDING_LEGAL_AND_PSEUDO_MATERIAL_REVIEW_NOT_PERMANENT_PHYSICAL_ILLEGALITY'))
    counts=dict(candidate_world_rows=len(result['candidate_world_rows']),inactive_rows=len(ids)+len(held),proposed_known_material_rows=len(ids),held_legacy_gate_rows=len(held),already_active_rows=len(result['already_active_world_rows']),
        material_init_required_rows=int(result['material_init_required'].sum()),max_legacy_vs_observed_RGB=float(np.abs(result['parent_effective_rgb_sparse']-result['observed_reference_rgb_sparse']).max()),
        original_source_pixel_UV_missing_rows=int(result['material_source_original_uv_missing'].sum()),
        source_type_counts={str(int(k)):int(v) for k,v in zip(*np.unique(result['material_source_type'],return_counts=True))},
        source_raw_counts={str(int(k)):int(v) for k,v in zip(*np.unique(result['material_source_raw'],return_counts=True))})
    write(args.output/'activation_proposal.json',dict(schema='road_activation_supervision_proposal.v1',parent_asset=str(args.parent_asset),counts=counts,frames=frame_rows,
        proposal_ready_for_builder=True,native_guard_qualified=False,geometry_normals_sigma_opacity_must_remain_parent=True,all_old_active_parameters_immutable=True,
        stage_A0='activate proposed rows with parent placeholder/legacy shader RGB: support-mechanism control only, not material acceptance',
        stage_A1='same activation plus initialize only NEW rows from immutable observed_reference_srgb; keep original prior ledger and all old active rows exact',
        pseudo_teacher='target-shadow uses frozen 2D C only as separate pseudo supervision/optional effective-color correction; not mixed into observed prior',
        observed_prior_traceability='source raw frame/world node/immutable RGB retained; legacy source_uv_native is missing and is not invented or claimed as raw pixel corroboration',
        held_semantics='118 parent legal/unknown rows are held for source-lineage audit, not called physically illegal. Old coverage_requested is recorded, never an existence gate.',
        RGB_old_new_record='parent_effective_rgb_sparse vs proposed_effective_rgb_sparse, keyed by proposed_world_rows; material_init_required explicit',
        required_next='actual newrow native footprint/K contribution envelope, current sigma/opacity, full-S/low-S/D/K before acceptance',
        GPU_operations=0,old_assets_modified=False,old_observed_ledger_modified=False,formal_updates=0))
    print(json.dumps(dict(output=str(args.output),counts=counts,native_guard_qualified=False),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
