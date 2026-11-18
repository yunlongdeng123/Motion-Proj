"""只激活已有世界道路锚点；原活动道路和观测来源台账保持不变。

CPU build 不加载原生场景。native 子命令必须持有新 campaign 的显式租约。
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np

CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_activation_20261118_r1')
PREVIOUS = Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_capacity_gpu_20261118_r1')
CPU_CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_capacity_cpu_20261117_r1')
PARENT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_asset_20261116_r1/phase2/formal_r1/asset')
REPO = Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
FRAMES = (86, 110, 134, 179, 198)


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def prepare_arrays(road, prior, proposed_world_rows, *, initialize_observed=False,
                   K_guard=None, near_guard=None):
    """纯 NumPy 构造；只允许 active，以及明确声明的新活动行 observed RGB。"""
    n = len(road['active'])
    ids = np.asarray(proposed_world_rows)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(np.unique(ids)) != len(ids):
        raise ValueError('proposal 需要无重复的完整世界行索引')
    if len(ids) and (ids.min() < 0 or ids.max() >= n):
        raise ValueError('proposal 世界行越界')
    if np.any(road['active'][ids]):
        raise ValueError('proposal 必须全部为父资产 inactive；不改变既有活动节点')
    requested = np.zeros(n, bool)
    requested[ids] = True
    legal = prior['surface_supported'] & prior['legal_write'] & ~prior['unknown']
    if np.any(~legal[ids] | ~prior['known'][ids] | (prior['source'][ids] != 1)):
        raise ValueError('本最小候选只消费已存在的合法 known/source1 表面节点')
    if np.any(prior['guarded_world_rows'][ids] | (road['minimum_model_depth'][ids] > 0)):
        raise ValueError('proposal 触及旧世界保护或 near 锁')
    guards = []
    for value in (K_guard, near_guard):
        value = np.zeros(n, bool) if value is None else np.asarray(value, bool)
        if value.shape != (n,):
            raise ValueError('guard 必须覆盖完整世界行域')
        guards.append(value)
    new = requested & ~guards[0] & ~guards[1]
    result = dict(road)
    result['active'] = road['active'] | new
    material_init = new.copy() if initialize_observed else np.zeros(n, bool)
    if initialize_observed:
        rgb = road['reference_srgb'].copy()
        observed = prior['observed_reference_srgb'][new]
        if not np.isfinite(observed).all() or np.any((observed < 0) | (observed > 1)):
            raise ValueError('冻结观测颜色必须有限且在 [0,1]')
        rgb[new] = observed
        result['reference_srgb'] = rgb
    allowed = {'active', 'reference_srgb'} if initialize_observed else {'active'}
    for key in road:
        if key not in allowed and not np.array_equal(result[key], road[key]):
            raise AssertionError('未授权道路字段变化: ' + key)
        if result[key].shape and result[key].shape[0] == n and not np.array_equal(result[key][road['active']], road[key][road['active']]):
            raise AssertionError('父活动道路发生变化: ' + key)
    p = dict(prior)
    p.update(parent_world_row_id=np.arange(n, dtype=np.int64),
             activation_parent_active=road['active'], activation_requested=requested,
             activation_new=new, activation_material_init=material_init,
             activation_K_guard=guards[0], activation_near_guard=guards[1],
             activation_physical_supported=prior['surface_supported'],
             activation_legal=prior['legal_write'], activation_unknown=prior['unknown'],
             activation_source_prior=prior['source'], activation_known_prior=prior['known'],
             activation_base_parent_rgb=road['reference_srgb'],
             activation_effective_rgb=result['reference_srgb'],
             effective_row_rgb=result['reference_srgb'], effective_active=result['active'],
             effective_coverage_requested=prior.get('effective_coverage_requested', prior.get('coverage_requested', prior.get('requested', np.zeros(n, bool)))) | new,
             allowed_color_refine=np.zeros(n, bool))
    kind = np.zeros(n, np.uint8)
    kind[material_init] = 1  # 0 unchanged / 1 immutable observed RGB consumed.
    p['activation_material_initialization_kind'] = kind
    return result, p


def build_activation_asset(parent_asset, proposal_npz, output_dir, *, guard_npz=None,
                           initialize_observed=False, pin=False):
    parent, proposal_path, output = map(Path, (parent_asset, proposal_npz, output_dir))
    if output.exists():
        raise FileExistsError('输出必须为新目录，保留旧现场: ' + str(output))
    road = read_npz(parent / 'road.npz')
    prior = read_npz(parent / 'provenance.npz')
    proposal = read_npz(proposal_path)
    guards = {} if guard_npz is None else read_npz(guard_npz)
    result, p = prepare_arrays(road, prior, proposal['proposed_world_rows'],
        initialize_observed=initialize_observed,
        K_guard=guards.get('world_K_guard'), near_guard=guards.get('world_near_guard'))
    output.mkdir(parents=True)
    np.savez_compressed(output / 'road.npz', **result)
    np.savez_compressed(output / 'provenance.npz', **p)
    ids = proposal['proposed_world_rows']
    delta = dict(parent_world_row_id=ids, activation_new=p['activation_new'][ids],
        activation_material_init=p['activation_material_init'][ids],
        K_guard=p['activation_K_guard'][ids], near_guard=p['activation_near_guard'][ids],
        material_source_prior=prior['source'][ids], known_prior=prior['known'][ids],
        material_donor_world_row=prior['source_world_row'][ids],
        material_source_raw=prior['source_raw'][ids], material_source_uv_native=prior['source_uv_native'][ids])
    for key in ('active', 'model_xyz', 'normal_world', 'gaussian_sigma_m', 'road_opacity', 'reference_srgb'):
        delta['parent_' + key] = road[key][ids]
        delta['effective_' + key] = result[key][ids]
    np.savez_compressed(output / 'activation_delta.npz', **delta)
    receipt = dict(schema='road_existing_anchor_activation.v1', parent_asset=str(parent),
        parent_road_sha256=sha256(parent / 'road.npz'), proposal=str(proposal_path),
        proposal_sha256=sha256(proposal_path), guard=None if guard_npz is None else str(guard_npz),
        proposed_rows=len(ids), parent_active_rows=int(road['active'].sum()),
        effective_active_rows=int(result['active'].sum()), new_active_rows=int(p['activation_new'].sum()),
        K_rejected_rows=int((p['activation_requested'] & p['activation_K_guard']).sum()),
        near_rejected_rows=int((p['activation_requested'] & p['activation_near_guard']).sum()),
        material_initialization='immutable observed RGB on new active rows' if initialize_observed else 'legacy placeholder unchanged; material qualification NOT claimed',
        observed_material_initialized_rows=int(p['activation_material_init'].sum()),
        only_road_fields_changed=['active', 'reference_srgb'] if initialize_observed else ['active'],
        old_active_all_fields_exact=True, all_world_geometry_sigma_opacity_exact=True,
        prior_source_and_observed_ledger_exact=True, optimizer_updates=0,
        new_geometry_rows=0, new_SD_or_TRELLIS_calls=0, quality_pass_claimed=False,
        material_qualified=bool(initialize_observed), native_qualification_pending=guard_npz is None)
    write_json(output / 'receipt.json', receipt)
    if pin:
        pin_asset(parent, output, receipt)
    return receipt


def pin_asset(parent, output, record):
    sys.path.insert(0, str(REPO / 'scripts/worldsim_v82'))
    from drivingeditor_dynamic_edit.persistent_road_handoff import save_persistent_road_handoff
    save_persistent_road_handoff(Path(parent) / 'persistent/scene.json', Path(parent) / 'persistent/proof.json',
        Path(output) / 'road.npz', Path(output) / 'persistent', training_record=record)


def coverage_record(alpha, mask):
    a = np.asarray(alpha)[np.asarray(mask, bool)]
    return dict(pixels=len(a), gap=int((a < .05).sum()), thin=int(((a >= .05) & (a <= .9)).sum()),
                supported=int((a > .9).sum()), alpha_mean=float(a.mean()) if len(a) else None)


def evaluation_masks(raw):
    with np.load(CPU_CAMPAIGN / f'ideal_targets_r2/geometry/{raw:03d}.npz') as d:
        D = d['fixed_evaluation_domain'].copy()
    masks = dict(D_full=D)
    if raw in (86, 110):
        d = read_npz(CPU_CAMPAIGN / f'supervision_v4_candidate_r1/{raw:03d}.npz')
        low = read_npz(PREVIOUS / f'check1/remaining_floor_cpu_r1/{raw:03d}_source_state_maps.npz')['fixed_low_S']
        masks.update(S_full=d['S_target_shadow'], S_photo=d['S_target_shadow_photo'], fixed_low_S=low)
    return masks


def native_validation(parent_asset, proposal, candidate_asset, output, *, lease_file, lease_owner):
    """真实 native 先验证父基准，再对新 packet 求 K 足迹并持久导出重载。"""
    sys.path.insert(0, str(PREVIOUS / 'code'))
    sys.path.insert(0, str(REPO / 'scripts/worldsim_v82'))
    from road_capacity_native import load_context, assert_gpu_lease
    assert_gpu_lease(lease_file, lease_owner)  # 在 Torch/native import 前执行。
    import torch
    from PIL import Image
    from drivingeditor_dynamic_edit.road_direct_teacher_training import conservative_retained_guard
    torch.set_num_threads(2)
    output, parent_asset, candidate_asset = map(Path, (output, parent_asset, candidate_asset))
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    calls = 0
    baseline = {}
    ctx = load_context(asset_dir=parent_asset, lease_file=lease_file, lease_owner=lease_owner)
    with torch.no_grad():
        for raw in FRAMES:
            r = ctx.render_road(raw)
            j = ctx.render_joint(raw)
            baseline[raw] = dict(road_rgb=r['rgb'].cpu().numpy(), road_alpha=r['alpha'][0].cpu().numpy(),
                joint_rgb=j['rgb'].cpu().numpy(), K=ctx.K_masks[raw])
            old = read_npz(PREVIOUS / f'delivery/public_native_r1/baseline/{raw:03d}_native.npz')
            error = max(float(np.abs(baseline[raw][key] - old[key]).max()) for key in ('road_rgb', 'road_alpha', 'joint_rgb'))
            if error > 1e-6:
                raise RuntimeError('父资产实际基准与冻结 oldS2 不一致: ' + str((raw, error)))
    calls += ctx.native_calls
    ctx.close()
    ctx = load_context(asset_dir=candidate_asset, lease_file=lease_file, lease_owner=lease_owner)
    guard, K_masks, guard_receipt = conservative_retained_guard(ctx.scene, ctx.provider,
        len(ctx.active_world_rows), np.ones(len(ctx.active_world_rows)), ctx.manifest.data['frames'])
    # helper 内的五次 native 调用不经过 context 计数器。
    calls += len(FRAMES)
    n = len(ctx.road['active'])
    world_guard = np.zeros(n, bool)
    world_guard[ctx.active_world_rows] = guard
    world_near = np.zeros(n, bool)
    near_records = []
    unit_scale = float(ctx.loader['unit_scale'])
    for raw in FRAMES:
        packet, camera = ctx.base_packet(raw), ctx.camera(raw)
        z = packet['means3D'] @ camera.world_view_transform[:3, 2] + camera.world_view_transform[3, 2]
        depth_m = z.detach().cpu().numpy() * unit_scale
        # 只把实际前向、但小于继承物理 near 的节点判为 near；相机后方不等于近裁剪违规。
        near = (depth_m > 0) & (depth_m <= .2)
        world_near[ctx.active_world_rows[near]] = True
        near_records.append(dict(raw=raw,positive_near_rows=int(near.sum())))
    np.savez_compressed(output / 'new_packet_guards.npz', world_K_guard=world_guard,
        world_near_guard=world_near, active_world_rows=ctx.active_world_rows)
    write_json(output / 'new_packet_guard.json', dict(K=guard_receipt, near=near_records,
        near_method='positive actual packet camera depth in meters <= inherited native .2m',
        K_mask_pixel_frames=sum(int(v.sum()) for v in K_masks.values())))
    calls += ctx.native_calls
    ctx.close()
    final_asset = output / 'asset_A1_guarded'
    build_activation_asset(parent_asset, proposal, final_asset, guard_npz=output/'new_packet_guards.npz',
        initialize_observed=True, pin=True)
    activation_only_asset = output / 'asset_A0_guarded'
    build_activation_asset(parent_asset, proposal, activation_only_asset, guard_npz=output/'new_packet_guards.npz',
        initialize_observed=False, pin=True)
    ctx = load_context(asset_dir=activation_only_asset, lease_file=lease_file, lease_owner=lease_owner)
    only_dir = output / 'A0'
    only_dir.mkdir()
    activation_only_rows = []
    activation_only_alpha = {}
    with torch.no_grad():
        for raw in FRAMES:
            r, j = ctx.render_road(raw), ctx.render_joint(raw)
            rgb, alpha, joint = r['rgb'].cpu().numpy(), r['alpha'][0].cpu().numpy(), j['rgb'].cpu().numpy()
            base = baseline[raw]
            difference = np.abs(joint - base['joint_rgb']).max(0)
            if np.any(difference[base['K']] != 0):
                raise RuntimeError('A0 激活对照联合 K 未保持 exact: ' + str(raw))
            activation_only_alpha[raw] = alpha.copy()
            masks = evaluation_masks(raw)
            activation_only_rows.append(dict(raw=raw, K_pixels=int(base['K'].sum()), joint_K_RGB_max=float(difference[base['K']].max()),
                regions={name: dict(before=coverage_record(base['road_alpha'], mask), after=coverage_record(alpha, mask))
                         for name, mask in masks.items()}))
            np.savez_compressed(only_dir/f'{raw:03d}_native.npz', road_rgb=rgb, road_alpha=alpha, joint_rgb=joint, **masks)
            for suffix, value in [('road', rgb), ('joint', joint)]:
                image = np.clip(np.rint(value.transpose(1, 2, 0) * 255), 0, 255).astype(np.uint8)
                Image.fromarray(image).save(only_dir/f'{raw:03d}_{suffix}.webp', lossless=True)
    calls += ctx.native_calls
    ctx.close()
    ctx = load_context(asset_dir=final_asset, lease_file=lease_file, lease_owner=lease_owner)
    frames = []
    with torch.no_grad():
        for raw in FRAMES:
            r, j = ctx.render_road(raw), ctx.render_joint(raw)
            rgb, alpha, joint = r['rgb'].cpu().numpy(), r['alpha'][0].cpu().numpy(), j['rgb'].cpu().numpy()
            base = baseline[raw]
            difference = np.abs(joint - base['joint_rgb']).max(0)
            if np.any(difference[base['K']] != 0):
                raise RuntimeError('新激活资产联合 K 未保持 exact: ' + str((raw, float(difference[base['K']].max()))))
            masks = evaluation_masks(raw)
            if not np.array_equal(alpha, activation_only_alpha[raw]):
                raise RuntimeError('A0/A1 仅颜色不同却 alpha 不一致: ' + str(raw))
            regions = {name: dict(before=coverage_record(base['road_alpha'], mask), after=coverage_record(alpha, mask),
                newly_supported=int((mask & (base['road_alpha'] <= .9) & (alpha > .9)).sum()),
                alpha_decreased_gt_1e_6=int((mask & (alpha < base['road_alpha'] - 1e-6)).sum())) for name, mask in masks.items()}
            frames.append(dict(raw=raw, regions=regions, K_pixels=int(base['K'].sum()), joint_K_RGB_max=float(difference[base['K']].max())))
            np.savez_compressed(output / f'{raw:03d}_native.npz', road_rgb=rgb, road_alpha=alpha,
                joint_rgb=joint, baseline_road_rgb=base['road_rgb'], baseline_road_alpha=base['road_alpha'], K=base['K'], **masks)
            for suffix, value in [('road', rgb), ('joint', joint)]:
                image = np.clip(np.rint(value.transpose(1, 2, 0) * 255), 0, 255).astype(np.uint8)
                Image.fromarray(image).save(output / f'{raw:03d}_{suffix}.webp', lossless=True)
    ctx.assert_frozen()
    reload_receipt = ctx.validate_public_reload(ctx.base_rgb, final_asset, output/'public_reload.json')
    calls += ctx.native_calls + 5  # public reload 的 actual 五次调用未进 context 计数器。
    ctx.close()
    totals = {name: sum(row['regions'].get(name, {}).get('before', {}).get('pixels', 0) for row in frames)
              for name in ['D_full', 'S_full', 'fixed_low_S']}
    if totals != dict(D_full=38416, S_full=3187, fixed_low_S=965):
        raise AssertionError('固定评价分母变化: ' + str(totals))
    result = dict(schema='road_activation_native_validation.v1', GPU_executed=True, optimizer_updates=0,
        parent_asset=str(parent_asset), candidate_asset=str(candidate_asset), final_asset=str(final_asset),
        activation_only_asset=str(activation_only_asset), activation_only_rows=activation_only_rows,
        A0_A1_alpha_all_exact=True,
        fixed_denominators=totals, K_pixel_frames=sum(row['K_pixels'] for row in frames),
        all_joint_K_exact=True, rows=frames, public_reload=reload_receipt,
        native_raster_calls=calls, seconds=time.monotonic()-start,
        geometry_sigma_opacity_and_old_active_RGB_changed=False, production_promotion=False, human_verdict=None)
    if result['K_pixel_frames'] != 77266:
        raise AssertionError('K 固定分母变化')
    write_json(output/'results.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build')
    build.add_argument('--parent', type=Path, default=PARENT)
    build.add_argument('--proposal', type=Path, required=True)
    build.add_argument('--output', type=Path, required=True)
    build.add_argument('--guard', type=Path)
    build.add_argument('--initialize-observed', action='store_true')
    build.add_argument('--pin', action='store_true')
    native = sub.add_parser('native')
    native.add_argument('--parent', type=Path, default=PARENT)
    native.add_argument('--proposal', type=Path, required=True)
    native.add_argument('--candidate', type=Path, required=True)
    native.add_argument('--output', type=Path, required=True)
    native.add_argument('--lease-file', type=Path, required=True)
    native.add_argument('--lease-owner', required=True)
    args = parser.parse_args()
    if args.command == 'build':
        result = build_activation_asset(args.parent, args.proposal, args.output, guard_npz=args.guard,
            initialize_observed=args.initialize_observed, pin=args.pin)
    else:
        result = native_validation(args.parent, args.proposal, args.candidate, args.output,
            lease_file=args.lease_file, lease_owner=args.lease_owner)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
