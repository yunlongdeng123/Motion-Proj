"""T2/A0-midpoint 唯一零更新入口；默认inspect不导入Torch或native。"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import time
import numpy as np

P = Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
T = P / 'trellis2'
G = P.parent / 'road_capacity_gpu_20261118_r1'
FRAMES = (86, 110, 134, 179, 198)
VARIANTS = ('T2_mid', 'A0_mid')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def bytes_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def compose_colors(parent_rgb, active_rows, eligible, cache):
    """完整共同许可消费过滤色；仅非许可行保留父场legacy。"""
    parent_rgb = np.asarray(parent_rgb)
    active_rows = np.asarray(active_rows)
    eligible = np.asarray(eligible, bool)
    if parent_rgb.shape != (len(active_rows), 3) or eligible.shape != (len(active_rows),):
        raise ValueError('parent/scope shape mismatch')
    if not np.array_equal(cache['active_world_rows'], active_rows):
        raise ValueError('cached/native world-row identity mismatch')
    result = {}
    for variant, key in [('T2_mid', 'filtered_rgb_T'), ('A0_mid', 'filtered_rgb_A0')]:
        value = np.asarray(cache[key])
        if value.dtype != np.float32 or value.shape != parent_rgb.shape or not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
            raise ValueError('invalid frozen material RGB')
        color = parent_rgb.copy()
        color[eligible] = value[eligible]
        if not bytes_equal(color[~eligible], parent_rgb[~eligible]):
            raise AssertionError('nonqualified legacy RGB changed')
        result[variant] = color
    return result


def prepare(output):
    import os
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('prepare requires explicit CPU-only CUDA_VISIBLE_DEVICES')
    output = Path(output)
    if output.exists():
        raise FileExistsError('preserve existing preparation')
    parent = Path(json.loads((P / 'execution_contract.json').read_text())['coverage_parent'])
    scope = load(P / 'method_scope_r1/scope.npz')
    cache = load(T / 'midpoint_full_r1/row_queries.npz')
    road = load(parent / 'road.npz')
    rows = np.flatnonzero(road['active'])
    if not np.array_equal(rows, scope['active_world_rows']) or int(scope['eligible'].sum()) != 84266:
        raise ValueError('scientific84266 scope mismatch')
    accepted = json.loads((T / 'midpoint_full_r1/acceptance.json').read_text())
    audit = json.loads((T / 'midpoint_audit_r1/receipt.json').read_text())
    if accepted['status'] != 'PASS' or not all(accepted['checks'].values()) or audit['status'] != 'PASS_MIDPOINT_CPU_QUALIFICATION':
        raise ValueError('midpoint numerical qualification missing')
    # 外patch不是父资产旧c0；它是同filteredA0。这里绑定主方法的准确零场初值。
    initial = load(P / 'cpu_exact_r1/initial_row_queries.npz')
    if not np.array_equal(initial['active_world_rows'], rows) or not bytes_equal(initial['filtered_A0_rgb'], cache['baseline_A0_GH15']):
        raise ValueError('exterior filteredA0 differs from main authoritative initialization')
    colors = compose_colors(road['reference_srgb'][rows], rows, scope['eligible'], cache)
    baseline_dir = parent.parent
    previous = json.loads((baseline_dir / 'results.json').read_text())
    if Path(previous['final_asset']).resolve() != parent.resolve() or previous.get('GPU_executed') is not True:
        raise ValueError('existing parent float renders do not bind current parent')
    baseline_files = {}
    Ktotal = 0
    for raw in FRAMES:
        path = baseline_dir / f'{raw:03d}_native.npz'
        native = load(path)
        if native['road_rgb'].shape != (3, 320, 480) or native['joint_rgb'].shape != (3, 320, 480) or native['road_alpha'].shape != (320, 480):
            raise ValueError('expected full precision native parent arrays')
        if native['road_rgb'].dtype != np.float32 or native['joint_rgb'].dtype != np.float32 or native['K'].dtype != bool:
            raise ValueError('parent float/mask dtype mismatch')
        if raw in (86, 110):
            if not bytes_equal(native['road_alpha'], load(P / f'stage0_native_r1/{raw:03d}_targets.npz')['fixed_alpha']):
                raise ValueError('parent coverage differs from fixed supervision')
        Ktotal += int(native['K'].sum())
        baseline_files[str(raw)] = str(path)
    if Ktotal != 77266:
        raise ValueError('full five-frame K denominator changed')
    output.mkdir(parents=True)
    np.savez_compressed(output / 'native_inputs.npz', active_world_rows=rows, eligible=scope['eligible'],
        parent_rgb=road['reference_srgb'][rows], **colors,
        material_T_weight_fraction=cache['material_T_weight_fraction'], common_valid_mass=cache['common_valid_mass'],
        copied_same_A0=cache['copied_same_A0'])
    sources = [Path(__file__), G / 'code/road_capacity_native.py', parent / 'road.npz', parent / 'provenance.npz',
        parent / 'persistent/scene.json', parent / 'persistent/proof.json', baseline_dir / 'results.json',
        P / 'method_scope_r1/scope.npz', P / 'cpu_exact_r1/initial_row_queries.npz', P / 'cpu_initial_r1/A0.npz',
        P / 'chart_plan_candidate_r1/geometry_plan.npz', T / 'midpoint_full_r1/row_queries.npz',
        T / 'midpoint_full_r1/receipt.json', T / 'midpoint_full_r1/acceptance.json', T / 'midpoint_audit_r1/receipt.json',
        T / 'world_prior_r1/mesh_uv_source.npz', T / 'world_prior_r1/basecolor_rgba.png',
        T / 'world_prior_r1/receipt.json', T / 'input_r1/manifest.json',
        *[Path(value) for value in baseline_files.values()]]
    manifest = dict(schema='TRELLIS2_zero_update_native_pair_preparation.v1', status='PASS_CPU_PREPARATION',
        coverage_parent=str(parent), frames=list(FRAMES), variants=list(VARIANTS), active_rows=len(rows), eligible_rows=84266,
        input_npz=str(output / 'native_inputs.npz'), input_sha256=sha(output / 'native_inputs.npz'),
        parent_float_files=baseline_files, protected_pixel_frames=77266,
        source_sha256={str(path): sha(path) for path in sources},
        color_policy='all84266 eligible consume selected filtered full-row colors; all other active rows exact N-parent legacy',
        exterior_filteredA0_same_main_initialization=True, runtime_mask_expansion=False,
        fields_frozen=['XYZ', 'normals', 'opacity', 'scales', 'active', 'static', 'vehicles', 'environment', 'cameras'],
        planned_native_forward_calls=dict(original_context_2columns_5road_5joint_each=20, public_clone_5road_each=10, total=30),
        public_reload_scope='actual public loader roadRGB/alpha/RGB8, compare saved floats without redundant expected forward; joint K audited directly for each full column',
        optimizer_updates=0, new_inference_calls=0, GPU_executed=False, human_verdict=None)
    write(output / 'manifest.json', manifest)
    print(json.dumps(dict(manifest=str(output / 'manifest.json'), status=manifest['status'], planned_native_forward_calls=30)), flush=True)


def augment_provenance(asset, variant, rows, eligible, cache, color):
    path = Path(asset) / 'provenance.npz'
    provenance = load(path)
    n = len(provenance['known'])
    write_mask = np.zeros(n, bool)
    write_mask[rows] = eligible
    material_T = np.zeros(n, np.float64)
    material_T[rows] = cache['material_T_weight_fraction'] if variant == 'T2_mid' else 0
    material_T[~write_mask] = 0
    t2 = write_mask & (material_T > 0)
    roles = np.zeros(n, np.uint8)
    roles[write_mask] = 1  # A0 filtered common base
    roles[t2] = 2  # source-normalized kernel contains T2
    provenance['effective_T2_prior'] = t2
    provenance['effective_T2_prior_world_rows'] = np.flatnonzero(t2)
    provenance['effective_T2_prior_weight_fraction'] = material_T
    provenance['effective_T2_prior_source_kind'] = roles
    provenance['effective_material_common_midpoint_write'] = write_mask
    provenance['effective_material_native_pair_variant'] = np.asarray(variant)
    provenance['effective_material_row_ci_is_derived'] = np.asarray(True)
    provenance['effective_T2_generated_alpha_changes_opacity'] = np.asarray(False)
    np.savez_compressed(path, **provenance)
    source = Path(asset) / 'material_source'
    source.mkdir()
    for original, name in [(T / 'world_prior_r1/mesh_uv_source.npz', 'mesh_uv_source.npz'),
        (T / 'world_prior_r1/basecolor_rgba.png', 'basecolor_rgba.png'),
        (P / 'cpu_initial_r1/A0.npz', 'A0.npz'), (T / 'input_r1/manifest.json', 'input_manifest.json'),
        (T / 'midpoint_full_r1/receipt.json', 'integration_recipe.json'),
        (T / 'midpoint_audit_r1/receipt.json', 'numerical_qualification.json')]:
        shutil.copyfile(original, source / name)
    np.savez_compressed(source / 'derived_row_ci.npz', active_world_rows=rows, eligible=eligible, color=color,
        common_valid_mass=cache['common_valid_mass'], material_T_weight_fraction=material_T[rows])
    write(Path(asset) / 'material_prior.json', dict(schema='TRELLIS2_fixed_mesh_material_prior.v1', variant=variant,
        authoritative_T2='material_source/mesh_uv_source.npz + material_source/basecolor_rgba.png',
        authoritative_A0='material_source/A0.npz', integral='material_source/integration_recipe.json',
        physical_Sigma_geometry=str(P / 'chart_plan_candidate_r1/geometry_plan.npz'),
        physical_Sigma_geometry_sha256=sha(P / 'chart_plan_candidate_r1/geometry_plan.npz'),
        derived_ci='material_source/derived_row_ci.npz', derived_ci_trainable=False,
        source_role_codes={'0': 'N-parent readonly legacy', '1': 'filtered A0', '2': 'kernel includes generated T2 material'},
        original_observation_ledger_immutable=True, generated_alpha_used_for_GS_opacity=False,
        effective_T2_rows=int(t2.sum()), common_write_rows=int(write_mask.sum()),
        training_method=False, zero_update_numerical_A0_control=(variant == 'A0_mid'), human_verdict=None))


def audit_asset(parent_road, parent_provenance, asset, rows, eligible, expected):
    road = load(Path(asset) / 'road.npz')
    p = load(Path(asset) / 'provenance.npz')
    checks = {f'road_exact_{key}': bytes_equal(value, road[key]) for key, value in parent_road.items() if key != 'reference_srgb'}
    checks['effective_RGB_exact'] = bytes_equal(road['reference_srgb'][rows], expected)
    world_mask = np.zeros(len(road['active']), bool)
    world_mask[rows] = eligible
    checks['all_other_world_RGB_exact'] = bytes_equal(road['reference_srgb'][~world_mask], parent_road['reference_srgb'][~world_mask])
    # effective_row_rgb is deliberately derived; every other original field remains exact.
    checks.update({f'prior_exact_{key}': bytes_equal(value, p[key]) for key, value in parent_provenance.items() if key != 'effective_row_rgb'})
    if not all(checks.values()):
        raise AssertionError('asset/ledger byte audit failed: ' + str([k for k, ok in checks.items() if not ok]))
    result = dict(status='PASS', checks=checks, source_known_unknown_observed_ledger_exact=True,
        all_geometry_coverage_exact=True, only_road_field_changed='reference_srgb', optimizer_updates=0)
    write(Path(asset) / 'byte_audit.json', result)
    return result


def execute(args):
    prepared = json.loads((args.preparation / 'manifest.json').read_text())
    contract = json.loads(args.contract.read_text())
    if contract.get('gpu_authorized') is not True or contract.get('optimizer_updates') != 0 or contract.get('frames') != list(FRAMES) or contract.get('variants') != list(VARIANTS):
        raise ValueError('explicit unique zero-update native-pair contract missing')
    if contract.get('prepared_manifest_sha256') != sha(args.preparation / 'manifest.json'):
        raise ValueError('native contract/preparation binding mismatch')
    for path, expected in prepared['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError('frozen source binding changed: ' + path)
    if sha(prepared['input_npz']) != prepared['input_sha256']:
        raise ValueError('prepared colors changed')
    if args.output.exists():
        raise FileExistsError('preserve previous native site')
    sys.path.insert(0, str(G / 'code'))
    from road_capacity_native import assert_gpu_lease, load_context
    assert_gpu_lease(args.lease_file, args.lease_owner)
    import torch
    from PIL import Image
    torch.set_num_threads(1)
    inputs = load(prepared['input_npz'])
    cache = load(T / 'midpoint_full_r1/row_queries.npz')
    rows, eligible = inputs['active_world_rows'], inputs['eligible']
    args.output.mkdir(parents=True)
    shutil.copyfile(args.preparation / 'manifest.json', args.output / 'preparation_manifest.json')
    shutil.copyfile(args.contract, args.output / 'execution_contract.json')
    shutil.copyfile(__file__, args.output / 'source.py')
    ctx = None
    total_calls = 0
    begin = time.monotonic()
    columns = {}
    try:
        ctx = load_context(asset_dir=prepared['coverage_parent'], frames=FRAMES, lease_file=args.lease_file, lease_owner=args.lease_owner)
        if not np.array_equal(ctx.active_world_rows, rows) or not bytes_equal(ctx.base_rgb.cpu().numpy(), inputs['parent_rgb']):
            raise ValueError('actual native packet differs from prepared parent')
        parent_road, parent_provenance = ctx.road, ctx.provenance
        for variant in VARIANTS:
            directory = args.output / variant
            rendered = directory / 'rendered'
            rendered.mkdir(parents=True)
            color = torch.as_tensor(inputs[variant], device=ctx.base_rgb.device)
            native_rows = []
            with torch.no_grad():
                for raw in FRAMES:
                    road, joint = ctx.render_road(raw, color), ctx.render_joint(raw, color)
                    baseline = load(prepared['parent_float_files'][str(raw)])
                    K = np.asarray(ctx.K_masks[raw], bool)
                    if not bytes_equal(K, baseline['K']):
                        raise ValueError('native K-mask differs from fixed denominator')
                    road_rgb = road['rgb'].cpu().numpy()
                    alpha = road['alpha'][0].cpu().numpy()
                    joint_rgb = joint['rgb'].cpu().numpy()
                    Kdiff = float(np.abs(joint_rgb[:, K].astype(np.float64) - baseline['joint_rgb'][:, K]).max())
                    alpha_diff = float(np.abs(alpha.astype(np.float64) - baseline['road_alpha']).max())
                    np.savez_compressed(rendered / f'{raw:03d}_native.npz', road_rgb=road_rgb, road_alpha=alpha,
                        joint_rgb=joint_rgb, joint_alpha=joint['alpha'][0].cpu().numpy(), K=K,
                        parent_road_alpha=baseline['road_alpha'])
                    for key, image in [('road', road_rgb), ('joint', joint_rgb)]:
                        Image.fromarray(np.round(image.transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)).save(rendered / f'{raw:03d}_{key}.webp', lossless=True)
                    native_rows.append(dict(raw=raw, role='reference' if raw in (86, 110) else 'diagnostic_not_blind',
                        K_pixels=int(K.sum()), joint_K_max=Kdiff, alpha_max=alpha_diff))
                    if Kdiff != 0 or alpha_diff != 0:
                        write(directory / 'failed_protection.json', native_rows)
                        raise RuntimeError('full native K/coverage exact gate failed')
            world_mask = np.zeros(len(ctx.road['active']), bool)
            world_mask[rows] = eligible
            asset = directory / 'asset'
            ctx.save_asset(color, world_mask, asset, training_record=dict(role=variant, optimizer_updates=0,
                new_inference_calls=0, authoritative_T2_meshUV_PNG=True, derived_ci_not_trainable=True,
                source_midpoint_cache=str(T / 'midpoint_full_r1/row_queries.npz')))
            augment_provenance(asset, variant, rows, eligible, cache, inputs[variant])
            byte_audit = audit_asset(parent_road, parent_provenance, asset, rows, eligible, inputs[variant])
            columns[variant] = dict(asset=str(asset), rendered=str(rendered), native_protection=native_rows,
                byte_checks=len(byte_audit['checks']), public_reload_pending=True)
        ctx.assert_frozen()
        total_calls += ctx.native_calls
        ctx.close()
        ctx = None
        # 逐列公共loader只重放5road，直接对已保存float，不重复expected forward。
        for variant in VARIANTS:
            column = columns[variant]
            ctx = load_context(asset_dir=column['asset'], frames=FRAMES, lease_file=args.lease_file, lease_owner=args.lease_owner)
            replay = []
            with torch.no_grad():
                for raw in FRAMES:
                    expected = load(Path(column['rendered']) / f'{raw:03d}_native.npz')
                    actual = ctx.render_road(raw)
                    rgb = actual['rgb'].cpu().numpy()
                    alpha = actual['alpha'][0].cpu().numpy()
                    errors = dict(raw=raw, RGB_max=float(np.abs(rgb.astype(np.float64) - expected['road_rgb']).max()),
                        alpha_max=float(np.abs(alpha.astype(np.float64) - expected['road_alpha']).max()),
                        RGB8_max=int(np.abs(np.round(rgb * 255) - np.round(expected['road_rgb'] * 255)).max()))
                    replay.append(errors)
                    if any(errors[key] != 0 for key in ['RGB_max', 'alpha_max', 'RGB8_max']):
                        raise RuntimeError('public road loader is not exact: ' + str(errors))
            result = dict(public_loader=True, GPU_executed=True, rows=replay, scope='roadRGB/alpha/RGB8 only; joint K verified by direct native column',
                all_exact=True, native_calls=ctx.native_calls, optimizer_updates=0)
            write(Path(column['asset']).parent / 'public_reload.json', result)
            column['public_reload_pending'] = False
            column['public_reload'] = result
            ctx.assert_frozen()
            total_calls += ctx.native_calls
            ctx.close()
            ctx = None
        if total_calls != 30:
            raise RuntimeError('unexpected native-call accounting: ' + str(total_calls))
        write(args.output / 'columns.json', columns)
        write(args.output / 'completed.json', dict(schema='TRELLIS2_zero_update_native_pair.v1', completed=True,
            GPU_executed=True, native_forward_calls=total_calls, native_backward_calls=0, optimizer_updates=0,
            new_inference_calls=0, protected_pixel_frames_per_column=77266, all_native_alpha_exact=True,
            all_native_joint_K_exact=True, all_public_road_RGB_alpha_RGB8_exact=True, columns=columns,
            elapsed_seconds=time.monotonic() - begin, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), production_promoted=False, human_verdict=None))
        print(json.dumps(dict(completed=True, output=str(args.output), native_forward_calls=total_calls)), flush=True)
    finally:
        if ctx is not None:
            ctx.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['inspect', 'prepare', 'native'], default='inspect')
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--preparation', type=Path, default=T / 'native_preparation_r1')
    ap.add_argument('--output', type=Path, default=T / 'native_columns_r1')
    ap.add_argument('--contract', type=Path, default=T / 'native_execution_contract.json')
    ap.add_argument('--lease-file', type=Path, default=P / 'gpu_lease.json')
    ap.add_argument('--lease-owner', default='static_handoff')
    args = ap.parse_args()
    if args.mode == 'inspect':
        print(json.dumps(dict(native_default=False, optimizer_updates=0, variants=list(VARIANTS), frames=list(FRAMES),
            common_eligible_rows=84266, planned_native_forward_calls=30)))
    elif args.mode == 'prepare':
        prepare(args.preparation)
    elif not args.execute:
        raise SystemExit('native requires --execute, explicit root contract and matching GPU lease')
    else:
        execute(args)


if __name__ == '__main__':
    main()
