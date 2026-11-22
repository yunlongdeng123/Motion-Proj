"""整理本轮完整链路/求解矩阵、packet 与恢复依赖；默认只准备，不压缩。"""
from pathlib import Path
import argparse
import datetime
import gzip
import importlib.metadata
import json
import os
import platform
import sys
import tarfile
import time
import numpy as np
from scipy.sparse import load_npz
from merge_resource_ledger import ROOT_DEFAULT, merge, read, write
from restore_manifest import sha, verify_archive


def within(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def row_matrix(stage, kind, receipt, contract):
    rows = []
    source_rows = [receipt] if kind == 'camera_solver' and receipt.get('evaluations') else receipt.get('rows', [])
    for r in source_rows:
        if kind == 'chain':
            rows.append(dict(stage=stage, kind=kind, patch=r.get('patch'), raw=r.get('raw'),
                reference=r.get('raw'), role='same_view_chain_control', score_n=r.get('score_n'),
                metrics=r.get('metrics'), stored_to_runtime_base_max_abs=r.get('stored_to_runtime_base_max_abs'),
                sample_to_saved_RGB_max_abs=r.get('sample_to_saved_RGB_max_abs'),
                direct_to_reloaded_render_max_abs=r.get('direct_to_reloaded_render_max_abs'),
                constant_render_max_abs=r.get('constant_render_max_abs'),
                non_RGB_fields_exact=r.get('non_RGB_fields_exact'), halo_RGB_exact=r.get('halo_RGB_exact'), alpha_exact=r.get('alpha_exact')))
        elif kind in ('solver', 'camera_solver'):
            for v in r.get('evaluations', []):
                rows.append(dict(stage=stage, kind=kind, patch=r.get('patch'), raw=v.get('raw'),
                    reference=r.get('reference'), role=v.get('role'), score_n=v.get('score_n'),
                    metrics=v.get('metrics'), raw_camera_metrics=v.get('raw_camera_metrics'), solver=r.get('solver'),
                    Brown_inverse_roundtrip_max_px=v.get('Brown_inverse_roundtrip_max_px'),
                    operator_linear_check_max_abs=r.get('operator_linear_check_max_abs'),
                    source_RGB_noncolor_exact=r.get('source_RGB_noncolor_exact', r.get('source_non_RGB_fields_exact')),
                    halo_RGB_exact=r.get('halo_RGB_exact'), alpha_exact=r.get('alpha_exact')))
        else:
            rows.append(dict(stage=stage, kind=kind, raw=r.get('raw'), patch=r.get('patch', contract.get('patch')),
                reference=contract.get('reference'),
                role='REFERENCE_SELECTION' if r.get('raw') == contract.get('reference') else 'QUERY_not_fit',
                score_n=r.get('score_n'), metrics=r.get('metrics'), original_row=r))
    return rows


def audit_matrix(root, expected):
    result, flat, problems = [], [], []
    declared = {s['directory']: s for s in expected['expected_stages']}
    discovered = {p.parent.name: p.parent for p in root.glob('*/contract.json')}
    for stage in sorted(set(declared) | set(discovered)):
        requirement = declared.get(stage)
        directory = root / stage
        contract = read(directory / 'contract.json') if (directory / 'contract.json').exists() else {}
        receipt = read(directory / 'receipt.json') if (directory / 'receipt.json').exists() else {}
        kind = (requirement or {}).get('kind', 'solver' if 'fixed_color_operator' in stage else 'chain' if 'chain_probe' in stage else 'diagnostic')
        records = row_matrix(stage, kind, receipt, contract)
        flat.extend(records)
        stage_problems = []
        if requirement is None:
            stage_problems.append('not_listed_in_expected_matrix')
        if not receipt.get('complete', False):
            stage_problems.append('missing_complete_receipt')
        if requirement:
            for patch, raws in requirement.get('patches', {}).items():
                actual = {r['raw'] for r in records if r.get('patch') == patch}
                if actual != set(raws):
                    stage_problems.append(f'{patch}: expected frames {raws}, actual {sorted(actual)}')
            for key in ('nativeF', 'nativeB'):
                if 'expected_' + key in requirement and receipt.get(key) != requirement['expected_' + key]:
                    stage_problems.append(key + '_does_not_match_expected')
            if 'reference' in requirement and any(r.get('reference') != requirement['reference'] for r in records):
                stage_problems.append('wrong_reference')
        if receipt.get('original_asset_modified', False):
            stage_problems.append('original_asset_modified')
        if receipt.get('new_diffusion_calls', 0):
            stage_problems.append('unexpected_diffusion')
        if receipt.get('target_migration', False):
            stage_problems.append('unexpected_target_migration')
        allowed_failure = bool(requirement and requirement.get('preserve_incomplete_as_failed', False))
        if not allowed_failure:
            problems.extend(stage + ': ' + p for p in stage_problems)
        result.append(dict(stage=stage, kind=kind, complete=bool(receipt.get('complete', False)),
            expected=requirement, contract=contract, receipt=str(directory / 'receipt.json'),
            nativeF=receipt.get('nativeF'), nativeB=receipt.get('nativeB'),
            matrix_rows=len(records), problems=stage_problems,
            incomplete_failure_preserved_by_contract=allowed_failure))
    if not expected.get('finalized_by_root', False):
        problems.append('Expected matrix is not finalized by root; compression remains disabled')
    for name in expected.get('required_final_artifacts', []):
        if not (root / name).is_file():
            problems.append('Missing final artifact: ' + name)
    return dict(complete=not problems, stages=result, rows=flat, problems=problems)


def packet_audit(root, selection, expected):
    baselines = {r['candidate_id']: Path(r['packet']) for r in selection['candidates']}
    stages = {s['directory']: s for s in expected['expected_stages']}
    results, problems = [], []
    for path in sorted(root.glob('**/*.npz')):
        if path.relative_to(root).parts[0] not in stages:
            continue
        if path.name not in ('source_direct_rgb.npz', 'source_fixed_RGB.npz', 'source_camera_corrected_RGB.npz') or 'archives' in path.parts:
            continue
        cid = next((part for part in path.parts if part in baselines), None)
        if cid is None and (path.parent / 'contract.json').exists():
            cid = read(path.parent / 'contract.json').get('patch')
        if cid is None:
            problems.append('Cannot bind derived packet: ' + str(path))
            continue
        with np.load(baselines[cid], allow_pickle=False) as src, np.load(path, allow_pickle=False) as dst:
            same_keys = set(src.files) == set(dst.files)
            noncolor = same_keys and all(np.array_equal(src[k], dst[k]) for k in src.files if k != 'reference_srgb')
            core = src['core_mask']
            halo = same_keys and np.array_equal(src['reference_srgb'][~core], dst['reference_srgb'][~core])
            colors = dst['reference_srgb'][core]
            finite = bool(np.isfinite(colors).all())
            bounded = bool((colors >= -1e-7).all() and (colors <= 1 + 1e-7).all())
            record = dict(packet=str(path), relative_packet=path.relative_to(root).as_posix(), source_candidate=cid,
                original_packet=str(baselines[cid]), same_keys=same_keys, all_noncolor_fields_exact=noncolor,
                halo_RGB_exact=halo, core_RGB_finite=finite, core_RGB_in_0_1=bounded,
                Gaussian_rows=int(len(core)), core_Gaussian_rows=int(core.sum()),
                core_RGB_min_max=[float(colors.min()), float(colors.max())],
                source_asset_field=str(src['source_asset']), full_parent_scene_in_packet=False)
            if not all([same_keys, noncolor, halo, finite, bounded]):
                problems.append('Derived packet invariant failed: ' + str(path))
        if path.name in ('source_fixed_RGB.npz', 'source_camera_corrected_RGB.npz'):
            state, operator = path.parent / 'linear_state.npz', path.parent / 'native_color_operator.npz'
            reused = path.name == 'source_camera_corrected_RGB.npz'
            if reused:
                prior = stages[path.parent.name]['reuses_operator_from']
                operator = root / prior / 'native_color_operator.npz'
                record['operator_reused_from_prior_branch'] = True
            record['linear_state'] = str(state)
            record['native_color_operator'] = str(operator)
            if not state.exists() or not operator.exists():
                problems.append('Missing solver recovery state: ' + str(path))
            else:
                A = load_npz(operator)
                with np.load(state, allow_pickle=False) as z:
                    if reused:
                        A = A[z['reused_operator_row_selection']]
                    record['operator_shape'] = list(A.shape)
                    record['operator_nnz'] = int(A.nnz)
                    record['linear_state_shapes'] = {k: list(z[k].shape) for k in z.files}
                    exact = np.array_equal(z['fitted_core_RGB'], colors)
                    record['linear_state_to_packet_core_RGB_exact'] = exact
                    if not exact or A.shape != (z['target_minus_fixed'].shape[0], colors.shape[0]):
                        problems.append('Solver matrix/packet mapping mismatch: ' + str(path))
        results.append(record)
    for path in sorted(root.glob('*/frozen*source.npz')):
        with np.load(path, allow_pickle=False) as z:
            fields = {k: dict(shape=list(z[k].shape), dtype=str(z[k].dtype),
                all_finite=bool(np.isfinite(z[k]).all()) if z[k].dtype.kind in 'fc' else None)
                for k in z.files}
            if any(v['all_finite'] is False for v in fields.values()):
                problems.append('Nonfinite original native frozen source: ' + str(path))
        results.append(dict(packet=str(path), relative_packet=path.relative_to(root).as_posix(),
            packet_kind='frozen_original_native_static_capture', fields=fields,
            original_parent_required=True, full_parent_scene_in_packet=False,
            geometry_was_captured_not_optimized=True))
    return dict(complete=not problems, packets=results, problems=problems)


def collect(root, campaign, files):
    def add(path, archive_path):
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(p)
        if archive_path in files and files[archive_path].resolve() != p.resolve():
            raise ValueError('Archive name collision: ' + archive_path)
        files[archive_path] = p

    def tree(directory, prefix):
        for p in sorted(Path(directory).rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix in ('.pyc', '.tmp'):
                continue
            add(p, prefix + '/' + p.relative_to(directory).as_posix())

    # Capture all present branches, including failed/incomplete attempts and exact
    # float arrays, sparse color operators, RGB states, source packets, and logs.
    static_dirs = {'code', 'photo_audit', 'color_chain_audit', 'publication', 'report', 'report_source', 'masks', 'runtime'}
    for directory in sorted(root.iterdir()):
        if directory.is_dir() and (directory.name in static_dirs or (directory / 'contract.json').exists()):
            tree(directory, 'run/' + directory.name)
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix in ('.json', '.jsonl', '.log', '.txt', '.md'):
            add(p, 'run/' + p.name)
    for p in sorted((root / 'recovery').glob('*')):
        if p.is_file() and (p.suffix == '.py' or p.name in ('expected_matrix.json', 'photo_audit_mirror_receipt.json', 'execution_environment.json', 'recovery_usage.json')):
            add(p, 'run/recovery/' + p.name)

    previous = campaign / 'road_gaussian_source_quality_20261122_r1'
    prior = campaign / 'road_gaussian_transfer_20261122_r2_isometric'
    for p, arc in [
        (previous / 'code/render_source1920.py', 'dependencies/source_quality_r1/code/render_source1920.py'),
        (previous / 'restore_manifest_compact.json', 'dependencies/source_quality_r1/restore_manifest_compact.json'),
        (prior / 'restore_manifest_compact.json', 'dependencies/isometric_r2/restore_manifest_compact.json'),
        (prior / 'code/native.py', 'dependencies/isometric_r2/code/native.py'),
        (campaign / 'road_surface_adaptive_20261121_r1/code/run_adaptive.py', 'dependencies/adaptive/code/run_adaptive.py'),
    ]:
        add(p, arc)
    for name in ('manifest.json', 'proposal_audit.json'):
        add(prior / 'source_selection' / name, 'dependencies/isometric_r2/source_selection/' + name)
    for p in (prior / 'source_selection/candidates').glob('*.npz'):
        add(p, 'dependencies/isometric_r2/source_selection/candidates/' + p.name)
    for name in ('manifest.json', 'patch_00_views.json', 'patch_02_views.json'):
        add(previous / 'source_view_ranking' / name, 'dependencies/photo_ranking/' + name)
    # Include actual calibration/ego transforms needed by the executed native
    # cameras, not only the K/T copied into the photo probe plan.
    bank = Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs')
    bank_receipt = read(bank / 'receipt.json')
    ids = {int(x['raw']): x['id'] for x in bank_receipt['frames']}
    raws = {int(r['raw']) for r in read(root / 'probe_plan.json')['records']}
    add(bank / 'receipt.json', 'inputs/source_bank/receipt.json')
    for raw in sorted(raws):
        obs = read(bank / f'observations/{raw:03d}.json')
        for rel in (f'observations/{raw:03d}.json', f'calib/{ids[raw]}.txt', obs['image_path']):
            add(bank / rel, 'inputs/source_bank/' + rel)
    return read(prior / 'source_selection/manifest.json')


def environment():
    versions = {}
    for package in ('numpy', 'scipy', 'Pillow', 'torch', 'safetensors', 'tensorrt'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return dict(python=sys.executable, python_version=sys.version, platform=platform.platform(),
                package_versions=versions, archiver_GPU_calls=0,
                native_python='/root/autodl-tmp/envs/drivingeditor/bin/python',
                CPU_python='/root/autodl-tmp/envs/motionproj/bin/python')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=ROOT_DEFAULT)
    parser.add_argument('--expected', default=None)
    parser.add_argument('--archive', action='store_true', help='Only after root finalizes all branches')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    recovery = root / 'recovery'
    recovery.mkdir(parents=True, exist_ok=True)
    assert root.name == 'road_source_fidelity_20261122_r2', root
    expected_path = Path(args.expected) if args.expected else recovery / 'expected_matrix.json'
    expected = read(expected_path)
    write(recovery / 'execution_environment.json', environment())
    resources = merge(root)
    write(recovery / 'resource_ledger_cumulative.json', resources)
    matrix = audit_matrix(root, expected)
    write(recovery / 'complete_result_matrix.json', matrix)
    files = {}
    selection = collect(root, root.parent, files)
    packets = packet_audit(root, selection, expected)
    write(recovery / 'packet_state_audit.json', packets)
    for name in ('resource_ledger_cumulative.json', 'complete_result_matrix.json', 'packet_state_audit.json'):
        files['run/recovery/' + name] = recovery / name
    inventory = []
    for arc, path in sorted(files.items()):
        before = path.stat()
        digest = sha(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError('File changed during snapshot: ' + str(path))
        inventory.append(dict(archive_path=arc, original_path=str(path), resolved_path=str(path.resolve()),
                              bytes=after.st_size, sha256=digest, mtime_ns=after.st_mtime_ns))
    prior_restore = read(root.parent / 'road_gaussian_transfer_20261122_r2_isometric/restore_manifest_compact.json')
    issues = matrix['problems'] + packets['problems'] + resources['accounting_problems']
    manifest = dict(schema='source_fidelity_curated_restore.v1', complete=not issues,
        run_id=root.name, original_root=str(root), final_matrix_contract=expected,
        created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        files=inventory, payload_files=len(inventory), payload_bytes=sum(r['bytes'] for r in inventory),
        readiness_issues=issues, resource_totals=resources['totals'],
        result_matrix='run/recovery/complete_result_matrix.json', packet_states='run/recovery/packet_state_audit.json',
        root_code_included=True, source_audit_full_native_inputs_included=True,
        full_float_arrays_color_operators_linear_states_and_packets_included=True,
        original_assets_written_by_archiver=False, new_diffusion_calls=0, target_training_updates=0,
        target_migration=False, production_promoted=False, power_operations=0,
        local_delivery_directory=expected.get('selected_output_directory', 'outputs/road_source_fidelity_20261122_r2'),
        prior_parent_recovery=prior_restore,
        external_dependency_policy='Full original persistent scene, static/dynamic/environment weights, GPU engine and environments remain external; restore them through prior_parent_recovery. This archive contains modified source packets, not a new full scene.',
        full_report_companion=read(root / 'report_companion_manifest.json') if (root / 'report_companion_manifest.json').exists() else None,
        path_prefix_map=[[str(root), 'run'], [str(root.parent/'road_gaussian_transfer_20261122_r2_isometric'), 'dependencies/isometric_r2'],
                         [str(root.parent/'road_gaussian_source_quality_20261122_r1'), 'dependencies/source_quality_r1'],
                         ['/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs', 'inputs/source_bank']],
        restore_steps=[
            'Run recovery/restore_manifest.py --archive ARCHIVE to verify every file before use; --extract-to requires a new empty directory.',
            'Restore the original parent persistent scene using prior_parent_recovery and keep engine/environment dependencies available.',
            'Map original_root to extracted run and map the explicit dependency/source-bank prefixes. Preserve archived code; change paths only in a separate working copy.',
            'For each direct packet, load source_direct_rgb.npz. For each solved packet, load source_fixed_RGB.npz with world_row_ids/core_mask and verify packet_state_audit invariants.',
            'native_color_operator.npz, linear_state.npz and contracts retain the measured fixed response and bounded LS state. There is no Adam/gradient optimizer checkpoint in this task.',
            'Use complete_result_matrix.json to separate reference fitting views and query views. No downstream target result or production promotion is implied.',
        ])
    prepared = recovery / 'restore_manifest_prepared.json'
    write(prepared, manifest)
    if not args.archive:
        print(json.dumps(dict(prepared=True, compressed=False, ready_for_archive=manifest['complete'],
            files=len(inventory), bytes=manifest['payload_bytes'], issues=issues, manifest=str(prepared)), ensure_ascii=False))
        return
    if issues:
        raise RuntimeError('Final archive is not ready: ' + json.dumps(issues, ensure_ascii=False))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination = root / 'archives' / f'{root.name}_curated_{stamp}.tar.gz'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not within(destination, root) or destination.exists():
        raise ValueError('Archive destination must be new and remain within current campaign')
    with tarfile.open(destination, 'w:gz', compresslevel=5, dereference=True) as tar:
        for row in inventory:
            path = Path(row['original_path'])
            if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
                raise RuntimeError('Payload changed before archiving: ' + str(path))
            tar.add(path, arcname=row['archive_path'], recursive=False)
        tar.add(prepared, arcname='restore_manifest.json', recursive=False)
    verify_archive(destination)
    # Force gzip CRC/trailer validation in addition to per-member validation.
    with gzip.open(destination, 'rb') as stream:
        for _ in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            pass
    receipt = dict(complete=True, archive=str(destination), archive_bytes=destination.stat().st_size,
                   archive_sha256=sha(destination), payload_files=len(inventory), verified_every_member=True,
                   manifest=str(prepared), GPU_calls=0, power_operations=0, git_operations=0)
    write(recovery / 'archive_receipt.json', receipt)
    compact = dict(schema='source_fidelity_compact_restore.v1', complete=True, run_id=root.name,
        archive=dict(path=receipt['archive'], filename=destination.name, bytes=receipt['archive_bytes'],
                     sha256=receipt['archive_sha256'], payload_files=receipt['payload_files'], verified_every_member=True),
        main_archive_role='本轮全部代码、原照与标定、颜色算子、线性状态、10份源packet、原生浮点/PNG、故障与CPU恢复证据，以及报告本轮补丁。',
        full_report_companion=manifest['full_report_companion'],
        report_companion_role='当前原报告HTML与全部历史/本轮资源的完整恢复备份；不建立新的查看入口。',
        source_gaussian_qualified_for_migration=False, adopted_source_asset=None,
        resources=resources['totals'], result_matrix='run/recovery/complete_result_matrix.json',
        packet_state_audit='run/recovery/packet_state_audit.json',
        packet_recovery_entries=[{k:v for k,v in r.items() if k in (
            'relative_packet','packet_kind','source_candidate','Gaussian_rows','core_Gaussian_rows',
            'all_noncolor_fields_exact','halo_RGB_exact','linear_state_to_packet_core_RGB_exact','operator_shape')}
            for r in packets['packets']],
        prior_parent_recovery=prior_restore, path_prefix_map=manifest['path_prefix_map'],
        restore_steps=manifest['restore_steps'] + [
            '完整原报告由已验证full_report_companion恢复；主包report_source/publication是本轮HTML与新增资源补丁。'],
        native_python='/root/autodl-tmp/envs/drivingeditor/bin/python',
        CPU_python='/root/autodl-tmp/envs/motionproj/bin/python',
        Git_operations=0, power_operations=0)
    write(root / 'restore_manifest_compact.json', compact)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
