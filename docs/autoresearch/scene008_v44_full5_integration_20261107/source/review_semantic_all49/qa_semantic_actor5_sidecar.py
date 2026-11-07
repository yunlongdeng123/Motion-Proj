"""独立 CPU 重算 BUILD 可见质量 sidecar；不渲染、不更改模型身份。"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import torch

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_full5_integration_20261107_r1')
FOLDER = ROOT / 'semantic_actor5_calibration4'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=FOLDER)
    parser.add_argument('--output', type=Path, default=ROOT / 'review/cpu_semantic_actor5_sidecar_audit.json')
    options = parser.parse_args()
    folder = options.folder.resolve()
    assert not torch.cuda.is_initialized()
    manifest = json.loads((folder / 'semantic_actor5_manifest.json').read_text())
    receipt = json.loads((folder / 'receipt.json').read_text())
    assert manifest['schema'] == 'scene008.semantic_actor5.visible_mass.v1'
    assert manifest['raw_ids'] in ([94, 105, 130, 137], list(range(89, 138)))
    assert manifest['threshold'] == dict(purity_strictly_greater_than=.5, total_mass_strictly_greater_than=0.)
    assert manifest['gaussian_parameter_updates'] == receipt['optimization_updates'] == 0
    expected_work = 4 if len(manifest['raw_ids']) == 4 else 45
    assert receipt['joint_forwards'] == receipt['dummy_feature_backwards'] == expected_work
    if len(manifest['raw_ids']) == 49:
        assert receipt['total_visibility_views'] == 49
        assert receipt['reused_raw_ids'] == [94, 105, 130, 137]
        for raw in receipt['reused_raw_ids']:
            assert sha(folder / f'point_mass/{raw:03d}.npz') == sha(FOLDER / f'point_mass/{raw:03d}.npz')
    for key in ['model_parameters_environment_and_original_identity_unchanged', 'source_files_unchanged', 'mask_files_unchanged']:
        assert receipt[key] is True
    for file_key, sha_key in [('selection_file', 'selection_sha256'), ('protocol_file', 'protocol_sha256'),
                              ('ancestry_file', 'ancestry_sha256')]:
        assert sha(folder / manifest[file_key]) == manifest[sha_key]
    source = Path(manifest['source_model'])
    for name, expected in manifest['gaussian_files_sha256'].items():
        assert sha(source / name) == expected
    with np.load(folder / manifest['selection_file'], allow_pickle=False) as saved:
        values = {key: saved[key].copy() for key in saved.files}
    with np.load(folder / manifest['ancestry_file'], allow_pickle=False) as saved:
        ancestry = {key: saved[key].copy() for key in saved.files}
    for branch, prefix in [('static', 'chkpnt'), ('dynamic', 'dynamic_chkpnt')]:
        payload, iteration = torch.load(str(source / f'{prefix}10000.pth'), map_location='cpu', weights_only=False, mmap=True)
        assert iteration == 10000 and len(payload[1]) == manifest['branch_rows'][branch]
        assert np.array_equal(payload[14]['sgcm_lineage_v1']['lineage_ids'].numpy(), ancestry[branch + '_lineage_ids'])
        if branch == 'dynamic':
            assert np.array_equal(payload[14]['dynamic_actor_ownership_v1']['actor_ids'].numpy(), ancestry['dynamic_original_owner_ids'])
            assert np.array_equal(payload[14]['dynamic_actor_ownership_v1']['source_actor_ids'].numpy(), ancestry['dynamic_source_owner_table'])
        del payload
    count_static = manifest['branch_rows']['static']
    inside = np.zeros(sum(manifest['branch_rows'].values()), dtype=np.float64)
    outside = inside.copy()
    max_conservation_error = 0.
    for frame in receipt['frames']:
        raw = frame['raw']
        with np.load(folder / f'point_mass/{raw:03d}.npz', allow_pickle=False) as saved:
            assert int(saved['branch_boundary']) == count_static
            a, b = saved['inside_mass'].astype(np.float64), saved['outside_mass'].astype(np.float64)
        assert np.isfinite(a).all() and np.isfinite(b).all() and (a >= 0).all() and (b >= 0).all()
        inside += a
        outside += b
        for actual, expected in [(float(a.sum()), frame['joint_alpha_inside_sum']),
                                 (float(b.sum()), frame['joint_alpha_outside_sum'])]:
            relative = abs(actual - expected) / max(abs(expected), 1e-12)
            assert relative <= 1e-5
            max_conservation_error = max(max_conservation_error, relative)
        assert frame['dummyones_alpha_max_abs'] == [0., 0.] and frame['model_grad_none']
    selected, matrix = {}, []
    owners = ancestry['dynamic_original_owner_ids']
    slices = dict(static=slice(0, count_static), dynamic=slice(count_static, None))
    for branch in ['static', 'dynamic']:
        a, total = values[branch + '_inside_mass'], values[branch + '_total_mass']
        assert np.array_equal(a, inside[slices[branch]])
        assert np.array_equal(total, (inside + outside)[slices[branch]])
        purity = np.divide(a, total, out=np.zeros_like(total), where=total > 0)
        expected = np.flatnonzero((total > 0) & (purity > .5))
        rows = values[branch + '_row_ids']
        assert rows.dtype == np.int64 and np.array_equal(rows, expected)
        selected[branch] = np.zeros(len(a), dtype=bool)
        selected[branch][rows] = True
        categories = [('static', np.ones(len(a), dtype=bool))] if branch == 'static' else [(str(i), owners == i) for i in [5, 6, 18, 26, 37]]
        for label, category in categories:
            chosen = selected[branch] & category
            matrix.append(dict(source=label, rows=int(category.sum()), selected_rows=int(chosen.sum()),
                unobserved_rows=int((category & (total == 0)).sum()),
                selected_inside_mass=float(a[chosen].sum()), selected_total_mass=float(total[chosen].sum())))
    all_inside = float(inside.sum())
    selected_inside = sum(row['selected_inside_mass'] for row in matrix)
    selected_total = sum(row['selected_total_mass'] for row in matrix)
    capture, spill = selected_inside / all_inside, (selected_total - selected_inside) / selected_total
    assert abs(capture - receipt['selection_capture_fraction']) < 1e-12
    assert abs(spill - receipt['selection_spill_fraction']) < 1e-12
    for row, original in zip(matrix, receipt['source_matrix']):
        for key, value in row.items():
            assert abs(value - original[key]) < 1e-8 if isinstance(value, float) else value == original[key]
    result = dict(status='PASS_INDEPENDENT_CPU_SEMANTIC_SIDECAR_AND_ANCESTRY_AUDIT',
        selected_rows={branch: int(mask.sum()) for branch, mask in selected.items()}, source_matrix=matrix,
        capture_fraction=capture, spill_fraction=spill, fixed_purity_threshold=.5, strict_greater=True,
        perpoint_mass_is_SUM_of_all_saved_BUILD_gradients_exact=True,
        total_visibility_views=len(manifest['raw_ids']), reused_visibility_views=0 if expected_work == 4 else 4,
        both_checkpoint_owner_and_lineage_arrays_unchanged=True, two_Gaussian_checkpoint_file_SHA256_exact=True,
        model_and_mask_source_guards_pass=True, maximum_mass_conservation_relative_error=max_conservation_error,
        dummyones_equals_joint_alpha_exact=True, new_native_forwards=expected_work, new_dummy_feature_backwards=expected_work,
        gaussian_parameter_updates=0, optimizer_created=False, source_ancestry_is_not_semantic_GT=True,
        selected_semantics_are_BUILD_majority_estimates_not_heldout_validation=True,
        CUDA_initialized=torch.cuda.is_initialized(), semantic_composition_quality_verdict=None)
    options.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
