"""Independent saved-8F check of the single prescribed L counterfactual."""
from pathlib import Path
import hashlib
import json
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np

B = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
R = B/'scene008_v44_a5_road_fusion_integration_20261107_r1'
OP = R/'J4_road_parent_operator'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    root = OP/'L_Cwarm12_render_v1'
    receipt_path = root/'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['status'] == 'COMPLETE_L_CWARM12_NATIVE8F_PENDING_ORIGINAL_94_GATES_AND_ROOT_REVIEW'
    assert receipt['actual_native_forwards'] == 8
    assert receipt['actual_dummy_backwards'] == receipt['new_solver_calls'] == receipt['Adam_updates'] == receipt['original_model_updates'] == 0
    contract = R/'contract/L_Cwarm12_counterfactual_v1.json'
    c = json.loads(contract.read_text())
    assert sha(contract) == receipt['contract_sha256']
    mpath = Path(c['scene_manifest'])
    m = json.loads(mpath.read_text())
    asset = mpath.parent/m['new_road']['file']
    assert sha(mpath) == c['scene_manifest_sha256']
    assert sha(asset) == c['asset_sha256'] == m['new_road']['sha256']
    source_files = [receipt_path, contract, mpath, asset, Path(__file__),
        OP/'operator_manifest.json', OP/'road_parent_operator.npz',
        OP/'K_single_v1/result.json', OP/'K_single_v1/K_candidate_parent_RGB.npz']
    input_hashes = {str(p): sha(p) for p in source_files}
    a = np.load(OP/'road_parent_operator.npz')
    k = np.load(OP/'K_single_v1/K_candidate_parent_RGB.npz')
    l = np.load(asset)
    table = l['parent_render_colours_srgb']
    parents = np.asarray(c['fixed_parent_ids'], np.int64)
    mask = np.ones(10098, bool)
    mask[parents] = False
    assert np.array_equal(table[mask], k['full_parent_RGB'][mask])
    warm_path = B/'scene008_road_fusion_bake_20261107_r1/engineer/canonical_edgeblend_v1/new_road_canonical_candidate.npz'
    with np.load(warm_path) as z:
        assert np.array_equal(table[parents], z['parent_render_colours_srgb'][parents])
    assert np.array_equal(l['render_colours_srgb'], np.repeat(table, 4, axis=0))
    assert np.count_nonzero(a['full_parent_A'][:, parents]) == 0
    delta = table[a['eligible_parent_ids']].astype(np.float64)-a['initial_parent_RGB'].astype(np.float64)
    expected = a['baseline_J']+a['A']@delta
    k_delta = k['full_parent_RGB'][a['eligible_parent_ids']].astype(np.float64)-a['initial_parent_RGB'].astype(np.float64)
    assert np.array_equal(expected, a['baseline_J']+a['A']@k_delta)
    cert = json.loads((OP/'K_single_v1/result.json').read_text())
    sys.path.insert(0, str(R/'source'))
    from registered_seam_pairs import seam_pairs
    world = B/'scene008_road_fusion_bake_20261107_r1/red_review/world_masks49_v1'
    pairs = {raw: seam_pairs(raw, world) for raw in [94, 105, 128, 137]}
    actual = np.empty_like(expected)
    k_actual = np.empty_like(expected)
    files = []
    points = []
    for raw in pairs:
        observed_path = B/f'scene008_instances_20261105_r1/observations/frames/{raw:03d}.npz'
        with np.load(observed_path) as z:
            observed = z['rgb'].transpose(2, 0, 1).copy()
        for state in ['keep', 'delete']:
            p = root/f'frames/{raw:03d}/{state}_float.npz'
            q = OP/f'K_candidate_render_v1/frames/{raw:03d}/{state}_float.npz'
            current, previous = np.load(p), np.load(q)
            for key in ['alpha', 'depth_sum', 'depth_m', 'group_alpha_T', 'added_owner_rgb_T']:
                assert current[key].dtype == previous[key].dtype
                assert current[key].shape == previous[key].shape
                assert current[key].tobytes() == previous[key].tobytes(), (raw, state, key)
            residual = current['rgb']-observed
            old_residual = previous['rgb']-observed
            for row in json.loads((OP/'operator_manifest.json').read_text())['seam_records']:
                if row['raw'] == raw and row['state'] == state:
                    pair = pairs[raw][row['raw_pair_index']]
                    actual[row['operator_row']] = (residual[:, pair['plus_mask']].mean(1)-residual[:, pair['minus_mask']].mean(1)).astype(np.float64)
                    k_actual[row['operator_row']] = (old_residual[:, pair['plus_mask']].mean(1)-old_residual[:, pair['minus_mask']].mean(1)).astype(np.float64)
            files.append(dict(raw=raw, state=state, path=str(p), sha256=sha(p),
                source_K_sha256=sha(q), alpha_depth_8groups_semanticRGB_byte_exact=True))
            if state == 'delete' and raw in [128, 137]:
                x, y = (301, 221) if raw == 128 else (300, 227)
                versions = dict(L=root, K=OP/'K_candidate_render_v1',
                    Cwarm=R/'H4_model_foreground_culling_v2', latest21K=R/'baseline4')
                values = {}
                gt = observed[:, y, x].astype(np.float64)
                for name, folder in versions.items():
                    with np.load(folder/f'frames/{raw:03d}/delete_float.npz') as z:
                        rgb = z['rgb'][:, y, x].astype(np.float64)
                    values[name] = dict(RGB=rgb.tolist(), point_MAE=float(np.abs(rgb-gt).mean()))
                warm_rgb = np.asarray(values['Cwarm']['RGB'])
                error = float(np.abs(np.asarray(values['L']['RGB'])-warm_rgb).max())
                assert error <= 1e-6
                points.append(dict(raw=raw, state=state, pixel_xy=[x, y], observed_RGB=gt.tolist(),
                    versions=values, actual_L_vs_saved_Cwarm_total_RGB_max_abs=error))
    affine_error = float(np.abs(actual-expected).max())
    signed_K_change = float(np.abs(actual-k_actual).max())
    assert affine_error <= 1e-6
    pools = []
    for group in cert['pools']:
        score = float(np.abs(actual[group['rows']]).mean())
        pools.append(dict(name=group['name'], state=group['state'], bins=group['bins'],
            predicted_score=float(np.abs(expected[group['rows']]).mean()), actual_score=score,
            original_target=group['original_target'], margin=group['original_target']-score,
            actual_original_pass=score <= group['original_target']))
    bins = []
    for group in cert['per_bin_constraints']:
        row = group['operator_row']
        score = float(np.abs(actual[row]).mean())
        target = group['original_no_regression_target']
        bins.append(dict(operator_row=row, raw=group['raw'], state=group['state'],
            kind=group['kind'], edge=group['edge'], tangent_bin=group['tangent_bin'],
            predicted_score=float(np.abs(expected[row]).mean()), actual_score=score,
            original_target=target, margin=target-score, actual_original_pass=score <= target))
    assert len(bins) == 106 and all(p['actual_original_pass'] for p in pools+bins)
    for p, expected_sha in input_hashes.items():
        assert sha(p) == expected_sha
    report = dict(status='PASS_L_REAL8F_SIX_POOLS_106_BINS_AND_TWO_POINT_CAUSAL_REPLAY_CPU',
        actual_candidate_native_forwards=8, audit_added_forwards=0,
        audit_backwards=0, audit_solver_calls=0, audit_optimizer_updates=0,
        fixed_union_parents=parents.tolist(), actual_changed_parent_count=int(np.any(table != k['full_parent_RGB'], axis=1).sum()),
        full106_by_union_A_all_exact_zero=True, predicted_signed_J_bit_equal_K=True,
        native_residual_affinity_max_abs_error=affine_error,
        actual_native_signed_J_vs_K_max_abs_change=signed_K_change,
        native_alpha_depth_8groups_semanticRGB_all_byte_exact_to_K=True,
        six_pools=pools, all_six_original_pools_actual_pass=True,
        all106_bin_checks=bins, all106_original_local_constraints_actual_pass=True,
        two_registered_DELETE_points=points, actual_files=files, source_files_sha256=input_hashes,
        original94_quality_gates='Independent evaluator owns the full registered image/geometry assessment; not inferred from these seam checks.',
        general_protection_fix_claim=False, quality_adopted=False,
        causal_scope='Only the two registered points and fixed twelve-parent Cwarm rollback are tested; no generic parent selection policy is established.',
        source_unchanged=True)
    out = R/'review/independent_L_real8F_affinity_cpu.json'
    assert not out.exists()
    out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(out), sha256=sha(out), affine_error=affine_error,
        actual_signed_K_change=signed_K_change, six_pools_pass=True, all106_pass=True,
        points=points), indent=2))


if __name__ == '__main__':
    main()
