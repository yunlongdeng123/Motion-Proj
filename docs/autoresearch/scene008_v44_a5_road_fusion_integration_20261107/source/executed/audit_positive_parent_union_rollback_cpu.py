"""One prescribed, in-memory Cwarm rollback; no solve, asset export, or CUDA."""
from pathlib import Path
import hashlib
import json
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np

B = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
R = B/'scene008_v44_a5_road_fusion_integration_20261107_r1'
NEW = B/'scene008_v44_a5_c49_rgb_20261107_r1'
OP = R/'J4_road_parent_operator'
PROBE = R/'white_point_parent_diagnostic_v1'
WARM = B/'scene008_road_fusion_bake_20261107_r1/engineer/canonical_edgeblend_v1/new_road_canonical_candidate.npz'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source_files = [Path(__file__), PROBE/'receipt.json', OP/'operator_manifest.json',
        OP/'road_parent_operator.npz', OP/'K_single_v1/result.json',
        OP/'K_single_v1/K_candidate_parent_RGB.npz', WARM,
        NEW/'parent_visibility49/qualification_manifest.json',
        NEW/'parent_visibility49/qualified_parent_rows.npz']
    original_hashes = {str(p): sha(p) for p in source_files}
    receipt = json.loads((PROBE/'receipt.json').read_text())
    cert = json.loads((OP/'K_single_v1/result.json').read_text())
    operator = np.load(OP/'road_parent_operator.npz')
    k = np.load(OP/'K_single_v1/K_candidate_parent_RGB.npz')
    warm = np.load(WARM)
    qualification = np.load(NEW/'parent_visibility49/qualified_parent_rows.npz')
    ids = operator['eligible_parent_ids']
    assert np.array_equal(ids, k['eligible_parent_ids'])
    assert np.array_equal(ids, qualification['eligible_parent_ids'])
    masses = {}
    for record in receipt['records']:
        path = PROBE/record['mass_file']
        assert sha(path) == record['mass_sha256']
        original_hashes[str(path)] = sha(path)
        z = np.load(path)
        mass = z['parent_alphaT'].astype(np.float64)
        positive = np.flatnonzero(mass > 0)
        assert np.array_equal(np.sort(z['positive_parent_ids']), positive)
        masses[record['raw']] = mass
    union = np.unique(np.concatenate([np.flatnonzero(a > 0) for a in masses.values()]))
    original = k['full_parent_RGB'].copy()
    counter = original.copy()
    counter[union] = warm['parent_render_colours_srgb'][union]
    unchanged = np.ones(10098, bool)
    unchanged[union] = False
    assert counter[unchanged].tobytes() == original[unchanged].tobytes()
    assert counter[union].tobytes() == warm['parent_render_colours_srgb'][union].tobytes()
    assert np.count_nonzero(operator['full_parent_A'][:, union]) == 0
    delta = counter[ids].astype(np.float64)-operator['initial_parent_RGB'].astype(np.float64)
    old_delta = original[ids].astype(np.float64)-operator['initial_parent_RGB'].astype(np.float64)
    predicted = operator['baseline_J']+operator['A']@delta
    previous = operator['baseline_J']+operator['A']@old_delta
    assert np.array_equal(predicted, previous)
    pools = []
    for group in cert['pools']:
        score = float(np.abs(predicted[group['rows']]).mean())
        pools.append(dict(name=group['name'], state=group['state'], bins=group['bins'],
            score=score, original_target=group['original_target'],
            original_pass=score <= group['original_target'],
            margin=group['original_target']-score))
    bins = []
    for group in cert['per_bin_constraints']:
        score = float(np.abs(predicted[group['operator_row']]).mean())
        target = group['original_no_regression_target']
        bins.append(dict(operator_row=group['operator_row'], raw=group['raw'], state=group['state'],
            kind=group['kind'], edge=group['edge'], tangent_bin=group['tangent_bin'],
            score=score, original_target=target, original_pass=score <= target,
            margin=target-score))
    assert len(bins) == 106 and all(a['original_pass'] for a in bins+pools)
    points = []
    for record in receipt['records']:
        raw = record['raw']
        x, y = record['pixel_xy']
        path = OP/f'K_candidate_render_v1/frames/{raw}/delete_float.npz'
        warm_path = R/f'H4_model_foreground_culling_v2/frames/{raw}/delete_float.npz'
        observed_path = B/f'scene008_instances_20261105_r1/observations/frames/{raw}.npz'
        for p in [path, warm_path, observed_path]:
            original_hashes[str(p)] = sha(p)
        z, w, obs = np.load(path), np.load(warm_path), np.load(observed_path)
        mass = masses[raw]
        road_rgb = mass@counter.astype(np.float64)
        total_rgb = z['rgb'][:, y, x].astype(np.float64) + mass@(counter.astype(np.float64)-original.astype(np.float64))
        gt = obs['rgb'][y, x].astype(np.float64)
        warm_total = w['rgb'][:, y, x].astype(np.float64)
        assert np.abs(total_rgb-warm_total).max() <= 1e-6
        points.append(dict(raw=raw, state='delete', pixel_xy=[x, y],
            K_total_RGB=z['rgb'][:, y, x].tolist(), counterfactual_total_RGB=total_rgb.tolist(),
            Cwarm_total_RGB=warm_total.tolist(), observed_RGB=gt.tolist(),
            K_point_MAE=float(np.abs(z['rgb'][:, y, x]-gt).mean()),
            counterfactual_point_MAE=float(np.abs(total_rgb-gt).mean()),
            Cwarm_point_MAE=float(np.abs(warm_total-gt).mean()),
            counterfactual_newroad_RGB_T=road_rgb.tolist(),
            total_RGB_reconstruction='cached K total + exact measured parent alphaT @ (counter-K RGB)',
            counterfactual_vs_saved_Cwarm_total_max_abs=float(np.abs(total_rgb-warm_total).max()),
            observation_is_diagnostic_only=True))
    per_view = {int(p): [] for p in union}
    for raw in range(89, 138):
        for state in ['keep', 'delete']:
            path = NEW/f'parent_visibility49/point_mass/{raw:03d}_{state}.npz'
            with np.load(path) as z:
                values = z['parent_mass']
                for p in union:
                    if values[p] > 0:
                        per_view[int(p)].append(dict(raw=raw, state=state, fit_mass=float(values[p])))
    parents = []
    for p in union:
        p = int(p)
        valid = warm['parent_source_valid'][:, p]
        source_rgb = warm['parent_source_rgb'][valid, p]
        parents.append(dict(parent_id=p, actually_changed=bool(np.any(counter[p] != original[p])),
            eligible_C49=bool(p in ids), C49_fit_mass=float(qualification['parent_mass'][p]),
            Cwarm_RGB=warm['parent_render_colours_srgb'][p].tolist(), K_RGB=original[p].tolist(),
            signed_A_column_all_zero=bool(np.count_nonzero(operator['full_parent_A'][:, p]) == 0),
            point_mass_by_raw={str(raw): float(mass[p]) for raw, mass in masses.items()},
            original_source_raw_ids=warm['parent_source_raw'][valid].tolist(),
            original_source_RGB=source_rgb.tolist(),
            original_source_RGB_mean=source_rgb.mean(0).tolist() if len(source_rgb) else None,
            original_source_valid_count49=int(warm['parent_source_valid_count49'][p]),
            C49_positive_fit_views=per_view[p]))
    report = dict(status='PASS_SINGLE_PRESCRIBED_IN_MEMORY_POSITIVE_PARENT_UNION_ROLLBACK_CPU',
        union_predicate='union of every parent with measured alphaT strictly > 0 at either registered DELETE pixel; no mass cutoff',
        parent_ids=union.tolist(), union_parent_count=len(union),
        actually_changed_parent_count=int(np.any(counter != original, axis=1).sum()),
        max_abs_RGB_change_from_K=float(np.abs(counter-original).max()),
        non_union_parent_RGB_bytes_unchanged=True, geometry_unchanged=True,
        full106_by_union_A_all_exact_zero=True,
        predicted_signed_J_bitwise_equal_to_K=True,
        predicted_signed_J_max_abs_change=float(np.abs(predicted-previous).max()),
        all_six_original_pools_pass=all(a['original_pass'] for a in pools),
        all106_original_local_constraints_pass=all(a['original_pass'] for a in bins),
        six_pools=pools, all106_bins=bins, two_points=points, parents=parents,
        original_K_tightened_certificate='The predicted signed RGBs are bitwise identical to the K point; the original certificate and its arithmetic tolerances are inherited without alteration.',
        observed_fact='The two diagnostic pixels are outside fit_mask and inside protect_mask, but their parent RGBs were eligible from positive legal fit mass in other pixels/views.',
        unresolved='The saved fit masses aggregate pixels. They do not identify the exact fit-pixel targets influencing each parent, so multi-view target conflict or misprojection is not established by this audit.',
        scope='CPU counterfactual only; no candidate asset, no solver, no new render, no quality adoption',
        source_files_sha256=original_hashes, all_input_files_unchanged=True,
        GPU_forwards=0, backwards=0, optimizer_updates=0, solver_calls=0, assets_written=0)
    for p, expected in original_hashes.items():
        assert sha(p) == expected
    output = R/'review/positive_parent_union_Cwarm_counterfactual_cpu_v1.json'
    assert not output.exists()
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output), sha256=sha(output),
        union=union.tolist(), changed=report['actually_changed_parent_count'],
        six_pools_pass=report['all_six_original_pools_pass'],
        all106_pass=report['all106_original_local_constraints_pass'],
        signed_change=report['predicted_signed_J_max_abs_change'], points=points), indent=2))


if __name__ == '__main__':
    main()
