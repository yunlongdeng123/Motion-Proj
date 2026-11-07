"""Evaluate the actual K8F original6pools plus corrected106 local constraints."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def key(row):
    return (row['raw'], row['state'], row['kind'], row['edge'], row['tangent_bin'])


def get_bin(review, variant, record):
    row = next(v for v in review['records'] if v['variant'] == variant and v['raw'] == record['raw'] and v['state'] == record['state'])
    field = 'primary128_seams' if record['kind'] == 'primary128' else 'world_seams'
    return next(v for v in row[field][record['edge']]['bins'] if v['tangent_bin'] == record['tangent_bin'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ['manifest', 'protocol', 'actual-review', 'current-review', 'j-review', 'candidate-parent', 'output']:
        p.add_argument('--' + arg, type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    m, protocol, actual, current, j = map(read, [a.manifest, a.protocol, a.actual_review, a.current_review, a.j_review])
    assert sha(a.protocol) == '90f669dde8e6b18cf886a6a3be21e6bc48ee55bb96d508f851d695f031742697'
    refs_path = a.protocol.parent / protocol['per_bin_reference_file']
    assert sha(refs_path) == protocol['per_bin_reference_sha256']
    refs = {key(r): r for r in read(refs_path)['records']}
    op = a.manifest.parent / m['matrix_file']
    assert sha(op) == m['matrix_sha256']
    with np.load(op, allow_pickle=False) as z:
        A, j0, c0, ids = [z[k].copy() for k in ['A', 'baseline_J', 'initial_parent_RGB', 'eligible_parent_ids']]
    with np.load(a.candidate_parent, allow_pickle=False) as z:
        rgb = z['full_parent_RGB'].copy()
    prediction = j0 + A @ (rgb[ids].astype(np.float64) - c0.astype(np.float64))
    bins = []
    for i, record in enumerate(m['seam_records']):
        value = get_bin(actual, 'K', record)
        old = get_bin(current, 'H', record)
        latest = get_bin(actual, 'baseline', record)
        previous = get_bin(j, 'J', record)
        ref = refs[key(record)]
        assert old['score'] == ref['current_H2_C2000_score'] and latest['score'] == ref['latest21K_score']
        target = min(old['score'], latest['score']) + 1e-6
        assert target == ref['original_no_regression_target']
        bins.append({**{k: record[k] for k in ['raw', 'state', 'kind', 'edge', 'tangent_bin', 'minus_pixels', 'plus_pixels']},
                     'latest_score': latest['score'], 'H2_C2000_score': old['score'], 'J_actual_score': previous['score'],
                     'K_actual_score': value['score'], 'delta_vs_latest': value['score'] - latest['score'],
                     'delta_vs_H2_C2000': value['score'] - old['score'], 'delta_vs_J': value['score'] - previous['score'],
                     'original_no_regression_target': target, 'tightened_reserve_target': target - 1e-7,
                     'margin_to_original': target - value['score'],
                     'margin_to_tightened': target - 1e-7 - value['score'],
                     'original_per_bin_pass': bool(value['score'] <= target),
                     'tightened_reserve_pass': bool(value['score'] <= target - 1e-7),
                     'actual_signed_residual_RGB': value['signed_residual_jump_rgb'],
                     'predicted_signed_residual_RGB': prediction[i].tolist()})
    pools = []
    for state in ['keep', 'delete']:
        for name, kind, edge, n in [('primary128', 'primary128', 'near_x', 6), ('world_near', 'world', 'near_x', 24), ('world_lateral', 'world', 'lateral_y', 23)]:
            chosen = [r for r in bins if r['state'] == state and r['kind'] == kind and r['edge'] == edge]
            assert len(chosen) == n
            ref = protocol['published_reference_metrics'][name][state]
            latest, cur = ref['latest'], ref['current_H2_C2000']
            target = min(cur, latest * .95, latest - 1e-4) if name == 'primary128' else min(cur, latest + 1e-6)
            score = sum(r['K_actual_score'] for r in chosen) / n
            pools.append(dict(name=name, state=state, bins=n, actual_original_score=score, original_target=target,
                              tightened_reserve_target=target - 1e-7, margin_to_original=target - score,
                              margin_to_tightened=target - 1e-7 - score, original_pass=bool(score <= target)))
    summaries = []
    for ref in ['latest', 'H2_C2000', 'J']:
        for state in ['keep', 'delete']:
            chosen = [r for r in bins if r['state'] == state]
            name = 'delta_vs_' + ref
            worst = max(chosen, key=lambda r: r[name])
            summaries.append(dict(reference=ref, state=state, bins=53,
                                  strict_sign_counts=dict(improved=sum(r[name] < 0 for r in chosen),
                                                          regressed=sum(r[name] > 0 for r in chosen), equal=sum(r[name] == 0 for r in chosen)),
                                  diagnostic_1e_6_counts=dict(improved=sum(r[name] < -1e-6 for r in chosen),
                                                            regressed=sum(r[name] > 1e-6 for r in chosen), within_1e_6=sum(abs(r[name]) <= 1e-6 for r in chosen)),
                                  worst_regression=worst))
    csv_fields = [k for k in bins[0] if not k.endswith('_RGB')]
    with (a.output / 'all106_bin_directions.csv').open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=csv_fields)
        w.writeheader()
        w.writerows({k: r[k] for k in csv_fields} for r in bins)
    inputs = [a.manifest, a.protocol, refs_path, a.actual_review, a.current_review, a.j_review, a.candidate_parent, op, Path(__file__)]
    result = {'schema': 'K_REAL8F_SIX_POOLS_AND_106_LOCAL_CONSTRAINTS_CPU_V1',
              'status': 'ALL_SIX_POOL_AND_106_LOCAL_ORIGINAL_TARGETS_PASSED_OTHER_QUALITY_SEPARATE'
                        if all(r['original_pass'] for r in pools) and all(r['original_per_bin_pass'] for r in bins)
                        else 'ACTUAL_POOL_OR_LOCAL_TARGET_FAILED',
              'actual_pools': pools, 'per_bin_original_pass_count': sum(r['original_per_bin_pass'] for r in bins),
              'per_bin_original_failed': [r for r in bins if not r['original_per_bin_pass']],
              'per_bin_summary': summaries, 'all106_bins': bins,
              'actual_vs_affine_prediction_max_abs': float(abs(np.asarray([r['actual_signed_residual_RGB'] for r in bins]) - prediction).max()),
              'original4560_failed_gates': actual['failed_gates'], 'new_solver_calls': 0,
              'new_GPU_forwards': 0, 'new_optimizer_updates': 0, 'quality_accepted': False, 'human_verdict': None,
              'inputs': [dict(path=str(path), sha256=sha(path)) for path in inputs],
              'limits': ['No thresholds were changed after rendering. Reserve margins and original1e-6 local tolerance are separate.',
                         'Other photo/geometry gates remain as the original4560 report, not replaced by seam feasibility.',
                         'These are reused BUILD views and overlapping metrics, not independent generalization observations.']}
    output = a.output.parent / (a.output.name + '.json')
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(output), 'status': result['status'], 'actual_pools': pools,
                      'local_pass': result['per_bin_original_pass_count'], 'local_failed': result['per_bin_original_failed']}))


if __name__ == '__main__':
    main()
