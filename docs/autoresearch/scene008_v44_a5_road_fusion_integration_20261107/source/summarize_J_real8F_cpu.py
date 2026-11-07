"""Small report-facing readout from unchanged, already evaluated J CPU evidence."""
from collections import Counter
from pathlib import Path
import csv
import hashlib
import json


ROOT = Path(__file__).resolve().parent.parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def main():
    review = ROOT / 'evidence/review'
    paths = [review / f'{name}.json' for name in [
        'J_candidate_fixed4560_cpu', 'J_actual_pools_bins_cpu_v1',
        'J_components_vs_latest_v1', 'J_components_vs_H2_C2000_v1']]
    gates, pools, latest, h2 = [read(p) for p in paths]
    statuses = Counter('unsupported' if r['passed'] is None else 'passed' if r['passed'] else 'failed'
                       for r in gates['numeric_gates'])
    failures = Counter(r['name'] for r in gates['numeric_gates'] if r['passed'] is False)
    table = []
    for reference, data in [('delivered21K', latest), ('H2_formalC2000', h2)]:
        for record in data['records']:
            for metric, value in record['original_fixed_domain_partial_metrics'].items():
                table.append({'reference': reference, 'raw': record['raw'], 'state': record['state'],
                              'metric': metric, 'pixels': value['baseline']['pixels'],
                              'reference_MAE': value['baseline']['mean'],
                              'J_MAE': value['candidate']['mean'],
                              'MAE_delta': value['MAE_delta'],
                              'units': 'm' if metric.endswith('_depth_m') else 'RGB_0_to_1_scale'})
    csv_path = ROOT / 'evidence/J_actual_photo_two_references.csv'
    with csv_path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    payload = {
        'status': 'FOUR_RAW_SEAM_POOL_FEASIBILITY_PASSED_FULL_QUALITY_GATES_FAILED',
        'scope': {'raws': [94, 105, 128, 137], 'states': ['keep', 'delete'],
                  'J_training_updates': 0, 'J_solver_calls': 1, 'J_actual_quality_forwards': 8,
                  'J_operator_forwards': 16, 'J_dummy_backwards': 16, 'J_affinity_forwards': 8,
                  'automatic_49_authorized': False},
        'original4560_gate_counts': dict(statuses),
        'failed_gate_categories': dict(failures),
        'original_pool_constraints': pools['actual_pools'],
        'per_bin_tail_summary': pools['per_bin_summary'],
        'same_H2_geometry': all(v['bitwise_equal'] for r in h2['records']
                                for v in r['global_geometry_vs_initial'].values()),
        'same_H2_foreground_RGB_pixel_max_abs': max(
            value['RGB_pixel_max_abs_difference']['max'] or 0.0
            for r in h2['records'] for value in r['foreground'].values()),
        'H2_foreground_union_newroad_mass_all_zero': all(
            r['foreground_union_newroad_mass']['zero_mass'] for r in h2['records']),
        'photo_table': str(csv_path.relative_to(ROOT)).replace('\\', '/'),
        'source_evidence': [{'path': str(p.relative_to(ROOT)).replace('\\', '/'),
                             'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
        'quality_accepted': False,
        'agent_visual_acceptance': 'PARENT_REVIEW_SEPARATE',
        'human_verdict': None,
        'limits': [
            'Pooled seam constraints allow local bin regressions; 53 bins/state are not independent observations.',
            'H2 equality refers to the fixed H2+C2000 geometry and foreground domains; geometry differs from delivered21K.',
            'The original photo/nonroad geometry failures are retained; no 49-frame J claim is made.',
            'The LP safety reserve is reported separately from the original gates; one real lateral pool exceeds reserve by about3.25e-9.',
        ],
    }
    out = ROOT / 'evidence/J_actual_cpu_readout_v1.json'
    out.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'path': str(out), 'csv_rows': len(table), 'counts': dict(statuses),
                      'same_H2_geometry': payload['same_H2_geometry']}, allow_nan=False))


if __name__ == '__main__':
    main()
