"""Summarize actual controlled diagnostic evaluations, without GPU work."""
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
BASE = HERE / 'continuation_diagnostic'
CASES = ['original30k_to30300', 'opacity_outside30k_to30300']


def extract(frame):
    full = frame['regions']['full']['metrics']
    outside = frame['regions']['non_sky_outside_actor']['metrics']
    return {
        'frame_id': frame['frame_id'],
        'composite_psnr_full': full['composite']['clamped_rgb_psnr_db'],
        'static_psnr_observed_non_sky_outside_actor': outside['static']['clamped_rgb_psnr_db'],
        'alpha_dynamic_observed_non_sky_outside_actor': outside['alpha_d']['mean'],
        'alpha_static_observed_non_sky_outside_actor': outside['alpha_s']['mean'],
        'static_environment_contribution_mean_observed_non_sky_outside_actor': outside['static_env_contribution']['mean'],
        'dynamic_rgb_support_observed_non_sky_outside_actor': outside['dynamic']['rgb_support_gt2_uint8_pixel_fraction'],
    }


def summarize(case):
    records = []
    for path in sorted((BASE / case / 'extra_eval').glob('*/diagnostic_eval.json')):
        data = json.loads(path.read_text(encoding='utf8'))
        frames = [extract(frame) for frame in data['frames']]
        if not data['all_rng_unchanged']:
            raise ValueError('Evaluation altered training RNG')
        records.append({
            'iteration': data['iteration'], 'frame_count': len(frames),
            'frames': frames,
            'means': {key: mean(f[key] for f in frames) for key in frames[0] if key != 'frame_id'},
        })
    return {'case': case, 'nodes': records}


def main():
    rows = [summarize(case) for case in CASES if (BASE / case).exists()]
    report = {
        'status': 'controlled_diagnostic_not_full_paper_reproduction',
        'metric_scope': 'Arithmetic mean of per-view region metrics; full49 terminal views, early5 fixed views. No GT claim under the current actor mask.',
        'hypothesis': 'Only additional outside-mask dynamic-opacity penalty coefficient0.05. Author intent not established.',
        'cases': rows,
    }
    if len(rows) == 2:
        a = {node['iteration']: node for node in rows[0]['nodes']}
        b = {node['iteration']: node for node in rows[1]['nodes']}
        delta = []
        for iteration in sorted(a.keys() & b.keys()):
            if [f['frame_id'] for f in a[iteration]['frames']] != [f['frame_id'] for f in b[iteration]['frames']]:
                raise ValueError('Different evaluation view order')
            delta.append({'iteration': iteration, 'frame_count': a[iteration]['frame_count'],
                          'candidate_minus_original': {key: b[iteration]['means'][key] - value for key, value in a[iteration]['means'].items()}})
        report['paired_deltas'] = delta
    (HERE / 'continuation_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'cases': [{'case': row['case'], 'nodes': [{key: node[key] for key in ['iteration', 'frame_count', 'means']} for node in row['nodes']]} for row in rows]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
