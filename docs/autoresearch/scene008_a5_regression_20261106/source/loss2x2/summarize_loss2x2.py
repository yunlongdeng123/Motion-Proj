"""Summarize the registered single-seed 2x2 without interpreting MAE as geometry."""
from pathlib import Path
import json

def locate_evidence():
    for parent in Path(__file__).resolve().parents:
        if parent.name=='a5_regression_20261106' and (parent/'evidence').is_dir():
            return parent/'evidence'
        candidate=parent/'outputs/a5_regression_20261106/evidence'
        if candidate.is_dir():return candidate
    raise FileNotFoundError('Expected the delivered a5_regression_20261106/evidence directory')


EVIDENCE = locate_evidence()


def main():
    rows = []
    summaries = {}
    for step in [1000, 5000]:
        for initialization, old_name, new_name in [
            ('original_static', 'support_control', 'loss5_control'),
            ('full_a5_real_source_support', 'support_full_a5_', 'loss5_full_a5_')]:
            for weight, prefix in [(1, old_name), (5, new_name)]:
                alias = prefix + f'{step // 1000}k'
                source = EVIDENCE / alias / 'summary.json'
                if not source.is_file():
                    continue
                data = json.loads(source.read_text(encoding='utf-8'))
                assert data['status'] == 'PASS_FRESH_NATIVE49_A5_EVALUATION'
                assert data['frame_count'] == 49 and data['fixed_denominator']['pixels'] == 121128
                assert data['fixed_denominator']['sha256'] == 'aa8e7d3b56a8c35ce6177ad1e3378df7afcbc83e6db5d813bdddbfc55a717419'
                assert data['optimization_updates'] == 0
                assert data['replay']['checked_images'] == 98 and data['replay']['max_uint8_difference'] <= 1
                assert data['source_integrity'] == 'PASS_UNCHANGED_BYTES_MTIME_SHA256'
                row = dict(initialization=initialization, static_rgb_weight=weight,
                           step=step, alias=alias, source=str(source), model=data['model'],
                           all49=data['all49'], boundary_all49=data['boundary_all49'],
                           source_integrity=data['source_integrity'], replay=data['replay'])
                rows.append(row)
                summaries[(initialization, weight, step)] = row
    contrasts = []
    for step in [1000, 5000]:
        deltas = {}
        for initialization in ['original_static', 'full_a5_real_source_support']:
            if (initialization, 5, step) not in summaries:
                continue
            base, candidate = [summaries[(initialization, w, step)] for w in [1, 5]]
            metrics = {}
            for key in ['keep_rgb_mae', 'delete_rgb_mae', 'keep_mean_alpha', 'delete_mean_alpha']:
                a, b = base['all49'][key], candidate['all49'][key]
                metrics[key] = dict(weight1=a, weight5=b, delta_weight5_minus1=b-a,
                                    relative_percent=(b-a) / a * 100)
            contrasts.append(dict(step=step, initialization=initialization, metrics=metrics))
            deltas[initialization] = metrics['keep_rgb_mae']['delta_weight5_minus1']
        if len(deltas) == 2:
            contrasts.append(dict(step=step, contrast='additive_mae_interaction',
                difference_of_weight_effects=(deltas['full_a5_real_source_support'] - deltas['original_static']),
                uncertainty='One seed, no variance estimate or significance claim'))
    result = dict(status='COMPLETE_ALL_REGISTERED_ENDPOINTS' if len(rows) == 8 else 'PARTIAL_AVAILABLE_ENDPOINTS',
                  rows=rows, contrasts=contrasts, fixed_a5_pixel_frames=121128,
                  intervention='Only hybrid_lambda_static_rgb 1->5 within each fixed initialization',
                  limitations=['Whole-static-complement supervision, not an a5-local weight',
                               'Initialization support includes local density/capacity and native KNN scale effects',
                               'Mean alpha and MAE do not establish car geometry or roof recovery',
                               'One seed; original static/dynamic priors, topology and renderer behavior remain different'],
                  architecture='../architecture.svg')
    output = EVIDENCE / 'loss2x2_summary.json'
    output.write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print(result['status'], len(rows), str(output))


if __name__ == '__main__':
    main()
