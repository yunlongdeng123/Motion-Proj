"""CPU-only audit of old Single vs new identity Full5 native10K float data."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).absolute().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--single', type=Path, default=root.parent/'a5_regression_20261106/evidence/reference_single10k')
    parser.add_argument('--full5', type=Path, default=root/'evidence/formal10k_keep_all49')
    parser.add_argument('--output', type=Path, default=root/'evidence/native10k_same_domain_cpu_audit.json')
    args = parser.parse_args()
    totals = {name: dict(full_error_sum=0., upper_error_sum=0.) for name in ('single', 'full5')}
    full_count = upper_count = 0
    visible = []
    records = []
    for raw in range(89, 138):
        paths = [base/'a5_float'/f'{raw:03d}.npz' for base in (args.single, args.full5)]
        with np.load(paths[0], allow_pickle=False) as single, np.load(paths[1], allow_pickle=False) as full5:
            assert int(single['raw']) == int(full5['raw']) == raw
            yx = single['yx']
            assert np.array_equal(yx, full5['yx'])
            assert np.array_equal(single['gt_rgb'], full5['gt_rgb'])
            assert str(single['mask_sha256']) == str(full5['mask_sha256'])
            time_error = abs(float(single['model_time'])-float(full5['model_time']))
            assert time_error <= 2e-7
            upper = yx[:, 0] <= yx[:, 0].min()+.25*(yx[:, 0].max()-yx[:, 0].min()) if len(yx) else np.zeros(0, dtype=bool)
            assert np.array_equal(upper, full5['upper_selected'])
            if len(yx):
                visible.append(raw)
            full_count += len(yx)
            upper_count += int(upper.sum())
            for name, data in (('single', single), ('full5', full5)):
                rgb = data['keep_rgb']
                gt = data['gt_rgb']
                assert np.isfinite(rgb).all() and np.isfinite(gt).all()
                # Saved native floats retain the raw renderer range; both
                # evaluators define primary RGB MAE after display-range clamp.
                error = np.abs(np.clip(rgb.astype(np.float64), 0, 1)-gt.astype(np.float64)).mean(axis=1)
                totals[name]['full_error_sum'] += float(error.sum(dtype=np.float64))
                totals[name]['upper_error_sum'] += float(error[upper].sum(dtype=np.float64))
            records.append(dict(raw=raw, pixels=len(yx), upper_pixels=int(upper.sum()),
                                yx_exact=True, gt_float_exact=True, upper_indices_exact=True,
                                timestamp_difference=time_error,
                                sources=[dict(path=str(p.absolute()), bytes=p.stat().st_size, sha256=sha(p)) for p in paths]))
    assert full_count == 121128 and upper_count == 25906 and visible == list(range(89, 113))
    for name, base in (('single', args.single), ('full5', args.full5)):
        data = totals[name]
        data['keep_rgb_mae'] = data['full_error_sum']/full_count
        data['upper_keep_rgb_mae'] = data['upper_error_sum']/upper_count
        summary = json.loads((base/'summary.json').read_text(encoding='utf-8-sig'))
        assert summary['iteration'] == 10000 and summary['frame_count'] == 49
        assert abs(data['keep_rgb_mae']-summary['all49']['keep_rgb_mae']) <= 1e-8
    result = dict(status='PASS_EXACT_FIXED_DOMAIN_NATIVE10K_COMPARISON',
                  frame_count=49, visible_a5_raw_frames=visible, full_pixels=full_count, upper_pixels=upper_count,
                  models=totals, frames=records,
                  all49_relative_mae_decrease=1-totals['full5']['keep_rgb_mae']/totals['single']['keep_rgb_mae'],
                  upper_relative_mae_decrease=1-totals['full5']['upper_keep_rgb_mae']/totals['single']['upper_keep_rgb_mae'],
                  upper_rule='original a5 mask intersect y<=min_y+.25*(max_y-min_y); not top25% by pixel count or alpha selection',
                  intervention='R, real initialization support, and native dynamic/static objective package change jointly; not one-loss causal isolation',
                  RGB_metric='Saved raw native floats; primary channel MAE uses clip(RGB,0,1), not uint8 PNG',
                  GPU_used=False, optimization_updates=0, human_verdict=None, script_sha256=sha(Path(__file__)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('status', 'frame_count', 'full_pixels', 'upper_pixels', 'models', 'all49_relative_mae_decrease', 'upper_relative_mae_decrease')}))


if __name__ == '__main__':
    main()
