"""Reference-only sky QA: original full M2/road/fine, 49 keep, no PNG or training.

Uses the SAME sky_statistics helper as the final candidate's existing renders.
No candidate frame is rendered again by this worker. Default is a plan.
"""
import argparse
import json
import os
from pathlib import Path
import sys


def execute(args):
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    import torch
    from final_runtime import ASSET_SOURCE, ASSET, sha, write
    from sky_metrics import sky_statistics
    if args.output.exists():
        raise FileExistsError('Use a fresh reference sky JSON; preserve existing evidence')
    sys.path.insert(0, str(ASSET_SOURCE))
    from scene_runtime import Runtime
    from sh_asset import SHAsset
    rows = []
    with torch.no_grad():
        runtime = Runtime(fine_road=True)
        asset = SHAsset.from_geometry(ASSET).cuda().requires_grad_(False)
        before = {key: value.detach().cpu().clone() for key, value in
                  (*asset.named_parameters(), *asset.named_buffers())}
        assert len(before) == 27
        asset_sha = sha(ASSET)
        try:
            for raw in range(89, 138):
                data = runtime.raw(raw)
                out = runtime.render_scene(raw, asset, delete=False)
                rows.append(dict(raw=raw, state='keep', timestamp=float(data['cam'].timestamp),
                                 **sky_statistics(runtime, raw, out['rgb'], data['rgb'])))
                runtime.assert_unchanged()
                print(f'SKY_REFERENCE raw{raw} keep', flush=True)
        finally:
            assert runtime.modelunchanged() and sha(ASSET) == asset_sha
            assert asset.assert_geometry_unchanged()
            after = dict((*asset.named_parameters(), *asset.named_buffers()))
            assert set(before) == set(after)
            for key, original in before.items():
                current = after[key].detach().cpu().contiguous()
                assert original.dtype == current.dtype and original.shape == current.shape
                assert torch.equal(original.contiguous().reshape(-1).view(torch.uint8),
                                   current.reshape(-1).view(torch.uint8)), key
        assert len(rows) == 49
        count = sum(row['sky_RGB']['count'] for row in rows)
        total = sum(row['sky_RGB']['sum'] for row in rows)
        prediction_sum = [sum(row['sky_prediction_RGB_sum'][k] for row in rows) for k in range(3)]
        truth_sum = [sum(row['sky_GT_RGB_sum'][k] for row in rows) for k in range(3)]
        report = dict(status='PASS_READ_ONLY_REFERENCE_SKY', accepted_current_M2_reference_only=True,
                      keep_raws=list(range(89, 138)), rows=rows,
                      pooled=dict(sky_RGB=dict(count=count, sum=total, mean=total/count),
                                  sky_prediction_RGB_mean=[v/count for v in prediction_sum],
                                  sky_GT_RGB_mean=[v/count for v in truth_sum]),
                      reference_scene=runtime.meta, M2_sha256=asset_sha,
                      M2_all_27_tensors_bitwise_unchanged_CPU=True, native_road_fine_unchanged=True,
                      helper_sha256=sha(Path(__file__).with_name('sky_metrics.py')),
                      script_sha256=sha(__file__), float_before_PNG=True, PNG_saved=False,
                      compare_scope='Candidate49 keep with same raw/maskSHA/count/GT; candidate4 delete reports own GT metrics, not paired against reference keep.',
                      training_updates=0, quality_accepted=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write(args.output, report)
    print(json.dumps(dict(status='PASS_READ_ONLY_REFERENCE_SKY', output=str(args.output), training_updates=0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='A fresh small .json output')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        execute(args)
    else:
        print(json.dumps(dict(output=str(args.output), reference_only=True, frames=49, edit='keep',
                              PNG_saved=False, candidate_rerendered=False, training_updates=0, execute=False), indent=2))


if __name__ == '__main__':
    main()
