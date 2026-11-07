"""CPU aggregation of the frozen a5 upper region from existing native floats.

python aggregate_fixed_a5_regions.py --evaluation-root EVAL --output upper_regions.json
No renderer, checkpoint, engine, or CUDA is loaded. Only the explicit output is written.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

PIXELS = 121128
UPPER_PIXELS = 25906
UPPER_SHA256 = 'd5c24aeee22bb804311e28f89de6f0652438465263c3e45a2f9d245d6b67379f'
GT_SHA256 = 'e1865e95d0f6e549bb104f7cec64f3a339bb23cb824ed471b269c2c6b8c0079a'


def aggregate(root):
    count = total = 0
    sums = dict(keep_rgb_error=0., delete_rgb_error=0., change=0., alpha=0.)
    domain_hash, gt_hash = hashlib.sha256(), hashlib.sha256()
    frames, sources = [], []
    for raw in range(89, 138):
        path = root / 'a5_float' / f'{raw:03d}.npz'
        before = path.stat()
        with np.load(path, allow_pickle=False) as data:
            yx = np.asarray(data['yx'])
            keep, deleted, gt = [np.asarray(data[key], dtype=np.float64)
                                 for key in ('keep_rgb', 'delete_rgb', 'gt_rgb')]
            alpha = np.asarray(data['keep_alpha'], dtype=np.float64)
        if yx.shape != (len(gt), 2) or keep.shape != gt.shape or deleted.shape != gt.shape or gt.shape[1:] != (3,) or alpha.shape != (len(gt),):
            raise ValueError('Invalid fixed-pixel schema: ' + str(path))
        if not all(np.isfinite(value).all() for value in (yx, keep, deleted, gt, alpha)):
            raise ValueError('Non-finite saved float: ' + str(path))
        total += len(gt)
        raw_bytes = np.asarray(raw, dtype='<i2').tobytes()
        gt_hash.update(raw_bytes)
        gt_hash.update(np.ascontiguousarray(gt, dtype='<f4').tobytes())
        domain_hash.update(raw_bytes)
        row = dict(raw=raw, full_mask_pixels=len(gt), pixels=0)
        if len(gt):
            select = yx[:, 0] <= yx[:, 0].min() + .25 * (yx[:, 0].max()-yx[:, 0].min())
            domain_hash.update(np.ascontiguousarray(yx[select], dtype='<i2').tobytes())
            a, b, g = np.clip(keep[select], 0, 1), np.clip(deleted[select], 0, 1), gt[select]
            n = int(select.sum())
            values = dict(keep_rgb_error=float(np.abs(a-g).mean(1).sum()),
                          delete_rgb_error=float(np.abs(b-g).mean(1).sum()),
                          change=float(np.abs(a-b).mean(1).sum()), alpha=float(alpha[select].sum()))
            for key, value in values.items():
                sums[key] += value
            count += n
            row.update(pixels=n, keep_rgb_mae=values['keep_rgb_error']/n,
                       delete_rgb_mae=values['delete_rgb_error']/n,
                       keep_delete_rgb_change=values['change']/n, keep_mean_alpha=values['alpha']/n)
        frames.append(row)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        after = path.stat()
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise ValueError('Input changed during CPU aggregation: ' + str(path))
        sources.append(dict(raw=raw, path=str(path), bytes=after.st_size, sha256=digest))
    if total != PIXELS or count != UPPER_PIXELS or domain_hash.hexdigest() != UPPER_SHA256 or gt_hash.hexdigest() != GT_SHA256:
        raise ValueError('The original frozen a5 domain or GT differs from this study')
    return dict(status='PASS_FIXED_A5_UPPER_AGGREGATION', frame_count=49,
                full_pixel_frames=total, pixels=count, keep_rgb_mae=sums['keep_rgb_error']/count,
                delete_rgb_mae=sums['delete_rgb_error']/count,
                keep_delete_rgb_change=sums['change']/count, keep_mean_alpha=sums['alpha']/count,
                upper25_indices_sha256=domain_hash.hexdigest(), gt_rgb_sha256=gt_hash.hexdigest(),
                rule='Original a5 mask intersected with upper25% HEIGHT of its per-frame bbox; not top25% pixel count, render-alpha selection, or semantic roof GT',
                metric_units='sRGB channel MAE in0..1, weighted by fixed pixel-frame count',
                alpha_caveat='Total alpha can be filled by background or incorrect surfaces and is not vehicle completeness',
                frames=frames, source_files=sources, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                GPU_used=False, optimizer_updates=0, quality_accepted=False, human_verdict=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.evaluation_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('status','pixels','keep_rgb_mae','upper25_indices_sha256','script_sha256')}, ensure_ascii=True))


if __name__ == '__main__':
    main()
