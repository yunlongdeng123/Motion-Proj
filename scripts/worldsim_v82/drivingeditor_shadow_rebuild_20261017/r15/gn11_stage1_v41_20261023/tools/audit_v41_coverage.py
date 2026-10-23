"""CPU evidence funnel with the complete silhouette denominator; no label/geometry changes."""
import argparse
import json
from pathlib import Path
import numpy as np

ACTORS = (5, 26, 18, 6, 37)
BUCKETS = ('no_fixed_surface', 'supported_not_R', 'R_missing_two_sources',
           'two_sources_color_disagree', 'labelled')


def evidence_funnel(target, supported, eligible, two, valid):
    target, supported, eligible, two, valid = [np.asarray(x, bool) for x in
        (target, supported, eligible, two, valid)]
    if (np.any(eligible & ~supported) or np.any(valid & ~(eligible & two))):
        raise ValueError('non-nested source evidence')
    masks = (target & ~supported, target & supported & ~eligible,
        target & eligible & ~two, target & eligible & two & ~valid, target & valid)
    counts = {key:int(mask.sum()) for key, mask in zip(BUCKETS, masks)}
    assert sum(counts.values()) == int(target.sum())
    return dict(target_pixels=int(target.sum()), **counts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dense', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.dense/'manifest.json').read_text())
    if len(manifest['frame_order']) != 49: raise ValueError('49 frames required')
    rows = []
    for frame in manifest['frame_order']:
        with np.load(args.dense/'frames'/f'{frame}.npz', allow_pickle=False) as z:
            actors = {str(a):evidence_funnel(z['actor_labels'] == i+1, z['supported'], z['candidate'],
                z['deletion_two_sources'], z['deletion_valid']) for i,a in enumerate(ACTORS)}
            for i,a in enumerate(ACTORS):
                assert actors[str(a)]['labelled'] == int(z['deletion_domains'][i].sum())
            rows.append(dict(frame=frame, actors=actors))
    totals = {}
    for actor in ACTORS:
        vals = [r['actors'][str(actor)] for r in rows]
        aggregate = {key:sum(r[key] for r in vals) for key in ('target_pixels', *BUCKETS)}
        aggregate['labelled_fraction'] = aggregate['labelled']/aggregate['target_pixels'] if aggregate['target_pixels'] else None
        aggregate['labelled_frames'] = sum(r['labelled'] > 0 for r in vals)
        totals[str(actor)] = aggregate
    result = dict(version='V41_COVERAGE_FUNNEL_V1', dense=str(args.dense.resolve()),
        denominator='49-frame original silhouette pixel sum; not world area or all known road pixels',
        no_fixed_surface='reference-ray intersection failed (range, missing four-corner support, normals or convergence); NOT proof of no raw LiDAR',
        supported_not_R='fixed supported surface exists but selected unit is not R eligible; NOT necessarily missing geometry',
        source_disagreement='first two geometry-valid sources disagree; not proof that all other sources disagree',
        frames=rows, totals=totals, GPU_used=False, optimizer_updates=0, data_changed=False)
    with args.output.open('x', encoding='utf-8') as f: json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(totals, indent=2))


if __name__ == '__main__': main()
