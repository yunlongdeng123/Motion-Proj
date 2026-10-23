"""Read-only all-source consensus audit; no new training labels and no geometry changes."""
import argparse
import json
from pathlib import Path
import numpy as np
from prepare_v41_revision import native_rgb, witness_at, sample_rgb


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dense', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.dense/'manifest.json').read_text())
    meta = json.loads(Path(manifest['source_bundle']).read_text())
    frame_order = manifest['frame_order']
    evidence = []
    for frame in frame_order:
        with np.load(args.dense/'frames'/f'{frame}_evidence.npz', allow_pickle=False) as z:
            evidence.append((z['box_clear_witness'].copy(), z['measured_patch_depth_m'].copy(),
                native_rgb(meta['frames'][frame]['original_rgb'])))
    rows = []
    for i,frame in enumerate(frame_order):
        with np.load(args.dense/'frames'/f'{frame}.npz', allow_pickle=False) as z:
            labels, supported = z['actor_labels'], z['supported']
            selected = supported & np.isin(labels, [4,5])
            yy,xx = np.where(selected)
            ids = np.ravel_multi_index((yy,xx),labels.shape)
            camera = meta['frames'][frame]; K = np.asarray(camera['intrinsic']); c2w = np.linalg.inv(camera['world_to_camera'])
            rays = np.c_[xx+.5, yy+.5, np.ones(len(xx))] @ np.linalg.inv(K).T
            world = (rays*z['fixed_surface_depth_m'][selected,None]) @ c2w[:3,:3].T+c2w[:3,3]
            samples = np.full((len(frame_order),len(xx),3),np.nan,np.float32)
            for j,source in enumerate(frame_order):
                if j == i: continue
                clear,depth,rgb = evidence[j]
                valid,uv = witness_at(world, meta['frames'][source], clear, depth)
                samples[j,valid] = sample_rgb(rgb,uv[valid])
            count = np.isfinite(samples[:,:,0]).sum(0)
            median = np.zeros((len(xx),3),np.float32)
            has_source = count > 0
            median[has_source] = np.nanmedian(samples[:,has_source],axis=0)
            residual = np.abs(samples-median[None]).mean(2)
            inliers = (residual <= .05).sum(0)
            # Majority of all admissible witnesses, not searching for any convenient matching pair.
            consensus = (inliers >= 3) & (inliers >= .6*count)
            actor_rows = {}
            for actor,label in ((6,4),(37,5)):
                sel = labels[selected] == label
                eligible = z['candidate'][selected]
                existing = z['deletion_valid'][selected]
                two = z['deletion_two_sources'][selected]
                def counts(mask):
                    mask = mask & sel
                    return dict(pixels=int(mask.sum()), any_source=int((mask & (count>0)).sum()),
                        at_least_three_sources=int((mask & (count>=3)).sum()),
                        consensus_candidate=int((mask & consensus).sum()))
                actor_rows[str(actor)] = dict(
                    supported_not_R=counts(~eligible),
                    first_two_disagree=counts(eligible & two & ~existing),
                    R_missing_two_sources=counts(eligible & ~two),
                    already_labelled=counts(existing))
            rows.append(dict(frame=frame,actors=actor_rows))
        print(json.dumps(rows[-1]),flush=True)
    totals = {str(a):{group:{key:sum(r['actors'][str(a)][group][key] for r in rows)
        for key in ('pixels','any_source','at_least_three_sources','consensus_candidate')}
        for group in ('supported_not_R','first_two_disagree','R_missing_two_sources','already_labelled')}
        for a in (6,37)}
    result = dict(version='V41_ALL_SOURCE_OPPORTUNITY_V1',dense=str(args.dense.resolve()),
        policy=dict(source='all other 48 frames; unchanged box-clear guarded measured-patch/depth gates',
            center='channelwise median of all valid source colors', tolerance=.05,
            minimum_inliers=3, minimum_inlier_fraction=.6,
            limitation='correlated video witnesses; no independent holdout, no exposure correction or spatial acceptance yet'),
        meaning='candidate evidence only; NOT reconstructed improvement or approved new labels',
        frames=rows,totals=totals,GPU_used=False,optimizer_updates=0,data_changed=False)
    with args.output.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(totals,indent=2))


if __name__ == '__main__': main()
