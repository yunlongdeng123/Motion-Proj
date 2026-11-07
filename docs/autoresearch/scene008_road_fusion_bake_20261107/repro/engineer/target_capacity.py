"""New atlas-target interval lower bounds from the existing shared80F cache.

This does not repeat root's original-real-RGB capacity probe. The independent
pixel interval relaxes the shared-world-colour constraint, so inside is not a
sufficiency certificate for a feasible 3D material solution.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from shared_fit_common import OUT,BUILD,load,read,write,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);args=p.parse_args()
    contract=read(args.output/'fit_contract.json');target=args.output/'atlas_target_capacity.json'
    if target.exists():raise FileExistsError('Preserve capacity evidence')
    if contract['placeholder_engineering_only']:raise ValueError('Use final scientific atlas targets only')
    assert read(contract['target_manifest'])['fusion_ready'] is True
    records=[]
    for raw in BUILD:
        path=Path(contract['targets_directory'])/f'raw{raw:03d}.npz';a=load(path)
        assert sha(path)==contract['target_files_sha256'][str(path.resolve())]
        for state in ('keep','delete'):
            baseline=load(Path(contract['baseline_cache_directory'])/f'raw{raw:03d}_{state}.npz')
            F=(baseline['rgb']-baseline['newroad_rgb_T']).transpose(1,2,0).astype(float)
            W=baseline['newroad_alpha_T'].astype(float);high=F+W[:,:,None]
            for variant in ('direct','fusion'):
                goal=a[variant+'_rgb'].astype(float);distance=np.maximum(F-goal,0)+np.maximum(goal-high,0)
                domains=dict(fixed_fit=a['fit_mask'],atlas_replaced_fit=a['fit_mask']&a['fusion_replace_mask'])
                for label,mask in domains.items():
                    count=int(mask.sum())
                    if not count:continue
                    value=distance[mask];weight=W[mask]
                    records.append(dict(raw=raw,state=state,variant=variant,domain=label,pixels=count,
                        exact_zero_W_pixels=int((weight==0).sum()),W_lt001_pixels=int((weight<.01).sum()),
                        relaxed_RGB_MAE_lower_bound=float(value.mean()),relaxed_RGB_MSE_lower_bound=float(np.square(value).mean()),
                        pixel_any_channel_outside_interval=int((value>0).any(1).sum()),
                        F_locked_RGB_mean=F[mask].mean(0).tolist(),W_mean=float(weight.mean())))
    pooled=[]
    for state in ('keep','delete'):
        for variant in ('direct','fusion'):
            for domain in ('fixed_fit','atlas_replaced_fit'):
                selected=[r for r in records if r['state']==state and r['variant']==variant and r['domain']==domain]
                count=sum(r['pixels'] for r in selected)
                if count:pooled.append(dict(state=state,variant=variant,domain=domain,pixels=count,
                    relaxed_MAE_lower_bound=sum(r['relaxed_RGB_MAE_lower_bound']*r['pixels'] for r in selected)/count,
                    relaxed_MSE_lower_bound=sum(r['relaxed_RGB_MSE_lower_bound']*r['pixels'] for r in selected)/count,
                    exact_zero_W_pixels=sum(r['exact_zero_W_pixels'] for r in selected),W_lt001_pixels=sum(r['W_lt001_pixels'] for r in selected)))
    write(target,dict(status='RELAXED_PIXEL_INTERVAL_LOWER_BOUND_ONLY',GPU_forwards=0,records=records,pooled=pooled,
        target_manifest_sha256=contract['target_manifest_sha256'],fit_contract_sha256=sha(args.output/'fit_contract.json'),
        range='RGB in[F,F+W] for each channel if newroad colour in[0,1]',
        limits=['Exact W=0 pixels cannot change through this road material.',
            'W<.01 is reported separately in the actual atlas replacement domain; asset-exterior zeros do not prove static-internal incapacity.',
            'The pixel interval ignores world colour sharing and is a lower bound only, not a feasibility guarantee.']))
    print(json.dumps(dict(path=str(target),sha256=sha(target),pooled=pooled)))


if __name__=='__main__':main()
