"""Full-image scope and fixed-band coverage review for an active-only child."""
from pathlib import Path
import argparse
import numpy as np
from local_gaussian_asset import bank,read,write,sha
from render_new_local_zero import pin,png


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['zero','unpruned-zero','baseline-measurement','masks','measurement-contract','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();r=read(a.zero/'receipt.json');cfg=read(a.measurement_contract)
    assert r['complete'] and r['nativeF']==r['nativeB']==5 and r['updates']==0
    assert not a.output.exists();a.output.mkdir(parents=True);rows=[]
    for raw in cfg['frames']:
        z=bank(a.zero/f'{raw:03d}/joint_render.npz');u=bank(a.unpruned_zero/f'{raw:03d}/joint_render.npz');b=bank(a.baseline_measurement/f'{raw:03d}/baseline_joint.npz')
        mask=bank(a.masks/f'{raw:03d}/scope_masks.npz');ev=bank(Path(cfg['fixed_eval'])/f'{raw:03d}/fixed_eval.npz')
        checks={}
        for name in ['outside_allowed_edit','actor_protected','K_protected']:
            d=mask[name];delta=abs(z['rgb']-b['rgb'])[:,d];ad=abs(z['alpha']-b['alpha'])[d]
            checks[name]=dict(pixels=int(d.sum()),RGB_byteexact=z['rgb'][:,d].tobytes()==b['rgb'][:,d].tobytes(),
                alpha_byteexact=z['alpha'][d].tobytes()==b['alpha'][d].tobytes(),RGB_changed_pixels=int((delta.max(0)>0).sum()),
                RGB_max_abs=float(delta.max()) if d.any() else 0.,alpha_max_abs=float(ad.max()) if d.any() else 0.)
        full=ev['full_original_band'];unknown=full&~ev['ground_eval'];coverage={}
        for name,ref in [('original_baseline',b),('unpruned_zero',u)]:
            new=full&(z['alpha']<.95)&(ref['alpha']>=.95)
            coverage[name]=dict(full_band_new_alpha95_holes=int(new.sum()),unknown_new_alpha95_holes=int((new&unknown).sum()),
                full_band_pixels=int(full.sum()),unknown_pixels=int(unknown.sum()))
        all_exact=all(x['RGB_byteexact'] and x['alpha_byteexact'] for x in checks.values())
        # New coverage relative to the original image must not be worse.
        # Any lost coverage relative to unpruned is retained separately, since
        # the original unpruned field itself touched forbidden pixels.
        no_holes=all(v['full_band_new_alpha95_holes']==0 for v in coverage.values())
        row=dict(raw=raw,checks_vs_original_baseline=checks,coverage=coverage,all_outside_car_K_RGBA_byteexact=all_exact,
            no_new_fixed_full_band_holes_vs_both_controls=no_holes,fixed_eval=pin(Path(cfg['fixed_eval'])/f'{raw:03d}/fixed_eval.npz'))
        if not all_exact:
            changed=(abs(z['rgb']-b['rgb']).max(0)>0)&mask['outside_allowed_edit']
            preview=np.zeros((*changed.shape,3),np.float32);preview[changed]=[1,.2,.1];png(a.output/f'{raw:03d}_outside_changes.png',preview)
        rows.append(row)
    accepted=all(x['all_outside_car_K_RGBA_byteexact'] and x['no_new_fixed_full_band_holes_vs_both_controls'] for x in rows)
    out=dict(schema='R10.pruned_zero_fixed_scope_review_CPU.v1',complete=True,decision='PASS_FOR_FRESH_FOOTPRINT_MEASUREMENT' if accepted else 'STOP_SCOPE_OR_COVERAGE_REVIEW',
        CPU_only=True,nativeF=0,nativeB=0,zero_receipt=pin(a.zero/'receipt.json'),unpruned_zero=pin(a.unpruned_zero/'receipt.json'),
        baseline_measurement=pin(a.baseline_measurement/'receipt.json'),mask_receipt=pin(a.masks/'receipt.json'),rows=rows,
        automatic_fresh_measurement_allowed=accepted,whole_joint_quality_accepted=False,old_scope_must_not_be_reused=True)
    write(a.output/'receipt.json',out);print(dict(decision=out['decision'],rows=rows))

if __name__=='__main__':main()
