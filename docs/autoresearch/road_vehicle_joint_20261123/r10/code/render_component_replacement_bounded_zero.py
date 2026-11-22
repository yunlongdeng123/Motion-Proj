"""Independent numeric-readout control; strict row eligibility is unchanged.

All prior strict failures are preserved. A pinned root readout contract is
required before this renderer may use a bounded outside-image comparison.
"""
from pathlib import Path
import argparse,sys
import numpy as np
import render_component_replacement_zero as generic
import render_replacement_contained_zero as contained
from local_joint_adapter import LocalJointAdapter
from local_gaussian_asset import bank,read,write,sha
from render_new_local_zero import npy,pin,png

ARGS=None
CONTRACT=None
OriginalQualify=generic.qualify


def qualify(a,asset,cfg):
    assert CONTRACT['complete'] and CONTRACT['decision']=='PASS'
    assert CONTRACT.get('candidate_scope_sha256',CONTRACT.get('scope_sha256'))==sha(a.policy/'scope.json')
    assert CONTRACT['zero_receipt_sha256']==sha(a.zero/'receipt.json')
    assert CONTRACT['outside_absolute_tolerance']==.5/255
    assert CONTRACT['K_and_car_RGBA_byteexact_required'] and CONTRACT['eligibility_strict_zero_unchanged']
    assert CONTRACT['actual_zero_step_authorized'] and CONTRACT['expected_nativeF']==5 and CONTRACT['expected_nativeB']==0
    return OriginalQualify(a,asset,cfg)


class BoundedReadoutAdapter(generic.ComponentAdapter):
    def render(self,raw,component='shared'):
        out=LocalJointAdapter.render(self,raw,component)
        if component!='shared':return out
        rgb=npy(out['rgb']);alpha=npy(out['alpha']).squeeze();ref=bank(ARGS.zero/f'{raw:03d}/joint_render.npz')
        maskroot=Path(self.diagnostic_proof['mask_receipt']['path']).parent;m=bank(maskroot/f'{raw:03d}/scope_masks.npz')
        row=dict(raw=raw,checks={},outside_registered_absolute_tolerance=CONTRACT['outside_absolute_tolerance'])
        for name in ['outside_allowed_edit','outside_replacement','actor_protected','K_protected']:
            d=m[name];delta=abs(rgb-ref['rgb'])[:,d];ad=abs(alpha-ref['alpha'])[d]
            exactRGB=rgb[:,d].tobytes()==ref['rgb'][:,d].tobytes();exactAlpha=alpha[d].tobytes()==ref['alpha'][d].tobytes()
            mx=float(delta.max()) if d.any() else 0.;ma=float(ad.max()) if d.any() else 0.
            row['checks'][name]=dict(pixels=int(d.sum()),RGB_byteexact=exactRGB,alpha_byteexact=exactAlpha,
                RGB_changed_exact_pixels=int((delta.max(0)>0).sum()),alpha_changed_exact_pixels=int((ad>0).sum()),
                RGB_max_abs=mx,alpha_max_abs=ma,
                registered_check_pass=(exactRGB and exactAlpha) if name in ['actor_protected','K_protected'] else max(mx,ma)<=CONTRACT['outside_absolute_tolerance'])
        row['all_outside_and_protection_RGBA_exact']=all(x['RGB_byteexact'] and x['alpha_byteexact'] for x in row['checks'].values())
        row['outside_bound_and_car_K_exact_pass']=all(x['registered_check_pass'] for x in row['checks'].values())
        cfg=read(ARGS.measurement_contract);ev=bank(Path(cfg['fixed_eval'])/f'{raw:03d}/fixed_eval.npz')
        holes=ev['full_original_band']&(alpha<.95)&(ref['alpha']>=.95)
        row['full_original_band_new_alpha95_holes_vs_v4zero']=int(holes.sum())
        row['unknown_new_alpha95_holes_vs_v4zero']=int((holes&~ev['ground_eval']).sum())
        row['fixed_full_band_coverage_pass']=row['full_original_band_new_alpha95_holes_vs_v4zero']<=CONTRACT['new_full_original_band_or_unknown_alpha95_holes_allowed']
        contained.CHECKS.append(row);write(ARGS.output/f'{raw:03d}_outside_invariance.json',row)
        if not (row['outside_bound_and_car_K_exact_pass'] and row['fixed_full_band_coverage_pass']):
            np.savez_compressed(ARGS.output/f'{raw:03d}_failed_RGBA.npz',rgb=rgb,alpha=alpha)
            png(ARGS.output/f'{raw:03d}_failed_outside_delta.png',np.clip(abs(rgb-ref['rgb'])*4,0,1))
        assert row['outside_bound_and_car_K_exact_pass'] and row['fixed_full_band_coverage_pass'],'Registered outside/car/K or fixed full-band coverage check failed'
        return out


def main():
    global ARGS,CONTRACT
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    for k in ['readout-contract','zero','output','policy','measurement-contract']:parser.add_argument('--'+k,type=Path,required=True)
    ARGS,_=parser.parse_known_args();CONTRACT=read(ARGS.readout_contract)
    cfg=read(ARGS.measurement_contract)
    assert cfg['frames']==[86,110,134,179,198]
    for raw in cfg['frames']:
        ev=bank(Path(cfg['fixed_eval'])/f'{raw:03d}/fixed_eval.npz')
        assert ev['full_original_band'].shape==ev['ground_eval'].shape==(768,1152)
    # The reusable base CLI owns all remaining arguments. The new contract
    # flag belongs exclusively to this independent registered variant.
    at=sys.argv.index('--readout-contract');sys.argv=sys.argv[:at]+sys.argv[at+2:]
    generic.qualify=qualify;generic.ComponentAdapter=BoundedReadoutAdapter
    try:generic.main()
    finally:
        if '--execute' in sys.argv and (ARGS.output/'receipt.json').is_file():
            r=read(ARGS.output/'receipt.json');r.update(schema='R10.component_replacement_bounded_numeric_joint_zero.v1',
                readout_contract=pin(ARGS.readout_contract),strict_eligibility_unchanged=True,
                outside_absolute_tolerance=CONTRACT['outside_absolute_tolerance'],
                actual_outside_car_K_checks=contained.CHECKS,
                all_five_outside_bound_and_car_K_exact_pass=len(contained.CHECKS)==5 and all(x['outside_bound_and_car_K_exact_pass'] for x in contained.CHECKS),
                exact_RGBA_pass_is_separately_reported=True,prior_strict_failure_preserved=True,
                full_joint_quality_not_accepted_by_numeric_readout=True,whole_scene_quality_accepted=False)
            write(ARGS.output/'receipt.json',r)

if __name__=='__main__':main()
