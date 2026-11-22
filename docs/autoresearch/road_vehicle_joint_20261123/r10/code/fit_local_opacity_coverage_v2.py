"""V2 changes only alpha protection readout direction, before new GPU work.

All V1 loss, Adam, masks, rows, sampler, geometry, source and lifecycle code is
reused unchanged. Original absolute-alpha FAIL and its code remain intact.
"""
import numpy as np
import fit_local_opacity as v1

original_checks=v1.checks


def coverage_checks(value,target,cfg,*,zero=False):
    result=original_checks(value,target,cfg,zero=zero)
    result['v1_absolute_alpha_readout_decision']=result['decision']
    alpha=value['alpha'];outside=target['outside_allowed_edit'];original=target['original_alpha']
    decrease=np.maximum(original-alpha,0)
    result.update(outside_alpha_decrease_max=float(decrease[outside].max(initial=0)),
        outside_alpha_increase_max=float(np.maximum(alpha-original,0)[outside].max(initial=0)),
        outside_original_alpha_ge95_pixels=int((outside&(original>=.95)).sum()),
        outside_new_alpha95_holes=int((outside&(original>=.95)&(alpha<.95)).sum()),
        alpha_finite_and_in_0_1=bool(np.isfinite(alpha).all() and (alpha>=0).all() and (alpha<=1).all()),
        protection_readout_version='v2_coverage_decrease_only_no_new_holes')
    passed=result['finite'] and result['alpha_finite_and_in_0_1'] and result['K_car_RGB_exact'] and result['K_car_alpha_exact']
    passed&=result['outside_RGB_max']<=cfg['outside_RGB_max_abs_bound']
    passed&=result['outside_alpha_decrease_max']<=cfg['outside_alpha_max_decrease_bound']
    passed&=result['outside_new_alpha95_holes']==0 and result['new_full_band_alpha95_holes']==0
    if zero:passed&=result['step0_full_RGBA_exact_parentzero']
    result['decision']='PASS' if passed else 'FAIL'
    return result


def main():
    v1.checks=coverage_checks
    v1.main()


if __name__=='__main__':main()
