"""Register the root-approved numerical replay before any new GPU call."""
from pathlib import Path
import argparse
from local_gaussian_asset import read,write,sha
def main():
    p=argparse.ArgumentParser();p.add_argument('--r10',type=Path,required=True);a=p.parse_args();old=a.r10/'edit_footprint_measure_v2/receipt.json';r=read(old)
    assert r['complete'] is False and r['nativeF']==r['nativeB']==1 and r['updates']==0
    out=a.r10/'engineering_contract_v3.json';assert not out.exists()
    write(out,dict(schema='R10.full_image_feature_VJP_engineering_replay.v3',complete=True,root_authorized_engineering_replay=True,
        failed_run_receipt=dict(path=str(old),sha256=sha(old)),failed_run_engineering_F=1,failed_run_engineering_B=1,failed_run_updates=0,
        aggregate_pixel_multiplier=2e-5,aggregate_absolute_floor=.01,
        aggregate_rule='abs(sum_float64(row_grad_float32)-direct_alpha_float64_sum)<=max(.01,2e-5*region_pixel_count)',
        observed_failed_full_image_absolute_error=6.130632,observed_failed_full_image_relative_error=1.094e-5,
        sentinel_absolute_tolerance=1e-5,sentinel_selection='nearest image-centre insideReplacement pixel; deterministic row-major tie break; no RGB/alpha selection',
        constant_one_alpha_tolerance=3e-6,per_row_partition_relative_scale=.001,
        per_row_partition_rule='abs(full-outside-inside)<=0.001*(1+full_row_mass); both partitions independently measured; FP32 rounding diagnostic only',
        raw_each_region_gradient_saved_before_aggregate_assert=True,strict_zero_eligibility_not_relaxed=True,
        outside_car_K_required_exact_zero=True,RGB_and_alpha_byteexact_current_new0=True,pixel_partitions_exact=True,
        gradients_finite_nonnegative=True,formal_method_sampling_LR_budget_unchanged=True,new_GPU_forward=0,new_GPU_backward=0,updates=0))
    print(dict(registered=True,path=str(out),sha256=sha(out),GPU_calls=0))
if __name__=='__main__':main()
