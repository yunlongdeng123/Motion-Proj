"""Intrinsic-coverage domain adapter; all v3 native VJP/FP32/render mathematics is reused unchanged."""
from pathlib import Path
import argparse,sys
import measure_local_edit_footprint as original
from prepare_intrinsic_coverage_masks_cpu import prepare_or_validate,pin,read,write

def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--intrinsic',type=Path,required=True);extra,remaining=p.parse_known_args()
    def domain(a,cfg,asset):
        a.intrinsic=extra.intrinsic
        return prepare_or_validate(a,cfg,asset.manifest)
    original.masks=domain
    sys.argv=[sys.argv[0],*remaining]
    output=Path(remaining[remaining.index('--output')+1]);execute='--execute' in remaining
    try:original.main()
    finally:
        rp=output/'receipt.json'
        if execute and rp.is_file():
            record=read(rp);record['inherited_measurement_schema']=record['schema'];record['schema']='R10.actual_intrinsic_coverage_footprint_alphaT_VJP.v1'
            record['intrinsic_coverage_receipt']=pin(extra.intrinsic/'receipt.json')
            record['mask_definition']='(original band union intrinsic local-only alpha>0) minus K/car; replacement uses intrinsic alpha>=.95'
            record['intrinsic_domain_adapter']=pin(__file__);record['unchanged_v3_measurement_source']=pin(original.__file__)
            record['new_native_raster_implementation']=False;record['original_band_evaluation_denominator_unchanged']=True
            write(rp,record)
if __name__=='__main__':main()
