"""Copy actual DGGT input, literal output and diagnostic PNGs without edits."""
from pathlib import Path
import argparse,shutil
from joint_rgb_common import read,write,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run=a.root/'dggt_no_diff_control_v1';receipt=read(run/'receipt.json');assert receipt['complete']
    inp=a.root/'inputs/forward_smoke_five_views';a.output.mkdir(parents=True,exist_ok=False);files=[]
    names=['official_literal_RGB','diagnostic_single_alpha_RGB','joint_RGB_premult','alpha','static_RGB_premult','dynamic_RGB_premult','static_alpha','dynamic_alpha','expected_depth_display']
    for raw in receipt['frames']:
        pairs=[('input',inp/f'{raw:03d}_dggt_input.png')]+[(name,run/f'{raw:03d}/{name}.png') for name in names]
        for name,source in pairs:
            target=a.output/f'{raw:03d}_{name}.png';shutil.copy2(source,target);assert sha(source)==sha(target)
            files.append(dict(raw=raw,kind=name,source=str(source),copy=str(target),sha256=sha(target),bytes=target.stat().st_size))
    refs=[('render_receipt.json',run/'receipt.json'),('original_forward_failed_receipt.json',a.root/'forward_smoke_five_views_v1/receipt.json'),
          ('recovered_forward_receipt.json',a.root/'forward_smoke_five_views_recovered_v2/receipt.json'),
          ('gauge_audit.json',a.root/'forward_smoke_five_views_v1/gauge_audit.json'),('input_contract.json',inp/'contract.json')]
    for name,source in refs:shutil.copy2(source,a.output/name)
    write(a.output/'manifest.json',dict(schema='R8.actual_DGGT_original_resolution_previews.v1',complete=True,files=files,
        image_wh=[518,350],PNG_bytes_exact=True,no_processing=True,literal_is_official_control=True,
        single_alpha_is_separately_named_diagnostic=True,task='same original input self reconstruction; original A9 is retained',
        no_road_deletion_success_claim=True,nativeF=0,nativeB=0,modelF=0,updates=0))
    print(dict(manifest=str(a.output/'manifest.json'),sha256=sha(a.output/'manifest.json'),PNG_files=len(files)))


if __name__=='__main__':main()
