"""Copy the fifteen actual final native PNGs without image processing."""
from pathlib import Path
import argparse,shutil
from joint_rgb_common import read,write,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    receipt=read(a.run/'receipt.json');cfg=read(a.run/'fit_contract.json')
    assert receipt['complete'] and receipt['logical_steps']==400 and receipt['actual_optimizer_updates']==400
    a.output.mkdir(parents=True,exist_ok=False);rows=[]
    for raw in cfg['frames']:
        for component in ['shared','road','actor']:
            source=a.run/f'final_readout/{raw:03d}/{component}.png';out=a.output/f'{raw:03d}_{component}.png'
            shutil.copy2(source,out);digest=sha(source);assert sha(out)==digest
            rows.append(dict(raw=raw,component=component,source=str(source),copy=str(out),sha256=digest,bytes=out.stat().st_size))
    write(a.output/'manifest.json',dict(schema='R8.actual_formal400_native_PNG_preview.v1',complete=True,
        source_run=str(a.run),run_receipt_sha256=sha(a.run/'receipt.json'),original_render_wh=[1152,768],
        files=rows,PNG_bytes_exact=True,no_processing=True,nativeF=0,nativeB=0,updates=0))
    print(dict(complete=True,manifest=str(a.output/'manifest.json'),sha256=sha(a.output/'manifest.json'),files=len(rows)))


if __name__=='__main__':main()
