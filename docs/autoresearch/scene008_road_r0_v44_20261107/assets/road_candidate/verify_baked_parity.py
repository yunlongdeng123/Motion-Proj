"""Exactly eight independent no-grad replay forwards for a baked candidate.

No numeric tolerance upgrades a non-identical result to an exact pass. All
nonzero float differences, missing reference groups and PNG differences remain
explicit in the receipt for review.
"""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4194304),b''):h.update(block)
    return h.hexdigest()


def write(path,v):Path(path).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def gpu_preflight():
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True).strip()
    unknown=[x for x in apps.splitlines() if x and int(x.split(',')[0]) not in (13030,15730,os.getpid())]
    if int(gpu.split(',')[0])<20*1024 or unknown:raise RuntimeError('GPU not released for independent parity: '+str((gpu,apps)))
    return dict(gpu=gpu,processes=apps,own_pid=os.getpid())


def difference(actual,expected):
    assert actual.shape==expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all()
    d=np.abs(actual-expected);where=np.unravel_index(d.argmax(),d.shape)
    return dict(maxabs=float(d.max()),meanabs=float(d.astype(np.float64).mean()),different_values=int((d!=0).sum()),
        maximum_index=list(map(int,where)),exact=np.array_equal(actual,expected))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--reference-results',type=Path,required=True)
    ap.add_argument('--reference-variant',default='calibrated')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();gpu=gpu_preflight();args.output.mkdir(parents=True,exist_ok=False)
    import torch
    from PIL import Image
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.30)
    sys.path.insert(0,str(args.manifest.parent));from baked_road_runtime import BakedRoadRuntime
    start=time.time();records=[];forward_count=0
    with torch.no_grad():
        model=BakedRoadRuntime(args.manifest)
        for raw in (105,113,130,137):
            for state in ('keep','delete'):
                if forward_count>=8:raise RuntimeError('Eight-forward replay budget exhausted')
                out=model.render_scene(raw,delete=state=='delete');forward_count+=1
                alpha=out['alpha'][0]
                arrays=dict(rgb=out['rgb'].cpu().numpy(),alpha=alpha.cpu().numpy(),
                    depth_m=(out['depth_sum'][0]/alpha.clamp_min(1e-8)/model.rt.scale).cpu().numpy(),
                    group_alpha_T=out['group_alpha_T'].cpu().numpy(),extension_alpha_T=out['extension_alpha_T'].cpu().numpy())
                reference=args.reference_results/f'{raw:03d}'/(args.reference_variant+'_'+state+'_float.npz')
                with np.load(reference,allow_pickle=False) as saved:
                    comparisons={key:difference(value,saved[key]) for key,value in arrays.items() if key in saved.files}
                    missing=[key for key in arrays if key not in saved.files]
                    group_rgb_reference_available='group_rgb_T' in saved.files
                    if group_rgb_reference_available:comparisons['group_rgb_T']=difference(out['group_rgb_T'].cpu().numpy(),saved['group_rgb_T'])
                folder=args.output/f'{raw:03d}';folder.mkdir(exist_ok=True)
                np.savez_compressed(folder/(state+'_float.npz'),**arrays)
                png=folder/(state+'.png');encoded=np.uint8(np.clip(arrays['rgb'].transpose(1,2,0),0,1)*255+.5)
                Image.fromarray(encoded).save(png)
                old_png=args.reference_results/f'{raw:03d}'/(args.reference_variant+'_'+state+'.png')
                old_pixels=np.asarray(Image.open(old_png).convert('RGB'))
                png_info=dict(sha256=sha(png),reference_sha256=sha(old_png),file_exact=sha(png)==sha(old_png),
                    pixel_exact=np.array_equal(encoded,old_pixels),different_channels=int((encoded!=old_pixels).sum()))
                old_metrics=json.loads((args.reference_results/f'{raw:03d}/metrics.json').read_text())
                expected_asset_hash=old_metrics['states'][state][args.reference_variant]['native_asset_rows_hash']
                row=dict(raw=raw,state=state,comparisons=comparisons,missing_reference_arrays=missing,
                    group_RGB_reference_available=group_rgb_reference_available,
                    group_RGB_reference_limitation=None if group_rgb_reference_available else 'Reference saved only group alpha; RGB group equality is not inferred',
                    PNG=png_info,reference_float_sha256=sha(reference),
                    frozen_vehicle_rows_exact=out['meta']['native_asset_row_hash']==expected_asset_hash,
                    A37_packed_rows=out['meta']['native_a37_packed_rows'],
                    alpha_group_conservation_maxabs=out['group_conservation_maxabs'],RGB_reconstruction_maxabs=out['color_reconstruction_maxabs'])
                records.append(row);write(folder/(state+'_comparison.json'),row)
                print(json.dumps(dict(raw=raw,state=state,maxabs={k:v['maxabs'] for k,v in comparisons.items()},PNG=png_info)),flush=True)
                del out;model.rt._data.clear();torch.cuda.empty_cache()
        exact=all(not r['missing_reference_arrays'] and all(v['exact'] for v in r['comparisons'].values()) and r['PNG']['file_exact'] and r['frozen_vehicle_rows_exact'] for r in records)
        receipt=dict(status='PASS_EXACT_BAKED_PARITY' if exact else 'BAKED_NUMERIC_DIFFERENCES_REVIEW_REQUIRED',
            research_candidate=True,adoption_claim=False,actual_forward_count=forward_count,maximum_forward_budget=8,
            reference_results=str(args.reference_results),reference_variant=args.reference_variant,
            records=records,runtime=model.receipt(),seconds=time.time()-start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),preflight_gpu=gpu,
            optimizer_updates=0,parameter_updates=0,shutdown_performed=False)
        assert forward_count==8;write(args.output/'receipt.json',receipt)
        print(json.dumps(dict(status=receipt['status'],forwards=forward_count,seconds=receipt['seconds'])),flush=True)


if __name__=='__main__':main()
