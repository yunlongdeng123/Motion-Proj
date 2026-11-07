"""Fresh-process 10-forward replay comparison against final saved float/PNG.

Every field records exact equality and actual max error. No relaxed tolerance
is silently labelled exact; old parameters and the intrinsic road are frozen.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import time
import numpy as np
from canonical_road_runtime import CanonicalRoadRuntime,sha,gpu_preflight


def image(path,array):
    from PIL import Image
    Image.fromarray(np.uint8(np.clip(array.transpose(1,2,0),0,1)*255+.5)).save(path)


def arrays(out,scale):
    d={key:out[key].detach().cpu().numpy() for key in ('rgb','alpha','group_alpha_T','group_rgb_T','env_rgb_T')}
    d.update(newroad_alpha_T=out['extra'][16].cpu().numpy(),newroad_rgb_T=out['extra'][17:20].cpu().numpy())
    d['depth_m']=(out['depth_sum'][0]/out['alpha'][0].clamp_min(1e-8)/scale).cpu().numpy()
    d['oldroad_alpha_T']=d['group_alpha_T'][2]-d['newroad_alpha_T']
    d['oldroad_rgb_T']=d['group_rgb_T'][2]-d['newroad_rgb_T']
    return d


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--raws',type=int,nargs=5,default=[105,113,128,130,137])
    p.add_argument('--gpu-fraction',type=float,default=.4);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
    preflight=gpu_preflight()
    import torch
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
    start=time.time();model=CanonicalRoadRuntime(args.manifest,verify_dependencies=False);rt=model.rt
    snap=args.output/'source_snapshot';snap.mkdir()
    for item in model.manifest['replay_sources'].values():shutil.copy2(args.manifest.parent/item['file'],snap/item['file'])
    records=[];count_before=rt.render_calls
    with torch.no_grad():
        for raw in args.raws:
            folder=args.output/f'{raw:03d}';folder.mkdir()
            for state in ('keep','delete'):
                reference=args.reference/f'{raw:03d}'/f'candidate_{state}.npz'
                ref_png=args.reference/f'{raw:03d}'/f'candidate_{state}.png'
                if not reference.exists() or not ref_png.exists():raise FileNotFoundError('Final reference requires both float arrays and PNG')
                out=model.render_scene(raw,delete=state=='delete');actual=arrays(out,rt.scale)
                with np.load(reference,allow_pickle=False) as z:expected={key:z[key] for key in z.files}
                assert set(actual)==set(expected),('array contract',set(actual),set(expected))
                fields={}
                for key,value in actual.items():
                    ref=expected[key];assert value.shape==ref.shape and value.dtype==ref.dtype,key
                    fields[key]=dict(exact=np.array_equal(value,ref),maxabs=float(np.max(np.abs(value.astype(float)-ref.astype(float)))),
                        changed_elements=int(np.count_nonzero(value!=ref)))
                target=folder/f'{state}.png';image(target,actual['rgb'])
                np.savez_compressed(folder/f'{state}_float.npz',**actual)
                record=dict(raw=raw,state=state,fields=fields,PNG_exact=target.read_bytes()==ref_png.read_bytes(),
                    reference_float_sha256=sha(reference),reference_PNG_sha256=sha(ref_png),actual_PNG_sha256=sha(target),
                    group_conservation_maxabs=out['group_conservation_maxabs'],RGB_conservation_maxabs=out['color_reconstruction_maxabs'])
                records.append(record);(folder/f'{state}_comparison.json').write_text(json.dumps(record,indent=2)+'\n')
                del out
            rt._data.clear();torch.cuda.empty_cache();print(json.dumps(dict(raw=raw,seconds=time.time()-start)),flush=True)
    receipt=dict(status='REPLAY_COMPARISON_COMPLETE',all_float_fields_exact=all(v['exact'] for r in records for v in r['fields'].values()),
        all_PNG_exact=all(r['PNG_exact'] for r in records),records=records,forwards=rt.render_calls-count_before,
        runtime=model.receipt(),seconds=time.time()-start,preflight=preflight,gpu_fraction=args.gpu_fraction,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),adoption_claim=False,numerical_policy='report exact and observed errors; no relaxed exact label')
    assert receipt['forwards']==10
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({key:receipt[key] for key in ('all_float_fields_exact','all_PNG_exact','forwards','seconds')}))


if __name__=='__main__':main()
