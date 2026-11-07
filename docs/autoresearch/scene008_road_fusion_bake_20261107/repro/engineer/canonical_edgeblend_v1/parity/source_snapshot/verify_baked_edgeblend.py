"""Exactly10 fresh-process forwards against the root-owned full49 evidence."""
from pathlib import Path
import argparse
import json
import os
import shutil
import time
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--variant',choices=['edgeblend'],default='edgeblend');p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
    import torch
    from PIL import Image
    from canonical_road_runtime import CanonicalRoadRuntime,sha,gpu_preflight
    for raw in (105,113,128,130,137):
        for state in ('keep','delete'):
            for suffix in ('npz','png'):
                if not (args.reference/f'{raw:03d}'/f'{args.variant}_{state}.{suffix}').is_file():raise FileNotFoundError('Await root full49 keys, do not duplicate evaluation')
    args.output.mkdir(parents=True,exist_ok=False);preflight=gpu_preflight()
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.4);started=time.time()
    model=CanonicalRoadRuntime(args.manifest,verify_dependencies=False);rt=model.rt;records=[]
    snap=args.output/'source_snapshot';snap.mkdir()
    for info in model.manifest['replay_sources'].values():shutil.copy2(args.manifest.parent/info['file'],snap/info['file'])
    with torch.no_grad():
        for raw in (105,113,128,130,137):
            folder=args.output/f'{raw:03d}';folder.mkdir()
            for state in ('keep','delete'):
                out=model.render_scene(raw,delete=state=='delete')
                data={key:out[key].cpu().numpy() for key in ('rgb','alpha','group_alpha_T','group_rgb_T','env_rgb_T')}
                data.update(depth_m=(out['depth_sum'][0]/out['alpha'][0].clamp_min(1e-8)/rt.scale).cpu().numpy(),
                    newroad_alpha_T=out['extra'][16].cpu().numpy(),newroad_rgb_T=out['extra'][17:20].cpu().numpy())
                data['oldroad_alpha_T']=data['group_alpha_T'][2]-data['newroad_alpha_T'];data['oldroad_rgb_T']=data['group_rgb_T'][2]-data['newroad_rgb_T']
                ref=args.reference/f'{raw:03d}'/f'{args.variant}_{state}.npz'
                with np.load(ref,allow_pickle=False) as z:expected={key:z[key] for key in z.files}
                assert set(data)==set(expected),(set(data),set(expected))
                fields={key:dict(exact=np.array_equal(value,expected[key]),maxabs=float(np.abs(value.astype(float)-expected[key].astype(float)).max()),
                    changed_elements=int((value!=expected[key]).sum())) for key,value in data.items()}
                target=folder/f'{state}.png';Image.fromarray(np.uint8(np.clip(data['rgb'].transpose(1,2,0),0,1)*255+.5)).save(target)
                refpng=args.reference/f'{raw:03d}'/f'{args.variant}_{state}.png'
                np.savez_compressed(folder/f'{state}_float.npz',**data)
                records.append(dict(raw=raw,state=state,fields=fields,PNG_byte_exact=target.read_bytes()==refpng.read_bytes(),
                    reference_float_sha256=sha(ref),reference_PNG_sha256=sha(refpng),actual_PNG_sha256=sha(target)))
            rt._data.clear();torch.cuda.empty_cache();print(json.dumps(dict(raw=raw,seconds=time.time()-started)),flush=True)
    receipt=dict(variant=args.variant,forwards=rt.render_calls,all_float_exact=all(f['exact'] for r in records for f in r['fields'].values()),
        all_PNG_byte_exact=all(r['PNG_byte_exact'] for r in records),records=records,seconds=time.time()-started,preflight=preflight,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),runtime=model.receipt(),
        original_parameter_updates=0,optimizer_updates=0,adoption_claim=False,numerical_policy='report actual exactness; no tolerance relaxation')
    assert receipt['forwards']==10
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({key:receipt[key] for key in ('variant','forwards','all_float_exact','all_PNG_byte_exact','seconds')}))


if __name__=='__main__':main()
