"""One runtime construction, zero raster forwards: exact persistent tuple capture.

Do not execute without root GPU scheduling. Final material is applied by CPU
exporter and does not require loading another runtime or geometry rebuild.
"""
from pathlib import Path
import argparse
import json
import os
import time
import numpy as np
from world_surface_sampling import sha,array_hash
from canonical_road_runtime import gpu_preflight


def capture(model,target):
    target=Path(target)
    if target.exists():raise FileExistsError('Preserve tuple capture')
    assert model.factor==2 and model.refined is not None
    assert model.opacity_fn is None,'Capture intrinsic geometry/opacity; view AA belongs to renderer'
    cv=lambda t:t.detach().cpu().contiguous().numpy().copy()
    child=model.refined;parent=model.parent.extension
    pairs={'render_model_xyz':'xyz','render_scales_model':'scaling','render_rotation_wxyz':'rotation',
        'render_opacity_logit':'opacity','render_colours_srgb':'_rgb'}
    values={key:cv(getattr(child,attribute)) for key,attribute in pairs.items()}
    values.update({f'parent_{key}':cv(getattr(parent,attribute)) for key,attribute in pairs.items()})
    values.update(child_to_parent_ids=cv(child.parent_indices),factor=np.array(2,np.int16),
        capture_origin=np.array('NATIVE_REFINED_PERSISTENT_TUPLE_NO_FORWARD'),
        parent_manifest_sha256=np.array(sha(model.manifest_path)),parent_asset_sha256=np.array(model.surface_sha),
        refined_source_sha256=np.array(sha(Path(__file__).with_name('refined_road_runtime.py'))),
        captured_model_scale=np.array(model.rt.scale),captured_V44_binding_sha256=np.array(model.rt.binding_sha256))
    assert child.unchanged() and parent.unchanged();model.rt.assert_unchanged(full=True)
    target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,**values)
    with np.load(target,allow_pickle=False) as z:
        checks={key:array_hash(value)==array_hash(z[key]) for key,value in values.items()}
    assert all(checks.values())
    return dict(path=str(target),sha256=sha(target),tuple_hashes={key:array_hash(value) for key,value in values.items()},
        exact_reload_checks=checks,forwards=model.rt.render_calls,original_parameter_updates=0,training_updates=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu-fraction',type=float,default=.4);args=p.parse_args()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
    preflight=gpu_preflight()
    import torch
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
    from refined_road_runtime import RefinedRoadRuntime
    start=time.time();model=RefinedRoadRuntime(args.parent_manifest,factor=2,verify_dependencies=False)
    record=capture(model,args.output);record.update(seconds=time.time()-start,preflight=preflight,gpu_fraction=args.gpu_fraction,
        GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        source_sha256=sha(__file__),runtime=model.receipt())
    receipt=args.output.with_suffix('.receipt.json');receipt.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),sha256=record['sha256'],forwards=record['forwards'],seconds=record['seconds'])))


if __name__=='__main__':main()
