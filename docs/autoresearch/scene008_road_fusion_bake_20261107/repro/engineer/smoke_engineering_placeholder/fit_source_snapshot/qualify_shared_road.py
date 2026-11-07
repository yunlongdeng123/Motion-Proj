"""One shared80F/B eligibility and baseline cache for direct/fusion comparison."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse
import json
import shutil
import time
import numpy as np
from shared_fit_common import OUT,BUILD,HELD,load,write,sha
from fusion_colour_runtime import FusionColourRuntime,PARENT

MASKS=PARENT.parents[2]/'scene008_road_r0_v44_20261107_r1/surface/material_cellfit_strip_v1/supervision_masks.npz'


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--masks',type=Path,default=MASKS)
    p.add_argument('--gpu-fraction',type=float,default=.4);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    masks=load(args.masks.resolve());assert sum(int(masks[f'raw{raw:03d}'].sum()) for raw in BUILD)==714546
    shutil.copy2(args.masks,args.output/'supervision_masks.npz')
    sources=Path(__file__).parent;source_hashes={name:sha(sources/name) for name in ('shared_fit_common.py','fusion_colour_runtime.py','qualify_shared_road.py')}
    snap=args.output/'qualification_source_snapshot';snap.mkdir()
    for name in source_hashes:shutil.copy2(sources/name,snap/name)
    import torch
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
    # Preflight before constructing any native CUDA model.
    import importlib.util
    spec=importlib.util.spec_from_file_location('_preflight_canonical',PARENT/'canonical_road_runtime.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);preflight=module.gpu_preflight()
    start=time.time();model=FusionColourRuntime();initial=model.initial
    cache=args.output/'baseline_cache';cache.mkdir();masses=np.zeros((2,10098),np.float64);counts=np.zeros((2,10098),np.int16)
    records=[];cache_hashes={}
    for raw in BUILD:
        domain=torch.as_tensor(masks[f'raw{raw:03d}'],device='cuda',dtype=torch.bool)
        for si,state in enumerate(('keep','delete')):
            probe=initial.detach().clone().requires_grad_(True);out=model.render(raw,state,probe)
            derivative=torch.autograd.grad(out['rgb'][:,domain].sum(),probe)[0]
            gradient=derivative.detach().cpu().numpy();assert np.isfinite(gradient).all() and gradient.min()>=0
            mass=gradient.mean(1);masses[si]+=mass;counts[si]+=(mass>0)
            wsum=float(out['newroad_alpha_T'][domain].sum().detach().cpu());difference=abs(float(mass.sum())-wsum)
            assert difference<.001+1e-5*max(wsum,1)
            target=cache/f'raw{raw:03d}_{state}.npz'
            np.savez_compressed(target,**{key:out[key].detach().cpu().numpy() for key in ('rgb','newroad_rgb_T','newroad_alpha_T','group_alpha_T')})
            cache_hashes[target.name]=sha(target)
            records.append(dict(raw=raw,state=state,pixels=int(domain.sum()),positive_parent_rows=int((mass>0).sum()),
                marker_mass=float(mass.sum()),road_W_sum=wsum,conservation_abs=difference,
                group_conservation=out['group_conservation_maxabs'],RGB_conservation=out['RGB_conservation_maxabs']))
            del out,probe,derivative,gradient
        model.rt._data.clear();torch.cuda.empty_cache()
        print(json.dumps(dict(raw=raw,seconds=time.time()-start)),flush=True)
    eligible=masses.sum(0)>0
    np.savez_compressed(args.output/'qualified_parents.npz',eligible=eligible,eligible_parent_ids=np.flatnonzero(eligible),
        observed_mass_by_state=masses,observed_frame_count_by_state=counts,initial_colours_srgb=initial.cpu().numpy())
    contract=dict(schema='SHARED_PARENT_ROAD_DIRECT_VS_FUSION_FIXED_GEOMETRY_V1',BUILD=list(BUILD),HELD=list(HELD),states=['keep','delete'],
        parent_manifest=str(model.manifest_path),parent_manifest_sha256=sha(model.manifest_path),
        canonical_asset_sha256=sha(model.model.asset_path),fit_masks_sha256=sha(args.output/'supervision_masks.npz'),
        qualified_parents_sha256=sha(args.output/'qualified_parents.npz'),source_sha256=source_hashes,
        target_schema='targets/projected_BUILD40/rawNNN.npz: direct_rgb/fusion_rgb float32HWC, fit_mask boolHW; shared keep/delete. Final target manifest must be ready; direct=unfused190source atlas, fusion=same190source Poisson atlas. Qualification uses mask only.',
        loss='.5*(mean RGB MSE on fixed fit_mask keep + mean RGB MSE on identical fit_mask delete); no regularizers',
        optimizer=dict(name='Adam',lr=.01,betas=[.9,.999],eps=1e-8,steps_per_variant=400,paired_states_per_step=2,epochs=10),
        parameterization='eligible parent RGB Parameter, projected to[0,1] after step; exact-zero columns absent from Parameter and Adam state',
        supervision_comparison='same190 nonHELD real source pool, parameter space, masks/sampler/budget; only atlas fusion operator changes',
        frozen='native/C2/oldroad/static/gates/environment/camera/trajectory/newroad geometry/sigma/rotation/opacity/view-AA',
        adoption_claim=False,hidden_GT_used=False)
    write(args.output/'shared_contract.json',contract);write(cache/'manifest.json',dict(files=cache_hashes,forwards=80,shared_by_both_variants=True))
    receipt=dict(status='PASS_SHARED_PREFIT_ELIGIBILITY',forwards=model.forwards,backwards=80,optimizer_steps=0,original_updates=0,
        eligible_parents=int(eligible.sum()),exact_zero_parents=int((~eligible).sum()),records=records,seconds=time.time()-start,
        preflight=preflight,gpu_fraction=args.gpu_fraction,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),runtime=model.receipt())
    assert model.forwards==80
    write(args.output/'qualification_receipt.json',receipt)
    print(json.dumps(dict(eligible_parents=receipt['eligible_parents'],exact_zero_parents=receipt['exact_zero_parents'],seconds=receipt['seconds'])))


if __name__=='__main__':main()
