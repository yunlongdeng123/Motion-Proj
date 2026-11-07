"""Register persistent native retirement from measured road and protection scores.

No native model training. The marker gradient is exactly the static alpha*T
mass on fixed evidence-derived target/protection rays in the single joint sort.
Mixed protected rows remain explicit rejected candidates, never per-frame holes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[k]='1'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--asset',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    from scipy.ndimage import binary_dilation
    from scipy.spatial import cKDTree
    from r0_runtime import R0Runtime
    from evaluate_r0 import write
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.55)
    started=time.time()
    contract=dict(schema='R0_PERSISTENT_REPLACEMENT_SCORES_V1',
        target='extension-solo alpha>=.98, excluding original a6 amodal+2px and retained-vehicle guard; delete a37 state',
        guard='observed protect outside target plus original a6 box, retained other vehicles and road markings; a37 keep protected by paired acceptance, not by treating its deleted silhouette as static truth',
        all49=True, marker_gradient_is_alphaT=True,
        geometric_candidate='native centre within0.8mXY of real-supported new road and height[-.2,3]m above local tangent',
        fit_score_min=.001, protected_to_fit_max=.01,
        coefficient=0., shared_across_time_camera_edit=True,
        mixed_rows='retain and list; no irreversible checkpoint edits; candidate gates require downstream paired validation',
        optimizer_updates=0)
    write(args.output/'contract.json',contract)
    model=R0Runtime(args.asset)
    rt=model.rt
    n=len(rt.n.static.get_xyz)
    fit=torch.zeros(n,device='cuda')
    protect=torch.zeros_like(fit)
    with np.load(args.asset) as z:
        local=z['local_xyz'][model.extension.ids].astype(float)
        normals=z['normal_local'][model.extension.ids].astype(float)
    transform=np.linalg.inv(np.asarray(rt.exp.meta['world_to_model'])@np.asarray(rt.exp.meta['coordinate_anchor']))
    centre=rt.n.static.get_xyz.detach().cpu().numpy()
    centre=centre@transform[:3,:3].T+transform[:3,3]
    distance,nn=cKDTree(local[:,:2]).query(centre[:,:2])
    height=((centre-local[nn])*normals[nn]).sum(1)
    geometric=(distance<=.8)&(height>=-.2)&(height<=3.)
    per_frame=[]
    for raw in range(89,138):
        data=rt.raw(raw);obs=data['obs']
        with torch.no_grad():
            solo=model.render(raw,solo=True)
            with np.load(rt.road_run/'inputs/views'/f'{raw:03d}.npz') as z:
                vehicle=z['other_vehicle_guard'].copy();markings=z['marking_mask'].copy()
            target=(solo['alpha'][0].cpu().numpy()>=.98)&~binary_dilation(obs['box_candidate'][0],iterations=2)&~vehicle
            guard=(obs['protect']&~target)|obs['box_candidate'][0]|vehicle|markings
            scene=model.scene(raw,delete=True)
            rows=scene['rows']
            assert scene['meta']['native_static_rows']==n
            del solo
        marker=torch.zeros((n,2),device='cuda',requires_grad=True)
        feature=rows['features'].clone()
        feature[:n,22:24]=marker
        kwargs=dict(rows,features=feature)
        out=rt.raster(scene['cam'],rt.pipe,0,kwargs,scene['bg'])
        loss=out['extra'][20,torch.as_tensor(target,device='cuda')].sum()+out['extra'][21,torch.as_tensor(guard,device='cuda')].sum()
        loss.backward()
        assert marker.grad is not None and bool(torch.isfinite(marker.grad).all())
        fit+=marker.grad[:,0].detach()
        protect+=marker.grad[:,1].detach()
        per_frame.append(dict(raw=raw,target_pixels=int(target.sum()),guard_pixels=int(guard.sum()),
                              static_fit_mass=float(marker.grad[:,0].sum()),static_guard_mass=float(marker.grad[:,1].sum())))
        del rows,scene,kwargs,out,feature,loss,marker
        rt._data.clear()
        torch.cuda.empty_cache()
        if raw%10==0:
            print(json.dumps(dict(raw=raw,seconds=time.time()-started)),flush=True)
    fit=fit.cpu().numpy();protect=protect.cpu().numpy()
    supported=geometric&(fit>contract['fit_score_min'])
    pure=supported&(protect<=contract['protected_to_fit_max']*fit)
    mixed=supported&~pure
    ids=np.flatnonzero(pure)
    np.savez_compressed(args.output/'retirement.npz',ids=ids,coefficients=np.zeros(len(ids),np.float32),
                        fit_score=fit[ids],protected_score=protect[ids])
    np.savez_compressed(args.output/'all_scores.npz',fit_score=fit,protected_score=protect,
        geometric_candidate=geometric,mixed=mixed,height_m=height,nearest_measured_XY_distance_m=distance)
    write(args.output/'receipt.json',dict(status='LEDGER_REGISTERED_NOT_QUALITY_ACCEPTED',
        original_static_rows=n,geometric_candidates=int(geometric.sum()),scored_candidates=int(supported.sum()),
        pure_retired_candidates=len(ids),mixed_retained_candidates=int(mixed.sum()),
        retirement_sha256=hashlib.sha256((args.output/'retirement.npz').read_bytes()).hexdigest(),
        selected_fit_mass=float(fit[pure].sum()),retained_mixed_fit_mass=float(fit[mixed].sum()),
        selected_protected_mass=float(protect[pure].sum()),frames=per_frame,
        seconds=time.time()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        contract=contract,runtime=model.receipt(),optimizer_updates=0,
        marker_backward_calls=49,no_new_identity_truth_claim=True,quality_accepted=False))


if __name__=='__main__':
    main()
