"""Frozen official PVD car completion: 200 fixed observations + 1848 DDPM points."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(16*1024**2),b''):h.update(data)
    return h.hexdigest()


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-contract',type=Path,required=True)
    ap.add_argument('--normalization-contract',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--resume-state',type=Path)
    args=ap.parse_args()
    root=Path(__file__).parent
    sys.path.insert(0,str(root/'vendor'));sys.path.insert(0,str(root))
    import numpy as np
    import torch
    from scipy.spatial import cKDTree
    from inference_core import Model,get_betas
    torch.set_num_threads(8)
    args.output.mkdir(parents=True,exist_ok=True)
    inp=json.loads(args.input_contract.read_text())
    norm=json.loads(args.normalization_contract.read_text())
    if norm.get('partial_bounds_scaled') is not False or norm.get('partial_bounds_centered') is not False:
        raise RuntimeError('Partial cloud normalization is prohibited')
    transform=np.asarray(norm['pvd_to_actor_native_4x4'],dtype=np.float64)
    inverse=np.linalg.inv(transform)
    points_path=Path(inp['points_npz'])
    if sha(points_path)!=inp['points_sha256']:raise RuntimeError('Original observations changed')
    data=np.load(points_path,allow_pickle=False)
    raw=np.asarray(data['observed_points_actor_native_m'],dtype=np.float64)
    size=np.asarray(data['size_lwh_m'],dtype=np.float64)
    if not np.allclose(size,norm['whole_box_lwh_m'],atol=1e-6):raise RuntimeError('Fixed whole box changed')
    rng=np.random.default_rng(args.seed)
    indices=rng.choice(len(raw),200,replace=False)
    partial_native=raw[indices]
    partial_pvd=(partial_native@inverse[:3,:3].T+inverse[:3,3]).astype(np.float32)
    checkpoint_path=root/'weights/car_completion_3199.pth'
    checkpoint_sha=sha(checkpoint_path)
    if checkpoint_sha!='567ec9e6f0c3b453e128d916363bb8f1ddf6a432eef13d10035ee006d732dc1a':
        raise RuntimeError('Author checkpoint hash drifted')
    signature=dict(seed=args.seed,checkpoint_sha256=checkpoint_sha,input_sha256=sha(args.input_contract),
                   normalization_sha256=sha(args.normalization_contract),partial_count=200,
                   new_noise_points=1848,total_points=2048,ddpm_steps=1000)
    np.savez_compressed(args.output/'fixed_partial_input.npz',raw_row_indices=indices,
                        points_actor_native_m=partial_native,points_pvd_normalized=partial_pvd)
    dump(args.output/'transform_contract.json',dict(norm,**signature,
         input_contract=str(args.input_contract),checkpoint=str(checkpoint_path)))
    started=time.monotonic()
    ckpt=torch.load(checkpoint_path,map_location='cpu')
    options=types.SimpleNamespace(nc=3,svpoints=200,embed_dim=64,attention=True,dropout=.1)
    model=Model(options,get_betas('linear',.0001,.02,1000),'mse','eps','fixedsmall')
    model.load_state_dict({key.replace('model.module.','model.',1):value for key,value in ckpt['model_state'].items()},strict=True)
    del ckpt
    model.requires_grad_(False);model.model.eval();model.training=False
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    model=model.cuda()
    partial=torch.from_numpy(partial_pvd.T[None].copy()).cuda()
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    checkpoints=args.output/'trajectory';checkpoints.mkdir(exist_ok=True)
    if args.resume_state:
        resume=torch.load(args.resume_state,map_location='cpu')
        if resume['signature']!=signature:raise RuntimeError('PVD resume contract mismatch')
        current=resume['points_pvd_normalized'].cuda()
        next_t=resume['next_t']
        torch.set_rng_state(resume['torch_cpu_rng_state'])
        torch.cuda.set_rng_state(resume['torch_cuda_rng_state'])
    else:
        current=torch.cat([partial,torch.randn((1,3,1848),device='cuda',dtype=torch.float32)],dim=-1)
        next_t=999
    def save_state(t,label):
        torch.save(dict(signature=signature,points_pvd_normalized=current.detach().cpu(),
                   fixed_partial_pvd_normalized=partial.detach().cpu(),next_t=t,
                   torch_cpu_rng_state=torch.get_rng_state(),torch_cuda_rng_state=torch.cuda.get_rng_state(),
                   checkpoint_path=str(checkpoint_path),normalization_contract=norm),checkpoints/f'{label}.pt')
    save_state(next_t,'initial_or_resumed')
    sampling_started=time.monotonic()
    completed=0
    with torch.no_grad():
        for t in reversed(range(next_t+1)):
            timestep=torch.full((1,),t,device='cuda',dtype=torch.int64)
            current=model.diffusion.p_sample(model._denoise,current,timestep,noise_fn=torch.randn,
                                            clip_denoised=False,return_pred_xstart=False)
            completed+=1
            if t==999 or t%100==0:
                if not torch.isfinite(current).all():raise RuntimeError(f'Nonfinite PVD state at timestep {t}')
                if not torch.equal(current[:,:,:200],partial):raise RuntimeError('Fixed PVD observations moved')
                save_state(t-1,f'after_t{t:04d}')
                if t==999:
                    dump(args.output/'first_step_finite_check.json',dict(finite=True,
                         fixed_partial_exact=True,timestep=999,total_points=2048,
                         peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                         elapsed_seconds=time.monotonic()-sampling_started))
                print(json.dumps(dict(timestep=t,completed_this_run=completed,elapsed_seconds=time.monotonic()-sampling_started)),flush=True)
    torch.cuda.synchronize()
    normalized=current[0].T.detach().cpu().numpy()
    native=normalized.astype(np.float64)@transform[:3,:3].T+transform[:3,3]
    if not np.isfinite(native).all():raise RuntimeError('PVD inverse transform is nonfinite')
    partial_error=float(np.max(np.abs(native[:200]-partial_native)))
    if partial_error>2e-5:raise RuntimeError('PVD partial point inverse mismatch')
    np.savez_compressed(args.output/'candidate_points_actor_native.npz',points_actor_native_m=native.astype(np.float32),
                        points_pvd_normalized=normalized,generated_mask=np.arange(2048)>=200,
                        observed_raw_row_indices=indices)
    nearest,_=cKDTree(native).query(raw,k=1,workers=4)
    inside=np.all(np.abs(native)<=size[None]/2+1e-6,axis=1)
    report=dict(schema_version='c2.frozen_pvd_car_completion.v1',model='PVD car_completion_3199',
        checkpoint_sha256=checkpoint_sha,source_commit='9747265a5f141e5546fd4f862bfa66aa59f1bd33',
        seed=args.seed,diffusion_steps_this_run=completed,official_total_diffusion_steps=1000,
        total_points=2048,fixed_original_points=200,new_diffused_points=1848,training_steps=0,
        trainable_prior_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        partial_coordinate_max_error_m=partial_error,all_points_finite=True,
        native_bounds_m=[native.min(0).tolist(),native.max(0).tolist()],inside_whole_box_fraction=float(inside.mean()),
        generated_inside_whole_box_fraction=float(inside[200:].mean()),
        observed_to_complete_approx_m=dict(mean=float(nearest.mean()),p50=float(np.quantile(nearest,.5)),p90=float(np.quantile(nearest,.9))),
        total_wall_seconds=time.monotonic()-started,sampling_wall_seconds=time.monotonic()-sampling_started,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        normalization_source=norm['normalization_source'],input_sha256=signature['input_sha256'],
        worker_sha256=sha(__file__),resumed_from=str(args.resume_state) if args.resume_state else None,
        quality_status='Actual frozen partial-to-complete proposal; downstream legal-box/observed constraints and RGB fit still required')
    dump(args.output/'proposal_report.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
