"""Inspect or decode the official Gaussian head from the already fixed TRELLIS SLAT.

The executable path performs no image encoding, diffusion sampling, or new seed draw.
GPU execution remains gated by the parent scheduler's later authorization.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import shutil

ROOT=Path(__file__).resolve().parent
CHECKPOINT=Path('/root/autodl-tmp/models/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16')
SLAT=ROOT/'runs/trellis_hq_rear_seed101_r1/slat.pt'
REGISTERED=ROOT/'cpu_audits/trellis_hq_rear_seed101_r1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(16<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--inspect-cpu',action='store_true')
    mode.add_argument('--execute',action='store_true')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.inspect_cpu:os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ.update(ATTN_BACKEND='xformers',SPARSE_ATTN_BACKEND='xformers',
                      SPARSE_CONV_BACKEND='spconv',SPCONV_ALGO='native',
                      OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                      HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import numpy as np
    import torch
    import xformers.ops as xops
    torch.set_num_threads(2)
    if not hasattr(xops.fmha,'BlockDiagonalMask'):
        from xformers.ops.fmha.attn_bias import BlockDiagonalMask
        xops.fmha.BlockDiagonalMask=BlockDiagonalMask
    sys.path.insert(0,str(ROOT/'repos/TRELLIS'))
    from trellis import models
    from trellis.modules.sparse import SparseTensor
    config=json.loads(CHECKPOINT.with_suffix('.json').read_text())
    latent=torch.load(SLAT,map_location='cpu',weights_only=True)
    init=json.loads((REGISTERED/'uniform/semantic_uniform_transform.json').read_text())
    reg=json.loads((REGISTERED/'registered/bounded_registration.json').read_text())
    raw_to_actor=np.asarray(reg['relative_initial_to_registered_4x4'])@np.asarray(init['raw_to_actor_uniform_4x4'])
    column_norms=np.linalg.norm(raw_to_actor[:3,:3],axis=0)
    uniform=float(column_norms.mean());rotation=raw_to_actor[:3,:3]/uniform
    if not np.allclose(column_norms,uniform,atol=1e-7) or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-7) or np.linalg.det(rotation)<.999:
        raise RuntimeError('Only the approved proper rotation plus uniform scale may register Gaussians.')
    started=time.time()
    decoder=models.from_pretrained(str(CHECKPOINT)).eval()
    load_seconds=time.time()-started
    count=int(latent['feats'].shape[0]*config['args']['representation_config']['num_gaussians'])
    plan={'schema':'a20.same_slat_gaussian_decode.v1','state':'cpu_interface_ready' if args.inspect_cpu else 'running',
          'execution_source_sha256':sha(__file__),
          'source_slat':str(SLAT),'source_slat_sha256':sha(SLAT),
          'source_slat_shapes':{k:list(v.shape) for k,v in latent.items()},
          'decoder_config':config,'decoder_checkpoint':str(CHECKPOINT.with_suffix('.safetensors')),
          'decoder_bytes':CHECKPOINT.with_suffix('.safetensors').stat().st_size,
          'decoder_sha256':sha(CHECKPOINT.with_suffix('.safetensors')),
          'decoder_parameter_bytes':sum(x.numel()*x.element_size() for x in decoder.parameters()),
          'decoder_buffer_bytes':sum(x.numel()*x.element_size() for x in decoder.buffers()),
          'strict_official_model_load':True,'cpu_load_seconds':load_seconds,
          'expected_gaussian_count':count,'activated_attribute_bytes_float32':count*14*4,
          'standard_ply_payload_bytes':count*17*4,
          'raw_to_registered_actor_4x4':raw_to_actor.tolist(),'uniform_scale_raw_to_actor':uniform,
          'no_diffusion_resampling':True,'no_DINO_required':True,'no_new_image_condition':True,
          'sampler_steps':0,'additional_candidate_count':0,
          'schema_fields':{
              'xyz':'float32[N,3], activated canonical raw position, not normalized _xyz',
              'features_dc':'float32[N,1,3], degree0 SH coefficients, not direct RGB',
              'scales':'float32[N,3], activated stddev including minimum 3D kernel',
              'rotation_wxyz':'float32[N,4], normalized quaternion after identity bias',
              'opacity':'float32[N,1], activated [0,1] after bias',
          },
          'standard_ply_fields':{'f_dc_0..2':'SH DC','opacity':'logit(activated opacity)',
                                 'scale_0..2':'log(activated scales)','rot_0..3':'wxyz quaternion after bias',
                                 'position_frame':'raw canonical; save_ply(transform=None), no default Y-up export'},
          'gpu_budget_estimate':{'reserved_GiB':4,'measured':False,
                                'reason':'single 171MB head, 12 sparse transformer blocks; no DINO/SS/SLAT/mesh grid residency'},
          'decode_time_estimate_seconds':[1,5],'estimate_measured':False,
          'gpu_executed':False,'torch_cuda_initialized':torch.cuda.is_initialized()}
    if args.inspect_cpu:
        (ROOT/'gaussian_decode_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        print(json.dumps(plan,indent=2));return
    if not args.out or args.out.exists():raise RuntimeError('Use a new explicit output directory.')
    if not torch.cuda.is_available():raise RuntimeError('No CUDA device scheduled.')
    args.out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,args.out/'decode_same_slat_gaussian.executed.py')
    (args.out/'frozen_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    decoder=decoder.cuda()
    sparse=SparseTensor(feats=latent['feats'].cuda(),coords=latent['coords'].cuda())
    torch.cuda.reset_peak_memory_stats();started=time.time()
    with torch.no_grad():
        gaussian=decoder(sparse)[0]
    torch.cuda.synchronize();seconds=time.time()-started
    activated={
        'xyz':gaussian.get_xyz.detach().float().cpu(),
        'features_dc':gaussian.get_features.detach().float().cpu(),
        'scales':gaussian.get_scaling.detach().float().cpu(),
        'rotation_wxyz':gaussian.get_rotation.detach().float().cpu(),
        'opacity':gaussian.get_opacity.detach().float().cpu(),
    }
    if not all(torch.isfinite(x).all() for x in activated.values()):raise RuntimeError('Nonfinite Gaussian decoder output.')
    if len(activated['xyz'])!=count:raise RuntimeError('Unexpected Gaussian count.')
    torch.save(activated,args.out/'gaussians_activated_raw.pt')
    internal={k:getattr(gaussian,k).detach().cpu() for k in ['_xyz','_features_dc','_scaling','_rotation','_opacity']}
    torch.save({'internal':internal,'init_params':gaussian.init_params},args.out/'gaussians_internal_raw.pt')
    gaussian.save_ply(str(args.out/'gaussians_raw.ply'),transform=None)
    # SH degree zero is rotation-independent. Transform physical ellipsoids once.
    from scipy.spatial.transform import Rotation
    positions=activated['xyz'].numpy()@raw_to_actor[:3,:3].T+raw_to_actor[:3,3]
    quaternions=activated['rotation_wxyz'].numpy()
    raw_rotation=Rotation.from_quat(quaternions[:,[1,2,3,0]])
    transformed_quaternion=(Rotation.from_matrix(rotation)*raw_rotation).as_quat()[:,[3,0,1,2]]
    actor=dict(activated,xyz=torch.from_numpy(positions.astype(np.float32)),
               scales=activated['scales']*uniform,
               rotation_wxyz=torch.from_numpy(transformed_quaternion.astype(np.float32)))
    torch.save(actor,args.out/'gaussians_activated_registered_actor.pt')
    plan.update(state='complete',gpu_executed=True,decode_seconds=seconds,
                gaussian_count=count,finite=True,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                files={q.name:{'bytes':q.stat().st_size,'sha256':sha(q)} for q in args.out.iterdir() if q.is_file()})
    (args.out/'decode_receipt.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(plan,indent=2))


if __name__=='__main__':
    main()
