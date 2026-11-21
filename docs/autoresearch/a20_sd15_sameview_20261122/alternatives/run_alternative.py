"""One official mesh-only candidate from the fixed common input, seed101.

Invoked explicitly by the parent GPU scheduler. No implicit model downloads.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import random
import time

ROOT = Path(__file__).resolve().parent
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  ATTN_BACKEND='xformers', SPARSE_ATTN_BACKEND='xformers',
                  SPARSE_CONV_BACKEND='spconv', SPCONV_ALGO='native',
                  OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(16<<20),b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['trellis','amodal3r'],required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=101)
    parser.add_argument('--execute',action='store_true',required=True)
    parser.add_argument('--replay-of',type=Path)
    args=parser.parse_args()
    if args.seed!=101:
        raise ValueError('This ablation authorizes one seed101 candidate per teacher.')
    if args.out.exists():
        raise RuntimeError('Preserve existing candidate; choose a new replay directory if needed.')
    manifest=json.loads((args.input/'input_manifest.json').read_text())
    preflight=json.loads((ROOT/'preflight_cpu.json').read_text())
    if not preflight['passed']:
        raise RuntimeError('CPU preflight must pass before GPU execution.')
    shared=args.input/'shared_rgba.png'
    if args.model=='trellis':
        if sha256(shared)!=manifest['source_sha256']:
            raise RuntimeError('Fixed common RGBA has changed.')
    else:
        if sha256(args.input/'amodal_rgb.png')!=manifest['files']['amodal_rgb.png']['sha256']:
            raise RuntimeError('Fixed real Amodal RGB has changed.')
        if sha256(args.input/'amodal_mask.png')!=manifest['files']['amodal_mask.png']['sha256']:
            raise RuntimeError('Fixed real Amodal mask has changed.')
    dino=ROOT/'dependencies/dinov2_vitl14_reg4_pretrain.pth'
    dino_record=json.loads((ROOT/'dino_download.json').read_text())
    if not dino.is_file() or dino.stat().st_size!=dino_record['bytes']:
        raise RuntimeError('Exact DINOv2 reg4 checkpoint is incomplete.')
    import torch
    torch.set_num_threads(4)
    import xformers.ops as xops
    compatibility={}
    if not hasattr(xops.fmha,'BlockDiagonalMask'):
        from xformers.ops.fmha.attn_bias import BlockDiagonalMask
        xops.fmha.BlockDiagonalMask=BlockDiagonalMask
        compatibility['xformers_BlockDiagonalMask']='same class exported from fmha.attn_bias; no attention math change'
    from PIL import Image
    import trimesh
    if not torch.cuda.is_available():
        raise RuntimeError('GPU execution must be scheduled by the parent.')
    args.out.mkdir(parents=True,exist_ok=False)
    started=time.time()
    status={'schema':'a20.alternative_candidate.v1','model':args.model,'seed':101,
            'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'input_manifest':str(args.input/'input_manifest.json'),
            'input_rgba_sha256':manifest.get('source_sha256'),'formats':['mesh'],
            'amodal_mask_protocol':manifest['amodal_mask_protocol'],
            'amodal_same_rgb_as_common':manifest['amodal_same_rgb_as_common'],
            'amodal_rgb_source_sha256':manifest['amodal_rgb_source_sha256'],
            'samplers':'official pipeline defaults, unmodified',
            'dino_checkpoint_sha256':dino_record['sha256'],
            'timings':{},'state':'running'}
    status['runtime_compatibility']=compatibility
    status['engineering_replay_of']=str(args.replay_of) if args.replay_of else None
    frozen_config=json.loads((ROOT/'model_configs'/args.model/'pipeline.json').read_text())
    status['official_pipeline_config']=frozen_config
    status['code_commit']=subprocess.check_output(['git','-C',str(ROOT/'repos'/('TRELLIS' if args.model=='trellis' else 'Amodal3R')),'rev-parse','HEAD'],text=True).strip()
    status['runtime']={'torch':torch.__version__,'cuda':torch.version.cuda,
                       'gpu':torch.cuda.get_device_name(),'pid':os.getpid()}
    def save():
        (args.out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    save()
    (args.out/'frozen_plan.json').write_text(json.dumps(status,indent=2)+'\n')
    try:
        original_hub_load=torch.hub.load
        def offline_dino(repo_or_dir, name, *extra, **kwargs):
            if repo_or_dir!='facebookresearch/dinov2' or name!='dinov2_vitl14_reg':
                raise RuntimeError(f'Unexpected Torch Hub request: {repo_or_dir}/{name}')
            with torch.device('meta'):
                model=original_hub_load('/root/autodl-tmp/third_party/dinov2-v51-stage-b',name,
                                        source='local',pretrained=False)
            model.load_state_dict(torch.load(dino,map_location='cpu',weights_only=True),strict=True,assign=True)
            if any(t.is_meta for t in list(model.parameters())+list(model.buffers())):
                raise RuntimeError('Exact DINO state left unmaterialized tensors.')
            return model
        torch.hub.load=offline_dino
        if args.model=='trellis':
            sys.path.insert(0,str(ROOT/'repos/TRELLIS'))
            from trellis.pipelines import TrellisImageTo3DPipeline as Pipeline
        else:
            sys.path.insert(0,str(ROOT/'repos/Amodal3R'))
            from amodal3r.pipelines import Amodal3RImageTo3DPipeline as Pipeline
        stage=time.time()
        pipeline=Pipeline.from_pretrained(str(ROOT/'model_configs'/args.model))
        pipeline.cuda()
        original_ss=pipeline.sample_sparse_structure
        def record_ss(*positional,**keywords):
            value=original_ss(*positional,**keywords)
            status['sparse_structure']={'shape':list(value.shape),'finite':bool(torch.isfinite(value).all())}
            if not status['sparse_structure']['finite']:
                raise RuntimeError('Sparse structure is non-finite.')
            torch.save(value.detach().cpu(),args.out/'sparse_structure_coords.pt')
            import numpy as np
            torch.save({'torch_cpu':torch.get_rng_state(),
                        'torch_cuda':torch.cuda.get_rng_state_all(),
                        'numpy':np.random.get_state(),'python':random.getstate()},
                       args.out/'rng_after_sparse_structure.pt')
            save()
            return value
        pipeline.sample_sparse_structure=record_ss
        original_slat=pipeline.sample_slat
        def record_slat(*positional,**keywords):
            value=original_slat(*positional,**keywords)
            finite=bool(torch.isfinite(value.feats).all())
            status['slat']={'features':list(value.feats.shape),'coords':list(value.coords.shape),
                            'features_dtype':str(value.feats.dtype),'finite':finite}
            if not finite:
                raise RuntimeError('SLAT features are non-finite.')
            torch.save({'feats':value.feats.detach().cpu(),'coords':value.coords.detach().cpu()},args.out/'slat.pt')
            save()
            return value
        pipeline.sample_slat=record_slat
        if args.model=='amodal3r':
            original_cond=pipeline.get_cond_w_masks
            def record_cond(*positional,**keywords):
                value=original_cond(*positional,**keywords)
                status.setdefault('actual_condition_tensors',[]).append({
                    'stage':keywords.get('stage'),
                    'cond':list(value['cond'].shape),
                    'neg_cond':list(value['neg_cond'].shape),
                    'dtype':str(value['cond'].dtype),
                })
                save()
                return value
            pipeline.get_cond_w_masks=record_cond
        else:
            original_cond=pipeline.get_cond
            def record_cond(*positional,**keywords):
                value=original_cond(*positional,**keywords)
                status.setdefault('actual_condition_tensors',[]).append({
                    'stage':'image','cond':list(value['cond'].shape),
                    'neg_cond':list(value['neg_cond'].shape),'dtype':str(value['cond'].dtype)})
                save()
                return value
            pipeline.get_cond=record_cond
        status['timings']['load_seconds']=time.time()-stage
        torch.cuda.reset_peak_memory_stats()
        stage=time.time()
        if args.model=='trellis':
            output=pipeline.run(Image.open(shared),num_samples=1,seed=101,formats=['mesh'])
        else:
            output=pipeline.run_multi_image([Image.open(args.input/'amodal_rgb.png').convert('RGB')],
                [Image.open(args.input/'amodal_mask.png').convert('L')],
                num_samples=1,seed=101,formats=['mesh'],erode_kernel_size=3)
        torch.cuda.synchronize()
        status['timings']['inference_seconds']=time.time()-stage
        mesh=output['mesh'][0]
        vertices=mesh.vertices.detach().float().cpu().numpy()
        faces=mesh.faces.detach().cpu().numpy()
        if not len(vertices) or not len(faces):
            raise RuntimeError('Official mesh decoder returned an empty mesh.')
        import numpy as np
        if not np.isfinite(vertices).all() or faces.min()<0 or faces.max()>=len(vertices):
            raise RuntimeError('Decoded triangle mesh contains invalid geometry.')
        geometry=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
        geometry.export(args.out/'mesh.ply')
        status.update(state='complete',output='official explicit triangle mesh, canonical coordinates; no material fitting',
            mesh={'vertices':len(vertices),'faces':len(faces),'bounds':geometry.bounds.tolist(),
                  'finite':True,
                  'watertight':bool(geometry.is_watertight),'bytes':(args.out/'mesh.ply').stat().st_size,
                  'sha256':sha256(args.out/'mesh.ply')},
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:
        status.update(state='failed',error=repr(error))
        raise
    finally:
        status['timings']['total_seconds']=time.time()-started
        status['finished_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        save()


if __name__=='__main__':
    main()
