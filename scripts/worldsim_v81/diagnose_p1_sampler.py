"""同一公开条件/初始噪声，仅改变反演与 CFG 配对；不训练、不改正式模式。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from motion_proj.worldsim_v81.infer_p1 import (FRAMES, SIZE, _official_pipeline_class,
                                              generate, load_sequence, save_outputs)
from motion_proj.worldsim_v81.model_bridge import (DEFAULT_EXTERNAL, load_components,
                                                  load_trainable_state)
from motion_proj.worldsim_v81.train_p1 import FORMAT


def stats(tensor: torch.Tensor) -> dict:
    value = tensor.detach().float()
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
            'finite': bool(value.isfinite().all()), 'rms': float(value.square().mean().sqrt()),
            'min': float(value.min()), 'max': float(value.max())}


@torch.no_grad()
def matched_cfg(components, pipeline, cache: dict, initial: torch.Tensor) -> tuple[torch.Tensor, list]:
    from diffusers import EulerDiscreteScheduler
    scheduler = EulerDiscreteScheduler.from_config(components.scheduler.config)
    scheduler.set_timesteps(25, device='cuda')
    latents = initial.clone().cuda()
    condition, embedding, time_ids = [cache[k].cuda() for k in ('condition', 'embedding', 'time_ids')]
    guidance = torch.linspace(1., 3., FRAMES, device='cuda', dtype=latents.dtype).view(1, FRAMES, 1, 1, 1)
    history = []
    for index, timestep in enumerate(scheduler.timesteps):
        pair = torch.cat([latents, latents])
        model_input = scheduler.scale_model_input(pair, timestep)
        model_input = torch.cat([model_input, condition], dim=2)
        with torch.autocast('cuda', dtype=torch.float16, cache_enabled=False):
            prediction = components.unet(model_input, timestep, embedding,
                                         added_time_ids=time_ids).sample
        unconditional, conditional = prediction.chunk(2)
        prediction = unconditional + guidance*(conditional-unconditional)
        latents = scheduler.step(prediction, timestep, latents).prev_sample
        if index in (0, 12, 24): history.append({'index':index,'timestep':float(timestep),**stats(latents)})
        if not torch.isfinite(latents).all(): raise FloatingPointError('controlled sampler 非有限值')
    with torch.autocast('cuda', dtype=torch.float16, cache_enabled=False):
        decoded = pipeline.decode_latents(latents, num_frames=FRAMES, decode_chunk_size=8)[0]
    return ((decoded.permute(1,0,2,3)+1)/2).clamp(0,1).cpu(), history


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--sequence-id', default='00f88c4f0a')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(REPO): raise ValueError('诊断输出必须在仓库外')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if state.get('format') != FORMAT: raise ValueError('仅接受 P1 checkpoint')
    _, target, visible_images, edges, refs, pairs = load_sequence(args.data_root,args.sequence_id,.33)
    components = load_components(); load_trainable_state(components,state['models'])
    cls = _official_pipeline_class(DEFAULT_EXTERNAL)
    pipeline = cls(vae=components.vae,image_encoder=components.image_encoder,
                   unet=components.unet,scheduler=components.scheduler,
                   feature_extractor=components.feature_extractor,fix_raft=components.raft,
                   vo_flow_complete=components.fcnet,lat_bi_propagator=components.propagator)
    rgb = np.stack([np.asarray(x) for x in visible_images])
    visible = torch.from_numpy(rgb.copy()).permute(0,3,1,2).unsqueeze(0).float().cuda()/127.5-1
    hole = torch.zeros(1,FRAMES,1,SIZE,SIZE,device='cuda');hole[...,:edges[0]]=1;hole[...,edges[1]:]=1
    visible *= 1-hole
    cache = {}; metrics = {}; original = pipeline.ddim_inversion_global
    def capture(latents, condition, masks, flows, embedding, time_ids):
        for name, value in [('initial',latents),('condition',condition),('embedding',embedding),('time_ids',time_ids)]:
            cache[name] = value.detach().cpu().clone();metrics[name]=stats(value)
        result = original(latents,condition,masks,flows,embedding,time_ids)
        cache['inverse'] = result.detach().cpu().clone();metrics['inverse']=stats(result)
        metrics['inverse_pair_mae'] = float((result[0]-result[1]).abs().mean())
        return result
    pipeline.ddim_inversion_global = capture
    literal = generate(components,pipeline,visible_images,visible,hole,pairs,
                       seed=2026,steps=25,mode='literal-public',amp='fp16')
    outputs = {'literal-public': literal}; histories = {}
    for mode, initial in [('no-inverse-matched-cfg',cache['initial']),
                          ('inverse-first-matched-cfg',cache['inverse'][0:1])]:
        outputs[mode], histories[mode] = matched_cfg(components,pipeline,cache,initial)
    args.output.mkdir(parents=True,exist_ok=True)
    for mode,prediction in outputs.items():
        saved = save_outputs(args.output/mode,target,visible_images,prediction,edges)
        obj={'status':'complete','mode':mode,'checkpoint_step':state['step'],'checkpoint':str(args.checkpoint),
             'sequence_id':args.sequence_id,'side_ratio_each':.33,'reference_indices':refs,'seed':2026,
             'shared_condition':'exact cached public CLIP/RAFT/FCNet/propagation/IDs, reused without re-encoding',
             'amp':'fp16','steps':25,'is_formal_metric':False,'output_paths':saved}
        (args.output/mode/'run.json').write_text(json.dumps(obj,indent=2))
    delta = {mode:float((pred-outputs['literal-public']).abs().mean()) for mode,pred in outputs.items()}
    report={'checkpoint_step':state['step'],'sequence_id':args.sequence_id,'stage_stats':metrics,
            'denoise_stats':histories,'pixel_mae_vs_literal':delta,
            'limits':'no-inverse changes reverse initialization and CFG pairing jointly; inverse-first fixes CFG but retains public inverse parameterization; one fixed validation clip, no quality verdict from MAE',
            'is_formal_metric':False}
    (args.output/'sampler_diagnostic.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__ == '__main__': main()
