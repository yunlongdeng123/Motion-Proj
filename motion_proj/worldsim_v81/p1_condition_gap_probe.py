"""固定权重的条件来源×噪声诊断；GT仅用于BUILD oracle，不是QUERY或新训练。"""

from __future__ import annotations

import argparse
from itertools import product
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import default_collate

from .model_bridge import (DEFAULT_EXTERNAL, DEFAULT_FCNET, DEFAULT_RAFT, DEFAULT_SVD,
                           encode_video, load_components, load_trainable_state)
from .p1_capacity_probe import CLIP_START, VIDEO_ID, save_teacher_frames
from .p1_data import YouTubeVOSP1Dataset
from .reference_propagation import build_reference_plan
from .train_p1 import (DEFAULT_DATA, FORMAT, _official_image_embedding, _official_util,
                       raft_flows_for_pairs, reference_indices_from_visible,
                       validate_resume_protocol)


def schedule_levels(scheduler, steps: int = 25) -> list[dict]:
    """从实际Euler schedule选三档，不把默认sigma=2当作高噪声能力。"""
    if steps < 3:
        raise ValueError('至少3个采样步才能选高/中/低噪声')
    config = scheduler.config
    if config.prediction_type != 'v_prediction' or config.timestep_type != 'continuous':
        raise ValueError('本诊断EDM公式只接受continuous/v_prediction，不能静默适配其他scheduler')
    scheduler.set_timesteps(steps, device='cpu')
    rows = []
    for label, index in [('high', 0), ('middle', steps // 2), ('low', steps - 1)]:
        sigma = float(scheduler.sigmas[index])
        timestep = float(scheduler.timesteps[index])
        if not math.isfinite(sigma) or sigma <= 0 or not math.isclose(
                timestep, 0.25 * math.log(sigma), abs_tol=1e-5):
            raise ValueError('实际sigma/timestep不满足已审计的EDM条件')
        rows.append({'level': label, 'schedule_index': index,
                     'sigma': sigma, 'timestep': timestep})
    return rows


def clip_pixels_from_visible(visible: torch.Tensor, hole: torch.Tensor) -> torch.Tensor:
    """两组共用CLIP预处理；QUERY的PIL黑洞在[-1,1]空间对应-1。"""
    return visible.masked_fill(hole.expand_as(visible).bool(), -1)


def latent_errors(denoised: torch.Tensor, target: torch.Tensor, hole: torch.Tensor,
                  sigma: float) -> dict:
    error = (denoised.float() - target.float()).square()
    mask = F.interpolate(hole.flatten(0, 1).float(), error.shape[-2:], mode='nearest')
    mask = mask.reshape(hole.shape[0], hole.shape[1], 1, *error.shape[-2:])
    count = error.shape[2]
    if mask.sum() == 0 or (1 - mask).sum() == 0:
        raise ValueError('洞区和可见区都必须非空')
    values = {'weighted_mse': float(error.mean() * ((1 + sigma**2) / sigma**2)),
              'hole_mse': float((error * mask).sum() / (mask.sum() * count)),
              'known_mse': float((error * (1 - mask)).sum() / ((1 - mask).sum() * count))}
    if not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError('诊断出现非有限误差')
    return values


@torch.no_grad()
def measure_fixed_input(unet, target, noise, condition, clip, ids, hole,
                        *, sigma: float, amp: str) -> tuple[dict, torch.Tensor]:
    """同一x0、epsilon和sigma，只替换condition/CLIP；无CFG及多步更新。"""
    device = next(unet.parameters()).device
    target, noise = target.to(device), noise.to(device)
    noisy = target + sigma * noise
    model_input = torch.cat((noisy / math.sqrt(1 + sigma**2), condition.to(device)), dim=2)
    enabled = device.type == 'cuda'
    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16}[amp]
    with torch.autocast(device.type, dtype=dtype, enabled=enabled):
        prediction = unet(model_input, torch.tensor([0.25 * math.log(sigma)], device=device),
                          clip.to(device), added_time_ids=ids.to(device)).sample
    denoised = (noisy / (1 + sigma**2)
                - sigma / math.sqrt(1 + sigma**2) * prediction.float())
    return latent_errors(denoised, target, hole.to(device), sigma), denoised.cpu()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--svd', type=Path, default=DEFAULT_SVD)
    parser.add_argument('--external', type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument('--raft-weight', type=Path, default=DEFAULT_RAFT)
    parser.add_argument('--fcnet-weight', type=Path, default=DEFAULT_FCNET)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--prepare-only', action='store_true',
                        help='仅读取指定scheduler配置并准备CPU计划；不声明远端/权重/数据已核实')
    return parser.parse_args(argv)


def main():
    args = parse_args()
    from diffusers import EulerDiscreteScheduler
    scheduler = EulerDiscreteScheduler.from_pretrained(args.svd, subfolder='scheduler')
    levels = schedule_levels(scheduler)
    if args.output_dir.exists():
        raise FileExistsError('诊断产物必须用新的仓库外目录')
    repo = Path(__file__).resolve().parents[2]
    if args.output_dir.resolve().is_relative_to(repo):
        raise ValueError('模型/图像/诊断完整产物不进入Git仓库')
    args.output_dir.mkdir(parents=True)
    report = {'status': 'cpu_prepared_gpu_not_run', 'kind': 'fixed_weight_condition_gap_teacher_only',
              'source_checkpoint': args.checkpoint.as_posix(), 'checkpoint_verified': False,
              'remote_actual_scheduler_verified': False, 'scheduler_config': dict(scheduler.config),
              'scheduler_source': str(args.svd / 'scheduler' / 'scheduler_config.json'),
              'levels': levels, 'clip_sources': ['full_gt_oracle', 'visible_black_hole'],
              'flow_sources': ['full_gt_oracle', 'visible_black_hole_official', 'visible_gray_hole_owned'],
              'video_id': VIDEO_ID, 'clip_start': CLIP_START, 'seed': args.seed,
              'shared_condition_vae': 'visible RGB, fixed .02 noise, mode, unscaled',
              'shared_flow_mask': '3x3 dilation, QUERY protocol; same for all three sources',
              'shared_time_ids': [6, 127, .02], 'optimizer_updates': 0, 'formal_eval': False,
              'human_verdict': None,
              'roles': 'All outputs are single-step noisy-GT BUILD probes; GT CLIP/flow cells are oracle diagnostic only, never deployable QUERY',
              'limitations': 'No CFG, no free Gaussian generation, no causal proof of multi-step sampling quality. FPS/augmentation/VAE are fixed to QUERY. CLIP uses the same training helper for both pixel sources; this is not an exact public test.py CLIP path. Gray-hole flow is an extra control for the owned/public source difference. Compare metrics within the same sigma; low-noise GT input is not independent reconstruction.'}
    destination = args.output_dir/'diagnostic.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    if args.prepare_only:
        print(json.dumps({'report': str(destination), 'levels': levels})); return
    if not torch.cuda.is_available():
        raise RuntimeError('GPU未开启；CPU计划保留，不运行模型或替换硬件')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if state.get('format') != FORMAT or state.get('step') != 100:
        raise ValueError('本轮只诊断正式双向step100，不能用旧协议或容量权重混报')
    validate_resume_protocol(state, 'paper-bidirectional-m4')
    amp = state['args']['amp']
    if amp not in {'bf16', 'fp16'}:
        raise ValueError('断点精度必须是bf16或fp16')
    if Path(state['args']['data_root']).resolve() != args.data_root.resolve():
        raise ValueError('必须使用step100的原始数据根，不接受同数量的替代数据池')
    dataset = YouTubeVOSP1Dataset(args.data_root)
    batch = default_collate([dataset[0]])
    if batch['video_id'] != [VIDEO_ID] or int(batch['clip_start'][0]) != CLIP_START:
        raise ValueError('固定训练片段来源发生变化')
    if any(state['data_profile'][key] != dataset.profile()[key]
           for key in ['rgb_root', 'videos', 'available_windows', 'seed', 'frames', 'size',
                       'mask_ratio_each_side']):
        raise ValueError('数据池与step100断点不一致')
    video_index, start = dataset.choice(0)
    input_frames = [str(path) for path in dataset.videos[video_index][1][start:start + 25]]
    report.update(status='gpu_running', checkpoint_verified=True,
                  input_frames=input_frames)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    components = load_components(args.svd, args.raft_weight, args.fcnet_weight, args.external)
    if schedule_levels(components.scheduler) != levels:
        raise ValueError('装载组件的实际scheduler与计划不同，不执行可疑对照')
    load_trainable_state(components, state['models']); del state
    for module in [components.vae, components.image_encoder, components.raft,
                   components.fcnet, components.propagator, components.unet]:
        module.eval()
    device = torch.device('cuda')
    generator = torch.Generator(device=device).manual_seed(args.seed)
    target, visible, hole = (batch[key].to(device) for key in ['target_rgb','visible_rgb','hole_mask'])
    refs = reference_indices_from_visible(visible, hole)
    plan = build_reference_plan(25, refs)
    torch.manual_seed(args.seed + 1)
    target_latent = encode_video(components.vae, target, scaled=True, sample=True).cpu()
    observation_noise = torch.randn(visible.shape, generator=generator, device=device)
    cond_latent = encode_video(components.vae, visible + .02 * observation_noise,
                               scaled=False, sample=False).cpu()
    noise = torch.randn(target_latent.shape, generator=generator, device=device).cpu()
    util = _official_util(args.external)
    clips = {'full_gt_oracle': _official_image_embedding(components, target, util).cpu(),
             'visible_black_hole': _official_image_embedding(components, clip_pixels_from_visible(visible, hole), util).cpu()}
    flow_inputs = {'full_gt_oracle': target,
                   'visible_black_hole_official': clip_pixels_from_visible(visible, hole),
                   'visible_gray_hole_owned': visible}
    conditions = {}
    for name, frames in flow_inputs.items():
        flow = raft_flows_for_pairs(components.raft, frames, list(plan.pairs), iters=20, pair_chunk=2)
        dilated = F.max_pool2d(hole.flatten(0, 1), 3, 1, 1).reshape_as(hole)
        # FCNet使用扩边mask，传播使用原mask，与现有QUERY相同。
        from .reference_propagation import static_fcnet_pair_masks, propagate_reference_latents
        pair_masks = static_fcnet_pair_masks(dilated, plan)
        predicted, _ = components.fcnet.forward_bidirect_flow(flow, pair_masks)
        completed = components.fcnet.combine_flow(flow, predicted, pair_masks)
        _, _, condition = propagate_reference_latents(components.propagator, cond_latent.to(device),
                                                       completed[0], completed[1], hole, plan)
        conditions[name] = condition.cpu()
        del flow, predicted, completed, condition
    ids = torch.tensor([[6,127,.02]], dtype=clips['full_gt_oracle'].dtype)
    shared = {'target_latent': target_latent, 'noise': noise, 'condition_latent': cond_latent,
              'observation_noise': observation_noise.cpu(), 'clips': clips,
              'conditions': conditions, 'time_ids': ids, 'mask': hole.cpu()}
    torch.save(shared, args.output_dir/'shared_inputs.pt')
    for module in [components.vae, components.image_encoder, components.raft,
                   components.fcnet, components.propagator]: module.to('cpu')
    del target, visible, hole, flow_inputs, observation_noise
    torch.cuda.empty_cache()
    from .infer_p1 import _official_pipeline_class
    pipeline = _official_pipeline_class(args.external)(
        vae=components.vae, image_encoder=components.image_encoder, unet=components.unet,
        scheduler=components.scheduler, feature_extractor=components.feature_extractor,
        fix_raft=components.raft, vo_flow_complete=components.fcnet, lat_bi_propagator=components.propagator)
    rows = []
    for level, clip_name, flow_name in product(levels, clips, conditions):
        score, denoised = measure_fixed_input(components.unet, target_latent, noise,
            conditions[flow_name], clips[clip_name], ids, shared['mask'], sigma=level['sigma'], amp=amp)
        folder = args.output_dir/f"{level['level']}__clip-{clip_name}__flow-{flow_name}"
        save_teacher_frames(pipeline, components, denoised, batch, folder)
        rows.append({**level, 'clip_source': clip_name, 'flow_source': flow_name,
                     'metrics': score, 'images': str(folder)})
        report.update(rows=rows)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(rows[-1]), flush=True)
    report.update(status='complete', checkpoint_verified=True,
                  remote_actual_scheduler_verified=True, rows=rows,
                  reference_indices=list(plan.refs), input_frames=input_frames,
                  shared_noise_exact=True, shared_x0_exact=True,
                  peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    with torch.no_grad(): main()
