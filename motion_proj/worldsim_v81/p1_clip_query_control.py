"""同一可见条件与初始高斯，只切换完整/可见首帧CLIP；完整分支是oracle。"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import default_collate

from .infer_p1 import _official_pipeline_class, _unet_noise, save_outputs
from .long_video_sampling import sample_global_latents
from .model_bridge import DEFAULT_EXTERNAL, DEFAULT_FCNET, DEFAULT_RAFT, DEFAULT_SVD, load_components, load_trainable_state
from .p1_capacity_probe import CLIP_START, VIDEO_ID, query_images_from_visible
from .p1_data import YouTubeVOSP1Dataset
from .train_p1 import DEFAULT_DATA, FORMAT, validate_resume_protocol


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2036)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('需要GPU，只由当前主任务启动')
    source = json.loads((args.source_dir/'diagnostic.json').read_text())
    if source['status'] != 'complete' or not source['remote_actual_scheduler_verified']:
        raise ValueError('必须使用已完成的18组条件来源诊断')
    if source['video_id'] != VIDEO_ID or source['clip_start'] != CLIP_START:
        raise ValueError('固定片段不得变化')
    if args.output_dir.exists() or args.output_dir.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('输出须为新的仓库外目录')
    state = torch.load(source['source_checkpoint'], map_location='cpu', weights_only=False)
    if state.get('format') != FORMAT or state.get('step') != 100:
        raise ValueError('只接受同一正式step100断点')
    validate_resume_protocol(state, 'paper-bidirectional-m4')
    dataset = YouTubeVOSP1Dataset(DEFAULT_DATA)
    batch = default_collate([dataset[0]])
    index, start = dataset.choice(0)
    files = [str(path) for path in dataset.videos[index][1][start:start+25]]
    if files != source['input_frames']:
        raise ValueError('原片段25张文件名单变化')
    cache = torch.load(args.source_dir/'shared_inputs.pt', map_location='cpu', weights_only=False)
    condition = cache['conditions']['visible_black_hole_official'].cuda()
    ids = cache['time_ids'].cuda()
    components = load_components(DEFAULT_SVD, DEFAULT_RAFT, DEFAULT_FCNET, DEFAULT_EXTERNAL)
    load_trainable_state(components, state['models'])
    amp = {'bf16': torch.bfloat16, 'fp16': torch.float16}[state['args']['amp']]
    del state
    for module in [components.vae, components.image_encoder, components.raft,
                   components.fcnet, components.propagator, components.unet]:
        module.eval()
    pipeline = _official_pipeline_class(DEFAULT_EXTERNAL)(
        vae=components.vae, image_encoder=components.image_encoder, unet=components.unet,
        scheduler=components.scheduler, feature_extractor=components.feature_extractor,
        fix_raft=components.raft, vo_flow_complete=components.fcnet, lat_bi_propagator=components.propagator)
    for module in [components.vae, components.image_encoder, components.raft,
                   components.fcnet, components.propagator]: module.to('cpu')
    captures = []
    original_prepare = pipeline.prepare_latents
    def capture_initial(*positional, **keywords):
        latent = original_prepare(*positional, **keywords)
        captures.append(latent.cpu().clone())
        return latent
    pipeline.prepare_latents = capture_initial
    target_images, visible_images, edges = query_images_from_visible(batch)
    args.output_dir.mkdir(parents=True)
    report = {'status': 'running', 'source_diagnostic': str(args.source_dir),
              'checkpoint': source['source_checkpoint'], 'checkpoint_step': 100,
              'video_id': VIDEO_ID, 'clip_start': CLIP_START, 'seed': args.seed,
              'flow_source': 'visible_black_hole_official', 'steps': 25,
              'cfg': [1, 3], 'input_frames': files, 'optimizer_updates': 0,
              'formal_eval': False, 'human_verdict': None,
              'role': 'full_gt_oracle is illegal for deployed QUERY; visible_black_hole is legal visible-only condition',
              'preprocessing': 'shared training CLIP helper, not exact public PIL replay',
              'shared': ['propagated visible condition', 'time IDs', 'scheduler', 'initial Gaussian latent']}
    destination = args.output_dir/'diagnostic.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    rows = []
    for name in ['full_gt_oracle', 'visible_black_hole']:
        components.unet.cuda()
        components.vae.cpu()
        generator = torch.Generator(device='cuda').manual_seed(args.seed)
        latent = sample_global_latents(pipeline, components.scheduler, condition,
            cache['clips'][name].cuda(), ids, num_channels_latents=components.unet.config.in_channels,
            height=256, width=256, num_inference_steps=25, generator=generator,
            predict_noise=lambda z,c,e,i,t,s: _unet_noise(components,z,c,e,i,t,s,cfg=True,amp_dtype=amp))
        components.unet.cpu(); components.vae.cuda()
        decoded = pipeline.decode_latents(latent, num_frames=25, decode_chunk_size=8)[0]
        prediction = ((decoded.permute(1,0,2,3).float().cpu()+1)/2).clamp(0,1)
        rows.append({'clip_source': name, 'outputs': save_outputs(args.output_dir/name,
                      target_images, visible_images, prediction, edges)})
        report.update(rows=rows)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    if len(captures) != 2 or not torch.equal(captures[0], captures[1]):
        raise RuntimeError('两组没有共享完全相同的初始高斯，不允许因果比较')
    torch.save(captures[0], args.output_dir/'initial_latent.pt')
    report.update(status='complete', same_initial_exact=True,
                  peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
