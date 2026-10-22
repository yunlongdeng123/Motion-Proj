"""冻结节点的三路输出、删车和固定区域评价。导入不执行；当前交付不运行。"""

from contextlib import contextmanager
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


@contextmanager
def preserve_transform(path):
    original = path.read_bytes() if path.exists() else None
    try:
        yield
    finally:
        if original is not None:
            path.write_bytes(original)
        elif path.exists():
            path.unlink()


def region_metrics(image, gt, mask):
    mask = mask.bool().squeeze(0)
    if not mask.any():
        return dict(pixels=0, mae=None, psnr=None)
    error = (image-gt)[:, mask]
    mse = float(error.square().mean())
    return dict(pixels=int(mask.sum()), mae=float(error.abs().mean()),
                psnr=-10*np.log10(max(mse, 1e-12)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--arm', choices=('B', 'R', 'RP'), required=True)
    parser.add_argument('--iteration', type=int, default=30000)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-eval', type=Path, help='同节点 B 的 evaluate_v4 输出')
    parser.add_argument('--dynamic-keep', type=Path, help='可选：对应该 checkpoint 动态行的 bool .npy')
    options = parser.parse_args()
    started = time.perf_counter()
    campaign = options.campaign.resolve()
    sys.path.insert(0, str(campaign/'engine'))
    from omegaconf import OmegaConf
    from PIL import Image
    from gaussian_renderer import render
    from scene import GaussianModel, Dynamic_GaussianModel, EnvLight
    from scene.waymo_loader2 import readWaymoInfo_add
    from utils.camera_utils import cameraList_from_camInfos
    from utils.general_utils import seed_everything
    from utils.paper_masks import load_dynamic_mask
    from utils.v4_road import load_bundle, classify_units, sample_map, RoadLayer, VERSION, POLICY, FRAMES
    from utils.v4_road_render import render_road, render_composite
    run = campaign/'arms'/options.arm/'run30k'
    config = json.loads((run/'effective_config.json').read_text(encoding='utf-8'))
    if config['v4_arm'] != options.arm:
        raise ValueError('错误实验组配置')
    if options.output.exists():
        raise FileExistsError('评价只写新目录')
    reference_summary = None
    if options.reference_eval:
        reference_summary = json.loads((options.reference_eval/'summary.json').read_text(encoding='utf-8'))
        if reference_summary['arm'] != 'B' or reference_summary['iteration'] != options.iteration:
            raise ValueError('原分支检查必须对照同节点 B')
    options.output.mkdir(parents=True)
    args = OmegaConf.create(dict(config, model_path=str(options.output)))
    seed_everything(config['seed'])
    with preserve_transform(Path(config['source_path'])/'transform_matrix.txt'):
        info = readWaymoInfo_add(args)
    cameras = sorted(cameraList_from_camInfos(info.train_cameras, 4, args), key=lambda c: c.timestamp)
    if [camera.image_name for camera in cameras] != FRAMES:
        raise ValueError('评价必须使用原49帧')
    static, dynamic = GaussianModel(args), Dynamic_GaussianModel(args)
    environment = EnvLight(resolution=args.env_map_res).cuda()
    for prefix, model in (('chkpnt', static), ('dynamic_chkpnt', dynamic), ('env_light_chkpnt', environment)):
        payload, iteration = torch.load(run/f'{prefix}{options.iteration}.pth', map_location='cpu')
        if iteration != options.iteration:
            raise ValueError('V1 checkpoint 步数不一致')
        payload = list(payload)
        if prefix == 'env_light_chkpnt':
            payload[0] = torch.nn.Parameter(payload[0].detach().cuda(), requires_grad=False)
        else:
            for index in range(1, 11):
                payload[index] = payload[index].detach().cuda()
            model.time_duration = info.time_duration
        model.restore(tuple(payload), training_args=None)
    road = None
    meta, arrays = load_bundle(config['v4_bundle'])
    observed, _ = classify_units(arrays)
    observed_ids = np.flatnonzero(observed)
    source_xyz = torch.as_tensor(arrays['model_xyz'][observed_ids], device='cuda', dtype=torch.float32)
    source_xyz = torch.cat((source_xyz, torch.ones_like(source_xyz[:, :1])), 1)
    if options.arm != 'B':
        road = RoadLayer(arrays, float(meta['meter_to_model']), options.arm, 'cuda')
        state = torch.load(run/f'road_chkpnt{options.iteration}.pth', map_location='cuda')
        if (state['version'] != VERSION or state['policy'] != POLICY or state['arm'] != options.arm
                or state['iteration'] != options.iteration or state['bundle'] != str(Path(config['v4_bundle']).resolve())):
            raise ValueError('道路 sidecar 与 V1 节点不一致')
        road.load_state_dict(state['layer'], strict=True)
        np.savez_compressed(options.output/'road_units.npz',
            unit_ids=road.unit_ids.cpu().numpy(), xyz=road.xyz.cpu().numpy(),
            scaling=road.scaling.cpu().numpy(), rotation=road.rotation.cpu().numpy(),
            color=road.colors().detach().cpu().numpy(), opacity=road.opacity.sigmoid().detach().cpu().numpy(),
            observed=road.observed.cpu().numpy(), connected_prior=road.prior.cpu().numpy(),
            source_seen=road.source_seen.cpu().numpy(), output_mask=road.output_mask().cpu().numpy(),
            unsupported_unit_ids=np.flatnonzero(~arrays['surface_supported']))
    keep = None
    if options.dynamic_keep:
        keep = torch.as_tensor(np.load(options.dynamic_keep, allow_pickle=False), device='cuda')
        if keep.dtype != torch.bool or keep.shape != (len(dynamic.get_xyz),):
            raise ValueError('动态 keep 文件须精确对应当前 checkpoint 行')
    background = torch.tensor([1., 1., 1.] if args.white_background else [0., 0., 0.], device='cuda')
    records = []
    max_native_difference = 0.
    max_reference_difference = 0. if options.reference_eval else None
    with torch.no_grad():
        for source_index, camera in enumerate(cameras):
            native = render(camera, static, dynamic, args, background, env_map=environment)
            replay = render_composite(camera, static, dynamic, None, args, background, environment)
            maximum = float((native['render']-replay['rgb']).abs().max())
            max_native_difference = max(max_native_difference, maximum)
            if maximum > 2e-5:
                raise ValueError('输出合成器未能重放原 V1 前向')
            combined = render_composite(camera, static, dynamic, road, args, background, environment)
            removed = render_composite(camera, static, dynamic if keep is not None else None,
                                       road, args, background, environment, keep)
            native_removed = render_composite(camera, static, dynamic if keep is not None else None,
                                              None, args, background, environment, keep)
            independent = render_road(camera, road, args) if road is not None else None
            gt = camera.original_image.cuda()
            mask = load_dynamic_mask(args.source_path, camera).float()
            if mask.ndim == 2:
                mask = mask[None]
            dilated = F.max_pool2d(mask[None], 7, 1, 3)[0] > .5
            interior = F.max_pool2d((1-mask)[None], 7, 1, 3)[0] < .5
            sky = camera.sky_mask.cuda().bool()[:1]
            domains = dict(full=torch.ones_like(mask, dtype=torch.bool), body=interior,
                           edge=dilated & ~interior, background=~dilated & ~sky)
            directory = options.output/camera.image_name
            directory.mkdir()
            native_cpu = native['render'].cpu().numpy()
            reference = None
            if options.reference_eval:
                with np.load(options.reference_eval/camera.image_name/'raw.npz', allow_pickle=False) as z:
                    reference = torch.as_tensor(z['v1'], device='cuda')
                max_reference_difference = max(max_reference_difference, float((native['render']-reference).abs().max()))
                domains['normal_background'] = domains['background'] & ((reference-gt).abs().mean(0, keepdim=True) <= .05)
            outputs = dict(v1=native['render'], road=None if independent is None else independent['rgb'],
                composite=combined['rgb'], delete=native_removed['rgb'], delete_with_road=removed['rgb'])
            metrics = {name: {region: region_metrics(value, gt, domain)
                       for region, domain in domains.items()} for name, value in outputs.items()
                       if value is not None and name in ('v1', 'composite')}
            # 隐藏路面没有整图 GT，不拿原车身 RGB 评价删车画面。
            record = dict(frame=camera.image_name, regions=metrics,
                road_visible_alpha_mean=float(combined['road_visible_alpha'].mean()),
                road_independent_alpha_mean=None if independent is None else float(independent['alpha'].mean()))
            # 固定同世界可靠来源分母；alpha 小或结果变差不剔除样本。
            clip = source_xyz @ camera.full_proj_transform
            source_uv = ((clip[:, :2]/clip[:, 3:].clamp_min(1e-8)+1)*source_xyz.new_tensor(
                [camera.image_width, camera.image_height])-1)/2
            source_weight = torch.as_tensor(arrays['base_weight'][source_index, observed_ids], device='cuda')
            target_rgb = torch.as_tensor(arrays['source_rgb'][source_index, observed_ids], device='cuda')
            weight_sum = float(source_weight.sum())
            road_metrics = dict(eligible_observations=int((source_weight > 0).sum()), weight_sum=weight_sum)
            if weight_sum:
                for name, image in (('v1', native['render']), ('composite', combined['rgb'])):
                    error = (sample_map(image, source_uv)-target_rgb).abs().mean(1)
                    road_metrics[name+'_source_mae'] = float((error*source_weight).sum())/weight_sum
                if independent is not None:
                    source_alpha = sample_map(independent['alpha'], source_uv)[:, 0]
                    source_color = sample_map(independent['rgb'], source_uv)/source_alpha[:, None].clamp_min(1e-5)
                    error = (source_color-target_rgb).abs().mean(1)
                    road_metrics['road_source_mae'] = float((error*source_weight).sum())/weight_sum
                    road_metrics['road_source_alpha'] = float((source_alpha*source_weight).sum())/weight_sum
                    source_depth = sample_map(independent['depth_sum'], source_uv)[:, 0]/source_alpha.clamp_min(1e-5)
                    target_depth = torch.as_tensor(arrays['source_depth'][source_index, observed_ids], device='cuda')
                    road_metrics['road_depth_consistency_m'] = float(
                        ((source_depth-target_depth).abs()*source_weight).sum())/weight_sum/float(meta['meter_to_model'])
            record['reliable_world_sources'] = road_metrics
            records.append(record)
            for name, value in outputs.items():
                if value is not None:
                    image = (value.clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8)
                    Image.fromarray(image).save(directory/f'{name}.png')
            np.savez_compressed(directory/'raw.npz', v1=native_cpu,
                composite=combined['rgb'].cpu().numpy(), v1_alpha=native['alpha'].cpu().numpy(),
                composite_alpha=combined['alpha'].cpu().numpy(),
                composite_depth_sum=combined['depth_sum'].cpu().numpy(),
                road_visible_alpha=combined['road_visible_alpha'].cpu().numpy())
    summary = dict(version=VERSION, arm=options.arm, iteration=options.iteration,
        V1_native_replay_max_abs=max_native_difference, V1_vs_B_max_abs=max_reference_difference,
        V1_vs_B_passed=None if max_reference_difference is None else max_reference_difference <= 2e-5,
        delete_scope='external_dynamic_keep' if keep is not None else 'all_dynamic',
        evaluation_wall_seconds=time.perf_counter()-started,
        optimizer_updates=0, frames=records, quality_accepted=False, human_verdict=None)
    (options.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    if summary['V1_vs_B_passed'] is False:
        raise SystemExit('原 V1 输出未通过 B/R/RP 一致性检查')


if __name__ == '__main__':
    main()
