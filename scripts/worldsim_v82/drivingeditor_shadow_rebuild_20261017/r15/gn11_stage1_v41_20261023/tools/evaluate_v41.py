"""V4.1 浮点49帧完整/道路/删车评价；本次仅准备，GPU启动前不执行。"""

from contextlib import contextmanager
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from v4_eval_metrics import METRIC_CONTRACT, region_metrics, full_perceptual_metrics, aggregate49
from v41_acceptance import CONTRACT, regression_summary, road_query_metrics, paired_deleted_metrics, deletion_summary, topology


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--arm', choices=('R', 'B'), required=True)
    parser.add_argument('--run', type=Path, help='可选：独立 smoke 输出目录')
    parser.add_argument('--iteration', type=int, default=30000)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dynamic-keep', type=Path, help='可选：对应该 checkpoint 动态行的 bool .npy')
    parser.add_argument('--audit-deletion-gradients', action='store_true', help='固定 raw102/129/137 的逐车只读梯度审计，零优化更新')
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
    from utils.v4_road import load_bundle, classify_units, sample_map, RoadLayer, FRAMES, POLICY as BASE_POLICY
    from utils.v41_objectives import VERSION, POLICY, ssim_map, valid_windows
    from utils.v41_vehicle import ACTORS
    from utils.v4_road_render import render_road, render_composite
    from v41_gradient_audit import AUDIT_FRAMES, domain_gradient_audit, a37_path_failures
    from utils.v41_render import render_training_deleted
    run = options.run.resolve() if options.run else campaign/'arms'/options.arm/'run30k'
    config = json.loads((run/'effective_config.json').read_text(encoding='utf-8'))
    if options.arm == 'B':
        if options.audit_deletion_gradients:
            raise ValueError('原V1没有道路参数，不做道路梯度审计')
        plan = json.loads((campaign/'plan.json').read_text())
        baseline = json.loads(Path(plan['baseline_config']).read_text())
        if config != dict(baseline, model_path=str(run)):
            raise ValueError('B须为原V1配置，训练不加道路字段')
        # 仅评价读取共同域和来源；不回写训练配置，不引入道路模型。
        config = dict(config, v4_arm='B', v4_bundle=plan['road_bundle'],
                      v41_dense_manifest=plan['dense_manifest'])
    dense_root = Path(config['v41_dense_manifest']).parent
    prefix = 'v1' if options.arm == 'B' else 'v41'
    node = json.loads((run/f'{prefix}_checkpoint{options.iteration}.json').read_text())
    expected_files = 4 if options.arm == 'B' else 5
    if node['iteration'] != options.iteration or len(node['files']) != expected_files or not all((run/f).is_file() for f in node['files']):
        raise ValueError('完整 V4.1 节点缺失')
    if config['v4_arm'] != options.arm:
        raise ValueError('错误实验组配置')
    if options.output.exists():
        raise FileExistsError('评价只写新目录')
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
        if (state['version'] != VERSION or state['policy'] != POLICY or state['base_policy'] != BASE_POLICY
                or state['dense_manifest'] != config['v41_dense_manifest'] or state['arm'] != options.arm
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
        if options.audit_deletion_gradients:
            raise ValueError('逐车梯度审计仅适用于默认 all_dynamic 删车')
        keep = torch.as_tensor(np.load(options.dynamic_keep, allow_pickle=False), device='cuda')
        if keep.dtype != torch.bool or keep.shape != (len(dynamic.get_xyz),):
            raise ValueError('动态 keep 文件须精确对应当前 checkpoint 行')
    background = torch.tensor([1., 1., 1.] if args.white_background else [0., 0., 0.], device='cuda')
    records = []
    color_sum = np.zeros(len(arrays['unit_ids']), np.float64)
    color_count = np.zeros(len(color_sum), np.int64)
    max_native_difference = 0.
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
            # B的零数组只表示道路不存在；道路颜色误差记null，不当作黑色道路预测。
            independent = render_road(camera, road, args) if road is not None else dict(
                rgb=torch.zeros_like(native['render']), alpha=torch.zeros_like(combined['alpha']))
            gt = camera.original_image.cuda()
            mask = load_dynamic_mask(args.source_path, camera).float()
            if mask.ndim == 2:
                mask = mask[None]
            dilated = F.max_pool2d(mask[None], 7, 1, 3)[0] > .5
            interior = F.max_pool2d((1-mask)[None], 7, 1, 3)[0] < .5
            sky = camera.sky_mask.cuda().bool()[:1]
            domains = dict(full=torch.ones_like(mask, dtype=torch.bool), body=interior,
                           edge=dilated & ~interior, background=~dilated & ~sky)
            with np.load(dense_root/'frames'/f'{camera.image_name}.npz', allow_pickle=False) as z:
                visible = torch.as_tensor(z['visible'], device='cuda')
                protected = torch.as_tensor(z['protect'], device='cuda')
                windows = torch.as_tensor(z['ssim_valid'], device='cuda')
                vehicle_body = torch.as_tensor(z['vehicle_body'], device='cuda')
                vehicle_edge = torch.as_tensor(z['vehicle_edge'], device='cuda')
                unit_ids = z['unit_id'].copy()
                deletion_domains = z['deletion_domains'].copy()
                deletion_kinds = z['deletion_label_kind'].copy()
                actor_labels = z['actor_labels'].copy()
                deletion_reference = torch.as_tensor(z['deletion_target_rgb'], device='cuda')
            domains['reliable_road'] = visible[None]
            domains['protected'] = protected[None]
            # 每身份始终保留同一键；远处小车内部为空也明确记录，不静默跳过身份。
            for i, actor in enumerate((*ACTORS, 'other')):
                domains[f'a{actor}_body'] = vehicle_body[i][None]
                domains[f'a{actor}_edge'] = vehicle_edge[i][None]
            domains['normal_background'] = domains['background'] & ((native['render']-gt).abs().mean(0, keepdim=True) <= .05)
            directory = options.output/camera.image_name
            directory.mkdir()
            native_cpu = native['render'].cpu().numpy()
            outputs = dict(v1=native['render'], road=None if independent is None else independent['rgb'],
                composite=combined['rgb'], delete=native_removed['rgb'], delete_with_road=removed['rgb'])
            outputs['road_surface'] = independent['rgb']/independent['alpha'].clamp_min(1e-5)
            metrics = {name: {region: region_metrics(value, gt, domain)
                       for region, domain in domains.items()} for name, value in outputs.items()
                       if value is not None and name in ('v1', 'composite')}
            raw_metrics = {name: {region: region_metrics(value, gt, domain, clamp=False)
                           for region, domain in domains.items()} for name, value in outputs.items()
                           if value is not None and name in ('v1', 'composite')}
            for name in ('v1', 'composite'):
                metrics[name]['full'].update(full_perceptual_metrics(outputs[name], gt))
            vehicle_structural = {}
            for key, domain in domains.items():
                if not key.startswith('a'): continue
                domain = domain[0]; valid = valid_windows(domain)
                vehicle_structural[key] = dict(valid_windows=int(valid.sum()),
                    **{name:float(ssim_map(outputs[name].clamp(0, 1), gt.clamp(0, 1), domain)[valid].mean())
                       if valid.any() else None for name in ('v1', 'composite')})
            # 隐藏路面没有整图 GT，不拿原车身 RGB 评价删车画面。
            record = dict(frame=camera.image_name, regions=metrics, raw_regions=raw_metrics, vehicle_masked_ssim=vehicle_structural,
                road_visible_alpha_mean=float(combined['road_visible_alpha'].mean()),
                road_independent_alpha_mean=None if independent is None else float(independent['alpha'].mean()))
            record['dense_road'] = dict(visible_pixels=int(visible.sum()), ssim_windows=int(windows.sum()),
                road_surface=region_metrics(outputs['road_surface'], gt, visible[None]) if road is not None else None,
                independent_alpha=float(independent['alpha'][0, visible].mean()),
                joint_road_contribution=float(combined['road_visible_alpha'][0, visible].mean()),
                alpha_below_point_one_fraction=float((independent['alpha'][0, visible] < .1).float().mean()),
                masked_ssim={name:float(ssim_map(outputs[name].clamp(0, 1), gt.clamp(0, 1), visible)[windows].mean())
                    for name in ('v1', 'composite')},
                protected_change_mae=float((combined['rgb']-native['render']).abs()[:, protected].mean()))
            # Keep the old yardstick alongside additions: expanded labels must not silently change comparison scope.
            for key, selector, source in (
                    ('crossview_deleted', deletion_kinds > 0, 'legacy plus blocked-consensus additions'),
                    ('crossview_deleted_legacy', deletion_kinds == 1, 'unchanged R5 two-view proxy'),
                    ('crossview_deleted_added', deletion_kinds == 2, 'fit-only median gated by disjoint validation blocks')):
                record[key] = dict(scope='all_dynamic', hidden_GT=False, source=source,
                    applicable=keep is None, actors={})
                if keep is None:
                    for a, actor in enumerate(ACTORS):
                        record[key]['actors'][str(actor)] = paired_deleted_metrics(
                            native_removed['rgb'].cpu().numpy(), removed['rgb'].cpu().numpy(),
                            deletion_reference.cpu().numpy(), actor_labels == a+1, deletion_domains[a] & selector,
                            removed['road_visible_alpha'][0].cpu().numpy())
            if options.audit_deletion_gradients and camera.image_name in AUDIT_FRAMES:
                # Extra diagnostic forward/backward only; autograd.grad does not fill .grad or step Adam.
                with torch.enable_grad():
                    probe = render_training_deleted(camera, static, road, args, background, environment)
                    record['deletion_gradient_audit'] = domain_gradient_audit(probe, deletion_reference,
                        torch.as_tensor(deletion_domains, device='cuda'),
                        torch.as_tensor(np.stack([actor_labels == a+1 for a in range(len(ACTORS))]), device='cuda'),
                        (road.dc,road.opacity), POLICY['deletion_contribution_target'])
                    del probe
                # Diagnostic only: retain geometry/native state, saturate road opacity, restore in finally.
                # A raster probe of available visibility, not a training update or guaranteed physical bound.
                saved_opacity = road.opacity.detach().clone()
                try:
                    road.opacity.fill_(20.)
                    opaque = render_training_deleted(camera, static, road, args, background, environment)
                    opaque_contribution = opaque['road_visible_alpha'][0].cpu().numpy()
                finally:
                    road.opacity.copy_(saved_opacity)
                for a,actor in enumerate(ACTORS):
                    record['deletion_gradient_audit'][str(actor)]['opaque_probe'] = {
                        str(t):dict(full_target=topology(actor_labels == a+1, opaque_contribution >= t),
                            labelled=topology(deletion_domains[a], opaque_contribution >= t))
                        for t in CONTRACT['coverage_thresholds']}
                del opaque
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
                if road is not None:
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
            pixel_ids = unit_ids[visible.cpu().numpy()]
            errors = (outputs['road_surface']-gt).abs().mean(0)[visible].cpu().numpy()
            if road is not None:
                np.add.at(color_sum, pixel_ids, errors)
                np.add.at(color_count, pixel_ids, 1)
            records.append(record)
            for name, value in outputs.items():
                if value is not None:
                    image = (value.clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8)
                    Image.fromarray(image).save(directory/f'{name}.png')
            np.savez_compressed(directory/'raw.npz', v1=native_cpu,
                gt=gt.cpu().numpy(), road_premultiplied=independent['rgb'].cpu().numpy(),
                road_surface=outputs['road_surface'].cpu().numpy(), road_alpha=independent['alpha'].cpu().numpy(),
                composite=combined['rgb'].cpu().numpy(), v1_alpha=native['alpha'].cpu().numpy(),
                composite_alpha=combined['alpha'].cpu().numpy(),
                composite_depth_sum=combined['depth_sum'].cpu().numpy(),
                road_visible_alpha=combined['road_visible_alpha'].cpu().numpy(),
                deleted_road_visible_alpha=removed['road_visible_alpha'].cpu().numpy(),
                delete=native_removed['rgb'].cpu().numpy(), delete_with_road=removed['rgb'].cpu().numpy())
    # 同世界颜色证据来自可见路面；不使用任何删车图上的车辆 RGB 作为隐藏路面真值。
    color_error = color_sum/np.maximum(color_count, 1)
    np.savez_compressed(options.output/'world_color_evidence.npz', mae=color_error, count=color_count)
    for i, record in enumerate(records):
        with np.load(options.output/record['frame']/'raw.npz', allow_pickle=False) as z:
            record['actor_road_queries'] = {str(actor):road_query_metrics(arrays, i, actor,
                z['road_visible_alpha'][0], z['deleted_road_visible_alpha'][0], z['road_alpha'][0],
                color_error, color_count) for actor in ACTORS}
    summary = dict(version=VERSION, metric_contract=METRIC_CONTRACT,
        aggregate49=aggregate49(records, FRAMES), arm=options.arm, iteration=options.iteration,
        V1_native_replay_max_abs=max_native_difference,
        V1_parameter_identity='separate CPU compare_v41_states --native-only B versus R; report CUDA trajectory differences without claiming bitwise identity',
        absent_road_arrays_are_zero=road is None,
        delete_scope='external_dynamic_keep' if keep is not None else 'all_dynamic',
        evaluation_wall_seconds=time.perf_counter()-started,
        optimizer_updates=0, frames=records, quality_accepted=False, human_verdict=None)
    summary['acceptance_contract'] = CONTRACT
    summary['quality_regressions'] = regression_summary(records)
    summary['crossview_deleted_comparison'] = deletion_summary(records)
    summary['crossview_deleted_legacy_comparison'] = deletion_summary(records, 'crossview_deleted_legacy')
    summary['crossview_deleted_added_comparison'] = deletion_summary(records, 'crossview_deleted_added')
    if options.audit_deletion_gradients:
        audits = {r['frame']:r['deletion_gradient_audit'] for r in records if 'deletion_gradient_audit' in r}
        summary['deletion_gradient_audit'] = dict(frames=audits, failures=a37_path_failures(audits),
            optimizer_updates=0, scope='training forward masks; same frozen checkpoint; separate actors',
            limitation='nonzero gradient is not improvement; raw137/a6 zero-label regions remain unknown; opaque probe uses same raster/fixed geometry, not a physical upper-bound proof')
    summary['a37_and_a6_verdict'] = 'REVIEW_REQUIRED: compare fixed-query coverage/color/connectivity to V4; opacity alone cannot pass'
    summary['dense_aggregate49'] = {key:float(np.mean([record['dense_road'][key] for record in records]))
        for key in ('visible_pixels', 'ssim_windows', 'independent_alpha', 'joint_road_contribution',
                    'alpha_below_point_one_fraction', 'protected_change_mae')}
    (options.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
