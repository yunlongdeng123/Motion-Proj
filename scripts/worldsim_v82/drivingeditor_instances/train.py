"""固定真实轨迹的实例重建。默认只验证，--run 才申请GPU，StreetCrafter优先。"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import random
import sys
import time

if __package__:
    from .runtime import validate_spec, load_models, masks_for_frame
    from .priority import inspect_priority, require_gpu_slot
    from .training_policy import resolve_training_policy, validate_resume_checkpoint
    from .restore_contract import verify_runtime_restore
else:
    from runtime import validate_spec, load_models, masks_for_frame
    from priority import inspect_priority, require_gpu_slot
    from training_policy import resolve_training_policy, validate_resume_checkpoint
    from restore_contract import verify_runtime_restore


def instance_mask_objective(package, masks, known, expected_instance_ids=None):
    if __package__:
        from .renderer import VISIBILITY_SEMANTICS
        from .losses import visible_instance_loss
    else:
        from renderer import VISIBILITY_SEMANTICS
        from losses import visible_instance_loss
    if package.get('instance_visibility_semantics') != VISIBILITY_SEMANTICS:
        raise ValueError('需要完整场景排序合成的实例可见贡献')
    if expected_instance_ids is not None and tuple(package['instance_ids']) != tuple(expected_instance_ids):
        raise ValueError('实例贡献通道与真实身份标签顺序不符')
    return visible_instance_loss(package['instance_visible_contribution'], masks, known)


def save_checkpoint(output, step, static, instances, optimizer, environment, sampler, spec, exclusion, *, training_policy, suffix=''):
    import numpy as np
    import torch
    path = Path(output) / ('checkpoint_%06d%s.pt' % (step, suffix))
    if path.exists():
        raise FileExistsError('拒绝覆盖checkpoint: ' + str(path))
    captured = {'schema': 'drivingeditor_rigid_scene_v1', 'step': step, 'spec': spec,
                'background': static.capture(), 'instances': instances.capture(optimizer),
                'environment': environment.capture(), 'background_exclusion': exclusion,
                'sampler': sampler.getstate(), 'python_rng': random.getstate(),
                'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all(),
                'training_policy': training_policy, 'restoration_verified': False}
    temporary = path.with_suffix('.pt.partial')
    torch.save(captured, temporary)
    os.replace(temporary, path)
    return path


def run(args, spec, readiness):
    slot, admission = require_gpu_slot(args.gpu_lock)
    monitor = None
    final_status, failure = 'failed', None
    output_created = False
    output = Path(args.output)
    try:
        if output.exists():
            raise FileExistsError('新实验目录已存在；恢复也使用新目录，保留旧资产')
        output.mkdir(parents=True)
        output_created = True
        import numpy as np
        import torch
        from PIL import Image
        if __package__:
            from .renderer import render, VISIBILITY_SEMANTICS
            from .losses import visible_instance_loss, supervision_contract, local_box_boundary
        else:
            from renderer import render, VISIBILITY_SEMANTICS
            from losses import visible_instance_loss, supervision_contract, local_box_boundary
        sys.path.insert(0, str(Path(spec['matrix_tools'])))
        from memory_monitor import MemoryMonitor
        monitor = MemoryMonitor(output, torch_module=torch, expected_iterations=args.iterations)
        monitor.start()
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        restored = torch.load(args.resume, map_location='cuda', weights_only=False) if args.resume else None
        if restored is not None and (restored['schema'] != 'drivingeditor_rigid_scene_v1' or restored['spec'] != spec):
            raise ValueError('恢复场景、输入与checkpoint合同不符')
        raw_reference = json.loads(Path(spec['reference_config']).read_text(encoding='utf-8-sig'))
        training_policy = resolve_training_policy(vars(args), raw_reference)
        resume_contract = None
        if restored is not None:
            resume_contract = validate_resume_checkpoint(restored, training_policy, args.iterations,
                legacy_effective_config=Path(args.resume).parent / 'effective_config.json',
                legacy_reference_config=spec['reference_config'],
                legacy_reference_provenance={'role': 'actual existing scene0008 GN11 reference used by the 300-step run',
                    'historical_raw_config_snapshot_was_not_saved_in_300_step_checkpoint': True})
            (output / 'resume_policy_contract.json').write_text(json.dumps(resume_contract, ensure_ascii=False, indent=2), encoding='utf-8')
        with monitor.phase('initialization'):
            config, scene, static, instances, optimizer, environment, exclusion = load_models(spec, output, freeze_background=not args.joint_background, restore=restored)
        from utils.hybrid30k_losses import compose_hybrid_loss
        from utils.paper_losses import nps_loss, boundary_distance, sky_loss, sparse_inverse_depth_loss
        from utils.loss_utils import ssim
        from utils.general_utils import get_expon_lr_func
        import kornia
        policy = dict(config)
        policy['hybrid_lambda_dynamic_opacity'] = 0.0  # 新目标下面明确加入逐实例可见贡献，不叠加旧孤立alpha目标。
        policy['hybrid_lambda_dynamic_rgb'] = 5.0
        xyz_schedule = get_expon_lr_func(lr_init=config.position_lr_init, lr_final=config.position_lr_final,
            lr_delay_mult=config.position_lr_delay_mult, max_steps=config.iterations)
        cameras = scene.getTrainCameras()
        with np.load(Path(spec['prepared']) / 'instances.npz', allow_pickle=False) as bundle:
            mask_known_all = bundle['mask_known'].copy()
            normalization_scale = float(np.linalg.norm(bundle['normalization'][:3, :3], axis=0).mean())
        if args.joint_background and not bool(mask_known_all.all()):
            raise ValueError('联合背景优化需要全部49xK真实可见mask，不能以缺激光标注当空mask')
        if restored is None:
            instances.pose_translation_limit = args.pose_translation_limit_m * normalization_scale
        selected_indices = np.flatnonzero(mask_known_all.any(axis=0)).tolist()
        selected_ids = [instances.instance_ids[i] for i in selected_indices]
        sampler = random.Random(args.seed)
        first = 1
        if restored is not None:
            first = resume_contract['first_step']
            sampler.setstate(restored['sampler'])
            random.setstate(restored['python_rng']); np.random.set_state(restored['numpy_rng'])
            torch.set_rng_state(restored['torch_rng'].cpu())
            torch.cuda.set_rng_state_all([state.cpu() for state in restored['cuda_rng']])
            # Keep a fresh golden load: legacy restore can consume optimizer
            # wrapper metadata in the original object passed to load_models.
            golden = torch.load(args.resume, map_location='cpu', weights_only=False)
            with monitor.phase('runtime_restore_verification', iteration=restored['step']):
                restore_receipt = verify_runtime_restore(golden, static, instances, optimizer, environment, sampler)
            restore_receipt['checkpoint_path'] = args.resume
            (output / 'runtime_restore_contract.json').write_text(json.dumps(restore_receipt, ensure_ascii=False, indent=2), encoding='utf-8')
            del golden
        effective = {'task': 'WS-V82-DRIVINGEDITOR-INSTANCES-20261013', 'spec': spec, 'arguments': vars(args),
                     'seed': args.seed, 'trajectory': 'fixed_genuine_track_pose_until_explicit_refine_stage',
                     'vehicle_PVG_motion': False, 'stationary_vehicles_included': True,
                     'background': 'GN11_30k_prior_pruned_by_selected_instance_boxes', 'background_frozen': not args.joint_background,
                     'background_exclusion': exclusion,
                     'new_SGCM_clones': False, 'SGCM': 'inherited_30k_background; rigid_clone_prune_is_separate',
                     'NPS': 'original_formula_if_enabled_in_reference; not_absolute_static_guarantee',
                     'opacity': '100 * equal_known_instance_mean(full_HxW_MSE_of_joint_visible_contribution)',
                     'unknown_mask_targets': 'excluded', 'pixel_masks': readiness['masks'],
                     'anisotropic_split': False, 'motion_model': 'fixed_local_Gaussians_plus_track_rigid_transform',
                     'GPU_admission': admission, 'restoration_verified': False}
        effective['training_policy'] = training_policy
        effective['resume_policy_contract'] = resume_contract
        (output / 'effective_config.json').write_text(json.dumps(effective, ensure_ascii=False, indent=2), encoding='utf-8')
        image_directory = output / 'preview'; image_directory.mkdir()
        started = time.monotonic()
        for step in range(first, args.iterations + 1):
            if step == first or step % 100 == 0:
                priority = inspect_priority()
                if not priority['allowed']:
                    with monitor.phase('priority_yield_checkpoint', iteration=step-1):
                        saved = save_checkpoint(output, step-1, static, instances, optimizer, environment, sampler, spec, exclusion,
                                                training_policy=training_policy, suffix='_priority_yield')
                    (output / 'status.json').write_text(json.dumps({'state': 'yielded_to_priority_work', 'step': step-1,
                        'resume_checkpoint': str(saved), 'priority': priority}, ensure_ascii=False, indent=2), encoding='utf-8')
                    final_status = 'yielded_to_priority_work'
                    return
            if args.refine_poses_after > 0 and step >= args.refine_poses_after and not instances.refine_poses:
                instances.enable_pose_refinement()
                # 保留现有几何Adam状态，添加此前没有优化的位姿参数。
                for name, parameter in (('pose_translation', instances.pose_translation_raw), ('pose_rotation', instances.pose_rotation_raw)):
                    optimizer.add_param_group({'params': [parameter], 'name': name, 'lr': 1e-4})
            camera = cameras[sampler.randrange(len(cameras))]
            for group in optimizer.param_groups:
                if group['name'] == 'xyz': group['lr'] = xyz_schedule(step)
            if args.joint_background: static.update_learning_rate(step)
            monitor.set_phase('training', iteration=step)
            raw_masks, known = masks_for_frame(spec, instances, camera)
            supervision_contract(known, freeze_background=not args.joint_background)
            other = [static.get_scaling_t.clamp_max(2), static.get_inst_velocity]
            package = render(camera, static, instances, config, torch.zeros(3, device='cuda'), env_map=environment,
                             other=other, instance_ids=selected_ids, max_features=args.max_features, is_training=True)
            if package['instance_visibility_semantics'] != VISIBILITY_SEMANTICS:
                raise ValueError('禁止用孤立实例alpha替代全场景可见贡献')
            gt = camera.original_image.cuda()
            masks = raw_masks[selected_indices]
            mask_known = known[selected_indices]
            union = masks[mask_known].sum(0, keepdim=True) if mask_known.any() else gt.new_zeros((1, *gt.shape[-2:]))
            if bool((union > 1).any()):
                raise ValueError('真实可见实例标签重叠')
            alpha = package['alpha']; depth = package['depth'] / alpha.clamp_min(1e-5)
            depth = (alpha / depth.clamp_min(1e-5) + (1-alpha) / 900).clamp_min(1e-5).reciprocal() if config.depth_blend_mode == 0 else alpha * depth + (1-alpha) * 900
            photo = lambda pred, target: .8 * (pred-target).abs().mean() + .2 * (1-ssim(pred, target))
            sky = camera.sky_mask.cuda().to(gt)
            if sky.ndim == 2: sky = sky[None]
            static_feature = package['static_feature'] / alpha.clamp_min(1e-5)
            nps_features = package['static_feature'][1:] / package['static_alpha'].clamp_min(1e-5)
            distance = boundary_distance(union) if policy.get('hybrid_nps_enabled', False) else None
            zero = package['render'].sum() * 0
            state = {'iteration': step, 'global_rgb': photo(package['render'], gt),
                     'dynamic_rgb': photo(package['render_d'] * union, gt * union) if mask_known.any() else zero,
                     'sky_rgb': photo(package['render_s'] * sky, gt * sky),
                     'static_rgb': photo(package['render_s'] * (1-union), gt * (1-union)),
                     'sky': sky_loss(package['static_alpha'], sky),
                     'depth': sparse_inverse_depth_loss(depth, camera.pts_depth.to(depth)),
                     'inverse_depth_smoothness': kornia.losses.inverse_depth_smoothness_loss((1/(depth+1e-5))[None], gt[None]),
                     'velocity_smoothness': kornia.losses.inverse_depth_smoothness_loss((package['feature'][1:] / alpha.clamp_min(1e-5))[None], gt[None]),
                     'dynamic_alpha': package['dynamic_visible_alpha'], 'dynamic_mask': union,
                     'static_t_map': static_feature[:1], 'static_v_map': static_feature[1:], 'alpha': alpha,
                     'nps_features': nps_features, 'nps_distances': distance}
            result = compose_hybrid_loss(state, policy, nps_fn=nps_loss)
            mask_loss = instance_mask_objective(package, masks, mask_known, selected_ids)
            boundary = local_box_boundary(instances)
            pose = instances.pose_regularization()
            loss = result.total + 100 * mask_loss + args.box_boundary_weight * boundary + args.pose_prior_weight * pose['total']
            if not torch.isfinite(loss): raise FloatingPointError('实例训练loss非finite')
            loss.backward()
            gradient_checks = [torch.isfinite(parameter.grad).all() for group in optimizer.param_groups
                               for parameter in group['params'] if parameter.grad is not None]
            if not gradient_checks or not bool(torch.stack(gradient_checks).all()):
                raise FloatingPointError('实例优化梯度缺失或非finite，停止并保留资产')
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if args.joint_background:
                static.optimizer.step(); static.optimizer.zero_grad(set_to_none=True)
                environment.optimizer.step(); environment.optimizer.zero_grad(set_to_none=True)
            if step % 1000 == 0: instances.oneupSHdegree()
            # 首版固定拓扑便于区分表示与增密；core提供保身份/Adam的clone/prune，
            # 不把它冒充原SGCM或自动套用PVG temporal densification。
            if step % 100 == 0 or step == first:
                record = {'step': step, 'frame': camera.image_name, 'loss': float(loss.detach()),
                          'instance_mask_loss': float(mask_loss.detach()), 'known_instances': int(mask_known.sum()),
                          'box_boundary': float(boundary.detach()), 'pose_prior': float(pose['total'].detach()),
                          'instance_points': instances.num_points, 'background_points': len(static.get_xyz),
                          'instance_optimizer_gradients_finite': True,
                          'steps_per_second': (step-first+1) / (time.monotonic()-started)}
                with (output / 'steps.jsonl').open('a', encoding='utf-8') as stream: stream.write(json.dumps(record) + '\n')
                (output / 'status.json').write_text(json.dumps(dict(record, state='running')), encoding='utf-8')
                print(json.dumps(record), flush=True)
            if step % 10000 == 0 or step == args.iterations:
                with monitor.phase('checkpoint', iteration=step):
                    save_checkpoint(output, step, static, instances, optimizer, environment, sampler, spec, exclusion,
                                    training_policy=training_policy)
                with monitor.phase('preview', iteration=step):
                    for name in ('render', 'render_s', 'raw_render_d'):
                        pixels = (package[name].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
                        Image.fromarray(pixels).save(image_directory / ('%06d_%s.png' % (step, name)))
        (output / 'status.json').write_text(json.dumps({'state': 'complete', 'step': args.iterations,
            'training_reconstruction_only': True, 'edit_visual_acceptance': 'pending', 'restoration_verified': False}), encoding='utf-8')
        final_status = 'complete'
    except BaseException as error:
        failure = error
        if output_created:
            (output / 'status.json').write_text(json.dumps({'state': 'engineering_failed', 'error': repr(error),
                'assets_preserved': True}), encoding='utf-8')
        raise
    finally:
        if monitor is not None: monitor.close(status=final_status, error=failure)
        slot.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--output')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--resume')
    parser.add_argument('--geometry-only-smoke', action='store_true', help='没有任何已知二维身份时，只允许显式<=300步几何/RGB冒烟')
    parser.add_argument('--iterations', type=int, default=30000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--joint-background', action='store_true')
    parser.add_argument('--refine-poses-after', type=int, default=0)
    parser.add_argument('--pose-translation-limit-m', type=float, default=.5)
    parser.add_argument('--box-boundary-weight', type=float, default=1.0)
    parser.add_argument('--pose-prior-weight', type=float, default=.01)
    parser.add_argument('--max-features', type=int, default=10)
    parser.add_argument('--gpu-lock', default='/root/autodl-tmp/drivingeditor_repro/campaign/gpu_slot.lock')
    args = parser.parse_args()
    spec, report = validate_spec(args.spec)
    if args.iterations < 1 or args.refine_poses_after < 0 or (args.refine_poses_after and args.refine_poses_after > args.iterations):
        raise ValueError('训练和位姿阶段预算无效')
    import math
    if args.seed < 0 or args.seed >= 2**32 or args.max_features < 8 or any(not math.isfinite(value) or value < 0 for value in (args.box_boundary_weight, args.pose_prior_weight)) or not math.isfinite(args.pose_translation_limit_m) or args.pose_translation_limit_m <= 0:
        raise ValueError('seed/feature容量/位姿范围/正则权重无效')
    if args.joint_background and not report['full_mask_supervision_ready']:
        raise ValueError('真实实例二维标签未齐全，联合背景阶段待准备')
    if args.run and report['mask_known_pairs'] == 0 and not (args.geometry_only_smoke and args.iterations <= 300):
        raise ValueError('没有真实二维身份映射；仅可显式<=300步geometry-only-smoke，不能静默跑30k')
    if not args.run:
        print(json.dumps({'status': 'CPU_SPEC_VALIDATED_GPU_NOT_STARTED', 'scene': spec['scene_id'],
                          'instances': report['instance_count'], 'known_masks': report['masks'],
                          'full_mask_supervision_ready': report['full_mask_supervision_ready'],
                          'default_background_frozen': not args.joint_background, 'trajectory_fixed': not args.refine_poses_after}, ensure_ascii=False, indent=2))
        return
    if not args.output: parser.error('--run需要新--output')
    run(args, spec, report)


if __name__ == '__main__':
    main()
