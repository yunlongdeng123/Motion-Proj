"""将已保存的paper基础设施接入2024继承目标；仅产出可复核局部diff。"""
import ast


HYBRID_BLOCK = '''        if paper_method:
            if not args.get("hybrid_method", False) or dynamic_gaussians is None:
                raise ValueError("此副本只用于含动态分支的hybrid30k实验")
            from utils.hybrid30k_losses import compose_hybrid_loss, diagnose_gradients, compare_objective_gradients
            from utils.hybrid30k_runtime import record_step, record_gradient
            from utils.paper_losses import nps_loss, boundary_distance, sky_loss, sparse_inverse_depth_loss
            legacy_static_feature = render_pkg["static_feature"] / alpha.clamp_min(EPS)
            legacy_static_t_map = legacy_static_feature[0:1]
            legacy_static_v_map = legacy_static_feature[1:]
            observed_static_mask = 1.0 - dynamic_mask
            static_rgb = ((1.0 - args.lambda_dssim) * F.l1_loss(static_image * observed_static_mask, gt_image * observed_static_mask)
                          + args.lambda_dssim * (1.0 - ssim(static_image * observed_static_mask, gt_image * observed_static_mask)))
            dynamic_rgb = ((1.0 - args.lambda_dssim) * F.l1_loss(dynamic_image, gt_dynamic_image)
                           + args.lambda_dssim * (1.0 - ssim(dynamic_image, gt_dynamic_image)))
            sky_rgb = ((1.0 - args.lambda_dssim) * F.l1_loss(static_image * sky_mask_1, sky_truth)
                       + args.lambda_dssim * (1.0 - ssim(static_image * sky_mask_1, sky_truth)))
            if viewpoint_cam.pts_depth is None:
                raise ValueError("LiDAR深度输入缺失")
            if args.hybrid_nps_enabled and not hasattr(viewpoint_cam, "_paper_nps_distance"):
                viewpoint_cam._paper_nps_distance = boundary_distance(dynamic_mask)
            inverse_depth = 1 / (depth + 1e-5)
            inverse_depth_smoothness = kornia.losses.inverse_depth_smoothness_loss(inverse_depth[None], gt_image[None])
            velocity_smoothness = kornia.losses.inverse_depth_smoothness_loss(v_map[None], gt_image[None])
            hybrid_state = {
                "iteration": iteration,
                "global_rgb": (1.0 - args.lambda_dssim) * loss_l1 + args.lambda_dssim * loss_ssim,
                "dynamic_rgb": dynamic_rgb, "sky_rgb": sky_rgb, "static_rgb": static_rgb,
                "sky": sky_loss(static_alpha, sky_mask),
                "depth": sparse_inverse_depth_loss(depth, viewpoint_cam.pts_depth.to(depth.device)),
                "inverse_depth_smoothness": inverse_depth_smoothness,
                "velocity_smoothness": velocity_smoothness,
                "dynamic_alpha": render_pkg["dynamic_alpha"], "dynamic_mask": dynamic_mask,
                "static_t_map": legacy_static_t_map, "static_v_map": legacy_static_v_map,
                "alpha": alpha, "nps_features": static_v_map,
                "nps_distances": getattr(viewpoint_cam, "_paper_nps_distance", None),
            }
            hybrid_result = compose_hybrid_loss(hybrid_state, args, nps_fn=nps_loss)
            loss = hybrid_result.total
            log_dict.update(hybrid_result.log_values())
            if iteration in args.hybrid_gradient_nodes:
                named = {name: render_pkg[name] for name in ("static_feature", "static_alpha", "alpha", "dynamic_alpha", "render", "render_s", "render_d") if render_pkg[name].requires_grad}
                preserved = [value for value in render_pkg.values() if torch.is_tensor(value) and value.requires_grad]
                diagnostic = diagnose_gradients(hybrid_result, named, preserve_grad_tensors=preserved)
                # 改写已有目标可能改变量级；在同一前向比较，不据文字假定影响很小。
                legacy_sky = (-sky_mask.float() * torch.log1p(-alpha.clamp(1e-6, 1-1e-6))).mean()
                points = viewpoint_cam.pts_depth.to(depth.device)
                valid = points > 0
                legacy_depth = (torch.abs(1 / (points[valid] + 1e-5) - 1 / (depth[valid] + 1e-5)).mean()
                                if valid.any() else depth.sum() * 0)
                depth_named = dict(named, normalized_depth=depth)
                diagnostic.update(compare_objective_gradients(legacy_sky * args.lambda_sky_opa, hybrid_result.weighted["sky"], named, prefix="rewrite_sky", preserve_grad_tensors=preserved))
                diagnostic.update(compare_objective_gradients(legacy_depth * args.lambda_lidar * hybrid_result.metadata["hybrid_depth_decay"], hybrid_result.weighted["depth"], depth_named, prefix="rewrite_depth", preserve_grad_tensors=preserved))
                record_gradient(args, iteration, diagnostic)
            record_step(args, iteration, viewpoint_cam, time_shift, gaussians, dynamic_gaussians, hybrid_result.log_values())
            # 仅保留旧可视化约定；loss和指标入口使用上面的raw alpha。
            dynamic_alpha = dynamic_alpha * dynamic_mask
'''


def once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('训练源码插入点不唯一: ' + before.splitlines()[0])
    return text.replace(before, after, 1)


def patch_train_source(text):
    text = text.replace('\r\n', '\n')
    if 'hybrid30k_losses' in text:
        raise ValueError('拒绝重复patch或复用旧hybrid副本')
    start = text.index('        if paper_method:\n            # Final-paper Eqs7-16.')
    end = text.index('        else:\n            if dynamic_gaussians is not None:', start)
    text = text[:start] + HYBRID_BLOCK + text[end:]
    text = once(text, '    scene = Scene(args, gaussians, dynamic_gaussians)\n',
                '    scene = Scene(args, gaussians, dynamic_gaussians)\n'
                '    from utils.hybrid30k_runtime import CameraSampler\n'
                '    scene.hybrid_sampler = CameraSampler(args.seed, scene.getTrainCameras())\n')
    text = once(text, 'create_sgcm_controller(args.source_path, args)',
                'create_sgcm_controller(args.source_path, args, training_views=scene.getTrainCameras(), gaussians=gaussians)')
    start = text.index('        if not viewpoint_stack:\n')
    end = text.index('        # time_shift = None\n', start) + len('        # time_shift = None\n')
    old = text[start:end]
    feature_start = old.index('        # render v and t scale map\n')
    feature_end = old.index('        if np.random.random() < args.lambda_self_supervision:')
    preserved_features = old[feature_start:feature_end]
    text = text[:start] + ('        viewpoint_cam, time_shift = scene.hybrid_sampler.select(scene.getTrainCameras(), args.lambda_self_supervision, scene.time_interval)\n'
                           + preserved_features) + text[end:]
    text = once(text, 'paper_sgcm.refine(gaussians, radii[:end_idx], iteration)',
                'paper_sgcm.refine(gaussians, radii[:end_idx], iteration, visibility_filter=visibility_filter[:end_idx])')
    text = once(text, '                        frame_records.append(record)\n',
                '                        from utils.hybrid30k_metrics import save_frame\n'
                '                        from utils.paper_masks import load_dynamic_mask\n'
                '                        record["hybrid_metrics"] = save_frame(outdir, record["frame_id"], render_pkg, gt_image, load_dynamic_mask(args.source_path, viewpoint), viewpoint.sky_mask)\n'
                '                        frame_records.append(record)\n')
    text = once(text, '    if (iteration % args.opacity_reset_interval == 0 or (\n',
                '    if iteration < args.densify_until_iter and (iteration % args.opacity_reset_interval == 0 or (\n')
    text = once(text, '    return evaluation_records\n',
                '    if iteration in test_iterations:\n'
                '        from utils.hybrid30k_static_time import run_static_time_check\n'
                '        run_static_time_check(scene, renderFunc, renderArgs, env_map, iteration)\n'
                '    return evaluation_records\n')
    ast.parse(text)
    return text


def patch_checkpoint_source(text):
    text = text.replace('\r\n', '\n')
    if 'state["hybrid_sampler"]' in text:
        raise ValueError('拒绝重复改写检查点源码')
    anchor = '    if sgcm is not None:\n'
    first = text.index(anchor, text.index('def capture_training_state('))
    text = text[:first] + ('    if hasattr(scene, "hybrid_sampler"):\n'
                           '        state["hybrid_sampler"] = scene.hybrid_sampler.capture()\n') + text[first:]
    text = once(text, '    if not isinstance(state, dict) or set(state) != required:\n',
                '    if args.get("hybrid_method", False):\n'
                '        required.add("hybrid_sampler")\n'
                '    if not isinstance(state, dict) or set(state) != required:\n')
    text = once(text, '    _validate_resume_configuration(state["config"], args)\n',
                '    _validate_resume_configuration(state["config"], args)\n'
                '    if args.get("hybrid_method", False):\n'
                '        from utils.hybrid30k_runtime import CameraSampler\n'
                '        probe = CameraSampler(args.seed, scene.getTrainCameras())\n'
                '        probe.restore(state["hybrid_sampler"])\n'
                '        if probe.steps != iteration:\n'
                '            raise ValueError("hybrid采样步数与检查点不同")\n')
    start = text.index('def restore_training_state(')
    # 只在原完整恢复返回前恢复额外独立采样流。
    return_start = text.index('    return ', start)
    text = text[:return_start] + ('    if args.get("hybrid_method", False):\n'
                                  '        if "hybrid_sampler" not in state or not hasattr(scene, "hybrid_sampler"):\n'
                                  '            raise ValueError("hybrid完整采样状态缺失，拒绝旧检查点")\n'
                                  '        scene.hybrid_sampler.restore(state["hybrid_sampler"])\n') + text[return_start:]
    ast.parse(text)
    return text
