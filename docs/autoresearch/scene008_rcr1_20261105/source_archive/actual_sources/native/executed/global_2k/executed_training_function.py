def training(args):

    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        tb_writer = None
        print("Tensorboard not available: not logging progress")
    vis_path = os.path.join(args.model_path, 'visualization')
    os.makedirs(vis_path, exist_ok=True)

    paper_method = bool(args.get("paper_method", False))
    if paper_method:
        from utils.paper_masks import load_dynamic_mask
        from utils.paper_losses import (sky_loss, dynamic_opacity_loss, boundary_distance,
                                       nps_loss, sparse_inverse_depth_loss, compose_paper_loss)

    gaussians = GaussianModel(args)
    if args.including_dynamic:
        dynamic_gaussians = Dynamic_GaussianModel(args)
    else:
        dynamic_gaussians = None
    scene = Scene(args, gaussians, dynamic_gaussians)
    from utils.hybrid30k_runtime import CameraSampler
    scene.hybrid_sampler = CameraSampler(args.seed, scene.getTrainCameras())
    gaussians.training_setup(args)
    if args.including_dynamic:
        dynamic_gaussians.training_setup(args)

    # classifier = torch.nn.Conv2d(gaussians.num_objects, 128, kernel_size=1)
    # cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    # cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-4)
    # classifier.cuda()

    if args.env_map_res > 0:
        env_map = EnvLight(resolution=args.env_map_res).cuda()
        env_map.training_setup(args)
    else:
        env_map = None

    first_iter = 0
    if args.start_checkpoint:
        (model_params, first_iter) = torch.load(args.start_checkpoint)
        gaussians.restore(model_params, args)
        if dynamic_gaussians is not None:
            dynamic_checkpoint = os.path.join(os.path.dirname(args.start_checkpoint),
                                             os.path.basename(args.start_checkpoint).replace("chkpnt", "dynamic_chkpnt", 1))
            (dynamic_params, dynamic_iter) = torch.load(dynamic_checkpoint)
            if dynamic_iter != first_iter:
                raise ValueError("Static and dynamic checkpoint iterations differ")
            dynamic_gaussians.restore(dynamic_params, args)

        if env_map is not None:
            env_checkpoint = os.path.join(os.path.dirname(args.start_checkpoint),
                                        os.path.basename(args.start_checkpoint).replace("chkpnt", "env_light_chkpnt"))
            (light_params, light_iter) = torch.load(env_checkpoint)
            if light_iter != first_iter:
                raise ValueError("Static and environment checkpoint iterations differ")
            env_map.restore(light_params, args)

    paper_sgcm = None
    if paper_method and args.get("paper_sgcm_enabled", False):
        from scene.paper_sgcm import create_sgcm_controller
        paper_sgcm = create_sgcm_controller(args.source_path, args, training_views=scene.getTrainCameras(), gaussians=gaussians)

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None

    ema_dict_for_log = defaultdict(int)
    if paper_method and args.start_checkpoint:
        from utils.paper_checkpoint import restore_training_state
        state_path = os.path.join(os.path.dirname(args.start_checkpoint), f"training_state{first_iter}.pth")
        viewpoint_stack, saved_ema = restore_training_state(
            state_path, first_iter, scene,
            {"static": gaussians, "dynamic": dynamic_gaussians, "environment": env_map},
            args, sgcm=paper_sgcm)
        ema_dict_for_log.update(saved_ema)
    progress_bar = tqdm(range(first_iter + 1, args.iterations + 1), desc="Training progress")

    for iteration in progress_bar:
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        if dynamic_gaussians is not None:
            dynamic_gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % args.sh_increase_interval == 0:
            gaussians.oneupSHdegree()
            if dynamic_gaussians is not None:
                dynamic_gaussians.oneupSHdegree()
        viewpoint_cam, time_shift = scene.hybrid_sampler.select(scene.getTrainCameras(), args.lambda_self_supervision, scene.time_interval)
        # render v and t scale map
        v = gaussians.get_inst_velocity
        t_scale = gaussians.get_scaling_t.clamp_max(2)
        other = [t_scale, v]
        # other = [t_scale, v]
        if dynamic_gaussians is not None:
            dynamic_v = dynamic_gaussians.get_inst_velocity
            dynamic_t_scale = dynamic_gaussians.get_scaling_t.clamp_max(2)
            dynamic_other = [dynamic_t_scale, dynamic_v]
        else:
            dynamic_other = []

        evidence, markers = _ownership.evidence(viewpoint_cam)
        render_pkg = _ownership.draw(viewpoint_cam, args, gaussians, dynamic_gaussians,
                                     background, env_map, other, dynamic_other,
                                     shift=time_shift, training=True, markers=markers)

        gt_image = viewpoint_cam.original_image.cuda()
        image = render_pkg["render"]
        depth = render_pkg["depth"]
        alpha = render_pkg["alpha"]
        # object = render_pkg["render_objects"]
        visibility_filter = render_pkg["radii"] > 0
        radii = render_pkg["radii"]
        if dynamic_gaussians is not None:
            gt_dynamic_image = viewpoint_cam.dynamic_image.cuda() if viewpoint_cam.dynamic_image is not None else None
            dynamic_image = render_pkg["render_d"]
            static_image = render_pkg["render_s"]
            static_alpha = render_pkg["static_alpha"]
            dynamic_alpha = render_pkg["dynamic_alpha"]
            static_feature = render_pkg['static_feature'] / (static_alpha if paper_method else alpha).clamp_min(EPS)
            static_t_map = static_feature[0:1]
            static_v_map = static_feature[1:]
            if paper_method:
                dynamic_mask = load_dynamic_mask(args.source_path, viewpoint_cam)
                gt_dynamic_image = gt_image * dynamic_mask
            else:
                dynamic_mask = convert_to_binary_mask(viewpoint_cam.dynamic_image.cuda())
            dynamic_image = extract_object_pixels(dynamic_image, dynamic_mask)
            dynamic_image_1 = extract_object_pixels(image, dynamic_mask)
            gt_dynamic_image = extract_object_pixels(gt_dynamic_image, dynamic_mask)
            bg_color_from_envmap_s = render_pkg["env_map_s"]
            bg_color_from_envmap = render_pkg["env_map"]
            # dynamic_alpha = extract_object_pixels(dynamic_alpha, dynamic_mask)

        log_dict = {}
        feature = render_pkg['feature'] / alpha.clamp_min(EPS)
        t_map = feature[0:1]
        v_map = feature[1:]
        if dynamic_gaussians is None:
            # The only branch is the static branch; retain its regularizers.
            static_t_map = t_map
            static_v_map = v_map

        sky_mask = viewpoint_cam.sky_mask.cuda() if viewpoint_cam.sky_mask is not None else torch.zeros_like(alpha, dtype=torch.bool)

        sky_depth = 900
        depth = depth / alpha.clamp_min(EPS)

        # gt_obj = viewpoint_cam.objects.cuda().long() if viewpoint_cam.objects is not None else None
        # print(gt_obj.max())
        # torch.backends.cudnn.enabled = False
        # logits = classifier(object)
        # loss_obj = cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze().mean()
        # loss_obj = loss_obj / torch.log(torch.tensor(128))

        if env_map is not None:
            if args.depth_blend_mode == 0:  # harmonic mean
                depth = 1 / (alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth).clamp_min(EPS)
            elif args.depth_blend_mode == 1:
                depth = alpha * depth + (1 - alpha) * sky_depth

        loss_l1 = F.l1_loss(image, gt_image)
        log_dict['loss_l1'] = loss_l1.item()
        loss_ssim = 1.0 - ssim(image, gt_image)
        log_dict['loss_ssim'] = loss_ssim.item()
        sky_mask_1 = convert_to_binary_mask(viewpoint_cam.sky_mask.long().cuda())
        sky_truth = extract_object_pixels(gt_image, sky_mask_1)
        if paper_method:
            if not args.get("hybrid_method", False) or dynamic_gaussians is None:
                raise ValueError("此副本只用于含动态分支的hybrid30k实验")
            from utils.hybrid30k_runtime import record_step
            terms = _ownership.terms(render_pkg, evidence, gt_image, viewpoint_cam.pts_depth,
                                     sky_mask, dynamic_mask, args, iteration)
            if iteration in (1, 1000):
                _ownership.audit_gradients(terms, gaussians, dynamic_gaussians, render_pkg,
                                           args, iteration)
            loss = sum(terms.values())
            log_dict.update({name: value.item() for name, value in terms.items()})
            log_dict['loss_instance_total'] = loss.item()
            record_step(args, iteration, viewpoint_cam, time_shift, gaussians, dynamic_gaussians, log_dict)
        else:
            if dynamic_gaussians is not None:
                if gt_dynamic_image.mean().max() > 0:
                    static_image_1 = extract_object_pixels(static_image, 1 - dynamic_mask)
                    static_truth = extract_object_pixels(gt_image, 1 - dynamic_mask)
                    sky_image = extract_object_pixels(static_image, sky_mask_1)
                    loss_ssim_d = 1.0 - ssim(dynamic_image, gt_dynamic_image)
                    loss_ssim_sky = 1.0 - ssim(sky_image, sky_truth)
                    loss_ssim_static = 1.0 - ssim(static_image_1, static_truth)
                    loss_l1_d = F.l1_loss(dynamic_image, gt_dynamic_image)
                    loss_l1_sky = F.l1_loss(sky_image, sky_truth)
                    loss_l1_static = F.l1_loss(static_image_1, static_truth)
                    loss_opacity = calculate_weighted_loss(static_alpha, gt_dynamic_image, 0.00001)
                    loss_v = calculate_weighted_loss(static_v_map, gt_dynamic_image, 0.00001)
                    # print(loss_dynamic)
                    loss = ((1.0 - args.lambda_dssim) * loss_l1 + args.lambda_dssim * loss_ssim
                            + 5 * ((1.0 - args.lambda_dssim) * loss_l1_d + args.lambda_dssim * loss_ssim_d)
                            + 5 * ((1.0 - args.lambda_dssim) * loss_l1_sky + args.lambda_dssim * loss_ssim_sky)
                            + ((1.0 - args.lambda_dssim) * loss_l1_static + args.lambda_dssim * loss_ssim_static))
                    # loss += loss_opacity
                    # loss += loss_v
                else:
                    sky_image = extract_object_pixels(image, sky_mask_1)
                    loss_ssim_sky = 1.0 - ssim(sky_image, sky_truth)
                    loss_l1_sky = F.l1_loss(sky_image, sky_truth)
                    loss = ((1.0 - args.lambda_dssim) * loss_l1 + args.lambda_dssim * loss_ssim +
                            5 * ((1.0 - args.lambda_dssim) * loss_l1_sky + args.lambda_dssim * loss_ssim_sky))
            else:
                sky_image = extract_object_pixels(image, sky_mask_1)
                loss_ssim_sky = 1.0 - ssim(sky_image, sky_truth)
                loss_l1_sky = F.l1_loss(sky_image, sky_truth)
                loss = ((1.0 - args.lambda_dssim) * loss_l1 + args.lambda_dssim * loss_ssim +
                        5 * ((1.0 - args.lambda_dssim) * loss_l1_sky + args.lambda_dssim * loss_ssim_sky))
            # alpha_1 = alpha.clone().detach()
            # loss_opacity = F.mse_loss(static_alpha, alpha_1)
            # total_content_loss = tv_loss(static_image, dynamic_image, 10)
            # print(total_content_loss)
            # loss += total_content_loss
            # if iteration % args.reg3d_interval == 0:
            #     # regularize at certain intervals
            #     logits3d = classifier(gaussians._objects_dc.permute(2,0,1))
            #     prob_obj3d = torch.softmax(logits3d,dim=0).squeeze().permute(1,0)
            #     loss_obj_3d = loss_cls_3d(gaussians._xyz.squeeze().detach(), prob_obj3d, args.reg3d_k, args.reg3d_lambda_val, args.reg3d_max_points, args.reg3d_sample_size)
            #     loss += loss_obj + loss_obj_3d
            #     log_dict['loss_obj'] = (loss_obj + loss_obj_3d).item()
            # else:
            #     loss += loss_obj
            #     log_dict['loss_obj'] = loss_obj.item()

            if args.lambda_lidar > 0:
                assert viewpoint_cam.pts_depth is not None
                pts_depth = viewpoint_cam.pts_depth.cuda()

                mask = pts_depth > 0
                loss_lidar = torch.abs(1 / (pts_depth[mask] + 1e-5) - 1 / (depth[mask] + 1e-5)).mean()
                if args.lidar_decay > 0:
                    iter_decay = np.exp(-iteration / 8000 * args.lidar_decay)
                else:
                    iter_decay = 1
                log_dict['loss_lidar'] = loss_lidar.item()
                loss += iter_decay * args.lambda_lidar * loss_lidar

            if args.lambda_t_reg > 0:
                loss_t_reg = 1/torch.abs(static_t_map).mean()
                log_dict['loss_t_reg'] = loss_t_reg.item()
                loss += args.lambda_t_reg * loss_t_reg

            if args.lambda_v_reg > 0:
                loss_v_reg = torch.abs(static_v_map).mean()
                log_dict['loss_v_reg'] = loss_v_reg.item()
                loss += args.lambda_v_reg * loss_v_reg

            if args.lambda_inv_depth > 0:
                inverse_depth = 1 / (depth + 1e-5)
                loss_inv_depth = kornia.losses.inverse_depth_smoothness_loss(inverse_depth[None], gt_image[None])
                log_dict['loss_inv_depth'] = loss_inv_depth.item()
                loss = loss + args.lambda_inv_depth * loss_inv_depth

            if args.lambda_v_smooth > 0:
                loss_v_smooth = kornia.losses.inverse_depth_smoothness_loss(v_map[None], gt_image[None])
                log_dict['loss_v_smooth'] = loss_v_smooth.item()
                loss = loss + args.lambda_v_smooth * loss_v_smooth

            if args.lambda_sky_opa > 0:
                o = alpha.clamp(1e-6, 1-1e-6)
                sky = sky_mask.float()
                loss_sky_opa = (-sky * torch.log(1 - o)).mean()
                log_dict['loss_sky_opa'] = loss_sky_opa.item()
                loss = loss + args.lambda_sky_opa * loss_sky_opa

            if args.lambda_opacity_entropy > 0:
                o = alpha.clamp(1e-6, 1 - 1e-6)
                loss_opacity_entropy = -(o*torch.log(o)).mean()
                log_dict['loss_opacity_entropy'] = loss_opacity_entropy.item()
                loss = loss + args.lambda_opacity_entropy * loss_opacity_entropy

            if args.lambda_opacity_entropy > 0 and args.including_dynamic:
                o = dynamic_alpha.clamp(1e-6, 1 - 1e-6)
                target = torch.ones_like(dynamic_alpha)  # 目标值为1
                target = target * dynamic_mask  # 目标值为1
                loss_convergence_to_one = nn.MSELoss()(dynamic_alpha, target)
                dynamic_alpha = dynamic_alpha * dynamic_mask
                # loss_convergence_to_one = -(dynamic_mask * (1-o) * torch.log(o)).sum() / dynamic_mask.sum()
                # loss_convergence_to_one += nn.MSELoss()(static_alpha_1, target_1)
                # log_dict['loss_dynamic_opacity_entropy'] = loss_dynamic_opacity_entropy.item()
                loss = loss + loss_convergence_to_one * 100

        # torch.autograd.set_detect_anomaly(True)
        loss.backward()
        log_dict['loss'] = loss.item()

        iter_end.record()

        with torch.no_grad():
            psnr_for_log = psnr(image, gt_image).double()
            log_dict["psnr"] = psnr_for_log
            for key in ['loss', "loss_l1", "psnr"]:
                ema_dict_for_log[key] = 0.4 * log_dict[key] + 0.6 * ema_dict_for_log[key]

            if iteration % 10 == 0:
                postfix = {k[5:] if k.startswith("loss_") else k:f"{ema_dict_for_log[k]:.{5}f}" for k, v in ema_dict_for_log.items()}
                postfix["scale"] = scene.resolution_scales[scene.scale_index]
                progress_bar.set_postfix(postfix)
            if iteration % 1000 == 0:
                print(gaussians.get_xyz.shape[0])
                if dynamic_gaussians is not None:
                    print(dynamic_gaussians.get_xyz.shape[0])

                # print(log_dict['loss_lidar'], log_dict['loss_v_reg'], log_dict['loss_inv_depth']
                #       , log_dict['loss_sky_opa'], log_dict['loss_opacity_entropy'])
                # print(log_dict['loss_t_reg'])
                # print(log_dict['loss_v_reg'])
                # print(log_dict['loss_inv_depth'])
                # print(log_dict['loss_v_smooth'])
                # print(log_dict['loss_sky_opa'])
                # print(log_dict['loss_opacity_entropy'])
            log_dict['iter_time'] = iter_start.elapsed_time(iter_end)
            log_dict['total_points'] = gaussians.get_xyz.shape[0]
            # Log and save
            if not paper_method:
                complete_eval(tb_writer, iteration, args.test_iterations, scene, render, (args, background),
                              log_dict, env_map=env_map, including_dynamic=args.including_dynamic)

            # Densification
            if iteration > args.densify_until_iter * args.time_split_frac:
                gaussians.no_time_split = False

            end_idx = gaussians.get_xyz.shape[0]
            static_xy = render_pkg['static_growth']
            static_radii = render_pkg['static_radii']
            static_t_grad = gaussians._t.grad
            gaussians._t.grad = render_pkg['static_growth_t'].grad
            static_time_parameter = gaussians._t
            try:
                densification_and_optimization(gaussians, args, iteration, static_xy,
                                               static_radii > 0, scene, static_radii, dynamic=False)
            finally:
                # Native topology changes replace Parameter objects and, as in
                # the control, those new objects receive no old step gradient.
                if gaussians._t is static_time_parameter:
                    gaussians._t.grad = static_t_grad
            dynamic_radii = render_pkg['dynamic_radii']
            densification_and_optimization(dynamic_gaussians, args, iteration,
                                           render_pkg['dynamic_growth'], dynamic_radii > 0,
                                           scene, dynamic_radii, dynamic=True)

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            if dynamic_gaussians is not None:
                dynamic_gaussians.optimizer.step()
                dynamic_gaussians.optimizer.zero_grad(set_to_none=True)
            # cls_optimizer.step()
            # cls_optimizer.zero_grad()
            if env_map is not None and iteration < args.env_optimize_until:
                env_map.optimizer.step()
                env_map.optimizer.zero_grad(set_to_none=True)
            if paper_sgcm is not None:
                # Controller uses an offset schedule: no generic topology change
                # on these iterations, so rendered static radii still align.
                sgcm_statistics = paper_sgcm.refine(gaussians, radii[:end_idx], iteration, visibility_filter=visibility_filter[:end_idx])
                if sgcm_statistics:
                    print("SGCM_STATS " + json.dumps(sgcm_statistics), flush=True)
                    if tb_writer:
                        for name, value in sgcm_statistics.items():
                            if isinstance(value, (int, float)):
                                tb_writer.add_scalar("sgcm/" + name, value, iteration)
            torch.cuda.empty_cache()
            if dynamic_gaussians is not None:
                if iteration % args.vis_step == 0 or iteration == 1:
                    other_img = []
                    feature = render_pkg['feature'] / alpha.clamp_min(1e-5)
                    t_map = feature[0:1]
                    v_map = feature[1:]
                    v_norm_map = v_map.norm(dim=0, keepdim=True)
                    static_v_norm_map = static_v_map.norm(dim=0, keepdim=True)
                    et_color = visualize_depth(t_map, near=0.01, far=1)
                    static_v_color = visualize_depth(static_v_norm_map, near=0.01, far=1)
                    static_et_color = visualize_depth(static_t_map, near=0.01, far=1)

                    v_color = visualize_depth(v_norm_map, near=0.01, far=1)
                    dynamic_opacity_color = visualize_depth(dynamic_alpha, near=0.01, far=1)
                    static_opacity_color = visualize_depth(static_alpha, near=0.01, far=1)
                    opacity_color = visualize_depth(alpha, near=0.01, far=1)
                    other_img.append(et_color)
                    other_img.append(static_v_color)
                    other_img.append(v_color)
                    other_img.append(dynamic_opacity_color)
                    other_img.append(static_opacity_color)
                    other_img.append(opacity_color)

                    if viewpoint_cam.pts_depth is not None:
                        depth_vis = visualize_depth(depth)
                        other_img.append(depth_vis)
                        other_img.append(bg_color_from_envmap)
                        other_img.append(bg_color_from_envmap_s)
                        pts_depth_vis = visualize_depth(viewpoint_cam.pts_depth)
                        other_img.append(pts_depth_vis)
                    grid = make_grid([
                        image,
                        gt_image,
                        dynamic_image,
                        static_image,
                        static_et_color,
                    ] + other_img, nrow=4)

                    save_image(grid, os.path.join(vis_path, f"{iteration:05d}_{viewpoint_cam.colmap_id:03d}.png"))
            else:
                if iteration % args.vis_step == 0 or iteration == 1:
                    other_img = []
                    feature = render_pkg['feature'] / alpha.clamp_min(1e-5)
                    t_map = feature[0:1]
                    v_map = feature[1:]
                    v_norm_map = v_map.norm(dim=0, keepdim=True)
                    opacity_color = visualize_depth(alpha, near=0.01, far=1)
                    et_color = visualize_depth(t_map, near=0.01, far=1)

                    v_color = visualize_depth(v_norm_map, near=0.01, far=1)
                    other_img.append(et_color)
                    other_img.append(v_color)

                    if viewpoint_cam.pts_depth is not None:
                        pts_depth_vis = visualize_depth(viewpoint_cam.pts_depth)
                        other_img.append(pts_depth_vis)

                    grid = make_grid([
                                         image,
                                         gt_image,
                                         alpha.repeat(3, 1, 1),
                                         torch.logical_not(sky_mask[:1]).float().repeat(3, 1, 1),
                                         visualize_depth(depth),
                                     ] + other_img, nrow=4)

                    save_image(grid, os.path.join(vis_path, f"{iteration:05d}_{viewpoint_cam.colmap_id:03d}.png"))
            if iteration % args.scale_increase_interval == 0:
                scene.upScale()

            if paper_method:
                from utils.paper_checkpoint import (save_components, capture_training_state,
                                                    publish_checkpoint_event)
                # In the paper path, either explicit save schedule requests a
                # full resumable bundle. PLY-only export is not a checkpoint.
                checkpoint_due = iteration in set(args.checkpoint_iterations) | set(args.save_iterations)
                models_for_checkpoint = {"static": gaussians, "dynamic": dynamic_gaussians,
                                         "environment": env_map}
                components = save_components(scene.model_path, iteration, models_for_checkpoint) if checkpoint_due else None
                eval_iterations = set(args.test_iterations)
                if checkpoint_due and args.get("paper_eval_train_each_checkpoint", False):
                    eval_iterations.add(iteration)
                # This is after optimizer/SGCM/scale changes. Eval and serialized
                # components therefore refer to exactly the same parameters.
                evaluations = complete_eval(tb_writer, iteration, eval_iterations, scene, render,
                                            (args, background), log_dict, env_map=env_map,
                                            including_dynamic=args.including_dynamic)
                if checkpoint_due:
                    continuation = capture_training_state(iteration, scene, viewpoint_stack,
                                                          models_for_checkpoint, args,
                                                          sgcm=paper_sgcm, ema=ema_dict_for_log)
                    publish_checkpoint_event(scene.model_path, iteration, components,
                                             continuation, evaluations, args, scene)
            elif iteration in args.checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                if dynamic_gaussians is not None:
                    torch.save((dynamic_gaussians.capture(), iteration), scene.model_path + "/dynamic_chkpnt" + str(iteration) + ".pth")
                if env_map is not None:
                    torch.save((env_map.capture(), iteration), scene.model_path + "/env_light_chkpnt" + str(iteration) + ".pth")
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                # scene.save(iteration, viewpoint_cam.timestamp, time_shift)
                point_cloud_path = os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration))
                makedirs(point_cloud_path, exist_ok=True)
                # torch.save(classifier.state_dict(), os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration),'classifier.pth'))

# def mkdir_p(folder_path):
#     # Creates a directory. equivalent to using mkdir -p on the command line
#     try:
#         makedirs(folder_path)
#     except OSError as exc:  # Python >2.5
#         if exc.errno == EEXIST and path.isdir(folder_path):
#             pass
#         else:
#             raisepython train.py --config configs/waymo_reconstruction.yaml source_path=/mnt/data/data/waymo_scenes/0147030/ model_path=/mnt/data/data/waymo_scenes/0147030/output/eval_output_4 including_dynamic=true
