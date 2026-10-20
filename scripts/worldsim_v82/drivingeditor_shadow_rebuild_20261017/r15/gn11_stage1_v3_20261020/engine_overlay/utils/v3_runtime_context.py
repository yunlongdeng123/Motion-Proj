"""GN11 v3 训练适配器：真实联合监督、深度顺序、动态泄漏与分支增密。"""

import functools
import json
from pathlib import Path

import numpy as np
import torch

from utils.v3_multiview_road import MultiViewRoad
from utils.v3_road_contract import ROAD_POLICY, ROAD_VERSION
from utils.v3_densification import accumulate_max_screen_gradient
from utils.v3_supervision_diagnostics import (
    center_in_boxes, supervision_metrics, sampled_front_contributions,
)
from utils.v3_regularizer_contract import (
    TEMPORAL_POLICY, TEMPORAL_VERSION, validate_v1_regularizers,
)
from utils.v3_objectives import (
    STATIC_BODY_RGB_GROUPS,
    body_joint_rgb_loss,
    dynamic_color_leak_losses,
    observed_rgb_loss,
    route_depth,
    static_front_occlusion_loss,
    vehicle_volume_opacity,
)


V3_METHOD_VERSION = "GN11_STAGE1_V3_20261020"
SUPERVISION_VERSION = "V3_RESPONSIBILITY_R2_20261020"
V3_POLICY = {
    "supervision_version": SUPERVISION_VERSION,
    "body_l1_normalization": "v1_full_image",
    "body_static_temporal_gradient": False,
    "body_static_road_gradient": False,
    "screen_gradient_accounting": "preserve_retained_buffers_then_add_routed_once",
    "body_joint_rgb_weight": 1.0,
    "static_front_occlusion_weight": 0.1,
    "static_front_depth_margin_ratio": 0.01,
    "static_front_temperature_ratio": 0.01,
    "dynamic_color_leak_weight": 1.0,
    "dynamic_color_ratio_weight": 0.1,
    "dynamic_boundary_ignore_px": 2,
    "dynamic_leak_top_fraction": 0.01,
    "dynamic_leak_top_weight": 0.1,
    "dynamic_radiance_ratio_cap": 4.0,
    "dynamic_ratio_slack": 1.0 / 255.0,
    "densification_fusion": "per_gaussian_max_branch_vs_joint_single_count",
}
ARM_ROAD_WEIGHTS = {
    "full": {
        "observed_rgb": 1.0,
        "prior_rgb": 0.1,
        "observed_depth": 0.1,
        "prior_depth": 0.01,
    },
    "real_only": {
        "observed_rgb": 1.0,
        "prior_rgb": 0.0,
        "observed_depth": 0.1,
        "prior_depth": 0.0,
    },
    "prior_only": {
        "observed_rgb": 0.0,
        "prior_rgb": 0.1,
        "observed_depth": 0.0,
        "prior_depth": 0.01,
    },
    "ownership_only": {
        "observed_rgb": 0.0,
        "prior_rgb": 0.0,
        "observed_depth": 0.0,
        "prior_depth": 0.0,
    },
}

ACTIVE_CONTEXT = None


def _json_clone(value):
    from omegaconf import OmegaConf
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    return json.loads(json.dumps(value))


def _append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")


def _optimizer_parameters(model, allowed_groups=None):
    result = []
    for group in model.optimizer.param_groups:
        name = group.get("name")
        if allowed_groups is not None and name not in allowed_groups:
            continue
        result.extend(parameter for parameter in group["params"] if parameter.requires_grad)
    return result


def _autograd(loss, inputs, screen_proxies=()):
    if not inputs or not loss.requires_grad:
        return [None] * len(inputs)
    # autograd.grad 虽不累积叶子参数 .grad，仍会触发非叶子 retain_grad。
    # 隔离所有分支代理，不只隔离本次 inputs 中的那个代理；保留先前梯度。
    saved = [(proxy, None if proxy.grad is None else proxy.grad.detach().clone())
             for proxy in screen_proxies]
    try:
        return torch.autograd.grad(loss, inputs, retain_graph=True, allow_unused=True)
    finally:
        for proxy, gradient in saved:
            proxy.grad = gradient


def _exclude_road_rows(gradient, road_mask):
    if gradient is None or road_mask is None:
        return gradient
    if gradient.ndim == 0 or gradient.shape[0] != road_mask.numel():
        raise RuntimeError("body static gradient / road topology mismatch")
    result = gradient.clone()
    result[road_mask] = 0
    return result


def _accumulate_parameter_gradients(parameters, gradients, label):
    with torch.no_grad():
        for parameter, gradient in zip(parameters, gradients):
            if gradient is None:
                continue
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f"non-finite {label} parameter gradient")
            if parameter.grad is None:
                parameter.grad = gradient.detach().clone()
            else:
                parameter.grad.add_(gradient)


def _accumulate_screen_gradient(proxy, gradient, label):
    if gradient is None:
        return
    if not torch.isfinite(gradient).all():
        raise FloatingPointError(f"non-finite {label} screen gradient")
    if proxy.grad is None:
        proxy.grad = gradient.detach().clone()
    else:
        proxy.grad.add_(gradient)


class V3Context:
    def __init__(self, args, scene, environment):
        bundle = Path(args["v3_bundle"])
        self.root = bundle.parent
        self.meta = json.loads(bundle.read_text(encoding="utf-8"))
        if self.meta["status"] != "CPU_MULTIVIEW_ROAD_INPUT_VERIFIED" or self.meta.get('version') != ROAD_VERSION:
            raise RuntimeError("real inputs not bound; synthetic tests cannot authorize training")
        if int(args.iterations) != 30000 or args.start_checkpoint:
            raise RuntimeError("v3 only authorizes fresh first-stage30K")
        if not all(bool(args.get(key, False)) for key in (
            "including_dynamic", "paper_method", "hybrid_method"
        )):
            raise RuntimeError("v3 requires the native hybrid static/dynamic training path")
        if set(self.meta["actors"]) != {6, 37, 5, 26, 18}:
            raise RuntimeError("wrong actor set")
        self.frames = self.meta["frames"]
        if set(self.frames) != {f"0008{i:03d}" for i in range(89, 138)}:
            raise RuntimeError("49 real frame contract failed")
        self.arm = str(args.get("v3_arm", ""))
        if self.arm not in ARM_ROAD_WEIGHTS:
            raise RuntimeError("unknown v3 comparison arm")
        supplied_policy = _json_clone(args.get("v3_policy", {}))
        if supplied_policy != V3_POLICY:
            raise RuntimeError("v3 policy differs from reviewed fixed policy")
        if float(args.get("hybrid_lambda_dynamic_opacity", -1)) != 100.0:
            raise RuntimeError("full raw dynamic-mask opacity MSE weight100 must remain enabled")
        if bool(args.get("white_background", False)):
            raise RuntimeError("dynamic_premultiplied_rgb requires the fixed black raster background")
        self.policy = supplied_policy
        validate_v1_regularizers(args)
        if _json_clone(args.get("v3_temporal_policy", {})) != TEMPORAL_POLICY:
            raise RuntimeError("V1 background regularizer policy mismatch")
        self.config = _json_clone(self.meta["config"])
        self.config["road"]["objective_weights"] = _json_clone(ARM_ROAD_WEIGHTS[self.arm])
        self.config["training"]["static_temporal_parameters"] = (
            "NATIVE_PVG_TRAINABLE_WITH_UNMODIFIED_V1_REGULARIZERS"
        )
        self.cache = {}

        from utils.hybrid30k_runtime import camera_id
        camera_ids = {Path(camera_id(camera)).stem for camera in scene.getTrainCameras()}
        if camera_ids != set(self.frames):
            raise RuntimeError("actual camera bundle mismatch")
        for camera in scene.getTrainCameras():
            frame_id = Path(camera_id(camera)).stem
            actual = camera.world_view_transform.detach().cpu().numpy().T
            expected = np.asarray(self.frames[frame_id]["world_to_camera"])
            model_from_world = np.asarray(self.meta["world_to_model"])
            derived = actual @ model_from_world
            derived[:3, :] /= float(self.meta["meter_to_model"])
            if not np.allclose(derived, expected, atol=2e-4, rtol=2e-4):
                raise RuntimeError("world/model camera transform mismatch: " + frame_id)

        if self.meta['policy'] != ROAD_POLICY:
            raise RuntimeError('road multiview policy mismatch')
        self.road = MultiViewRoad(bundle, scene, args, args.model_path)

        self.log = Path(args.model_path) / "v3_objectives.jsonl"
        self.densification_log = Path(args.model_path) / "v3_densification.jsonl"
        self.gate_only = bool(args.get("v3_zero_update_gate", False))
        self.models = [scene.gaussians, scene.dynamic_gaussians, environment]
        self.optimizer_steps = {}
        for name, model in zip(["static", "dynamic", "environment"], self.models):
            if model is None:
                continue
            self.optimizer_steps[name] = 0
            original_step = model.optimizer.step

            def counted_step(*positional, _name=name, _original=original_step, **keyword):
                result = _original(*positional, **keyword)
                self.optimizer_steps[_name] += 1
                return result

            model.optimizer.step = functools.wraps(original_step)(counted_step)
        self.gate_details = {}
        self.output = Path(args.model_path)
        global ACTIVE_CONTEXT
        ACTIVE_CONTEXT = self

    def select_gate_camera(self, scene, camera, time_shift):
        if not self.gate_only:
            return camera, time_shift
        return next(
            candidate for candidate in scene.getTrainCameras()
            if self.frame(candidate)[0] == "0008089"
        ), None

    def gate_forward(self, scene, render, camera, args, background, environment):
        if not self.gate_only:
            return
        import copy

        static = scene.gaussians
        times = [
            min(float(candidate.timestamp) for candidate in scene.getTrainCameras()),
            max(float(candidate.timestamp) for candidate in scene.getTrainCameras()),
        ]
        images = []
        with torch.no_grad():
            if not all(getattr(static, field).requires_grad for field in (
                "_t", "_scaling_t", "_velocity"
            )):
                raise RuntimeError("background temporal freedom was disabled")
            groups = {group["name"]: group for group in static.optimizer.param_groups}
            if not all(groups[name]["lr"] > 0 for name in ("t", "scaling_t", "velocity")):
                raise RuntimeError("background temporal learning rate was zeroed")
            for timestamp in times:
                angular = 2 * np.pi / static.T
                expected_xyz = static._xyz + static._velocity * (
                    torch.sin((timestamp - static._t) * angular) / angular
                )
                expected_alpha = torch.exp(
                    -0.5 * (static._t - timestamp).square() / static.get_scaling_t.square()
                )
                expected_xyz = torch.where(static.road_unit_mask[:, None], static._xyz, expected_xyz)
                expected_alpha = torch.where(static.road_unit_mask[:, None], torch.ones_like(expected_alpha), expected_alpha)
                if (not torch.allclose(static.get_xyz_SHM(timestamp), expected_xyz)
                        or not torch.allclose(static.get_marginal_t(timestamp), expected_alpha)):
                    raise RuntimeError("background no longer follows the native PVG forward")
                probe = copy.copy(camera)
                probe.timestamp = timestamp
                package = render(
                    probe,
                    static,
                    scene.dynamic_gaussians,
                    args,
                    background,
                    env_map=environment,
                    other=[static.get_scaling_t.clamp_max(2), static.get_inst_velocity],
                    dynamic_other=[
                        scene.dynamic_gaussians.get_scaling_t.clamp_max(2),
                        scene.dynamic_gaussians.get_inst_velocity,
                    ],
                    time_shift=None,
                    is_training=False,
                )
                required = {
                    "dynamic_premultiplied_rgb",
                    "static_viewspace_points",
                    "dynamic_viewspace_points",
                    "static_visibility_filter",
                    "dynamic_visibility_filter",
                    "background_means3D",
                    "background_opacities",
                }
                if not required.issubset(package):
                    raise RuntimeError("v3 renderer contract is incomplete")
                if not torch.equal(package["render_d"], package["dynamic_premultiplied_rgb"]):
                    raise RuntimeError("evaluation dynamic render must equal pre-EnvMap dynamic color")
                images.append(package["render_s"].detach().clone())
                del package
            difference = float((images[0] - images[1]).abs().max())
            if not np.isfinite(difference):
                raise RuntimeError("background time diagnostic is non-finite")
        self.gate_details.update(
            method_version=V3_METHOD_VERSION,
            arm=self.arm,
            v3_policy=self.policy,
            actual_road_weights=self.config["road"]["objective_weights"],
            static_fixed_camera_max_RGB_delta=difference,
            static_times_tested=times,
            background_temporal_version=TEMPORAL_VERSION,
            background_temporal_policy=TEMPORAL_POLICY,
            background_temporal_parameters_trainable=True,
            fixed_camera_time_variation_is_allowed=True,
            gate_raw_frame=89,
            road_version=ROAD_VERSION,
            road_policy=ROAD_POLICY,
            road_unit_count=self.road.seed_count,
            road_world_ids_are_time_shared=True,
        )

    def frame(self, camera):
        from utils.hybrid30k_runtime import camera_id

        frame_id = Path(camera_id(camera)).stem
        return frame_id, self.frames[frame_id]

    def partition_state(self, state, camera, package, target, mask, depth, args, ssim_fn):
        from utils.paper_losses import sparse_inverse_depth_loss
        import kornia

        state = dict(state)
        state["global_rgb"] = observed_rgb_loss(
            package["render"], target, mask, ssim_fn, args.lambda_dssim
        )
        routed_depth = route_depth(
            depth, package["dynamic_depth"], package["dynamic_alpha"], mask
        )
        state["depth"] = sparse_inverse_depth_loss(
            routed_depth, camera.pts_depth.to(depth.device)
        )
        state["inverse_depth_smoothness"] = (
            kornia.losses.inverse_depth_smoothness_loss(
                (1 / (routed_depth + 1e-5))[None], target[None]
            )
        )
        state["alpha"] = torch.where(mask.bool(), package["dynamic_alpha"], package["alpha"])
        joint_velocity = package["feature"][1:] / package["alpha"].clamp_min(1e-5)
        dynamic_velocity = (
            package["dynamic_feature"][1:] / package["dynamic_alpha"].clamp_min(1e-5)
        )
        routed_velocity = torch.where(mask.bool(), dynamic_velocity, joint_velocity)
        state["velocity_smoothness"] = kornia.losses.inverse_depth_smoothness_loss(
            routed_velocity[None], target[None]
        )
        return state

    def objectives(self, camera, package, static, mask, target, args, ssim_fn, iteration):
        frame_id, frame = self.frame(camera)

        def tensor(value):
            return torch.as_tensor(value, device=static._xyz.device, dtype=static._xyz.dtype)

        boxes = frame["selected_boxes_model"]
        centers = tensor([box["center"] for box in boxes]).reshape(-1, 3)
        axes = tensor([box["axes"] for box in boxes]).reshape(-1, 3, 3)
        half_sizes = tensor([box["half"] for box in boxes]).reshape(-1, 3)
        plane = self.meta["ground_plane_model"]
        scale = float(self.meta["meter_to_model"])
        training = self.config["training"]
        volume = vehicle_volume_opacity(
            package["background_means3D"],
            package["background_opacities"],
            centers,
            axes,
            half_sizes,
            tensor(plane[:3]),
            float(plane[3]),
            training["road_clearance_m"] * scale,
            training["volume_edge_m"] * scale,
        )

        road, road_terms = self.road.objective(static, args, iteration, self.config['road']['objective_weights'])

        body_joint = body_joint_rgb_loss(
            package["render"], target, mask, ssim_fn, args.lambda_dssim,
            self.policy["body_l1_normalization"],
        )
        front, front_metrics = static_front_occlusion_loss(
            package["static_depth"],
            package["static_alpha"],
            camera.pts_depth,
            mask,
            self.policy["static_front_depth_margin_ratio"],
            self.policy["static_front_temperature_ratio"],
        )
        leak_color, leak_ratio, leak_metrics = dynamic_color_leak_losses(
            package["dynamic_premultiplied_rgb"],
            package["dynamic_alpha"],
            mask,
            self.policy["dynamic_boundary_ignore_px"],
            self.policy["dynamic_leak_top_fraction"],
            self.policy["dynamic_leak_top_weight"],
            self.policy["dynamic_radiance_ratio_cap"],
            self.policy["dynamic_ratio_slack"],
        )
        base_extra = (
            training["volume_exclusion_weight"] * volume
            + self.policy["static_front_occlusion_weight"] * front
            + self.policy["dynamic_color_leak_weight"] * leak_color
            + self.policy["dynamic_color_ratio_weight"] * leak_ratio
        )
        routed_body = self.policy["body_joint_rgb_weight"] * body_joint

        if self.gate_only or iteration == 1 or iteration % 100 == 0:
            with torch.no_grad():
                diagnostic_metrics, diagnostic_maps = supervision_metrics(
                    package, target, mask, camera.pts_depth, scale,
                    self.policy['dynamic_boundary_ignore_px'])
            record = {
                "step": iteration,
                "frame": frame_id,
                "arm": self.arm,
                "body_joint_rgb_raw": float(body_joint.detach()),
                "body_joint_rgb_weight": self.policy["body_joint_rgb_weight"],
                "ownership_volume_raw": float(volume.detach()),
                "ownership_volume_weight": training["volume_exclusion_weight"],
                "front_occlusion_raw": float(front.detach()),
                "front_occlusion_weight": self.policy["static_front_occlusion_weight"],
                "dynamic_color_leak_raw": float(leak_color.detach()),
                "dynamic_color_leak_weight": self.policy["dynamic_color_leak_weight"],
                "dynamic_color_ratio_raw": float(leak_ratio.detach()),
                "dynamic_color_ratio_weight": self.policy["dynamic_color_ratio_weight"],
                "full_dynamic_opacity_mse_weight": 100.0,
                "road": road_terms,
                "road_version": ROAD_VERSION,
                "road_geometry_is_common_to_four_arms": True,
                "road_weights": self.config["road"]["objective_weights"],
                "road_total": float(road.detach()),
                "road_updates_dynamic": False,
                "road_updates_environment": False,
                "body_joint_updates_static_color": False,
                "body_joint_updates_environment": False,
                "body_joint_static_groups": sorted(STATIC_BODY_RGB_GROUPS),
                "body_l1_normalization": self.policy["body_l1_normalization"],
                "body_area_fraction": float(mask.float().mean()),
                "body_joint_updates_background_time": False,
                "body_joint_updates_known_road_rows": False,
                "background_temporal_version": TEMPORAL_VERSION,
                "background_temporal_policy": TEMPORAL_POLICY,
                "selective_temporal_objective_enabled": False,
            }
            record.update({key: float(value.detach()) for key, value in front_metrics.items()})
            record.update({key: float(value.detach()) for key, value in leak_metrics.items()})
            record['supervision_diagnostics'] = diagnostic_metrics
            _append_jsonl(self.log, record)
        if self.gate_only or iteration == 1 or iteration % 5000 == 0:
            # 只读节点/准入抽样，发生在当前步更新前。实际开销单列，不改变损失。
            from gaussian_renderer import render_contribution_features
            with torch.no_grad():
                inside = center_in_boxes(package['background_means3D'], centers, axes, half_sizes)
                arrays, sampled_metrics = sampled_front_contributions(
                    package, mask, camera.pts_depth, inside, static.road_unit_mask,
                    lambda flags, joint: render_contribution_features(package, flags, joint),
                    self.policy['static_front_depth_margin_ratio'])
                destination = self.output/'supervision_audit'/f'iteration_{iteration:06d}_{frame_id}'
                destination.mkdir(parents=True, exist_ok=False)
                diagnostic_maps.update(arrays)
                diagnostic_maps.update(joint_rgb=package['render'], target_rgb=target,
                                       dynamic_raw_rgb=package['dynamic_premultiplied_rgb'],
                                       dynamic_alpha=package['dynamic_alpha'], static_alpha=package['static_alpha'])
                np.savez_compressed(destination/'maps.npz', **{
                    key: value.detach().cpu().numpy() for key, value in diagnostic_maps.items()})
                volume_count = int(inside.sum())
                height = package['background_means3D'] @ tensor(plane[:3])+float(plane[3])
                penalizable = inside & (height > training['road_clearance_m']*scale)
                metadata = dict(supervision_version=SUPERVISION_VERSION, iteration=iteration,
                                completed_optimizer_updates=iteration-1, frame=frame_id, arm=self.arm,
                                meter_to_model=scale, body_l1_normalization=self.policy['body_l1_normalization'],
                                static_centers_inside_selected_boxes=volume_count,
                                static_centers_above_road_clearance_inside_boxes=int(penalizable.sum()),
                                static_total_centers=len(inside),
                                diagnostics=diagnostic_metrics, sampled_front=sampled_metrics,
                                sample_depth_units='model units; divide by meter_to_model for metres',
                                depth_maps='opacity-normalized camera-z before sky/environment depth blending',
                                sample_policy='up to14 deterministic evenly spaced valid body LiDAR pixels',
                                front_definition='native center camera-z < measured_depth*(1-margin_ratio), native alpha*T; all geometry retained',
                                static_and_joint_transmittance_reported_separately=True,
                                gradient_or_optimization_target_added=False, quality_accepted=False, human_verdict=None)
                (destination/'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
                if self.gate_only:
                    self.gate_details['sampled_native_front_diagnostic'] = sampled_metrics
                    if sampled_metrics['sampled_pixels'] == 0:
                        raise RuntimeError('CUDA gate has no valid measured body depth for front diagnostic')
        return {
            "base_extra": base_extra,
            "body_joint": routed_body,
            "road": road,
        }

    def backward(self, base_loss, body_joint, road, static, dynamic, package, iteration):
        static_body_parameters = _optimizer_parameters(static, STATIC_BODY_RGB_GROUPS)
        static_all_parameters = _optimizer_parameters(static)
        dynamic_parameters = _optimizer_parameters(dynamic)
        joint_screen = package["viewspace_points"]
        static_screen = package["static_viewspace_points"]
        proxies = [package[key] for key in (
            "viewspace_points", "static_viewspace_points", "dynamic_viewspace_points")]

        body_inputs = static_body_parameters + dynamic_parameters + [joint_screen]
        body_gradients = list(_autograd(body_joint, body_inputs, proxies))
        road_mask = getattr(static, "road_unit_mask", None)
        for index in range(len(static_body_parameters)):
            body_gradients[index] = _exclude_road_rows(body_gradients[index], road_mask)
        # 道路固定容量，本身不增密；也不把车身对道路的屏幕梯度计入需求。
        if body_gradients[-1] is not None and road_mask is not None:
            body_gradients[-1] = body_gradients[-1].clone()
            static_count = road_mask.numel()
            body_gradients[-1][:static_count][road_mask] = 0
        road_inputs = static_all_parameters + [static_screen]
        road_gradients = _autograd(road, road_inputs, proxies)

        base_loss.backward()
        body_parameter_count = len(static_body_parameters) + len(dynamic_parameters)
        _accumulate_parameter_gradients(
            static_body_parameters + dynamic_parameters,
            body_gradients[:body_parameter_count],
            "body-joint-routed",
        )
        _accumulate_screen_gradient(
            joint_screen, body_gradients[-1], "body-joint-routed"
        )
        _accumulate_parameter_gradients(
            static_all_parameters, road_gradients[:-1], "road-static-only"
        )
        _accumulate_screen_gradient(
            static_screen, road_gradients[-1], "road-static-only"
        )

        if self.gate_only:
            counts = []
            for model in self.models:
                if model is None:
                    continue
                parameters = [
                    parameter
                    for group in model.optimizer.param_groups
                    for parameter in group["params"]
                    if parameter.grad is not None
                ]
                if not parameters or not all(torch.isfinite(parameter.grad).all() for parameter in parameters):
                    raise FloatingPointError("GPU backward contract failed")
                counts.append(len(parameters))
            for field in ["_t", "_scaling_t", "_velocity"]:
                parameter = getattr(static, field)
                if (not parameter.requires_grad or parameter.grad is None
                        or not torch.isfinite(parameter.grad).all()):
                    raise RuntimeError("background temporal gradient is missing or non-finite")
            for key in ["viewspace_points", "static_viewspace_points", "dynamic_viewspace_points"]:
                gradient = package[key].grad
                if gradient is None or not torch.isfinite(gradient).all():
                    raise RuntimeError("missing finite v3 screen gradient: " + key)
            self.gate_details.update(
                finite_gradient_parameter_groups=counts,
                optimizer_updates=0,
                optimizer_step_counts=dict(self.optimizer_steps),
                background_temporal_gradients_finite=True,
                background_temporal_version=TEMPORAL_VERSION,
                body_joint_static_color_gradient_routed=False,
                body_joint_environment_gradient_routed=False,
                independent_branch_screen_gradients_present=True,
                supervision_version=SUPERVISION_VERSION,
                screen_gradient_double_accumulation_prevented=True,
                body_joint_background_time_gradient_routed=False,
                body_joint_known_road_gradient_routed=False,
            )
            if any(self.optimizer_steps.values()):
                raise RuntimeError("zero-update gate executed an optimizer step")
            if hasattr(self, 'road'):
                if self.road.gradient_l2_cumulative.max() <= 0 and self.config['road']['objective_weights']['observed_rgb'] > 0:
                    raise RuntimeError('real multiview road RGB path has no sampled texture gradient')
                self.gate_details['road_source_frames_tested'] = [89+index for index in self.road.last_frames]
                self.gate_details['road_raster_forward_backward_finite'] = True
                self.road.export(static, iteration=0, optimizer_updates=0)
            (self.output / "gpu_gate_details.json").write_text(
                json.dumps(self.gate_details, indent=2), encoding="utf-8"
            )
            raise ZeroUpdateGateComplete(
                "actual GPU v3 forward/backward complete; optimizer updates=0"
            )

    def accumulate_densification(
        self,
        model,
        branch,
        iteration,
        branch_proxy,
        branch_visibility,
        branch_radii,
        joint_proxy,
        joint_visibility,
        joint_radii,
    ):
        statistics = accumulate_max_screen_gradient(
            model,
            branch_proxy,
            branch_visibility,
            branch_radii,
            joint_proxy,
            joint_visibility,
            joint_radii,
        )
        if iteration == 1 or iteration % 100 == 0:
            record = {
                "step": iteration,
                "branch": branch,
                "fusion": self.policy["densification_fusion"],
                "denominator_increment_per_active_point": 1,
            }
            record.update({key: float(value.detach()) for key, value in statistics.items()})
            _append_jsonl(self.densification_log, record)


class ZeroUpdateGateComplete(RuntimeError):
    pass
