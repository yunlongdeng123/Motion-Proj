#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel, Dynamic_GaussianModel
from scene.cameras import Camera
from utils.sh_utils import eval_sh
import copy

def render(viewpoint_camera: Camera, pc: GaussianModel, pc_dyn: Dynamic_GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
           override_color=None, env_map=None,
           time_shift=None, other=[], dynamic_other=[], mask=None, is_training=False):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    # dynamic_screenspace_points = torch.zeros_like(pc_dyn.get_xyz, dtype=pc_dyn.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    #
    # try:
    #     screenspace_points.retain_grad()
    # except:
    #     pass

    # Set up rasterization configuration
    if pipe.neg_fov:
        # we find that set fov as -1 slightly improves the results
        tanfovx = math.tan(-0.5)
        tanfovy = math.tan(-0.5)
    else:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color if env_map is not None else torch.zeros(3, device="cuda"),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )
    feature_list = other
    dynamic_feature_list = dynamic_other
    if pc_dyn is not None:
        if len(dynamic_feature_list) > 0 and len(feature_list) > 0:
            features = torch.cat(feature_list, dim=1)
            dynamic_features = torch.cat(dynamic_feature_list, dim=1)
            features = torch.cat([features, dynamic_features], dim=0)
            S_other = features.shape[1]
        else:
            features = torch.zeros_like(pc.get_xyz[:, :0])
            dynamic_features = torch.zeros_like(pc_dyn.get_xyz[:, :0])
            features = torch.cat([features, dynamic_features], dim=0)
            S_other = features.shape[1]
    else:
        if len(feature_list) > 0:
            features = torch.cat(feature_list, dim=1)
            S_other = features.shape[1]
        else:
            features = torch.zeros_like(pc.get_xyz[:, :0])
            S_other = 0
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    if pipe.including_dynamic or pipe.adding:
        static_render_kwargs = prepare_rasterization(viewpoint_camera, pc, pipe, scaling_modifier, override_color,
                                                         time_shift, other, mask, split=False)
        dynamic_render_kwargs = prepare_rasterization(viewpoint_camera, pc_dyn, pipe, scaling_modifier, override_color,
                                                      time_shift, dynamic_other, mask, split=False, dynamic=True)
        render_kwargs = merge_kwargs(static_render_kwargs, dynamic_render_kwargs)
        static_render_kwargs_split = prepare_rasterization(viewpoint_camera, pc, pipe, scaling_modifier, override_color,
                                                           time_shift, other, mask, split=False)
        static_screenspace_points = torch.zeros_like(static_render_kwargs["means3D"], dtype=static_render_kwargs["means3D"].dtype, requires_grad=True, device="cuda") + 0
        screenspace_points = torch.zeros_like(render_kwargs["means3D"], dtype=render_kwargs["means3D"].dtype, requires_grad=True, device="cuda") + 0
        dyn_screenspace_points = torch.zeros_like(dynamic_render_kwargs["means3D"], dtype=dynamic_render_kwargs["means3D"].dtype, requires_grad=True, device="cuda") + 0

        try:
            static_screenspace_points.retain_grad()
            screenspace_points.retain_grad()
            dyn_screenspace_points.retain_grad()
        except:
            pass
        render_kwargs["means2D"] = screenspace_points
        dynamic_render_kwargs["means2D"] = dyn_screenspace_points
        static_render_kwargs["means2D"] = static_screenspace_points
        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        contrib, rendered_image, rendered_feature, radii = rasterizer(**render_kwargs)


        # if split:
        contrib_d, rendered_image_d, rendered_feature_d, radii_d = rasterizer(**dynamic_render_kwargs)
        #
        contrib_s, rendered_image_s, rendered_feature_s, radii_s = rasterizer(**static_render_kwargs)

        rendered_other, rendered_depth, rendered_opacity = rendered_feature.split([S_other, 1, 1], dim=0)
        rendered_other_s, rendered_depth_s, rendered_opacity_s = rendered_feature_s.split([S_other, 1, 1], dim=0)
        rendered_other_d, rendered_depth_d, rendered_opacity_d = rendered_feature_d.split([S_other, 1, 1], dim=0)

        # 固定语义：独立动态光栅化后的预乘颜色，尚未混入 EnvMap。
        # 训练路径中的 render_d 会继续保留历史环境填充行为；泄漏监督与诊断
        # 必须使用这个字段，避免把环境颜色误算成动态高斯颜色。
        dynamic_premultiplied_rgb = rendered_image_d

        if env_map is not None:
            bg_color_from_envmap = env_map(viewpoint_camera.get_world_directions(is_training).permute(1, 2, 0)).permute(2, 0, 1)
            if is_training:
                rendered_image_d = rendered_image_d + (1 - rendered_opacity_d) * bg_color_from_envmap.detach()
            rendered_image = rendered_image + (1 - rendered_opacity) * bg_color_from_envmap
            rendered_image_s = rendered_image_s + (1 - rendered_opacity_s) * bg_color_from_envmap

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image,
                "render_d": rendered_image_d,
                "dynamic_premultiplied_rgb": dynamic_premultiplied_rgb,
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii,
                "static_viewspace_points": static_screenspace_points,
                "static_visibility_filter": radii_s > 0,
                "static_radii": radii_s,
                "background_means3D": static_render_kwargs["means3D"],
                "background_opacities": static_render_kwargs["opacities"],
                "dynamic_viewspace_points": dyn_screenspace_points,
                "dynamic_visibility_filter": radii_d > 0,
                "dynamic_radii": radii_d,
                "contrib": contrib,
                "depth": rendered_depth,
                "alpha": rendered_opacity,
                "feature": rendered_other,
                "render_s": rendered_image_s,
                "static_alpha": rendered_opacity_s,
                "static_feature": rendered_other_s,
                "dynamic_alpha": rendered_opacity_d,
                "static_depth": rendered_depth_s,
                "dynamic_depth": rendered_depth_d,
                "dynamic_feature": rendered_other_d,
                "env_map_s": (1 - rendered_opacity_s) * bg_color_from_envmap,
                "env_map": (1 - rendered_opacity) * bg_color_from_envmap,
                }
    else:
        render_kwargs = prepare_rasterization(viewpoint_camera, pc, pipe, scaling_modifier, override_color,
                                                         time_shift, other, mask, split=False)
        screenspace_points = torch.zeros_like(render_kwargs["means3D"], dtype=render_kwargs["means3D"].dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        render_kwargs["means2D"] = screenspace_points
        contrib, rendered_image, rendered_feature, radii = rasterizer(**render_kwargs)
        rendered_other, rendered_depth, rendered_opacity = rendered_feature.split([S_other, 1, 1], dim=0)
        if env_map is not None:
            bg_color_from_envmap = env_map(viewpoint_camera.get_world_directions(is_training).permute(1, 2, 0)).permute(2, 0, 1)
            rendered_image = rendered_image + (1 - rendered_opacity) * bg_color_from_envmap
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii,
                "contrib": contrib,
                "depth": rendered_depth,
                "alpha": rendered_opacity,
                "feature": rendered_other
                }
def render_road_only(camera, pc, pipe):
    """同一背景参数中的道路行；不另建颜色网络，不把真实遮挡从正常前向裁掉。"""
    rows = torch.nonzero(pc.road_unit_mask, as_tuple=False).flatten()
    if not len(rows):
        raise RuntimeError('multiview road renderer has no world-unit primitives')
    # 在建图前取道路行，避免每个额外来源都为全部背景构造一遍大图。
    class RoadView:
        active_sh_degree = pc.active_sh_degree
        get_max_sh_channels = pc.get_max_sh_channels
        get_xyz = pc._xyz[rows]
        get_opacity = pc.opacity_activation(pc._opacity[rows])
        get_scaling = pc.scaling_activation(pc._scaling[rows])
        get_scaling_t = pc.scaling_t_activation(pc._scaling_t[rows])
        get_rotation = pc.rotation_activation(pc._rotation[rows])
        get_features = torch.cat((pc._features_dc[rows], torch.zeros_like(pc._features_rest[rows])), 1)

        def get_xyz_SHM(self, timestamp):
            return self.get_xyz

        def get_marginal_t(self, timestamp):
            return torch.ones_like(self.get_xyz[:, :1])

        def get_covariance(self, modifier):
            return pc.covariance_activation(self.get_scaling, modifier, pc._rotation[rows])

    kwargs = prepare_rasterization(camera, RoadView(), pipe, other=[], mask=None)
    kwargs = {name: value.contiguous() if value is not None else None for name, value in kwargs.items()}
    proxy = torch.zeros_like(kwargs['means3D'], requires_grad=True) + 0
    if proxy.requires_grad:
        proxy.retain_grad()
    kwargs['means2D'] = proxy
    settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height), image_width=int(camera.image_width),
        tanfovx=math.tan(-.5 if pipe.neg_fov else camera.FoVx*.5),
        tanfovy=math.tan(-.5 if pipe.neg_fov else camera.FoVy*.5),
        bg=torch.zeros(3, device=pc._xyz.device), scale_modifier=1.,
        viewmatrix=camera.world_view_transform, projmatrix=camera.full_proj_transform,
        sh_degree=pc.active_sh_degree, campos=camera.camera_center, prefiltered=False, debug=pipe.debug)
    _, rgb, features, radii = GaussianRasterizer(raster_settings=settings)(**kwargs)
    depth_sum, alpha = features.split([1, 1], dim=0)
    return dict(rgb=rgb, depth_sum=depth_sum, alpha=alpha, proxy=proxy, rows=rows, radii=radii)


def merge_kwargs(render_kwargs, render_kwargs_box):
    merged_kwargs = {}
    for k, v in render_kwargs.items():
        if v is not None and render_kwargs_box[k] is not None:
            merged_kwargs[k] = torch.cat((v, render_kwargs_box[k]), dim=0).contiguous()
    return merged_kwargs

def prepare_rasterization(viewpoint_camera, pc: GaussianModel, pipe, scaling_modifier=1.0, override_color=None,
                          time_shift=None, other=[], mask=None, split=False, dynamic=False):
    if dynamic:
        means3D = pc.get_xyz
        opacity = pc.get_opacity
        scales = None
        rotations = None
        cov3D_precomp = None
        # time_shift = None
        num_rows = means3D.shape[0]
        # 计算前一半的终止索引
        mid_index = (num_rows + 1) // 2  # 如果是奇数，多一行分给前一半
        # 分割张量
        # means3D_orin = means3D[:mid_index]
        # means3D_orin = torch.cat((means3D_orin, means3D_orin), dim=0)
        # distances_orin = torch.sqrt(torch.sum(torch.square(means3D_orin), dim=1, keepdim=True))
        # distances_add = torch.sqrt(torch.sum(torch.square(means3D), dim=1, keepdim=True))
        # times = distances_add / distances_orin
        if time_shift is not None:
            means3D = pc.get_xyz_SHM(viewpoint_camera.timestamp - time_shift)
            means3D = means3D + pc.get_inst_velocity * time_shift
            marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp - time_shift)
        else:
            means3D = pc.get_xyz_SHM(viewpoint_camera.timestamp)
            marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp)
        opacity = opacity * marginal_t

        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

            # mask3d = (pc.get_scaling_t[:, 0] < 0.2) | (marginal_t[:, 0] < 0.05)
        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None

        if override_color is None:
            if pipe.convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, pc.get_max_sh_channels)
                dir_pp = (means3D.detach() - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0],
                                                                                   1)).detach()
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = override_color

        feature_list = other

        if len(feature_list) > 0:
            features = torch.cat(feature_list, dim=1)
            S_other = features.shape[1]
        else:
            features = torch.zeros_like(means3D[:, :0])
            S_other = 0

        # Prefilter
        if mask is None:
            mask = marginal_t[:, 0] > 0.05
        else:
            mask = mask & (marginal_t[:, 0] > 0.05)
        masked_means3D = means3D[mask]
        # sh_objs = sh_objs[mask]
        masked_xyz_homo = torch.cat([masked_means3D, torch.ones_like(masked_means3D[:, :1])], dim=1)
        masked_depth = (masked_xyz_homo @ viewpoint_camera.world_view_transform[:, 2:3])
        depth_alpha = torch.zeros(means3D.shape[0], 2, dtype=torch.float32, device=means3D.device)
        depth_alpha[mask] = torch.cat([
            masked_depth,
            torch.ones_like(masked_depth)
        ], dim=1)
        features = torch.cat([features, depth_alpha], dim=1)
        # means3D[:, 1] += 1
        mask3d = (pc.get_scaling_t[:, 0] < 0.2) | (marginal_t[:, 0] < 0.05)
        mask3d = ~mask3d
        return {"means3D": means3D,
                "shs": shs if shs is not None else None,
                "colors_precomp": colors_precomp if colors_precomp is not None else None,
                "features": features,
                "opacities": opacity,
                "scales": scales,
                "rotations": rotations,
                "cov3D_precomp": cov3D_precomp if cov3D_precomp is not None else None,
                "mask": mask}
    elif split:
        means3D = pc.get_xyz
        opacity = pc.get_opacity
        scales = None
        rotations = None
        cov3D_precomp = None
        time_shift = None
        if time_shift is not None:
            means3D = pc.get_xyz_SHM(viewpoint_camera.timestamp - time_shift)
            means3D = means3D + pc.get_inst_velocity * time_shift
            marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp - time_shift)
        else:
            means3D = pc.get_xyz_SHM(viewpoint_camera.timestamp)
            marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp)
        opacity = opacity * marginal_t

        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

            # mask3d = (pc.get_scaling_t[:, 0] < 0.2) | (marginal_t[:, 0] < 0.05)
        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None

        if override_color is None:
            if pipe.convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, pc.get_max_sh_channels)
                dir_pp = (means3D.detach() - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0],
                                                                                   1)).detach()
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = override_color

        feature_list = other

        if len(feature_list) > 0:
            features = torch.cat(feature_list, dim=1)
            S_other = features.shape[1]
        else:
            features = torch.zeros_like(means3D[:, :0])
            S_other = 0

        # Prefilter
        if mask is None:
            mask = marginal_t[:, 0] > 0.05
        else:
            mask = mask & (marginal_t[:, 0] > 0.05)
        masked_means3D = means3D[mask]
        # sh_objs = sh_objs[mask]
        masked_xyz_homo = torch.cat([masked_means3D, torch.ones_like(masked_means3D[:, :1])], dim=1)
        masked_depth = (masked_xyz_homo @ viewpoint_camera.world_view_transform[:, 2:3])
        depth_alpha = torch.zeros(means3D.shape[0], 2, dtype=torch.float32, device=means3D.device)
        depth_alpha[mask] = torch.cat([
            masked_depth,
            torch.ones_like(masked_depth)
        ], dim=1)
        features = torch.cat([features, depth_alpha], dim=1)
        mask3d = (pc.get_scaling_t[:, 0] < 0.2) | (marginal_t[:, 0] < 0.05)
        mask3d = ~mask3d
        return {"means3D": means3D[mask3d],
                "shs": shs[mask3d] if shs is not None else None,
                "colors_precomp": colors_precomp[mask3d] if colors_precomp is not None else None,
                "features": features[mask3d],
                "opacities": opacity[mask3d],
                "scales": scales[mask3d],
                "rotations": rotations[mask3d],
                "cov3D_precomp": cov3D_precomp[mask3d] if cov3D_precomp is not None else None,
                "mask": mask[mask3d]}
    else:
        means3D = pc.get_xyz
        opacity = pc.get_opacity
        scales = None
        rotations = None
        cov3D_precomp = None
        time_shift = None
        if time_shift is not None:
            means3D = pc.get_xyz_SHM(viewpoint_camera.timestamp - time_shift)
            means3D = means3D + pc.get_inst_velocity * time_shift
            marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp - time_shift)
        else:
            means3D = pc.get_xyz_SHM(viewpoint_camera.timestamp)
            marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp)
        opacity = opacity * marginal_t

        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

            # mask3d = (pc.get_scaling_t[:, 0] < 0.2) | (marginal_t[:, 0] < 0.05)
        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None

        if override_color is None:
            if pipe.convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, pc.get_max_sh_channels)
                dir_pp = (means3D.detach() - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0],
                                                                                   1)).detach()
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = override_color

        feature_list = other

        if len(feature_list) > 0:
            features = torch.cat(feature_list, dim=1)
            S_other = features.shape[1]
        else:
            features = torch.zeros_like(means3D[:, :0])
            S_other = 0

        # Prefilter
        if mask is None:
            mask = marginal_t[:, 0] > 0.05
        else:
            mask = mask & (marginal_t[:, 0] > 0.05)
        masked_means3D = means3D[mask]
        # sh_objs = sh_objs[mask]
        masked_xyz_homo = torch.cat([masked_means3D, torch.ones_like(masked_means3D[:, :1])], dim=1)
        masked_depth = (masked_xyz_homo @ viewpoint_camera.world_view_transform[:, 2:3])
        depth_alpha = torch.zeros(means3D.shape[0], 2, dtype=torch.float32, device=means3D.device)
        depth_alpha[mask] = torch.cat([
            masked_depth,
            torch.ones_like(masked_depth)
        ], dim=1)
        features = torch.cat([features, depth_alpha], dim=1)
        mask3d = (pc.get_scaling_t[:, 0] < 0.2) | (marginal_t[:, 0] < 0.05)
        mask3d = ~mask3d
        return {"means3D": means3D,
                "shs": shs,
                "colors_precomp": colors_precomp,
                "features": features,
                "opacities": opacity,
                "scales": scales,
                "rotations": rotations,
                "cov3D_precomp": cov3D_precomp,
                "mask": mask}
