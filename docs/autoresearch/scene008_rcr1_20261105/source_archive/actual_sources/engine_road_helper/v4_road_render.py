"""道路独立训练光栅与输出合成；合成使用同一次原生深度排序。"""

import math
import torch


def settings(camera, pipe, degree, background, scaling_modifier=1.):
    from diff_gaussian_rasterization import GaussianRasterizationSettings
    return GaussianRasterizationSettings(
        image_height=int(camera.image_height), image_width=int(camera.image_width),
        tanfovx=math.tan(-.5) if pipe.neg_fov else math.tan(camera.FoVx*.5),
        tanfovy=math.tan(-.5) if pipe.neg_fov else math.tan(camera.FoVy*.5),
        bg=background, scale_modifier=scaling_modifier,
        viewmatrix=camera.world_view_transform, projmatrix=camera.full_proj_transform,
        sh_degree=degree, campos=camera.camera_center, prefiltered=False, debug=pipe.debug)


def road_kwargs(camera, road, pipe, sh_channels=1, precomputed_color=False, for_training=False):
    xyz = road.xyz
    homogeneous = torch.cat((xyz, torch.ones_like(xyz[:, :1])), 1)
    depth = homogeneous @ camera.world_view_transform[:, 2:3]
    covariance = None
    if pipe.compute_cov3D_python:
        from utils.general_utils import build_scaling_rotation, strip_symmetric
        matrix = build_scaling_rotation(road.scaling, road.rotation)
        covariance = strip_symmetric(matrix @ matrix.transpose(1, 2))
    shs = None
    if not precomputed_color:
        rest = road.dc.new_zeros((len(xyz), sh_channels-1, 3))
        shs = torch.cat((road.dc, rest), 1)
    return dict(means3D=xyz, means2D=torch.zeros_like(xyz), shs=shs,
        colors_precomp=road.colors() if precomputed_color else None,
        features=torch.cat((depth, torch.ones_like(depth)), 1),
        opacities=road.opacity.sigmoid(), scales=None if covariance is not None else road.scaling,
        rotations=None if covariance is not None else road.rotation, cov3D_precomp=covariance,
        mask=road.active if for_training else road.output_mask())


def merge_rows(left, right):
    result = {}
    if set(left) != set(right):
        raise ValueError('光栅字段不匹配')
    for name in left:
        a, b = left[name], right[name]
        if a is None and b is None:
            result[name] = None
        elif a is None or b is None:
            raise ValueError('光栅表示不一致：'+name)
        else:
            result[name] = torch.cat((a, b), 0).contiguous()
    return result


def raster(camera, pipe, degree, kwargs, background):
    from diff_gaussian_rasterization import GaussianRasterizer
    _, rgb, features, radii = GaussianRasterizer(
        raster_settings=settings(camera, pipe, degree, background))(**kwargs)
    return dict(rgb=rgb, depth_sum=features[:1], alpha=features[1:2],
                extra=features[2:], radii=radii)


def render_road(camera, road, pipe, for_training=False):
    """不传入 V1 或 EnvMap，颜色为预乘 RGB，深度为加权和。"""
    return raster(camera, pipe, 0, road_kwargs(camera, road, pipe, for_training=for_training),
                  road.xyz.new_zeros(3))


@torch.no_grad()
def render_composite(camera, static, dynamic, road, pipe, background, env_map=None,
                     dynamic_keep=None):
    from gaussian_renderer import prepare_rasterization
    native_background = background if env_map is not None else torch.zeros_like(background)
    kwargs = prepare_rasterization(camera, static, pipe)
    kwargs['means2D'] = torch.zeros_like(kwargs['means3D'])
    if dynamic is not None:
        dynamic_kwargs = prepare_rasterization(camera, dynamic, pipe, dynamic=True)
        dynamic_kwargs['means2D'] = torch.zeros_like(dynamic_kwargs['means3D'])
        if dynamic_keep is not None:
            if dynamic_keep.dtype != torch.bool or dynamic_keep.shape != dynamic_kwargs['mask'].shape:
                raise ValueError('删车必须给出对应动态高斯的 bool keep mask')
            dynamic_kwargs['mask'] = dynamic_kwargs['mask'] & dynamic_keep
        kwargs = merge_rows(kwargs, dynamic_kwargs)
    if road is not None:
        channels = kwargs['shs'].shape[1] if kwargs['shs'] is not None else 1
        road_values = road_kwargs(camera, road, pipe, channels,
            precomputed_color=kwargs['colors_precomp'] is not None)
        kwargs['features'] = torch.cat((kwargs['features'], torch.zeros_like(kwargs['features'][:, :1])), 1)
        road_values['features'] = torch.cat((road_values['features'], torch.ones_like(road_values['features'][:, :1])), 1)
        kwargs = merge_rows(kwargs, road_values)
    # xyz、opacity 和相机均未按图像掩码改写；CUDA 统一按实际深度排序。
    package = raster(camera, pipe, static.active_sh_degree, kwargs, native_background)
    package['road_visible_alpha'] = package['extra'] if road is not None else torch.zeros_like(package['alpha'])
    package['raw_rgb'] = package['rgb']
    if env_map is not None:
        environment = env_map(camera.get_world_directions(False).permute(1, 2, 0)).permute(2, 0, 1)
        package['rgb'] = package['rgb']+(1-package['alpha'])*environment
    return package
