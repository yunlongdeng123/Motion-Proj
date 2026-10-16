"""刚性实例与 DrivingEditor 静态 PVG 的联合渲染入口。

实例必须先由 ``evaluate(timestamp, edits)`` 转到世界坐标。颜色在 Python
计算：静态 SH 使用世界方向，实例 SH 使用物体局部方向。CUDA 只接收联合
点集及预计算颜色；其 tile/depth 排序和 alpha*T 合成同时用于 RGB 和身份
feature。身份通道不足时，每批仍提交同一个完整场景，不能只渲染该批车辆。

``dynamic_alpha`` 保留原 DrivingEditor 动态分支独立渲染的 raw union，供完整
body opacity 约束使用。``instance_visible_contribution`` 则来自全场景合成，
供可见实例 mask 监督使用。两者的监督语义不同，不得互换。
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple, Sequence

import torch
import torch.nn.functional as F


# 当前已下载源码缺少已安装二进制对应的 config.h。保守默认仍须在 CUDA 验收时
# 对照实际 MAX_FEATURES；上游当前值不能证明本机已编译版本的容量。
DEFAULT_MAX_FEATURES = 10
VISIBILITY_SEMANTICS = "full_scene_transmittance_weighted_alpha"


class RasterizationSettings(NamedTuple):
    """字段与当前 DrivingEditor feature rasterizer 的 settings ABI 一致。"""

    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool


def evaluate_sh_rgb(features: torch.Tensor, directions: torch.Tensor, degree: int) -> torch.Tensor:
    """与当前 3DGS/DrivingEditor 相同的 degree 0--3 SH 常量和 +0.5 约定。"""
    if degree < 0 or degree > 3:
        raise ValueError("当前 DrivingEditor rasterizer 只支持 SH degree 0--3")
    if features.ndim != 3 or features.shape[2] != 3 or features.shape[1] < (degree + 1) ** 2:
        raise ValueError("SH features 必须为 [N, C, 3] 且包含 active degree 的系数")
    if directions.shape != (features.shape[0], 3):
        raise ValueError("SH directions 必须为 [N, 3]")
    sh = features.transpose(1, 2)
    result = 0.28209479177387814 * sh[..., 0]
    if degree > 0:
        x, y, z = directions.unbind(-1)
        x, y, z = x[:, None], y[:, None], z[:, None]
        result = result + 0.4886025119029199 * (
            -y * sh[..., 1] + z * sh[..., 2] - x * sh[..., 3]
        )
        if degree > 1:
            xx, yy, zz, xy, yz, xz = x*x, y*y, z*z, x*y, y*z, x*z
            result = result + (
                1.0925484305920792 * xy * sh[..., 4]
                - 1.0925484305920792 * yz * sh[..., 5]
                + 0.31539156525252005 * (2*zz - xx - yy) * sh[..., 6]
                - 1.0925484305920792 * xz * sh[..., 7]
                + 0.5462742152960396 * (xx - yy) * sh[..., 8]
            )
            if degree > 2:
                result = result + (
                    -0.5900435899266435 * y * (3*xx - yy) * sh[..., 9]
                    + 2.890611442640554 * xy * z * sh[..., 10]
                    - 0.4570457994644658 * y * (4*zz - xx - yy) * sh[..., 11]
                    + 0.3731763325901154 * z * (2*zz - 3*xx - 3*yy) * sh[..., 12]
                    - 0.4570457994644658 * x * (4*zz - xx - yy) * sh[..., 13]
                    + 1.445305721320277 * z * (xx - yy) * sh[..., 14]
                    - 0.5900435899266435 * x * (xx - 3*yy) * sh[..., 15]
                )
    return torch.clamp_min(result + 0.5, 0.0)


def _feature_columns(items, xyz, row_ids=None):
    if items is None or (not torch.is_tensor(items) and len(items) == 0):
        return xyz.new_empty((len(xyz), 0))
    items = [items] if torch.is_tensor(items) else list(items)
    columns = []
    for column in items:
        if column.ndim == 1:
            column = column[:, None]
        if column.ndim != 2:
            raise ValueError("other features 必须为 [N, C]")
        if row_ids is not None:
            # dynamic_other 定义在当前 packed 参数表；替换可以复制源行且总点数不变，
            # 因而不能仅凭 N 相等就省略 row_ids 重排。
            if row_ids.shape != (len(xyz),) or (row_ids.numel() and (int(row_ids.min()) < 0 or int(row_ids.max()) >= len(column))):
                raise ValueError("other features 的点数与当前 Gaussian 点集不同")
            column = column[row_ids]
        elif len(column) != len(xyz):
            raise ValueError("other features 的点数与当前 Gaussian 点集不同")
        if column.device != xyz.device or column.dtype != xyz.dtype:
            raise ValueError("other features 必须与 Gaussian 点集使用相同 device/dtype")
        columns.append(column)
    return torch.cat(columns, dim=1)


def _boolean_mask(mask, count, device):
    if mask is None:
        return torch.ones(count, dtype=torch.bool, device=device)
    if mask.numel() != count:
        raise ValueError("mask 点数与静态 Gaussian 点集不同")
    return mask.to(device=device, dtype=torch.bool).reshape(count)


def prepare_static(viewpoint_camera, pc, pipe, scaling_modifier=1.0, other=None, mask=None):
    """保留原静态 PVG 的 SHM、时间边缘权重和 marginal > .05 预筛选。"""
    # 原静态 prepare_rasterization 会将 time_shift 置空；这里仍用相机 timestamp。
    xyz = pc.get_xyz_SHM(viewpoint_camera.timestamp)
    marginal = pc.get_marginal_t(viewpoint_camera.timestamp).reshape(-1, 1)
    point_mask = _boolean_mask(mask, len(xyz), xyz.device) & (marginal[:, 0] > 0.05)
    direction = xyz - viewpoint_camera.camera_center
    # 原 Python SH 路径 detach 方向；原 CUDA SH 路径则保留颜色对位置的梯度。
    # 联合颜色必须预计算，但仍按旧 pipe 开关保留各自的静态梯度语义。
    if getattr(pipe, "convert_SHs_python", False):
        direction = direction.detach()
    colors = evaluate_sh_rgb(
        pc.get_features,
        F.normalize(direction, dim=-1, eps=1e-8),
        int(pc.active_sh_degree),
    )
    use_covariance = bool(getattr(pipe, "compute_cov3D_python", False))
    return dict(
        means3D=xyz, colors_precomp=colors, shs=None,
        opacities=pc.get_opacity * marginal,
        scales=None if use_covariance else pc.get_scaling,
        rotations=None if use_covariance else pc.get_rotation,
        cov3D_precomp=pc.get_covariance(scaling_modifier) if use_covariance else None,
        mask=point_mask, other=_feature_columns(other, xyz),
    )


def prepare_instances(viewpoint_camera, model, pipe, scaling_modifier=1.0, other=None, edits=None):
    """只使用刚性世界状态，不向车辆施加 PVG 时间 envelope 或 velocity。"""
    world = model.evaluate(viewpoint_camera.timestamp, edits=edits)
    xyz = world.xyz
    use_covariance = bool(getattr(pipe, "compute_cov3D_python", False))
    kwargs = dict(
        means3D=xyz,
        colors_precomp=evaluate_sh_rgb(
            world.features, world.local_view_directions(viewpoint_camera.camera_center),
            int(world.active_sh_degree),
        ),
        shs=None, opacities=world.opacity,
        scales=None if use_covariance else world.scaling,
        rotations=None if use_covariance else world.rotation,
        cov3D_precomp=world.covariance(scaling_modifier, packed=True) if use_covariance else None,
        mask=torch.ones(len(xyz), dtype=torch.bool, device=xyz.device),
        other=_feature_columns(other, xyz, world.row_ids),
    )
    return kwargs, world


def _merge(static, dynamic):
    result = {}
    for name in ("means3D", "colors_precomp", "opacities", "scales", "rotations", "cov3D_precomp", "mask", "other"):
        left, right = static[name], dynamic[name]
        if left is None and right is None:
            result[name] = None
        elif left is None or right is None:
            raise ValueError(f"静态与实例的 {name} 表示必须一致")
        else:
            if left.device != right.device or left.dtype != right.dtype:
                raise ValueError(f"静态与实例的 {name} 必须使用相同 device/dtype")
            result[name] = torch.cat((left, right), dim=0).contiguous()
    result["shs"] = None
    return result


def _screenspace(xyz):
    result = torch.zeros_like(xyz, requires_grad=True) + 0
    if result.requires_grad:
        result.retain_grad()
    return result


def _payload(scene, camera, dynamic_membership, point_ids, channel_ids):
    xyz = scene["means3D"]
    homogeneous = torch.cat((xyz, torch.ones_like(xyz[:, :1])), dim=1)
    depth = homogeneous @ camera.world_view_transform[:, 2:3]
    # 与原代码保持一致：depth 与 alpha channel 的 mask 外点先置零。
    depth_alpha = torch.cat((depth, torch.ones_like(depth)), dim=1) * scene["mask"][:, None]
    labels = torch.stack([(point_ids == i).to(xyz.dtype) for i in channel_ids], dim=1) if channel_ids else xyz.new_empty((len(xyz), 0))
    return torch.cat((scene["other"], depth_alpha, dynamic_membership[:, None], labels), dim=1).contiguous()


def _rasterize(rasterizer, scene, payload, settings, max_features):
    if payload.shape[1] > max_features:
        raise ValueError("feature payload 超过配置的 CUDA MAX_FEATURES；不能调用有溢出风险的 kernel")
    if len(scene["means3D"]) == 0:
        h, w = settings.image_height, settings.image_width
        xyz = scene["means3D"]
        return (
            torch.zeros((1, h, w), dtype=torch.int32, device=xyz.device),
            settings.bg[:, None, None].expand(3, h, w),
            xyz.new_zeros((payload.shape[1], h, w)),
            torch.zeros(0, dtype=torch.int32, device=xyz.device),
        )
    kwargs = {name: value for name, value in scene.items() if name != "other"}
    result = rasterizer(**kwargs, features=payload)
    if not isinstance(result, (tuple, list)) or len(result) != 4:
        raise RuntimeError("需要当前 DrivingEditor 四输出 feature rasterizer (contrib, rgb, feature, radii)")
    if result[2].shape[0] != payload.shape[1]:
        raise RuntimeError("rasterizer 未返回请求的 feature channels")
    return result


def render(
    viewpoint_camera,
    pc,
    instances,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    env_map=None,
    time_shift=None,
    other=None,
    dynamic_other=None,
    mask=None,
    is_training=False,
    *,
    edits=None,
    instance_ids: Sequence[str] | None = None,
    max_features: int = DEFAULT_MAX_FEATURES,
    render_separate: bool = True,
    rasterizer_factory=None,
) -> dict[str, Any]:
    """一次联合点集的 RGB 与全场景可见实例贡献；可附加原独立分支输出。

    ``instance_ids`` 只筛选输出通道，不移除遮挡物。实际删除/平移/旋转/替换
    通过模型命名 track 的 edits 接口进入同一个 evaluate 和联合 render。
    ``time_shift`` 为旧调用签名保留；刚性实例使用明确的 camera.timestamp。
    ``max_features`` 必须不大于已编译 rasterizer 的 MAX_FEATURES。
    """
    if not isinstance(max_features, int) or max_features < 3:
        raise ValueError("max_features 必须是至少 3 的整数")
    static = prepare_static(viewpoint_camera, pc, pipe, scaling_modifier, other, mask)
    xyz = static["means3D"]
    if xyz.dtype != torch.float32:
        raise ValueError("当前 DrivingEditor CUDA rasterizer 要求 float32 Gaussian 数据")
    if bg_color.shape != (3,) or bg_color.device != xyz.device or bg_color.dtype != xyz.dtype:
        raise ValueError("bg_color 必须为与 Gaussian 点集相同 device/dtype 的 [3]")
    if instances is None:
        if edits or instance_ids:
            raise ValueError("没有实例模型时不能指定 edits 或 instance_ids")
        dynamic = {name: value[:0] if torch.is_tensor(value) else value for name, value in static.items()}
        track_ids, world = (), None
    else:
        dynamic, world = prepare_instances(viewpoint_camera, instances, pipe, scaling_modifier, dynamic_other, edits)
        track_ids = tuple(instances.instance_ids)
        if len(set(track_ids)) != len(track_ids):
            raise ValueError("instance_ids 必须唯一")
        if world.point_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("world.point_ids 必须为整数 instance indices")
        if world.point_ids.numel() and (int(world.point_ids.min()) < 0 or int(world.point_ids.max()) >= len(track_ids)):
            raise ValueError("world.point_ids 越出 stable instance_ids 范围")
    # 静态 t/velocity features 保留。刚性车辆没有同义的 PVG 值，未提供则显式零填充。
    extra_count = max(static["other"].shape[1], dynamic["other"].shape[1])
    for branch in (static, dynamic):
        missing = extra_count - branch["other"].shape[1]
        if missing:
            if branch["other"].shape[1] != 0:
                raise ValueError("静态和动态 other features 的 channel 数必须一致或一侧为空")
            branch["other"] = branch["means3D"].new_zeros((len(branch["means3D"]), extra_count))
    available = max_features - extra_count - 3  # depth, alpha, dynamic visible union
    selected = track_ids if instance_ids is None else tuple(instance_ids)
    if len(set(selected)) != len(selected) or any(name not in track_ids for name in selected):
        raise ValueError("请求的 instance_ids 必须唯一且存在于模型中")
    if available < 0 or (selected and available < 1):
        raise ValueError("MAX_FEATURES 不足以容纳 other/depth/alpha/union 和至少一个身份通道")
    selected_indices = [track_ids.index(name) for name in selected]
    chunks = [selected_indices[i:i+available] for i in range(0, len(selected_indices), available)] if selected_indices else [[]]

    full = _merge(static, dynamic)
    static_count, dynamic_count = len(xyz), len(dynamic["means3D"])
    screen = _screenspace(full["means3D"])
    full["means2D"] = screen
    if override_color is not None:
        color = override_color.expand(len(full["means3D"]), 3) if override_color.shape == (3,) else override_color
        if color.shape != full["colors_precomp"].shape or color.device != xyz.device or color.dtype != xyz.dtype:
            raise ValueError("override_color 必须是同 device/dtype 的 [3] 或联合点集 [N,3]")
        full["colors_precomp"] = color
        static["colors_precomp"], dynamic["colors_precomp"] = color[:static_count], color[static_count:]
    tanfovx = math.tan(-0.5 if getattr(pipe, "neg_fov", False) else viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(-0.5 if getattr(pipe, "neg_fov", False) else viewpoint_camera.FoVy * 0.5)
    settings = RasterizationSettings(
        image_height=int(viewpoint_camera.image_height), image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx, tanfovy=tanfovy,
        bg=bg_color if env_map is not None else torch.zeros_like(bg_color),
        scale_modifier=scaling_modifier, viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform, sh_degree=int(pc.active_sh_degree),
        campos=viewpoint_camera.camera_center, prefiltered=False, debug=bool(getattr(pipe, "debug", False)),
    )
    if rasterizer_factory is None:
        from diff_gaussian_rasterization import GaussianRasterizer
        rasterizer_factory = GaussianRasterizer
    rasterizer = rasterizer_factory(raster_settings=settings)
    dynamic_ids = world.point_ids if world is not None else torch.empty(0, dtype=torch.long, device=xyz.device)
    ids = torch.cat((torch.full((static_count,), -1, dtype=torch.long, device=xyz.device), dynamic_ids))
    membership = torch.cat((xyz.new_zeros(static_count), xyz.new_ones(dynamic_count)))
    label_maps, first = [], None
    for channel_ids in chunks:
        # 不按 channel_ids 筛选任何几何：其他车辆和静态点仍在同一排序/合成中。
        result = _rasterize(rasterizer, full, _payload(full, viewpoint_camera, membership, ids, channel_ids), settings, max_features)
        if first is None:
            first = result
        label_maps.append(result[2][extra_count+3:])
    contrib, image, rendered_features, radii = first
    feature = rendered_features[:extra_count]
    depth, alpha = rendered_features[extra_count:extra_count+1], rendered_features[extra_count+1:extra_count+2]
    visible_union = rendered_features[extra_count+2:extra_count+3]
    visible_labels = torch.cat(label_maps, dim=0)
    env_image = torch.zeros_like(image)
    if env_map is not None:
        env_image = env_map(viewpoint_camera.get_world_directions(is_training).permute(1, 2, 0)).permute(2, 0, 1)
        image = image + (1-alpha) * env_image
    package = dict(
        render=image, raw_render=first[1], viewspace_points=screen, visibility_filter=radii > 0, radii=radii,
        contrib=contrib, depth=depth, alpha=alpha, feature=feature,
        env_map=(1-alpha) * env_image,
        instance_ids=selected, instance_visible_contribution=visible_labels,
        visible_instance_alpha=visible_labels,
        instance_visible_alpha={name: visible_labels[i:i+1] for i, name in enumerate(selected)},
        dynamic_visible_alpha=visible_union, static_visible_alpha=alpha-visible_union,
        instance_visibility_semantics=VISIBILITY_SEMANTICS,
        instance_feature_passes=len(chunks), rasterizer_feature_capacity=max_features,
        static_point_count=static_count, instance_point_ids=dynamic_ids,
        instance_row_ids=world.row_ids if world is not None else dynamic_ids,
    )
    if render_separate:
        for suffix, branch, is_dynamic in (("s", static, False), ("d", dynamic, True)):
            branch["means2D"] = _screenspace(branch["means3D"])
            branch_ids = dynamic_ids if is_dynamic else ids[:static_count]
            branch_membership = branch["means3D"].new_full((len(branch["means3D"]),), float(is_dynamic))
            isolated = _rasterize(rasterizer, branch, _payload(branch, viewpoint_camera, branch_membership, branch_ids, []), settings, max_features)
            branch_image, branch_features = isolated[1], isolated[2]
            branch_alpha = branch_features[extra_count+1:extra_count+2]
            # 原实现只在训练时给独立动态 RGB 补环境背景；raw alpha 从不乘 GT mask。
            if env_map is not None and (not is_dynamic or is_training):
                branch_image = branch_image + (1-branch_alpha) * env_image
            label = "dynamic" if is_dynamic else "static"
            package.update({
                f"render_{suffix}": branch_image, f"raw_render_{suffix}": isolated[1], f"{label}_alpha": branch_alpha,
                f"{label}_feature": branch_features[:extra_count],
                f"{label}_depth": branch_features[extra_count:extra_count+1],
                f"{label}_viewspace_points": branch["means2D"], f"{label}_radii": isolated[3],
                f"env_map_{suffix}": (1-branch_alpha) * env_image,
            })
        package["dynamic_alpha_raw"] = package["dynamic_alpha"]
        package["dynamic_alpha_semantics"] = "isolated_all_instances_raw_union_body_opacity"
    return package


def render_with_edits(viewpoint_camera, pc, instances, pipe, bg_color, *, edits, **kwargs):
    """命名 track 的删除、平移、旋转、替换共用正常全局渲染路径。"""
    return render(viewpoint_camera, pc, instances, pipe, bg_color, edits=edits, **kwargs)
