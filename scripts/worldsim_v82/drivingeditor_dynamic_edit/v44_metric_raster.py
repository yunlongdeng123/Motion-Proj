"""显式启用的统一光栅单位适配；不单独移动道路，不改持久参数。

旧 CUDA 的固定 near=0.2 使用模型单位。统一缩放所有输入和相机平移，
可把它换算到指定物理单位；features 内的深度仍保持原模型单位。原生
行在合并前保留旧 near 可见集合，避免凭此改变已经验收的原生重建。
此适配不修复已有表示的深度，必须另行检查 KEEP 遮挡关系。
"""
from __future__ import annotations
from types import SimpleNamespace


def metric_raster(native_raster, unit_scale):
    """包装一次统一排序；RGBA/alpha 的尺度控制可直接与 native 比较。"""
    import math
    if not callable(native_raster) or isinstance(unit_scale, bool) or not math.isfinite(unit_scale) or unit_scale <= 0:
        raise ValueError('metric_raster requires a callable native raster and finite positive unit_scale')
    def raster(camera, pipe, degree, rows, background):
        view = camera.world_view_transform.clone()
        view[3, :3] = view[3, :3] * unit_scale
        projection = getattr(camera, 'projection_matrix', None)
        if projection is None:
            projection = camera.world_view_transform.inverse() @ camera.full_proj_transform
        scaled_camera = SimpleNamespace(
            image_height=camera.image_height, image_width=camera.image_width,
            FoVx=camera.FoVx, FoVy=camera.FoVy, world_view_transform=view,
            full_proj_transform=view @ projection, camera_center=camera.camera_center * unit_scale)
        scaled = dict(rows, means3D=rows['means3D'] * unit_scale)
        if rows['scales'] is not None: scaled['scales'] = rows['scales'] * unit_scale
        if rows['cov3D_precomp'] is not None: scaled['cov3D_precomp'] = rows['cov3D_precomp'] * (unit_scale * unit_scale)
        return native_raster(scaled_camera, pipe, degree, scaled, background)
    return raster
