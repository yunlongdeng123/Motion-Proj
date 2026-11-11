"""道路实拍认证的独立 CPU 数组接口；无场景、帧号、actor 或文件路径绑定。

公式来源：v41_dense_geometry.measured_patch_evidence / ray_box_blocked，
fullclip_road_cpu.Surface，build_expanded_evidence 的父格调用，以及
prepare_road37 的父格→子格继承、双线性 RGB、上辐射簇及邻近沥青推断。
所有 xyz、法向、OBB 和相机原点必须在同一个米制正交坐标系中。相机为
CV xyz（z 为前向），像素约定是 K 投影后减 0.5，门限使用交付源图尺度。

重要：实测三角形认证发生在父格。子格继承父格的 source_valid，然后只
重新投影并检查源前景 guard；它们不是各自具有独立 LiDAR 测量的新观测。
本模块不加载传感器、模型、旧 runtime 或 GPU，也不生成场景资产。
"""
from __future__ import annotations

import warnings
import numpy as np
from scipy.ndimage import binary_dilation, map_coordinates
from scipy.spatial import Delaunay, QhullError, cKDTree


POLICY = {
    "triangle_max_edge_m": 1.6,
    "triangle_max_edge_px": 24.0,
    "triangle_normal_cosine": 0.9,
    "depth_tolerance_m": 0.3,
    "nearest_return_radius_px": 1.5,
    "minimum_depth_m": 0.1,
    "minimum_view_cosine": 0.12,
    "evidence_patch_m": 0.2,
    "minimum_evidence_area_px": 0.25,
    "minimum_source_quality": 0.2,
    "foreground_box_margin_px": 2,
    "parent_cell_m": 0.08,
    "child_cell_m": 0.02,
}


def transform(xyz, matrix):
    xyz, matrix = np.asarray(xyz), np.asarray(matrix)
    return xyz @ matrix[:3, :3].T + matrix[:3, 3]


def project(xyz, K, world_to_camera):
    """同原实现：CV 相机的 native pixel-center 投影（减 0.5）。"""
    camera = transform(xyz, world_to_camera)
    homogeneous = camera @ np.asarray(K).T
    return homogeneous[:, :2] / np.maximum(camera[:, 2:], 1e-8) - 0.5, camera[:, 2]


def _box_arrays(box_centers, box_halves, box_axes):
    if box_centers is None and box_halves is None and box_axes is None:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3, 3))
    if any(value is None for value in (box_centers, box_halves, box_axes)):
        raise ValueError("box_centers/box_halves/box_axes must be provided together")
    centers = np.asarray(box_centers, dtype=float).reshape(-1, 3)
    halves = np.asarray(box_halves, dtype=float).reshape(-1, 3)
    axes = np.asarray(box_axes, dtype=float).reshape(-1, 3, 3)
    if len(centers) != len(halves) or len(centers) != len(axes):
        raise ValueError("OBB array lengths differ")
    if not all(np.isfinite(a).all() for a in (centers, halves, axes)) or (halves <= 0).any():
        raise ValueError("OBBs must be finite with positive half sizes")
    return centers, halves, axes


def ray_box_blocked(xyz, camera_origin, box_centers=None, box_halves=None, box_axes=None):
    """原 OBB 线段检验；axes 的列为盒体局部轴，不额外膨胀 3D 盒。"""
    xyz = np.asarray(xyz)
    origin = np.asarray(camera_origin)
    centers, halves, axes = _box_arrays(box_centers, box_halves, box_axes)
    blocked = np.zeros(len(xyz), bool)
    for center, half, axis in zip(centers, halves, axes):
        start = (origin - center) @ axis
        delta = (xyz - origin) @ axis
        parallel = np.abs(delta) < 1e-10
        safe = np.where(parallel, 1.0, delta)
        a, b = (-half - start) / safe, (half - start) / safe
        near = np.where(parallel, -np.inf, np.minimum(a, b)).max(1)
        far = np.where(parallel, np.inf, np.maximum(a, b)).min(1)
        outside = (parallel & (np.abs(start) > half)).any(1)
        blocked |= ~outside & (far >= np.maximum(near, 0)) & (near < 0.999) & (far > 0)
    return blocked


def build_foreground_guard(image_shape, boxes_xyxy=None, foreground_guard=None, margin_px=2):
    """源图 2D 注释矩形并集膨胀，再并入调用方已准备的遮挡 mask。

    boxes_xyxy 使用该源图尺度的 [x0,y0,x1,y1]，右/下边界不包含。
    foreground_guard 按原提取链作为已准备好的 guard 使用，不在此二次膨胀。
    """
    height, width = map(int, image_shape)
    mask = np.zeros((height, width), bool)
    for x0, y0, x1, y1 in np.asarray([] if boxes_xyxy is None else boxes_xyxy).reshape(-1, 4):
        x0, x1 = max(0, int(np.floor(x0))), min(width, int(np.ceil(x1)))
        y0, y1 = max(0, int(np.floor(y0))), min(height, int(np.ceil(y1)))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    if margin_px > 0:
        mask = binary_dilation(mask, iterations=int(margin_px))
    if foreground_guard is not None:
        guard = np.asarray(foreground_guard, dtype=bool)
        if guard.shape != mask.shape:
            raise ValueError("foreground guard and image_shape differ")
        mask |= guard
    return mask


def measured_patch_evidence(lidar_xyz, lidar_ground, K, world_to_camera, query_xyz,
                            query_pixels, query_normals, *, image_shape,
                            lidar_pixels=None, lidar_depth=None, query_depth=None):
    """父格实测小三角形认证，返回 (valid, harmonic_depth, triangle_ids, conflict)。

    triangle_ids 指向输入 lidar_xyz；无法构建三角网时返回 -1，其余只在
    valid 为真时构成合格来源。可显式提供已投影的
    LiDAR XY/depth，避免重复投影，但必须与同一个 K/相机/减0.5约定一致。
    query_normals 必须为单位法向；保持原公式，不在此另行归一化。
    """
    cloud = np.asarray(lidar_xyz)
    ground = np.asarray(lidar_ground, dtype=bool)
    query = np.asarray(query_xyz)
    pixels = np.asarray(query_pixels)
    normals = np.asarray(query_normals)
    height, width = map(int, image_shape)
    count = len(query)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or ground.shape != (len(cloud),):
        raise ValueError("LiDAR must be Nx3 with an N ground mask")
    if query.shape != (count, 3) or pixels.shape != (count, 2) or normals.shape != (count, 3):
        raise ValueError("query xyz/pixels/normals have incompatible shapes")
    acceptable = np.zeros(count, bool)
    interpolation = np.zeros(count)
    triangles = np.full((count, 3), -1, dtype=np.int64)
    conflict = np.zeros(count, bool)
    if count == 0:
        return acceptable, interpolation, triangles, conflict
    if (lidar_pixels is None) != (lidar_depth is None):
        raise ValueError("lidar_pixels and lidar_depth must be provided together")
    if lidar_pixels is None:
        uv, depth = project(cloud, K, world_to_camera)
    else:
        uv, depth = np.asarray(lidar_pixels), np.asarray(lidar_depth)
        if uv.shape != (len(cloud), 2) or depth.shape != (len(cloud),):
            raise ValueError("projected LiDAR shapes differ")
    qdepth = (transform(query, world_to_camera)[:, 2] if query_depth is None else np.asarray(query_depth))
    if qdepth.shape != (count,):
        raise ValueError("query_depth shape differs")
    inside = ((depth > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < width - 1)
              & (uv[:, 1] >= 0) & (uv[:, 1] < height - 1))
    indices = np.flatnonzero(inside)
    pixel = np.floor(uv[indices] + 0.5).astype(np.int64)
    # 同一像素只保留最近的真实返回，不能以其后方 ground 穿过前方物体。
    order = np.argsort(depth[indices], kind="stable")
    _, first = np.unique(pixel[order, 1] * width + pixel[order, 0], return_index=True)
    nearest = indices[order[first]]
    if not len(nearest):
        return acceptable, interpolation, triangles, conflict
    qi = np.flatnonzero(np.isfinite(pixels).all(1) & np.isfinite(qdepth))
    if not len(qi):
        return acceptable, interpolation, triangles, conflict
    distances, hit = cKDTree(uv[nearest]).query(pixels[qi])
    hit = nearest[hit]
    direct = distances <= 1.5
    conflict[qi] = direct & ((~ground[hit]) | (np.abs(depth[hit] - qdepth[qi]) > 0.3))
    vertices = nearest[ground[nearest]]
    if len(vertices) < 3:
        return acceptable, interpolation, triangles, conflict
    try:
        triangulation = Delaunay(uv[vertices])
    except QhullError:
        return acceptable, interpolation, triangles, conflict
    simplex = triangulation.find_simplex(pixels[qi])
    safe = np.maximum(simplex, 0)
    ids = vertices[triangulation.simplices[safe]]
    matrix = triangulation.transform[safe]
    first_two = np.einsum("nij,nj->ni", matrix[:, :2], pixels[qi] - matrix[:, 2])
    barycentric = np.c_[first_two, 1 - first_two.sum(1)]
    measured = 1 / np.maximum((barycentric / depth[ids]).sum(1), 1e-8)
    world_edges = np.stack([np.linalg.norm(cloud[ids[:, a]] - cloud[ids[:, b]], axis=1)
                            for a, b in ((0, 1), (1, 2), (2, 0))], 1)
    pixel_edges = np.stack([np.linalg.norm(uv[ids[:, a]] - uv[ids[:, b]], axis=1)
                            for a, b in ((0, 1), (1, 2), (2, 0))], 1)
    normal = np.cross(cloud[ids[:, 1]] - cloud[ids[:, 0]], cloud[ids[:, 2]] - cloud[ids[:, 0]])
    norm = np.linalg.norm(normal, axis=1)
    acceptable[qi] = ((simplex >= 0) & np.isfinite(measured) & (norm > 1e-8)
        & (world_edges.max(1) <= 1.6) & (pixel_edges.max(1) <= 24.0)
        & (np.abs((normal * normals[qi]).sum(1)) / np.maximum(norm, 1e-8) >= 0.9)
        & (np.abs(measured - qdepth[qi]) <= 0.3) & ~conflict[qi])
    interpolation[qi], triangles[qi] = measured, ids
    return acceptable, interpolation, triangles, conflict


def source_view_score(xyz, normals, camera_origin, depth, valid=None):
    """原细格颜色选择分数 abs(n dot view_unit)/z²。"""
    ray = np.asarray(camera_origin) - np.asarray(xyz)
    length = np.linalg.norm(ray, axis=1)
    cosine = np.abs((ray / np.maximum(length[:, None], 1e-8) * np.asarray(normals)).sum(1))
    score = cosine / np.maximum(np.asarray(depth), 0.1) ** 2
    return np.where(np.asarray(valid, bool), score, 0) if valid is not None else score


def certify_parent_sources(*, lidar_xyz, lidar_ground, K, world_to_camera,
                           query_xyz, query_normals, foreground_guard, surface_supported,
                           box_centers=None, box_halves=None, box_axes=None, source_quality=1.0,
                           query_pixels=None, query_depth=None, lidar_pixels=None,
                           lidar_depth=None, camera_origin=None):
    """008 父格完整来源资格；前置 footprint gate 仍使用 20 cm 证据尺度。

    先筛局部面支撑、in-frame、双线性 guard==0、cos>=.12、area>=.25px、
    quality>=.2，再做 measured_patch_evidence 及所有 3D OBB 的 ray veto。
    不附加网络语义类别、颜色借用、场景范围或 actor 过滤。
    """
    xyz, normals = np.asarray(query_xyz), np.asarray(query_normals)
    guard = np.asarray(foreground_guard, dtype=bool)
    supported = np.asarray(surface_supported, dtype=bool)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or normals.shape != xyz.shape or supported.shape != (len(xyz),):
        raise ValueError("parent xyz/normals/support have incompatible shapes")
    if guard.ndim != 2:
        raise ValueError("foreground_guard must be HxW")
    if (query_pixels is None) != (query_depth is None):
        raise ValueError("query_pixels and query_depth must be provided together")
    uv, depth = project(xyz, K, world_to_camera) if query_pixels is None else (np.asarray(query_pixels), np.asarray(query_depth))
    origin = np.linalg.inv(world_to_camera)[:3, 3] if camera_origin is None else np.asarray(camera_origin)
    ray = origin - xyz
    length = np.linalg.norm(ray, axis=1)
    cosine = np.abs((ray / np.maximum(length[:, None], 1e-9) * normals).sum(1))
    area = np.asarray(K)[0, 0] * np.asarray(K)[1, 1] * 0.2 ** 2 * cosine / np.maximum(depth * depth, 1e-8)
    height, width = guard.shape
    inside = ((depth > 0.1) & np.isfinite(uv).all(1) & (uv[:, 0] >= 0)
              & (uv[:, 0] < width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < height - 1))
    safe_uv = np.where(np.isfinite(uv), uv, -1)
    clear_guard = map_coordinates(guard.astype(float), [safe_uv[:, 1], safe_uv[:, 0]],
                                 order=1, mode="constant", cval=1) == 0
    candidate = supported & inside & clear_guard & (cosine >= 0.12) & (area >= 0.25) & (np.asarray(source_quality) >= 0.2)
    ids = np.flatnonzero(candidate)
    valid = np.zeros(len(xyz), bool)
    measured_depth = np.zeros(len(xyz))
    triangle_ids = np.full((len(xyz), 3), -1, np.int64)
    measured_conflict = np.zeros(len(xyz), bool)
    box_blocked = np.zeros(len(xyz), bool)
    if len(ids):
        ok, measured, triangles, conflict = measured_patch_evidence(
            lidar_xyz, lidar_ground, K, world_to_camera, xyz[ids], uv[ids], normals[ids],
            image_shape=guard.shape, lidar_pixels=lidar_pixels, lidar_depth=lidar_depth, query_depth=depth[ids])
        blocked = ray_box_blocked(xyz[ids], origin, box_centers, box_halves, box_axes)
        valid[ids] = ok & ~blocked
        measured_depth[ids], triangle_ids[ids] = measured, triangles
        measured_conflict[ids], box_blocked[ids] = conflict, blocked
    return dict(valid=valid, candidate=candidate, measured_depth=measured_depth,
                triangle_ids=triangle_ids, measured_conflict=measured_conflict,
                box_blocked=box_blocked, score=source_view_score(xyz, normals, origin, depth, valid),
                cosine=cosine, evidence_area_px=area, query_pixels=uv, query_depth=depth)


def inherit_child_sources(parent_valid, parent_ids, child_pixels, child_depth, foreground_guard):
    """忠实细格认证：继承逻辑父格来源，再检查子格 in-frame 与 order0 guard。

    parent_ids 是逻辑 8 cm 父格 ID；不得换成用于补几何的最近其他父格 ID。
    不重做细格 LiDAR/三角形/3D OBB 认证。无父格（-1）保持不合格。
    """
    parent_valid = np.asarray(parent_valid, dtype=bool)
    parent_ids = np.asarray(parent_ids)
    uv, depth = np.asarray(child_pixels), np.asarray(child_depth)
    guard = np.asarray(foreground_guard, dtype=bool)
    count = len(parent_ids)
    if parent_valid.ndim != 1 or parent_ids.shape != (count,) or parent_ids.dtype.kind not in "iu":
        raise ValueError("parent_valid must be 1D and parent_ids must be 1D integer indices")
    if uv.shape != (count, 2) or depth.shape != (count,) or guard.ndim != 2:
        raise ValueError("child pixels/depth/guard have incompatible shapes")
    height, width = guard.shape
    inside_parent = (parent_ids >= 0) & (parent_ids < len(parent_valid))
    valid = np.zeros(count, bool)
    ids = np.flatnonzero(inside_parent)
    valid[ids] = parent_valid[parent_ids[ids]]
    valid &= ((depth > 0.1) & np.isfinite(uv).all(1) & (uv[:, 0] >= 0)
              & (uv[:, 0] < width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < height - 1))
    safe_uv = np.where(np.isfinite(uv), uv, -1)
    valid &= map_coordinates(guard.astype(float), [safe_uv[:, 1], safe_uv[:, 0]],
                             order=0, mode="constant", cval=1) < 0.5
    return valid


class LocalRoadSurface:
    """原局部面：已去重 ground pool 的 8-NN 残差 IDW；不产生实拍来源资格。

    plane 可为 z=a*x+b*y+c 的长度3数组，或 n·xyz+d=0 的长度4数组。
    pool 的地面资格、ALL OBB 点内排除及 10 cm XY 去重由调用方完成。
    """
    def __init__(self, ground_xyz, plane):
        self.points = np.asarray(ground_xyz, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 3 or not len(self.points):
            raise ValueError("ground_xyz must be nonempty Nx3")
        coefficients = np.asarray(plane, dtype=float)
        if coefficients.shape == (4,):
            if abs(coefficients[2]) < 1e-12:
                raise ValueError("road plane has no finite z(x,y)")
            coefficients = -coefficients[[0, 1, 3]] / coefficients[2]
        if coefficients.shape != (3,):
            raise ValueError("plane must have three height coefficients or four plane coefficients")
        self.plane = coefficients
        self.tree = cKDTree(self.points[:, :2])

    def height(self, xy):
        xy = np.asarray(xy)
        distance, ids = self.tree.query(xy, k=8, distance_upper_bound=0.8)
        valid = np.isfinite(distance)
        points = self.points[np.minimum(ids, len(self.points) - 1)]
        residual = points[..., 2] - points[..., :2] @ self.plane[:2] - self.plane[2]
        weight = np.where(valid, 1 / np.maximum(distance, 0.05) ** 2, 0)
        height = xy @ self.plane[:2] + self.plane[2] + (weight * residual).sum(1) / np.maximum(weight.sum(1), 1e-12)
        return height, valid.sum(1) >= 3

    def normal(self, xy):
        xy = np.asarray(xy)
        dx = (self.height(xy + [0.05, 0])[0] - self.height(xy - [0.05, 0])[0]) / 0.1
        dy = (self.height(xy + [0, 0.05])[0] - self.height(xy - [0, 0.05])[0]) / 0.1
        normal = np.c_[-dx, -dy, np.ones(len(xy))]
        return normal / np.linalg.norm(normal, axis=1)[:, None]


def linear(srgb):
    srgb = np.asarray(srgb)
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def srgb(linear_rgb):
    linear_rgb = np.asarray(linear_rgb)
    return np.where(linear_rgb <= 0.0031308, linear_rgb * 12.92,
                    1.055 * np.maximum(linear_rgb, 1e-10) ** (1 / 2.4) - 0.055)


def sample_linear_rgb(image_srgb, pixels):
    """原顺序：双线性采样源 sRGB，再转 linear RGB；uint8 或 float[0,1]。"""
    image = np.asarray(image_srgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_srgb must be HxWx3")
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    pixels = np.asarray(pixels)
    sampled = np.stack([map_coordinates(image[:, :, channel], [pixels[:, 1], pixels[:, 0]],
                                       order=1, mode="nearest") for channel in range(3)], 1)
    return linear(sampled)


def select_reference(rgb_linear, source_valid, scores, *, chunk_size=65536):
    """按原 q75−.025 上辐射簇、最佳分辨率分数和 RGB<.08 的三帧一致选色。

    输入 [sources,cells,3] / [sources,cells]，可为 memmap；按 cells 分块，
    避免对全源全细格创建额外巨型 RGB 差数组。best_source_index 仅 real 有效；
    chosen_index 保留 argmax 的原结果供完整重放。
    """
    rgb, valid, score = np.asarray(rgb_linear), np.asarray(source_valid, dtype=bool), np.asarray(scores)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or valid.shape != rgb.shape[:2] or score.shape != valid.shape or not rgb.shape[0]:
        raise ValueError("source RGB/valid/scores have incompatible shapes")
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    sources, cells = valid.shape
    reference = np.zeros((cells, 3), dtype=rgb.dtype)
    agree = np.zeros((sources, cells), bool)
    chosen = np.zeros(cells, np.int64)
    q75_all = np.full(cells, np.nan)
    for start in range(0, cells, int(chunk_size)):
        stop = min(cells, start + int(chunk_size))
        colors, eligible = rgb[:, start:stop], valid[:, start:stop]
        luma = colors @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            q75 = np.nanquantile(np.where(eligible, luma, np.nan), 0.75, axis=0)
        clean = eligible & (luma >= q75[None] - 0.025)
        best = np.argmax(np.where(clean, score[:, start:stop], -1), axis=0)
        selected = colors[best, np.arange(stop - start)]
        reference[start:stop] = selected
        agree[:, start:stop] = clean & (np.abs(colors - selected[None]).max(2) < 0.08)
        chosen[start:stop], q75_all[start:stop] = best, q75
    real = agree.sum(0) >= 3
    return dict(reference_linear=reference, real=real, source_agree=agree,
                chosen_index=chosen, best_source_index=np.where(real, chosen, -1), q75=q75_all)


def infer_asphalt_neighbours(xyz, reference_linear, real):
    """原 8-NN 沥青 IDW；仅最近邻距离≤0.60m，推断 confidence=.25。

    注意：原实现并不要求另外七个邻居也在0.60m内。推断不是 observed truth，
    不改变 real/source_valid/source_agree，调用方须继续保存独立 inferred 标签。
    """
    xy = np.asarray(xyz)[:, :2]
    reference = np.array(reference_linear, copy=True)
    real = np.asarray(real, dtype=bool)
    color = srgb(reference)
    asphalt = real & (np.ptp(color, axis=1) < 0.18) & (color.max(1) < 0.65)
    ids = np.flatnonzero(asphalt)
    inferred = np.zeros(len(xy), bool)
    if len(ids) >= 8:
        distance, near = cKDTree(xy[ids]).query(xy, k=8)
        inferred = ~real & (distance[:, 0] <= 0.60)
        weight = 1 / np.maximum(distance, 0.02) ** 2
        neighbors = ids[near[inferred]]
        reference[inferred] = ((reference[neighbors] * weight[inferred, :, None]).sum(1)
                               / weight[inferred].sum(1)[:, None])
    confidence = np.where(real, 1.0, np.where(inferred, 0.25, 0.0)).astype(np.float32)
    return dict(reference_linear=reference, inferred=inferred, confidence=confidence, asphalt=asphalt)
