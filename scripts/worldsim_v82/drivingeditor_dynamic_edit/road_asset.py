"""固定道路上的观测优先材质图集与独立表面覆盖修复。

图集是参考曝光下的 sRGB 外观，不是光照已分离的反射率。几何与颜色
分别保存；未知颜色有显式标志，不能因原资产有有限 RGB 而伪装成实测。
"""
from pathlib import Path
import json

import numpy as np


SOURCE_UNKNOWN, SOURCE_OBSERVED, SOURCE_RETRIEVED, SOURCE_GENERATED = range(4)
SOURCE_NAMES = {0: "unknown", 1: "observed", 2: "retrieved", 3: "generated"}


def infer_measured_plane_surface(world_xyz, normal_world, candidate_mask, measurement_world,
                                 plane_normal, plane_offset, radius_m=1.5,
                                 maximum_edge_m=3., maximum_plane_residual_m=.05):
    """在独立道路测距支持凸包中补固定平面，不改任何既有活跃节点。

    高斯下方无直接测距不等于路面未知：已冻结测距平面的真实内点包围、
    最近距离和支持三角跨度共同限定可插值范围；新节点始终标记inferred。
    """
    from scipy.spatial import cKDTree, Delaunay
    world = np.asarray(world_xyz, np.float64)
    normals = np.asarray(normal_world, np.float64)
    measurement = np.asarray(measurement_world, np.float64)
    normal = np.asarray(plane_normal, np.float64)
    normal /= np.linalg.norm(normal)
    if normal[2] < .9:
        raise ValueError("道路平面不符合有限近水平高度场")
    selected = np.flatnonzero(np.abs(measurement @ normal - plane_offset) <= maximum_plane_residual_m)
    if len(selected) < 3:
        raise ValueError("独立测距平面缺少非退化支持")
    pool = measurement[selected]
    triangulation = Delaunay(pool[:, :2])
    ids = np.flatnonzero(candidate_mask)
    simplex = triangulation.find_simplex(world[ids, :2])
    inside = simplex >= 0
    triangles = triangulation.simplices[np.maximum(simplex, 0)]
    vertices = pool[triangles]
    edge = np.max(np.stack([np.linalg.norm(vertices[:, 0, :2] - vertices[:, 1, :2], axis=1),
        np.linalg.norm(vertices[:, 1, :2] - vertices[:, 2, :2], axis=1),
        np.linalg.norm(vertices[:, 2, :2] - vertices[:, 0, :2], axis=1)], axis=1), axis=1)
    nearest, _ = cKDTree(pool[:, :2]).query(world[ids, :2])
    okay = inside & (edge <= maximum_edge_m) & (nearest <= radius_m)
    accepted = np.zeros(len(world), bool)
    accepted[ids[okay]] = True
    target = world.copy()
    target[accepted, 2] = (plane_offset - target[accepted, :2] @ normal[:2]) / normal[2]
    output_normal = normals.copy()
    output_normal[accepted] = normal
    anchors = np.full((len(world), 3), -1, np.int64)
    anchors[ids[okay]] = selected[triangles[okay]]
    distance = np.full(len(world), np.nan, np.float32)
    distance[ids] = nearest
    maximum_edge = np.full(len(world), np.nan, np.float32)
    maximum_edge[ids] = edge
    inside_hull = np.zeros(len(world), bool)
    inside_hull[ids] = inside
    residual = np.full(len(world), np.nan, np.float32)
    residual[ids[okay]] = np.max(np.abs(vertices[okay] @ normal - plane_offset), axis=1)
    return target, output_normal, dict(surface_interpolated=accepted, interpolation_anchor_ids=anchors,
        interpolation_radius_m=distance, interpolation_triangle_max_edge_m=maximum_edge,
        interpolation_inside_measurement_hull=inside_hull, interpolation_height_residual_m=residual), dict(
        method="independent measured-road plane within inlier-support Delaunay hull",
        plane_normal_world=normal.tolist(), plane_offset_world=float(plane_offset),
        plane_inlier_anchor_count=int(len(pool)), maximum_anchor_plane_residual_m=maximum_plane_residual_m,
        maximum_nearest_anchor_distance_m=radius_m, maximum_support_triangle_edge_m=maximum_edge_m,
        support_triangle_anchor_count=3, accepted_nodes=int(accepted.sum()),
        new_surface_semantics="inferred from fixed measured plane, not directly measured nodes")


def interpolate_legal_surface(world_xyz, normal_world, anchor_mask, candidate_mask,
                              radius_m=1.5, max_height_residual_m=.025):
    """在邻近同一可信路面内补节点；仅候选节点高度改变，XY与原锚点固定。

    四周锚点必须包围候选、近水平、法向一致，各锚切平面的预测高度也
    一致。失败点保留未认证状态，不扩大未知道路轮廓。
    """
    from scipy.spatial import cKDTree
    world = np.asarray(world_xyz, np.float64)
    normal = np.asarray(normal_world, np.float64)
    anchor = np.asarray(anchor_mask, bool)
    candidate = np.asarray(candidate_mask, bool) & ~anchor
    n = len(world)
    norm = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    norm *= np.where(norm[:, 2:3] < 0, -1, 1)
    usable = anchor & (norm[:, 2] >= .94)
    anchors = np.flatnonzero(usable)
    ids = np.flatnonzero(candidate)
    accepted = np.zeros(n, bool)
    out_world, out_normal = world.copy(), normal.copy()
    anchor_ids = np.full((n, 25), -1, np.int64)
    residual = np.full(n, np.nan, np.float32)
    radius = np.full(n, np.nan, np.float32)
    if len(anchors) < 8 or not len(ids):
        return out_world, out_normal, dict(surface_interpolated=accepted, interpolation_anchor_ids=anchor_ids,
            interpolation_height_residual_m=residual, interpolation_radius_m=radius)
    # 从三个米制半径、八个方向查询锚点，避免k近邻只覆盖孔洞单侧。
    angles = np.arange(8) * (np.pi / 4)
    directions = np.c_[np.cos(angles), np.sin(angles)]
    offsets = np.concatenate([np.zeros((1, 2))] + [directions * radius_m * fraction for fraction in [1 / 3, 2 / 3, 1]])
    tips = world[ids, None, :2] + offsets[None]
    _, index = cKDTree(world[anchors, :2]).query(tips.reshape(-1, 2))
    neighbor_ids = anchors[index.reshape(len(ids), 25)]
    point = world[neighbor_ids]
    normals = norm[neighbor_ids]
    delta = point[:, :, :2] - world[ids, None, :2]
    distance = np.linalg.norm(delta, axis=2)
    valid = distance <= radius_m + 1e-8
    for column in range(1, 25):
        valid[:, column] &= np.all(neighbor_ids[:, :column] != neighbor_ids[:, column:column + 1], axis=1)
    base_weight = np.where(valid, 1 / np.maximum(distance, .10) ** 2, 0)
    weight = base_weight / np.maximum(base_weight.sum(axis=1, keepdims=True), 1e-12)
    design = np.concatenate([delta, np.ones((*delta.shape[:2], 1))], axis=2)
    def fit(current_weight):
        lhs = np.einsum("nki,nk,nkj->nij", design, current_weight, design)
        rhs = np.einsum("nki,nk,nk->ni", design, current_weight, point[:, :, 2])
        eigenvalue = np.linalg.eigvalsh(lhs)
        solvable = eigenvalue[:, 0] > 1e-6
        lhs[~solvable] = np.eye(3)
        return np.linalg.solve(lhs, rhs[..., None])[..., 0], solvable
    # 原parent的局部IDW面存在少量粗糙法向；只让一致位置子集定义补全面。
    # 不移动或丢弃原表示，只在新节点的几何认证中拒绝这些离群锚点。
    for _ in range(5):
        coef, solvable = fit(weight)
        fitted = np.einsum("nki,ni->nk", design, coef)
        residual_height = np.abs(point[:, :, 2] - fitted)
        robust = np.minimum(1., max_height_residual_m / np.maximum(residual_height, 1e-8))
        weight = base_weight * robust
        weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-12)
    height_inlier = valid & (residual_height <= max_height_residual_m)
    height_inlier_ratio = height_inlier.sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
    plane_normal = np.c_[-coef[:, :2], np.ones(len(ids))]
    plane_normal /= np.linalg.norm(plane_normal, axis=1, keepdims=True)
    selected = height_inlier & (np.sum(normals * plane_normal[:, None], axis=2) >= .98)
    weight = base_weight * selected
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-12)
    coef, solvable = fit(weight)
    z = coef[:, 2]
    fitted = np.einsum("nki,ni->nk", design, coef)
    valid = selected
    error = np.max(np.where(valid, np.abs(point[:, :, 2] - fitted), 0), axis=1)
    plane_normal = np.c_[-coef[:, :2], np.ones(len(ids))]
    plane_normal /= np.linalg.norm(plane_normal, axis=1, keepdims=True)
    angular = np.min(np.where(valid, np.sum(normals * plane_normal[:, None], axis=2), 1), axis=1)
    bearing = np.mod(np.arctan2(delta[:, :, 1], delta[:, :, 0]), 2 * np.pi)
    ordered = np.sort(np.where(valid, bearing, np.inf), axis=1)
    count = valid.sum(axis=1)
    with np.errstate(invalid="ignore"):
        differences = np.diff(ordered, axis=1)
    internal_gap = np.max(np.where(np.arange(24)[None] < count[:, None] - 1, differences, 0), axis=1)
    last = ordered[np.arange(len(ids)), np.maximum(count - 1, 0)]
    first = np.where(count > 0, ordered[:, 0], 0)
    last = np.where(count > 0, last, 0)
    gap = np.maximum(internal_gap, first + 2 * np.pi - last)
    surrounds = gap < np.pi - .02
    okay = solvable & (height_inlier_ratio >= .6) & (valid.sum(axis=1) >= 8) & surrounds & (error <= max_height_residual_m) & (angular >= .98)
    take = ids[okay]
    out_world[take, 2] = z[okay]
    out_normal[take] = plane_normal[okay]
    accepted[take] = True
    residual[take] = error[okay]
    radius[take] = np.max(np.where(valid[okay], distance[okay], 0), axis=1)
    anchor_ids[take] = np.where(valid[okay], neighbor_ids[okay], -1)
    diagnostics = {}
    for name, value in [("anchor_count", valid.sum(axis=1)), ("height_error", error),
                        ("normal_cosine", angular), ("angular_gap", gap), ("solvable", solvable),
                        ("height_inlier_ratio", height_inlier_ratio)]:
        full = np.full(n, np.nan, np.float32)
        full[ids] = value
        diagnostics["interpolation_candidate_" + name] = full
    return out_world, out_normal, dict(surface_interpolated=accepted, interpolation_anchor_ids=anchor_ids,
        interpolation_height_residual_m=residual, interpolation_radius_m=radius, **diagnostics)


def srgb_to_linear(value):
    value = np.asarray(value)
    return np.where(value <= .04045, value / 12.92, ((value + .055) / 1.055) ** 2.4)


def linear_to_srgb(value):
    value = np.clip(value, 0, 1)
    return np.where(value <= .0031308, 12.92 * value, 1.055 * value ** (1 / 2.4) - .055)


def _bilinear_grid(array, uv, derivatives=False):
    """整数为纹理像素中心；边界内插，不把出界点当作有效材料。"""
    height, width = array.shape[:2]
    safe = np.clip(uv, [0, 0], [width - 1, height - 1])
    lo = np.minimum(np.floor(safe).astype(np.int64), [width - 2, height - 2])
    frac = safe - lo
    x, y = lo.T
    fx, fy = frac.T
    vals = np.stack([array[y, x], array[y, x + 1], array[y + 1, x], array[y + 1, x + 1]], axis=1)
    weight = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], axis=1)
    expand = (slice(None), slice(None)) + (None,) * (array.ndim - 2)
    value = (vals * weight[expand]).sum(axis=1)
    if not derivatives:
        return value, vals, weight, lo
    shape = (slice(None),) + (None,) * (array.ndim - 2)
    du = (vals[:, 1] - vals[:, 0]) * (1 - fy)[shape] + (vals[:, 3] - vals[:, 2]) * fy[shape]
    dv = (vals[:, 2] - vals[:, 0]) * (1 - fx)[shape] + (vals[:, 3] - vals[:, 1]) * fx[shape]
    return value, du, dv


class WorldRoadMaterialAtlas:
    """按米定义的二维材质场；查询只接收世界位置，不接收视频帧号。

    既有道路节点存在毫米级 XY 曲率，不能直接用一个近似仿射矩阵取色。
    图集保留每个固定世界节点，先反解双线性世界 XY 网格，再取双线性
    线性 RGB。world_from_uv 是参考坐标，surface_world_xyz 是精确定位。
    """
    schema = "world_road_material_atlas.v1"

    def __init__(self, rgb_srgb, confidence, source, legal, surface_world_xyz,
                 world_from_uv, grid_world_row=None, source_raw=None,
                 source_world_row=None, source_distance_m=None):
        self.rgb_srgb = np.asarray(rgb_srgb, np.float32).copy()
        self.confidence = np.asarray(confidence, np.float32).copy()
        self.source = np.asarray(source, np.uint8).copy()
        self.legal = np.asarray(legal, bool).copy()
        self.surface_world_xyz = np.asarray(surface_world_xyz, np.float64).copy()
        self.world_from_uv = np.asarray(world_from_uv, np.float64).copy()
        shape = self.rgb_srgb.shape[:2]
        if (self.rgb_srgb.shape != (*shape, 3) or min(shape) < 2
                or any(a.shape != shape for a in [self.confidence, self.source, self.legal])
                or self.surface_world_xyz.shape != (*shape, 3)
                or self.world_from_uv.shape != (4, 4)):
            raise ValueError("图集数组形状不一致，且每轴至少两个节点")
        if (not np.isfinite(self.rgb_srgb).all() or np.any((self.rgb_srgb < 0) | (self.rgb_srgb > 1))
                or not np.isfinite(self.confidence).all() or np.any((self.confidence < 0) | (self.confidence > 1))
                or not np.isfinite(self.surface_world_xyz).all() or not np.isfinite(self.world_from_uv).all()
                or np.any(self.source > 3) or np.any((self.source != 0) & ~self.legal)
                or np.any((self.source == 0) & (self.confidence != 0))):
            raise ValueError("图集颜色、来源、合法性或置信度不符合契约")
        if abs(np.linalg.det(self.world_from_uv[:2, :2])) < 1e-12:
            raise ValueError("世界 XY 到图集的参考变换退化")
        self.uv_from_xy = np.linalg.inv(self.world_from_uv[:2, :2])
        defaults = {"grid_world_row": -1, "source_raw": -1, "source_world_row": -1,
                    "source_distance_m": np.inf}
        for name, value in [("grid_world_row", grid_world_row), ("source_raw", source_raw),
                            ("source_world_row", source_world_row), ("source_distance_m", source_distance_m)]:
            value = np.full(shape, defaults[name]) if value is None else np.asarray(value)
            if value.shape != shape:
                raise ValueError(name + " 与图集网格不一致")
            setattr(self, name, value.copy())
        self._linear = srgb_to_linear(self.rgb_srgb)

    def query(self, world_xyz, surface_tolerance_m=.01):
        world = np.asarray(world_xyz, np.float64)
        if world.ndim != 2 or world.shape[1] != 3 or not np.isfinite(world).all():
            raise ValueError("世界位置必须是有限 N×3 米制数组")
        uv = (world[:, :2] - self.world_from_uv[:2, 3]) @ self.uv_from_xy.T
        height, width = self.confidence.shape
        uv = np.clip(uv, [0, 0], [width - 1, height - 1])
        # 原网格 XY 偏离仿射至多毫米级，局部 Newton 反解保留几何原位置。
        for _ in range(8):
            pos, du, dv = _bilinear_grid(self.surface_world_xyz[..., :2], uv, derivatives=True)
            residual = pos - world[:, :2]
            det = du[:, 0] * dv[:, 1] - du[:, 1] * dv[:, 0]
            if np.any(np.abs(det) < 1e-12):
                raise ValueError("世界 XY 网格发生折叠或退化")
            delta = np.c_[(dv[:, 1] * residual[:, 0] - dv[:, 0] * residual[:, 1]) / det,
                          (-du[:, 1] * residual[:, 0] + du[:, 0] * residual[:, 1]) / det]
            uv = np.clip(uv - delta, [0, 0], [width - 1, height - 1])
            if not len(delta) or np.max(np.abs(delta)) < 1e-8:
                break
        rounded = np.rint(uv)
        uv = np.where(np.abs(uv - rounded) < 1e-6, rounded, uv)
        inside = ((uv >= -1e-6) & (uv <= [width - 1 + 1e-6, height - 1 + 1e-6])).all(axis=1)
        position, _, weight, lo = _bilinear_grid(self.surface_world_xyz, uv)
        inside &= np.linalg.norm(position[:, :2] - world[:, :2], axis=1) < 1e-5
        inside &= np.abs(position[:, 2] - world[:, 2]) <= surface_tolerance_m
        color, _, _, _ = _bilinear_grid(self._linear, uv)
        conf, _, _, _ = _bilinear_grid(self.confidence, uv)
        _, sources, _, _ = _bilinear_grid(self.source, uv)
        _, legality, _, _ = _bilinear_grid(self.legal, uv)
        contributing = weight > 1e-9
        valid = inside & np.all(~contributing | (legality & (sources > 0)), axis=1)
        conf *= valid
        rgb = linear_to_srgb(color).astype(np.float32)
        nearest = np.clip(np.rint(uv).astype(np.int64), [0, 0], [width - 1, height - 1])
        x, y = nearest.T
        exact = (np.abs(uv - np.rint(uv)) < 1e-8).all(axis=1) & inside
        rgb[exact] = self.rgb_srgb[y[exact], x[exact]]
        source = self.source[y, x].copy()
        source[~valid] = 0
        mix = np.zeros((len(world), 4), np.float32)
        for category in range(4):
            mix[:, category] = (weight * (sources == category)).sum(axis=1)
        return dict(rgb_srgb=rgb, rgb_linear=color.astype(np.float32), confidence=conf.astype(np.float32),
                    source=source, source_fraction=mix, unknown=~valid, inside=inside, uv=uv,
                    source_raw=self.source_raw[y, x], source_world_row=self.source_world_row[y, x],
                    source_distance_m=self.source_distance_m[y, x])

    def query_xy(self, world_xy):
        """F(x,y)外观查询；表面高度/覆盖资格由独立几何支路判断。"""
        xy = np.asarray(world_xy, np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("世界XY必须是N×2米制位置")
        return self.query(np.c_[xy, np.zeros(len(xy))], surface_tolerance_m=np.inf)

    def save(self, path):
        np.savez_compressed(path, **{k: getattr(self, k) for k in ["rgb_srgb", "confidence", "source", "legal",
                            "surface_world_xyz", "world_from_uv", "grid_world_row", "source_raw",
                            "source_world_row", "source_distance_m"]},
                            known=self.source == SOURCE_OBSERVED, unknown=self.source == SOURCE_UNKNOWN)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as saved:
            return cls(**{k: saved[k] for k in ["rgb_srgb", "confidence", "source", "legal",
                "surface_world_xyz", "world_from_uv", "grid_world_row", "source_raw", "source_world_row", "source_distance_m"]})


def build_observation_priority_atlas(world_xyz, world_keys, observed_rgb, observed_mask,
                                     legal, generated_rgb=None, generated_mask=None,
                                     observed_source_raw=None, generated_source_raw=None,
                                     retrieval_radius_m=.60):
    """观测锁定 → 近邻实测沥青检索 → 可选冻结生成；没有证据就保留 unknown。

    检索只复制一个可审计的原观测节点，限制世界 XY 距离；不延长标线或
    混合非沥青像素。该局部材质先验不能标记为隐藏道路真实颜色。
    """
    from scipy.spatial import cKDTree
    world = np.asarray(world_xyz, np.float64)
    keys = np.asarray(world_keys, np.int64)
    n = len(world)
    observed_rgb = np.asarray(observed_rgb, np.float32)
    observed_mask = np.asarray(observed_mask, bool)
    legal = np.asarray(legal, bool)
    if world.shape != (n, 3) or keys.shape != (n, 2) or observed_rgb.shape != (n, 3) or any(a.shape != (n,) for a in [legal, observed_mask]):
        raise ValueError("道路节点、颜色和掩码形状不匹配")
    if (not np.isfinite(world).all() or not np.isfinite(observed_rgb).all()
            or np.any((observed_rgb < 0) | (observed_rgb > 1)) or retrieval_radius_m <= 0
            or np.any(observed_mask & ~legal)):
        raise ValueError("观测必须合法且全部输入有限；检索半径必须为正")
    local_keys = keys - keys.min(axis=0)
    width, height = local_keys.max(axis=0) + 1
    grid = np.full((height, width), -1, np.int64)
    grid[local_keys[:, 1], local_keys[:, 0]] = np.arange(n)
    if n != width * height or np.any(grid < 0):
        raise ValueError("要求完整且无重复的二维道路格点；未知材料由source标志表达")
    design = np.c_[local_keys, np.ones(n)]
    coef = np.linalg.lstsq(design, world[:, :2], rcond=None)[0]
    matrix = np.eye(4)
    matrix[:2, :2] = coef[:2].T
    matrix[:2, 3] = coef[2]
    matrix[2, 3] = np.median(world[:, 2])
    rgb = np.full((n, 3), .5, np.float32)
    confidence = np.zeros(n, np.float32)
    source = np.zeros(n, np.uint8)
    raw = np.full(n, -1, np.int16)
    donor = np.full(n, -1, np.int64)
    distance = np.full(n, np.inf, np.float32)
    rgb[observed_mask] = observed_rgb[observed_mask]
    confidence[observed_mask] = 1
    source[observed_mask] = SOURCE_OBSERVED
    donor[observed_mask] = np.flatnonzero(observed_mask)
    distance[observed_mask] = 0
    if observed_source_raw is not None:
        raw[observed_mask] = np.asarray(observed_source_raw)[observed_mask]
    lum = observed_rgb @ np.array([.2126, .7152, .0722], np.float32)
    # 与已有道路推断一致的近场沥青候选；避免将道路漆线扩成大块颜色。
    asphalt = observed_mask & (observed_rgb.max(axis=1) < .65) & (np.ptp(observed_rgb, axis=1) < .18) & (lum > .06)
    candidates = np.flatnonzero(asphalt)
    query_ids = np.flatnonzero(legal & ~observed_mask)
    if len(candidates) and len(query_ids):
        dist, index = cKDTree(world[candidates, :2]).query(world[query_ids, :2])
        accept = dist <= retrieval_radius_m
        ids, selected = query_ids[accept], candidates[index[accept]]
        rgb[ids] = rgb[selected]
        source[ids] = SOURCE_RETRIEVED
        confidence[ids] = .55 * np.exp(-.5 * (dist[accept] / retrieval_radius_m) ** 2)
        donor[ids] = selected
        distance[ids] = dist[accept]
        raw[ids] = raw[selected]
    if generated_rgb is not None:
        generated_rgb = np.asarray(generated_rgb, np.float32)
        mask = np.asarray(generated_mask, bool) & legal & (source == SOURCE_UNKNOWN)
        if generated_rgb.shape != (n, 3) or not np.isfinite(generated_rgb).all() or np.any((generated_rgb < 0) | (generated_rgb > 1)):
            raise ValueError("生成颜色必须是有限 N×3 [0,1]")
        rgb[mask] = generated_rgb[mask]
        source[mask] = SOURCE_GENERATED
        confidence[mask] = .25
        if generated_source_raw is not None:
            raw[mask] = np.asarray(generated_source_raw)[mask]
    atlas = WorldRoadMaterialAtlas(rgb[grid], confidence[grid], source[grid], legal[grid], world[grid], matrix,
        grid_world_row=grid, source_raw=raw[grid], source_world_row=donor[grid], source_distance_m=distance[grid])
    receipt = dict(schema=atlas.schema, grid_hw=[int(height), int(width)],
                   grid_cell_m=np.linalg.norm(matrix[:2, :2], axis=0).tolist(),
                   reference_affine_max_xy_residual_m=float(np.abs(design @ coef - world[:, :2]).max()),
                   exact_world_xy_inverse=True, retrieval_radius_m=float(retrieval_radius_m),
                   retrieval="single nearest legal observed asphalt node; world XY metric radius; no hidden answer",
                   source_counts={SOURCE_NAMES[k]: int((source == k).sum()) for k in range(4)},
                   observed_rgb_preserved=bool(np.array_equal(rgb[observed_mask], observed_rgb[observed_mask])),
                   new_diffusion_forwards=0, new_training_updates=0)
    return atlas, receipt


def repair_road_coverage(base, sampled, requested, surface_supported, trusted_geometry,
                         protected_geometry=None, cell_m=.02, opacity_floor=.99, world_keys=None):
    """覆盖支路只操作 active/面内足迹/透明度；固定位置、法向与保留几何不动。

    独立于 RGB 拟合，按网格间距设置非可信补全面的最低足迹。边界内缩
    检查保证扩大后的3σ足迹留在合法支持区；不能用扩大的高斯越过禁区。
    未知材质只报告，不用占位颜色增加新表面。
    """
    from scipy.ndimage import distance_transform_edt
    target = {k: np.array(v, copy=True) for k, v in base.items()}
    n = len(target["active"])
    requested, support, trusted = map(lambda a: np.asarray(a, bool), [requested, surface_supported, trusted_geometry])
    protected = np.zeros(n, bool) if protected_geometry is None else np.asarray(protected_geometry, bool)
    if any(a.shape != (n,) for a in [requested, support, trusted, protected]):
        raise ValueError("覆盖掩码必须与完整道路行一致")
    legal = requested & support & ~protected
    material = ~np.asarray(sampled["unknown"], bool)
    # 单位方格在格角的四邻峰值保证有限不透明度，不改法线方向厚度。
    minimum_sigma = float(cell_m) * .85
    distance_to_edge = np.full(n, np.inf)
    if world_keys is not None:
        keys = np.asarray(world_keys, np.int64)
        ij = keys - keys.min(axis=0)
        width, height = ij.max(axis=0) + 1
        domain = np.zeros((height + 2, width + 2), bool)
        domain[ij[:, 1] + 1, ij[:, 0] + 1] = legal
        distance_to_edge = distance_transform_edt(domain, sampling=cell_m)[ij[:, 1] + 1, ij[:, 0] + 1]
    final_extent = 3 * np.maximum(target["gaussian_sigma_m"][:, :2], minimum_sigma).max(axis=1)
    added = legal & material & ~target["active"] & (distance_to_edge > final_extent)
    target["active"][added] = True
    repair = legal & material & target["active"] & ~trusted & (distance_to_edge > final_extent)
    before_sigma = target["gaussian_sigma_m"].copy()
    target["gaussian_sigma_m"][repair, :2] = np.maximum(target["gaussian_sigma_m"][repair, :2], minimum_sigma)
    target["road_opacity"][repair] = np.maximum(target["road_opacity"][repair], float(opacity_floor))
    # 具体投影的前景/裁剪约束仍由持久场景与统一深度排序负责。
    # 所有新增覆盖仅可用原始固定格点，绝不从 H/W 膨胀推导几何。
    repaired = np.any(before_sigma != target["gaussian_sigma_m"], axis=1) | (target["road_opacity"] != base["road_opacity"])
    if np.any(repaired & (trusted | protected)) or np.any(added & ~legal):
        raise RuntimeError("覆盖修复越过固定几何边界")
    unchanged = [k for k in base if k not in ["active", "gaussian_sigma_m", "road_opacity"]]
    if not all(np.array_equal(target[k], base[k], equal_nan=True) for k in unchanged):
        raise RuntimeError("覆盖分支改变了固定几何或材质")
    return target, dict(legal_coverage=legal, coverage_added=added, coverage_repaired=repaired,
                       trusted_geometry=trusted, protected_geometry=protected,
                       unsupported_requested=requested & ~support, unknown_requested=legal & ~material), dict(
        schema="fixed_road_surface_coverage.v1", requested_rows=int(requested.sum()),
        supported_requested_rows=int((requested & support).sum()), unsupported_requested_rows=int((requested & ~support).sum()),
        newly_active_existing_nodes=int(added.sum()), footprint_repaired_nodes=int(repaired.sum()),
        unknown_material_requested_rows=int((legal & ~material).sum()), minimum_tangent_sigma_m=minimum_sigma,
        position_and_normal_updates=0, trusted_sigma_opacity_updates=0, added_nodes_outside_fixed_grid=0,
        note="surface support and material support are separate; unsupported requested geometry remains explicit")


def apply_persistent_road_guard(base, candidate, guard):
    """按真实保留物体成像贡献回退完整世界行，跨帧固定；不裁剪最终图像。

    调用方需保存真实同排序 alpha·T 的证据。此处只执行明确的完整行
    回退；独立材质图集及计划来源保留，渲染覆盖策略另行标记。
    """
    guard = np.asarray(guard, bool)
    n = len(base["active"])
    if guard.shape != (n,) or len(candidate["active"]) != n:
        raise ValueError("保留保护必须绑定同一完整道路行拓扑")
    target = {key: np.array(value, copy=True) for key, value in candidate.items()}
    for key in ["model_from_world", "meter_to_model"]:
        if key in base and (key not in target or not np.array_equal(target[key], base[key])):
            raise ValueError("保留保护不能绑定不同世界/模型变换 " + key)
    fields = []
    for key, original in base.items():
        if key not in target:
            raise ValueError("新道路缺少父资产字段 " + key)
        if original.ndim and original.shape[0] == n:
            if target[key].shape != original.shape:
                raise ValueError("完整道路行形状不一致 " + key)
            target[key][guard] = original[guard]
            fields.append(key)
    return target, dict(schema="persistent_road_retained_guard.v1", guarded_world_rows=int(guard.sum()),
        reverted_fields=fields, disabled_new_nodes=int((guard & candidate["active"] & ~base["active"]).sum()),
        appearance_policy="planned atlas preserved; guarded render rows use complete legacy state",
        image_space_paste=False, guard_fixed_across_frames=True)
