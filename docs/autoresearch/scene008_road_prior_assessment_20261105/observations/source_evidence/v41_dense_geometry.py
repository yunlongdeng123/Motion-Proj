"""CPU 构造监督证据；不修改道路高斯，不将插值称为实测密集深度。"""

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, map_coordinates, convolve
from scipy.spatial import Delaunay, QhullError, cKDTree

MASK_VERSION = 'V41_FIXED_DENSE_EVIDENCE_R2'
MASK_POLICY = dict(cell_m=.2, width=480, height=320, boundary_margin_px=2,
    measured_ground_plane_distance_m=.15, measured_triangle_max_edge_m=1.6,
    measured_triangle_max_edge_px=24., measured_triangle_normal_cosine=.9,
    measured_depth_tolerance_m=.3, nearest_return_radius_px=1.5,
    minimum_confirmed_views=2, ssim_window=11, ssim_min_valid_fraction=.8,
    pixel_convention='native_pixel_center_K_projection_minus_0.5',
    missing_support='unknown_no_gap_fill', semantic_ground_truth=False)


def transform(x, matrix):
    return np.asarray(x) @ matrix[:3, :3].T+matrix[:3, 3]


def project(world, K, w2c):
    camera = transform(world, w2c)
    homogeneous = camera @ K.T
    return homogeneous[:, :2]/np.maximum(camera[:, 2:], 1e-8)-.5, camera[:, 2]


def ray_box_blocked(world, origin, boxes):
    blocked = np.zeros(len(world), bool)
    for box in boxes:
        start = (origin-box['center']) @ box['axes']
        delta = (world-origin) @ box['axes']
        parallel = np.abs(delta) < 1e-10
        safe = np.where(parallel, 1., delta)
        a, b = (-box['half']-start)/safe, (box['half']-start)/safe
        near = np.where(parallel, -np.inf, np.minimum(a, b)).max(1)
        far = np.where(parallel, np.inf, np.maximum(a, b)).min(1)
        outside = (parallel & (np.abs(start) > box['half'])).any(1)
        blocked |= ~outside & (far >= np.maximum(near, 0)) & (near < .999) & (far > 0)
    return blocked


class FixedSurface:
    """原始固定中心的局部双线性参考面；四角缺任意支撑便不接受。"""
    def __init__(self, meta, units):
        self.anchor = np.asarray(meta['coordinate_anchor'])
        self.inverse = np.linalg.inv(self.anchor)
        self.keys = units['keys'].astype(np.int64)
        self.lower, upper = self.keys.min(0), self.keys.max(0)
        self.grid = np.full(tuple(upper-self.lower+1), -1, np.int64)
        self.grid[tuple((self.keys-self.lower).T)] = np.arange(len(self.keys))
        self.local = transform(units['world'], self.inverse)
        self.supported = units['surface_supported'].astype(bool)
        self.observed = self.supported & (units['provenance'] == 1) & ((units['base_weight'] > 0).sum(0) >= 2)
        self.normals = units['normal_world'] @ self.anchor[:3, :3]
        self.plane = np.linalg.lstsq(np.c_[self.local[self.supported, :2],
            np.ones(self.supported.sum())], self.local[self.supported, 2], rcond=None)[0]

    def ids(self, keys):
        indices = keys-self.lower
        inside = ((indices >= 0) & (indices < np.asarray(self.grid.shape))).all(1)
        indices = np.clip(indices, 0, np.asarray(self.grid.shape)-1)
        return np.where(inside, self.grid[tuple(indices.T)], -1)

    def height(self, xy):
        q = np.clip(np.nan_to_num(xy/.2-.5), -1e6, 1e6)
        key = np.floor(q).astype(np.int64)
        fraction = q-key
        ids = np.stack([self.ids(key+offset) for offset in ((0, 0), (1, 0), (0, 1), (1, 1))], 1)
        safe = np.maximum(ids, 0)
        weights = np.stack(((1-fraction[:, 0])*(1-fraction[:, 1]),
            fraction[:, 0]*(1-fraction[:, 1]), (1-fraction[:, 0])*fraction[:, 1],
            fraction[:, 0]*fraction[:, 1]), 1)
        heights = (self.local[safe, 2]*weights).sum(1)
        valid = (ids >= 0).all(1) & self.supported[safe].all(1)
        normal = self.normals[safe]
        valid &= ((normal*normal[:, :1]).sum(2) >= .9).all(1)
        return heights, valid

    def camera(self, K, w2c):
        yy, xx = np.mgrid[:320, :480]
        pixels = np.c_[xx.ravel(), yy.ravel()]
        c2w = np.linalg.inv(w2c)
        rays = np.c_[pixels+.5, np.ones(len(pixels))] @ np.linalg.inv(K).T
        rays = rays @ c2w[:3, :3].T @ self.inverse[:3, :3].T
        origin = transform(c2w[None, :3, 3], self.inverse)[0]
        denominator = rays[:, 2]-rays[:, :2] @ self.plane[:2]
        numerator = origin[:2] @ self.plane[:2]+self.plane[2]-origin[2]
        distance = np.divide(numerator, denominator, out=np.full(len(rays), 1e5), where=np.abs(denominator) > 1e-7)
        initial = origin+distance[:, None]*rays
        potential = ((distance > .1) & (distance < 120) & (initial[:, 0] >= 3)
                     & (initial[:, 0] <= 90) & (np.abs(initial[:, 1]) <= 8))
        for _ in range(8):
            point = origin+distance[:, None]*rays
            height, supported = self.height(point[:, :2])
            correction = np.divide(height-point[:, 2], denominator, out=np.zeros_like(distance),
                                   where=(np.abs(denominator) > 1e-7) & supported & potential)
            distance += np.clip(correction, -10, 10)
        local = origin+distance[:, None]*rays
        height, support = self.height(local[:, :2])
        unit = self.ids(np.floor(np.clip(local[:, :2]/.2, -1e6, 1e6)).astype(np.int64))
        valid = potential & support & (np.abs(height-local[:, 2]) < .03) & (unit >= 0)
        observed = valid & self.observed[np.maximum(unit, 0)]
        return dict(world=transform(local, self.anchor), depth=distance, potential=potential,
                    supported=valid, observed=observed, unit_id=unit, pixels=pixels)


def measured_patch_evidence(world, ground, K, w2c, query_world, query_pixels, query_normals):
    """用同帧最近地面回波的小三角形验证，保留三顶点来源；不外插。"""
    uv, depth = project(world, K, w2c)
    inside = (depth > .1) & (uv[:, 0] >= 0) & (uv[:, 0] < 479) & (uv[:, 1] >= 0) & (uv[:, 1] < 319)
    indices = np.flatnonzero(inside)
    pixel = np.floor(uv[indices]+.5).astype(np.int64)
    # 每个像素只保留最近回波，不能让后方路面穿过当前非路面回波。
    order = np.argsort(depth[indices], kind='stable')
    _, first = np.unique(pixel[order, 1]*480+pixel[order, 0], return_index=True)
    nearest = indices[order[first]]
    query_depth = transform(query_world, w2c)[:, 2]
    if not len(nearest):
        return np.zeros(len(query_world), bool), np.zeros(len(query_world)), np.full((len(query_world), 3), -1), np.zeros(len(query_world), bool)
    distances, hit = cKDTree(uv[nearest]).query(query_pixels)
    hit = nearest[hit]
    direct = (distances <= MASK_POLICY['nearest_return_radius_px'])
    measured_conflict = direct & ((~ground[hit]) | (np.abs(depth[hit]-query_depth) > .3))
    vertices = nearest[ground[nearest]]
    if len(vertices) < 3:
        return np.zeros(len(query_world), bool), np.zeros(len(query_world)), np.full((len(query_world), 3), -1), measured_conflict
    try:
        triangulation = Delaunay(uv[vertices])
    except QhullError:
        return np.zeros(len(query_world), bool), np.zeros(len(query_world)), np.full((len(query_world), 3), -1), measured_conflict
    simplex = triangulation.find_simplex(query_pixels)
    safe = np.maximum(simplex, 0)
    ids = vertices[triangulation.simplices[safe]]
    matrix = triangulation.transform[safe]
    first_two = np.einsum('nij,nj->ni', matrix[:, :2], query_pixels-matrix[:, 2])
    barycentric = np.c_[first_two, 1-first_two.sum(1)]
    interpolation = 1/np.maximum((barycentric/depth[ids]).sum(1), 1e-8)
    world_edges = np.stack([np.linalg.norm(world[ids[:, a]]-world[ids[:, b]], axis=1)
                           for a, b in ((0, 1), (1, 2), (2, 0))], 1)
    pixel_edges = np.stack([np.linalg.norm(uv[ids[:, a]]-uv[ids[:, b]], axis=1)
                           for a, b in ((0, 1), (1, 2), (2, 0))], 1)
    normal = np.cross(world[ids[:, 1]]-world[ids[:, 0]], world[ids[:, 2]]-world[ids[:, 0]])
    norm = np.linalg.norm(normal, axis=1)
    acceptable = ((simplex >= 0) & np.isfinite(interpolation) & (norm > 1e-8)
        & (world_edges.max(1) <= MASK_POLICY['measured_triangle_max_edge_m'])
        & (pixel_edges.max(1) <= MASK_POLICY['measured_triangle_max_edge_px'])
        & (np.abs((normal*query_normals).sum(1))/np.maximum(norm, 1e-8)
           >= MASK_POLICY['measured_triangle_normal_cosine'])
        & (np.abs(interpolation-query_depth) <= MASK_POLICY['measured_depth_tolerance_m'])
        & ~measured_conflict)
    return acceptable, interpolation, ids, measured_conflict


def finalize_masks(single, source_count, protect):
    visible = single & (source_count >= MASK_POLICY['minimum_confirmed_views']) & ~protect
    count = convolve(visible.astype(np.float32), np.ones((11, 11), np.float32), mode='constant', cval=0)
    interior = np.zeros_like(visible); interior[5:-5, 5:-5] = True
    windows = (count >= 121*MASK_POLICY['ssim_min_valid_fraction']) & visible & interior
    return visible, windows
