"""米制世界道路几何；仅相机/LiDAR/三维框，不生成或融合 RGB。"""

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, map_coordinates
from scipy.spatial import cKDTree


def required_occluder_masks(engineering, fid):
    """恢复V2的三路输入；缺任一路就拒绝，不能静默降级。"""
    from pathlib import Path
    from PIL import Image
    root = Path(engineering)
    paths = [root/'full5_pipeline/inputs/scene0008_089_137/matched_training_labels'/f'{fid}.png',
             root/'all_vehicle_pipeline/inputs/scene0008_089_137/dynamic_masks/dynamic_mask_0'/f'{fid}.png',
             root/'all_vehicle_native49/native_predictions'/f'{fid}.npz']
    first, second = [np.asarray(Image.open(path).resize((480, 320), Image.Resampling.NEAREST)) > 0
                     for path in paths[:2]]
    with np.load(paths[2], allow_pickle=False) as archive:
        masks = archive['masks']
        if masks.ndim != 3 or not np.isfinite(masks).all():
            raise ValueError('invalid original native proposal masks')
        native = masks.astype(bool).any(0)
    native = np.asarray(Image.fromarray(native).resize((480, 320), Image.Resampling.NEAREST)) > 0
    return first | second | native, dict(paths=[str(path) for path in paths],
        pixels=[int(first.sum()), int(second.sum()), int(native.sum())],
        native_extra_pixels=int((native & ~(first | second)).sum()), complete=True)


def transform(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def project(world, intrinsic, world_to_camera, wh=(480, 320)):
    camera = transform(world, world_to_camera)
    homogeneous = camera @ intrinsic.T
    uv = homogeneous[:, :2] / np.maximum(homogeneous[:, 2:], 1e-12)
    inside = ((camera[:, 2] > .1) & (uv[:, 0] >= 0) & (uv[:, 0] < wh[0]-1)
              & (uv[:, 1] >= 0) & (uv[:, 1] < wh[1]-1))
    return uv, camera[:, 2], inside


def sample(array, uv):
    uv = np.where(np.isfinite(uv), uv, -1)
    if array.ndim == 3:
        return np.stack([sample(array[..., channel], uv) for channel in range(array.shape[-1])], -1)
    return map_coordinates(array.astype(float), [uv[:, 1], uv[:, 0]], order=1,
                           mode='constant', cval=0)


def ray_box_blocked(world, origin, boxes):
    blocked = np.zeros(len(world), bool)
    for box in boxes:
        start = (origin-box['center']) @ box['axes']
        delta = (world-origin) @ box['axes']
        parallel = np.abs(delta) < 1e-10
        safe = np.where(parallel, 1, delta)
        first, second = (-box['half']-start)/safe, (box['half']-start)/safe
        near = np.where(parallel, -np.inf, np.minimum(first, second)).max(1)
        far = np.where(parallel, np.inf, np.maximum(first, second)).min(1)
        outside = (parallel & (np.abs(start) > box['half'])).any(1)
        blocked |= ~outside & (far >= np.maximum(near, 0)) & (near < .999) & (far > 0)
    return blocked


class MeasuredRoadSurface:
    def __init__(self, anchor, plane, points, radius=.8):
        self.anchor, self.inverse = anchor, np.linalg.inv(anchor)
        self.plane, self.points, self.radius = plane, points, radius
        self.tree = cKDTree(points[:, :2])

    def height(self, xy):
        z = xy @ self.plane[:2] + self.plane[2]
        distances, indices = self.tree.query(xy, k=8, distance_upper_bound=self.radius)
        valid = np.isfinite(distances)
        points = self.points[np.minimum(indices, len(self.points)-1)]
        residual = points[..., 2]-points[..., :2] @ self.plane[:2]-self.plane[2]
        weights = np.where(valid, 1/np.maximum(distances, .05)**2, 0)
        z += (weights*residual).sum(1)/np.maximum(weights.sum(1), 1e-12)
        return z, valid.sum(1) >= 3

    def intersect(self, pixels, intrinsic, world_to_camera):
        c2w = np.linalg.inv(world_to_camera)
        direction = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(intrinsic).T @ c2w[:3, :3].T
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        origin = transform(c2w[None, :3, 3], self.inverse)[0]
        direction = direction @ self.inverse[:3, :3].T
        denom = direction[:, 2]-direction[:, :2] @ self.plane[:2]
        numer = origin[:2] @ self.plane[:2]+self.plane[2]-origin[2]
        distance = np.divide(numer, denom, out=np.full(len(pixels), np.nan), where=np.abs(denom)>1e-7)
        for _ in range(12):
            points = origin+distance[:, None]*direction
            height, _ = self.height(np.nan_to_num(points[:, :2]))
            distance += np.divide(height-points[:, 2], denom, out=np.zeros_like(distance), where=np.abs(denom)>1e-7)
        points = origin+distance[:, None]*direction
        height, support = self.height(np.nan_to_num(points[:, :2]))
        valid = (np.isfinite(points).all(1) & (distance > .1) & (distance < 120)
                 & (np.abs(points[:, 2]-height) < .03))
        return points, valid, support

    def normals(self, local_xy):
        delta = .05
        dx = (self.height(local_xy+[delta, 0])[0]-self.height(local_xy-[delta, 0])[0])/(2*delta)
        dy = (self.height(local_xy+[0, delta])[0]-self.height(local_xy-[0, delta])[0])/(2*delta)
        normal = np.c_[-dx, -dy, np.ones(len(local_xy))] @ self.anchor[:3, :3].T
        return normal/np.linalg.norm(normal, axis=1)[:, None]


def sparse_depth_maps(world, ground, intrinsic, world_to_camera):
    uv, depth, inside = project(world, intrinsic, world_to_camera)
    order = np.flatnonzero(inside)
    order = order[np.argsort(depth[order])[::-1]]
    pixels = np.floor(uv[order]+.5).astype(int)
    nearest = np.full((320, 480), np.inf)
    road = np.zeros((320, 480), bool)
    for index, (x, y) in zip(order, pixels):
        nearest[y, x], road[y, x] = depth[index], ground[index]
    valid = np.isfinite(nearest)
    if not valid.any():
        return np.zeros_like(nearest), valid, road
    distance, indices = distance_transform_edt(~valid, return_indices=True)
    valid = distance <= 1.5
    return np.where(valid, nearest[tuple(indices)], 0), valid, road[tuple(indices)] & valid


def source_observation(world, normals, support, frame, policy):
    uv, depth, inside = project(world, frame['K'], frame['w2c'])
    origin = np.linalg.inv(frame['w2c'])[:3, 3]
    direction = origin-world
    distance = np.linalg.norm(direction, axis=1)
    cosine = np.abs((direction/np.maximum(distance[:, None], 1e-9)*normals).sum(1))
    area = frame['K'][0, 0]*frame['K'][1, 1]*policy['cell_m']**2*cosine/np.maximum(depth**2, 1e-6)
    mask = binary_dilation(frame['occluder'], iterations=policy['mask_margin_px'])
    measured = sample(frame['depth'], uv)
    gates = dict(in_view=inside, surface_supported=support,
                 mask_clear=sample(mask, uv) == 0,
                 road_support=sample(frame['road'], uv) > .99,
                 measured_depth_match=(sample(frame['valid'], uv) > .99) & (np.abs(measured-depth) < policy['depth_tolerance_m']),
                 boxes_clear=~ray_box_blocked(world, origin, frame['boxes']),
                 angle_ok=cosine >= policy['minimum_cosine'],
                 resolution_ok=area >= policy['minimum_area_px'],
                 quality_ok=np.full(len(world), frame['quality'] >= policy['minimum_quality']))
    accepted = np.logical_and.reduce(list(gates.values()))
    weight = np.where(accepted, cosine**2*frame['quality']/np.maximum(distance, 1)**2, 0)
    # 各帧原始 sRGB 保留；不做色彩一致性融合或按跨帧颜色差拒绝真实观测。
    return sample(frame['rgb']/255., uv), uv, measured, weight, gates
