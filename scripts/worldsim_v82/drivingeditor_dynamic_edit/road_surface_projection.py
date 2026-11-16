"""固定道路网格的精确三角投影；最近正深度替代易振荡的高度迭代。"""
import numpy as np
from numba import njit


@njit(cache=True)
def _raster(world, projected, depth, supported, minimum, height, width, image_h, image_w):
    zbuffer = np.full((image_h, image_w), np.inf)
    world_hit = np.full((image_h, image_w, 3), np.nan)
    face = np.full((image_h, image_w), -1, np.int64)
    triangle_rows = np.full((image_h, image_w, 3), -1, np.int64)
    barycentric = np.zeros((image_h, image_w, 3))
    for y in range(height - 1):
        for x in range(width - 1):
            for kind in range(2):
                a = y * width + x
                if kind == 0:
                    b, c = a + 1, a + width + 1
                else:
                    b, c = a + width + 1, a + width
                if not (supported[a] and supported[b] and supported[c]):
                    continue
                if depth[a] <= minimum[a] or depth[b] <= minimum[b] or depth[c] <= minimum[c]:
                    continue
                ax, ay = projected[a]
                bx, by = projected[b]
                cx, cy = projected[c]
                denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
                if abs(denominator) < 1e-12:
                    continue
                minx = max(int(np.ceil(min(ax, bx, cx) - .5)), 0)
                maxx = min(int(np.floor(max(ax, bx, cx) - .5)), image_w - 1)
                miny = max(int(np.ceil(min(ay, by, cy) - .5)), 0)
                maxy = min(int(np.floor(max(ay, by, cy) - .5)), image_h - 1)
                for py in range(miny, maxy + 1):
                    for px in range(minx, maxx + 1):
                        u, v = px + .5, py + .5
                        l0 = ((by - cy) * (u - cx) + (cx - bx) * (v - cy)) / denominator
                        l1 = ((cy - ay) * (u - cx) + (ax - cx) * (v - cy)) / denominator
                        l2 = 1 - l0 - l1
                        if min(l0, l1, l2) < -1e-9:
                            continue
                        w0, w1, w2 = l0 / depth[a], l1 / depth[b], l2 / depth[c]
                        total = w0 + w1 + w2
                        z = 1 / total
                        if z >= zbuffer[py, px]:
                            continue
                        w0, w1, w2 = w0 / total, w1 / total, w2 / total
                        zbuffer[py, px] = z
                        world_hit[py, px] = world[a] * w0 + world[b] * w1 + world[c] * w2
                        face[py, px] = 2 * (y * (width - 1) + x) + kind
                        triangle_rows[py, px, 0] = a
                        triangle_rows[py, px, 1] = b
                        triangle_rows[py, px, 2] = c
                        barycentric[py, px, 0] = w0
                        barycentric[py, px, 1] = w1
                        barycentric[py, px, 2] = w2
    return world_hit, face, triangle_rows, barycentric, zbuffer


def project_surface_mesh(atlas, geometry_supported, camera_K, world_to_camera, image_hw, minimum_depth_m=None):
    """三角内透视校正世界点，保留逐像素面编号/三个原世界节点及重心权重。"""
    grid = atlas.grid_world_row
    height, width = grid.shape
    world = atlas.surface_world_xyz.reshape(-1, 3)
    matrix = np.asarray(world_to_camera, np.float64)
    intrinsic = np.asarray(camera_K, np.float64)
    camera = world @ matrix[:3, :3].T + matrix[:3, 3]
    projected_h = camera @ intrinsic.T
    projected = projected_h[:, :2] / np.maximum(projected_h[:, 2:3], 1e-9)
    support = np.asarray(geometry_supported, bool)[grid].ravel()
    minimum = np.full(len(world), .2) if minimum_depth_m is None else np.maximum(.2, np.asarray(minimum_depth_m)[grid].ravel())
    hit, face, corner, weights, depth = _raster(world, projected, camera[:, 2], support, minimum,
        height, width, int(image_hw[0]), int(image_hw[1]))
    valid = face >= 0
    actual_ids = np.full_like(corner, -1)
    actual_ids[valid] = grid.ravel()[corner[valid]]
    reprojection = np.full(image_hw, np.nan)
    if valid.any():
        xyz = hit[valid] @ matrix[:3, :3].T + matrix[:3, 3]
        uvh = xyz @ intrinsic.T
        uv = uvh[:, :2] / uvh[:, 2:3]
        y, x = np.nonzero(valid)
        reprojection[valid] = np.max(np.abs(uv - np.c_[x + .5, y + .5]), axis=1)
    return dict(world_xyz=hit, valid=valid, triangle_face_id=face, triangle_world_rows=actual_ids,
        triangle_weights=weights, camera_depth_m=depth, reprojection_error_px=reprojection)
