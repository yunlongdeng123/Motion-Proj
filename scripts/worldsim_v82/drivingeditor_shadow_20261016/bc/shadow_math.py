"""NumPy replay of the saved rasterizer equations. Approximate until FP32 checked."""
import math
import numpy as np


def rotation_matrix(q):
    q = np.asarray(q, np.float32)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = q.T
    return np.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                     2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                     2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)


def project_points(xyz, camera):
    p = np.asarray(xyz, np.float32)
    view = np.asarray(camera['world_view_transform'], np.float32)
    full = np.asarray(camera['full_proj_transform'], np.float32)
    cam = p @ view[:3, :3] + view[3, :3]
    clip = p @ full[:3, :] + full[3, :]
    wh = np.array([camera['width'], camera['height']], np.float32)
    uv = ((clip[:, :2] / (clip[:, 3:4] + np.float32(1e-7)) + 1) * wh - 1) * .5
    return uv, cam[:, 2]


def project_gaussians(data, camera, neg_fov):
    uv, depth = project_points(data['xyz'], camera)
    view = np.asarray(camera['world_view_transform'], np.float32)
    cam = data['xyz'] @ view[:3, :3] + view[3, :3]
    ids = np.flatnonzero(data['mask'] & (depth > .2))
    uv, cam, z = uv[ids], cam[ids], depth[ids]
    rot, scale = rotation_matrix(data['rotation'][ids]), data['scales'][ids]
    cov = (rot * scale[:, None, :] ** 2) @ rot.transpose(0, 2, 1)
    cov_cam = view[:3, :3].T @ cov @ view[:3, :3]
    tx = math.tan(-.5 if neg_fov else camera['FoVx'] * .5)
    ty = math.tan(-.5 if neg_fov else camera['FoVy'] * .5)
    fx, fy = np.float32(camera['width']/(2*tx)), np.float32(camera['height']/(2*ty))
    # Preserve the actual inherited negative-FOV min/max order, even for tx<0.
    x = np.minimum(np.float32(1.3*tx), np.maximum(np.float32(-1.3*tx), cam[:, 0]/z))*z
    y = np.minimum(np.float32(1.3*ty), np.maximum(np.float32(-1.3*ty), cam[:, 1]/z))*z
    j = np.zeros((len(ids), 2, 3), np.float32)
    j[:, 0, 0], j[:, 1, 1] = fx/z, fy/z
    j[:, 0, 2], j[:, 1, 2] = -fx*x/(z*z), -fy*y/(z*z)
    c2 = j @ cov_cam @ j.transpose(0, 2, 1)
    c2[:, 0, 0] += .3
    c2[:, 1, 1] += .3
    a, b, c = c2[:, 0, 0], c2[:, 0, 1], c2[:, 1, 1]
    det, mid = a*c-b*b, .5*(a+c)
    root = np.sqrt(np.maximum(.1, mid*mid-det))
    radius = np.ceil(3*np.sqrt(np.maximum(mid+root, mid-root)))
    grid = np.array([(camera['width']+15)//16, (camera['height']+15)//16])
    rmin = np.maximum(0, np.minimum(grid, np.trunc((uv-radius[:, None])/16))).astype(int)
    rmax = np.maximum(0, np.minimum(grid, np.trunc((uv+radius[:, None]+15)/16))).astype(int)
    good = (det != 0) & np.all(rmax > rmin, axis=1)
    idx = ids[good]
    return dict(ids=idx, xyz=data['xyz'][idx], scales=data['scales'][idx],
                rotation=data['rotation'][idx], opacity=data['opacity'][idx], colors=data['colors'][idx],
                uv=uv[good], depth=z[good], radius=radius[good], rmin=rmin[good], rmax=rmax[good],
                conic=np.stack([c/det, -b/det, a/det], 1)[good])


def blend(alpha):
    """Actual forward.cu thresholds, including skipping the terminating Gaussian."""
    a = np.minimum(.99, np.asarray(alpha, np.float32)).copy()
    a[a < 1/255] = 0
    trans = np.cumprod(1-a, axis=1, dtype=np.float32)
    previous = np.concatenate((np.ones((len(a), 1), np.float32), trans[:, :-1]), 1)
    weights = a * previous * (trans >= .0001)
    return weights, weights.sum(1)


def replay(projected, pixels, batch=32):
    pixels = np.asarray(pixels, np.float32)
    mass = np.zeros(len(projected['ids']), np.float64)
    rgb, alphas, probes = [], [], []
    for first in range(0, len(pixels), batch):
        p = pixels[first:first+batch]
        tiles = (p//16).astype(int)
        ids = np.flatnonzero(np.all(projected['rmax'] > tiles.min(0), 1) &
                             np.all(projected['rmin'] <= tiles.max(0), 1))
        ids = ids[np.argsort(projected['depth'][ids], kind='stable')]
        dx = projected['uv'][ids, 0][None]-p[:, :1]
        dy = projected['uv'][ids, 1][None]-p[:, 1:2]
        c = projected['conic'][ids]
        power = -.5*(c[:, 0][None]*dx*dx+c[:, 2][None]*dy*dy)-c[:, 1][None]*dx*dy
        hit = np.all(projected['rmin'][ids][None] <= tiles[:, None], -1) & np.all(projected['rmax'][ids][None] > tiles[:, None], -1)
        alpha = projected['opacity'][ids][None] * np.exp(np.minimum(power, 0))
        alpha[(power > 0) | ~hit] = 0
        weights, a = blend(alpha)
        rgb.append(weights @ projected['colors'][ids])
        alphas.append(a)
        mass[ids] += weights.sum(0, dtype=np.float64)
        if len(pixels) <= 16:
            for point, w in zip(p, weights):
                top = np.argsort(w)[::-1][:20]
                probes.append({'xy': point.astype(int).tolist(), 'contributors': [
                    {'static_id': int(projected['ids'][ids[k]]), 'weight': float(w[k]),
                     'depth_model': float(projected['depth'][ids[k]]), 'RGB': projected['colors'][ids[k]].tolist()}
                    for k in top if w[k] > 0]})
    return np.concatenate(rgb), np.concatenate(alphas), mass, probes


def ray_plane(pixel, camera, plane):
    """Use actual full projection, not a guessed focal length or pixel convention."""
    x, y = pixel
    ndc = np.array([(2*x+1)/camera['width']-1, (2*y+1)/camera['height']-1, .5, 1.])
    p = ndc @ np.linalg.inv(np.asarray(camera['full_proj_transform'], np.float64))
    p = p[:3]/p[3]
    origin = np.asarray(camera['camera_center'], np.float64)
    direction = p-origin
    plane = np.asarray(plane, np.float64)
    denom = float(direction @ plane[:3])
    if abs(denom) < 1e-10:
        return None
    distance = -float(origin @ plane[:3]+plane[3])/denom
    return origin+distance*direction if distance > 0 else None
