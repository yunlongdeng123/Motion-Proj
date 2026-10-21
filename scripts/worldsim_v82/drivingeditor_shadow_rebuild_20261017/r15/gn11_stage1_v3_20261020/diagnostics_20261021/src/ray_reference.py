"""CPU reference for the frozen DrivingEditor compact rasterizer.

Not a replacement for a CUDA trace. All thresholds, negative-FOV covariance,
tile bounds and early termination follow the inspected native source. Numeric
agreement with saved native alpha/depth is reported per query, never assumed.
"""
import numpy as np


def rotation(q):
    q = np.asarray(q, np.float32)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q.T
    return np.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                     2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                     2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), 1).reshape(-1, 3, 3)


def covariance(scales, quat):
    r = rotation(quat)
    return (r * np.asarray(scales)[:, None, :]) @ np.transpose(r * np.asarray(scales)[:, None, :], (0, 2, 1))


def sh_color(sh, directions, degree):
    d = directions / np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    x, y, z = d.T
    out = np.float32(.28209479177387814)*sh[:, 0]
    def add(v, i):
        nonlocal out
        out = out + v[:, None]*sh[:, i]
    if degree > 0:
        add(-.4886025119029199*y, 1); add(.4886025119029199*z, 2); add(-.4886025119029199*x, 3)
    if degree > 1:
        xx, yy, zz = x*x, y*y, z*z
        add(1.0925484305920792*x*y, 4); add(-1.0925484305920792*y*z, 5)
        add(.31539156525252005*(2*zz-xx-yy), 6); add(-1.0925484305920792*x*z, 7)
        add(.5462742152960396*(xx-yy), 8)
    if degree > 2:
        add(-.5900435899266435*y*(3*xx-yy), 9); add(2.890611442640554*x*y*z, 10)
        add(-.4570457994644658*y*(4*zz-xx-yy), 11)
        add(.3731763325901154*z*(2*zz-3*xx-3*yy), 12)
        add(-.4570457994644658*x*(4*zz-xx-yy), 13)
        add(1.445305721320277*z*(xx-yy), 14); add(-.5900435899266435*x*(xx-3*yy), 15)
    return np.maximum(out+np.float32(.5), 0).astype(np.float32)


def project(xyz, cov, opacity, mask, world_view, intrinsic, neg_fov=True, width=480, height=320):
    """Conservative full primitive preprocessing, not center-nearest attribution."""
    w = np.asarray(world_view, np.float32)
    camera = xyz @ w[:3, :3].T + w[:3, 3]
    z = camera[:, 2]
    selected = np.flatnonzero(mask & (z > .2) & np.isfinite(camera).all(1))
    c, v = camera[selected], cov[selected]
    fx, fy, cx, cy = intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2]
    # Native center-shift projection and ndc2Pix yield a -0.5 pixel offset.
    uv = np.stack((fx*c[:, 0]/c[:, 2]+cx-.5, fy*c[:, 1]/c[:, 2]+cy-.5), 1).astype(np.float32)
    tx, ty = (np.tan(-.5), np.tan(-.5)) if neg_fov else (width/(2*fx), height/(2*fy))
    focal_x, focal_y = width/(2*tx), height/(2*ty)
    # Do NOT np.clip with reversed limits: preserve native min(lim,max(-lim,x)).
    xz = np.minimum(1.3*tx, np.maximum(-1.3*tx, c[:, 0]/c[:, 2]))
    yz = np.minimum(1.3*ty, np.maximum(-1.3*ty, c[:, 1]/c[:, 2]))
    j = np.zeros((len(c), 2, 3), np.float32)
    j[:, 0, 0] = focal_x/c[:, 2]; j[:, 0, 2] = -focal_x*xz/c[:, 2]
    j[:, 1, 1] = focal_y/c[:, 2]; j[:, 1, 2] = -focal_y*yz/c[:, 2]
    t = j @ w[:3, :3]
    cv = t @ v @ t.transpose(0, 2, 1)
    a, b, d = cv[:, 0, 0]+.3, cv[:, 0, 1], cv[:, 1, 1]+.3
    det = a*d-b*b
    conic = np.stack((d/det, -b/det, a/det), 1)
    mid = .5*(a+d)
    radius = np.ceil(3*np.sqrt(np.maximum(mid+np.sqrt(np.maximum(.1, mid*mid-det)), 0)))
    grid = np.asarray([(width+15)//16, (height+15)//16])
    lo = np.minimum(grid, np.maximum(0, np.trunc((uv-radius[:, None])/16))).astype(np.int32)
    hi = np.minimum(grid, np.maximum(0, np.trunc((uv+radius[:, None]+15)/16))).astype(np.int32)
    keep = (det > 0) & (hi > lo).all(1) & np.isfinite(conic).all(1)
    return dict(row=selected[keep], uv=uv[keep], conic=conic[keep],
                opacity=opacity[selected[keep]], depth=z[selected[keep]],
                lo=lo[keep], hi=hi[keep], radius=radius[keep],
                cov2=np.stack((a, b, d), 1)[keep])


def candidates(p, x, y):
    tile = np.asarray([x//16, y//16])
    idx = np.flatnonzero(((p['lo'] <= tile) & (p['hi'] > tile)).all(1))
    delta = p['uv'][idx]-np.asarray([x, y], np.float32)
    a, b, d = p['conic'][idx].T
    dx, dy = delta.T
    power = -.5*(a*dx*dx+d*dy*dy)-b*dx*dy
    alpha = np.minimum(.99, p['opacity'][idx]*np.exp(np.minimum(power, 0)))
    valid = (power <= 0) & (alpha >= 1/255)
    idx, alpha = idx[valid], alpha[valid]
    order = np.argsort(p['depth'][idx], kind='stable')
    return idx[order], alpha[order].astype(np.float32)


def composite(alpha, colors, depth, removed=None):
    """Recompute T for every surviving primitive, including formerly occluded ones."""
    t, rgb, dz = np.float32(1), np.zeros(3, np.float32), np.float32(0)
    accepted, weights, transmittance = [], [], []
    removed = set(removed or ())
    for i, a in enumerate(alpha):
        if i in removed:
            continue
        test = np.float32(t*(np.float32(1)-a))
        if test < .0001:
            break  # native excludes the primitive that crosses the termination threshold
        weight = np.float32(a*t)
        accepted.append(i); weights.append(float(weight)); transmittance.append(float(t))
        rgb += colors[i]*weight; dz += depth[i]*weight; t = test
    return dict(rgb=rgb, depth_sum=float(dz), alpha=float(np.sum(np.asarray(weights, np.float32), dtype=np.float32)),
                T=float(t), indices=accepted, weights=weights, transmittance=transmittance)
