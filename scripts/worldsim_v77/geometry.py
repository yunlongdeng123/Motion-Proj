"""V7.7 的坐标、GT 框选择与解析编辑。所有点使用列向量变换的行存储约定。"""
import numpy as np


def transform(points, matrix):
    p = np.asarray(points, dtype=np.float64)
    t = np.asarray(matrix, dtype=np.float64)
    return p @ t[:3, :3].T + t[:3, 3]


def box_mask(points, pose, size):
    local = transform(points, np.linalg.inv(pose))
    return (np.abs(local) <= np.asarray(size) / 2 + 1e-8).all(axis=-1)


def resized_intrinsics(k, original_wh, hw):
    h, w = hw
    ow, oh = original_wh
    # PIL resize 的像素中心映射；本实验1600x900不触发极端长宽比裁剪。
    affine = np.array([[w / ow, 0, (w / ow - 1) / 2],
                       [0, h / oh, (h / oh - 1) / 2], [0, 0, 1]])
    return affine @ np.asarray(k)


def unproject(depth, k, c2w):
    h, w = depth.shape
    yy, xx = np.mgrid[:h, :w]
    pixels = np.stack([xx, yy, np.ones_like(xx)], -1)
    return transform((pixels @ np.linalg.inv(k).T) * depth[..., None], c2w)


def fit_sim3(source, target):
    """相机中心的最小二乘正尺度Sim(3)，不使用对象或LiDAR。"""
    x, y = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    xc, yc = x - x.mean(0), y - y.mean(0)
    u, singular, vt = np.linalg.svd(yc.T @ xc)
    correction = np.diag([1., 1., np.linalg.det(u @ vt)])
    rotation = u @ correction @ vt
    denom = np.sum(xc * xc)
    if denom < 1e-12:
        raise ValueError('预测相机中心退化，无法确定米制尺度')
    scale = float(np.sum(singular * np.diag(correction)) / denom)
    if scale <= 0:
        raise ValueError('非正Sim3尺度')
    translation = y.mean(0) - scale * rotation @ x.mean(0)
    residual = x @ rotation.T * scale + translation - y
    return scale, rotation, translation, np.linalg.norm(residual, axis=1)


def target_pose(source, delta_xyz=(0., 0., 0.), yaw_deg=0.):
    """yaw绕对象中心、世界z轴；平移为世界坐标米制位移。"""
    t = np.asarray(source, dtype=np.float64).copy()
    theta = np.deg2rad(yaw_deg)
    r = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    t[:3, :3] = r @ t[:3, :3]
    t[:3, 3] += np.asarray(delta_xyz)
    return t


def edit_actor(points, colors, selected, source, target, operation):
    """不做学习、补全或背景修改；保留每个输出点的源索引。"""
    ids = np.arange(len(points))
    selected = np.asarray(selected, dtype=bool)
    if operation == 'DELETE':
        keep = ~selected
        return points[keep].copy(), colors[keep].copy(), ids[keep]
    moved = transform(points[selected], target @ np.linalg.inv(source))
    if operation == 'MOVE':
        result = np.asarray(points, dtype=np.float64).copy()
        result[selected] = moved
        return result, colors.copy(), ids
    if operation == 'INSERT':
        return np.concatenate([points, moved]), np.concatenate([colors, colors[selected]]), np.concatenate([ids, ids[selected]])
    raise ValueError(operation)


def project_bbox(pose, size, c2w, k, hw):
    signs = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]])
    corners = transform(signs * np.asarray(size) / 2, pose)
    cam = transform(corners, np.linalg.inv(c2w))
    if cam[:, 2].min() <= .2:
        return None  # 本轮只选完整在近裁面前的框，不凭部分投影造巨框。
    projected = cam @ k.T
    uv = projected[:, :2] / projected[:, 2:3]
    h, w = hw
    lo = np.maximum(uv.min(0), [0, 0])
    hi = np.minimum(uv.max(0), [w - 1, h - 1])
    if np.any(hi <= lo):
        return None
    return [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])]
