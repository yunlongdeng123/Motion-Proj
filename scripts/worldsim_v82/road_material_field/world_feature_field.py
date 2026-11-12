"""固定米制表面的相机投影与稠密特征融合。RGB/特征不参与几何求解。"""
from __future__ import annotations
import numpy as np


def transform_xyz(xyz, matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    return np.asarray(xyz) @ matrix[:3, :3].T + matrix[:3, 3]


def camera_from_observation(observation):
    """Waymo X前/Y左/Z上转换为OpenCV X右/Y下/Z前，保留原图Brown畸变。"""
    pose = np.asarray(observation['camera_pose_world']).reshape(4, 4)
    extrinsic = np.asarray(observation['camera_extrinsic']).reshape(4, 4)
    axis = np.array([[0., -1., 0., 0.], [0., 0., -1., 0.],
                     [1., 0., 0., 0.], [0., 0., 0., 1.]])
    intrinsic = observation['camera_intrinsic_Brown']
    fx, fy, cx, cy = intrinsic[:4]
    return dict(camera_K_native=[[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                world_to_camera=(axis @ np.linalg.inv(pose @ extrinsic)).tolist(),
                distortion_Brown=list(intrinsic[4:]),
                image_hw=[observation['camera_size'][1], observation['camera_size'][0]],
                camera_model='opencv_brown')


def project_world_to_camera(world_xyz, camera_K_native, world_to_camera,
                            image_hw=None, distortion_Brown=None):
    xyz = transform_xyz(world_xyz, world_to_camera)
    depth = xyz[:, 2]
    xy = xyz[:, :2] / np.maximum(depth[:, None], 1e-9)
    if distortion_Brown is not None:
        k1, k2, p1, p2, k3 = distortion_Brown
        x, y = xy.T; r2 = x*x + y*y
        radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
        xy = np.column_stack((x*radial + 2*p1*x*y + p2*(r2+2*x*x),
                              y*radial + p1*(r2+2*y*y) + 2*p2*x*y))
    K = np.asarray(camera_K_native)
    uv = xy @ K[:2, :2].T + K[:2, 2]
    inside = np.isfinite(uv).all(1) & (depth > .1)
    if image_hw is not None:
        h, w = image_hw
        inside &= (uv[:, 0] >= .5) & (uv[:, 1] >= .5)
        inside &= (uv[:, 0] < w-1.5) & (uv[:, 1] < h-1.5)
    return uv, depth, inside


def undistort_pixels(uv_native, camera_K_native, distortion_Brown):
    K = np.asarray(camera_K_native)
    target = (np.asarray(uv_native)-K[:2, 2]) @ np.linalg.inv(K[:2, :2]).T
    x, y = target[:, 0].copy(), target[:, 1].copy()
    k1, k2, p1, p2, k3 = distortion_Brown
    for _ in range(8):
        r2 = x*x+y*y; radial = 1+k1*r2+k2*r2*r2+k3*r2*r2*r2
        dx = 2*p1*x*y+p2*(r2+2*x*x); dy = p1*(r2+2*y*y)+2*p2*x*y
        x = (target[:, 0]-dx)/np.maximum(radial, 1e-5)
        y = (target[:, 1]-dy)/np.maximum(radial, 1e-5)
    return np.column_stack((x, y, np.ones(len(x))))


def camera_ground_rays(camera, wh, local_from_world, plane):
    """相机像素中心射线和固定局部平面的交点，输出H×W局部坐标。"""
    w, h = wh; nh, nw = camera['image_hw']
    yy, xx = np.mgrid[:h, :w]
    uv = np.c_[((xx.ravel()+.5)*nw/w), ((yy.ravel()+.5)*nh/h)]
    cv_rays = undistort_pixels(uv, camera['camera_K_native'], camera['distortion_Brown'])
    local_from_camera = np.asarray(local_from_world) @ np.linalg.inv(np.asarray(camera['world_to_camera']))
    rays = cv_rays @ local_from_camera[:3, :3].T
    origin = local_from_camera[:3, 3]
    den = rays[:, 2] - rays[:, :2] @ np.asarray(plane[:2])
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = (plane[2]+np.asarray(plane[:2])@origin[:2]-origin[2])/den
    xyz = origin+depth[:, None]*rays
    return xyz.reshape(h, w, 3), depth.reshape(h, w)


def bilinear_sample(array, uv_native, native_hw):
    """像素坐标定义为边界原点，像素中心为0.5；支持RGB或任意维特征。"""
    array = np.asarray(array)
    h, w = array.shape[:2]; nh, nw = native_hw
    xy = np.asarray(uv_native)*[w/nw, h/nh]-.5
    xy = np.clip(np.nan_to_num(xy), [0, 0], [max(w-1.00001, 0), max(h-1.00001, 0)])
    low = np.floor(xy).astype(np.int64); high = np.minimum(low+1, [w-1, h-1])
    frac = xy-low; x, y = low.T; x1, y1 = high.T; fx, fy = frac.T
    return ((1-fx)[:, None]*(1-fy)[:, None]*array[y, x]
            +fx[:, None]*(1-fy)[:, None]*array[y, x1]
            +(1-fx)[:, None]*fy[:, None]*array[y1, x]
            +fx[:, None]*fy[:, None]*array[y1, x1])


def fuse_dense_features(world_xyz, features, cameras, per_view_observations,
                        exclude_world_mask=None, normalize=True):
    """仅融合已通过可见性筛选的原图特征；禁区先于encoder屏蔽，函数再剔除世界cell。

    features为[S,Hf,Wf,C]或等价列表。每帧observations提供point_ids、uv_native、weight。
    返回field[N,C]、weight[N]、observation_count[N]。不隐式读取RGB、target或测试真值。
    """
    count = len(world_xyz); channels = np.asarray(features[0]).shape[-1]
    total = np.zeros((count, channels), np.float32)
    weight_sum = np.zeros(count, np.float32); observation_count = np.zeros(count, np.int32)
    excluded = np.zeros(count, bool) if exclude_world_mask is None else np.asarray(exclude_world_mask, bool)
    for feature, camera, observation in zip(features, cameras, per_view_observations):
        ids = np.asarray(observation['point_ids'], np.int64)
        weights = np.asarray(observation['weight'], np.float32)
        usable = (weights > 0) & ~excluded[ids]
        ids, weights = ids[usable], weights[usable]
        values = bilinear_sample(np.asarray(feature, np.float32),
                                 np.asarray(observation['uv_native'])[usable], camera['image_hw'])
        # 同一帧同一世界cell只允许一条记录，避免由重复采样人为增加权重。
        if len(np.unique(ids)) != len(ids):
            raise ValueError('Duplicate same-world observations in one camera')
        total[ids] += values*weights[:, None]; weight_sum[ids] += weights
        observation_count[ids] += 1
    total /= np.maximum(weight_sum[:, None], 1e-12)
    if normalize:
        total /= np.maximum(np.linalg.norm(total, axis=1, keepdims=True), 1e-12)
    return dict(field=total, weight=weight_sum, observation_count=observation_count,
                known=weight_sum > 0, unknown=weight_sum <= 0)
