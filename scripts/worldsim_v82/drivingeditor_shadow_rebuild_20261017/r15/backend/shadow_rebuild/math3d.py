"""坐标约定：米、列向量变换；相机 +Z 朝前；四元数 wxyz。"""
import math
import torch
import torch.nn.functional as F


def normalize(x):
    return F.normalize(x, dim=-1, eps=1e-10)


def quaternion_matrix(q):
    w, x, y, z = normalize(q).unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def srgb_to_linear(x):
    return torch.where(x <= .04045, x/12.92, ((x.clamp_min(0)+.055)/1.055).pow(2.4))


def linear_to_srgb(x):
    return torch.where(x <= .0031308, 12.92*x, 1.055*x.clamp_min(1e-10).pow(1/2.4)-.055)


def logit(x):
    return torch.logit(x.clamp(1e-5, 1-1e-5))


def covariance(q, scales):
    rotation = quaternion_matrix(q)
    return (rotation * scales.square()[..., None, :]) @ rotation.transpose(-1, -2)


def rotation_z(angle, *, device, dtype):
    a = torch.as_tensor(angle, device=device, dtype=dtype)
    c, s, z, o = a.cos(), a.sin(), a*0, a*0+1
    return torch.stack((c, -s, z, s, c, z, z, z, o)).reshape(3, 3)


def normal_basis(normal):
    """可微正交基，用于有限角太阳盘采样。"""
    normal = normalize(normal)
    seed = torch.eye(3, device=normal.device, dtype=normal.dtype)[normal.detach().abs().argmin()]
    u = normalize(torch.linalg.cross(normal, seed))
    return u, torch.linalg.cross(normal, u)


def disk_directions(direction, radius, count):
    if count < 1:
        raise ValueError("太阳采样数必须为正")
    u, v = normal_basis(direction)
    i = torch.arange(count, device=direction.device, dtype=direction.dtype)
    r = torch.sqrt((i+.5)/count) if count > 1 else i
    theta = i * (math.pi*(3-math.sqrt(5)))
    return normalize(direction + radius*r[:, None]*(theta.cos()[:, None]*u+theta.sin()[:, None]*v))
