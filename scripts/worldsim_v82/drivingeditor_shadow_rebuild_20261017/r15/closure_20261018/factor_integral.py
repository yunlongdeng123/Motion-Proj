"""R15 阴影积分的局部因子参考；尚未接入生产训练。

不构造/求逆世界协方差，不对负二次型进行补救式 clamp。
rotation 可以包含保存姿态的微小非正交误差：solve 的对象只是旋转因子。
主参考使用 FP64；FP32 是待验收候选，不能假设已修复。
输入方向必须单位化；尺度为正；opacity 沿用现有最薄轴光学厚度约定。
"""
import math
import torch


def factor_integral(origins, directions, means, rotation, scales, opacity,
                    lower=0.0, upper=None, computation_dtype=torch.float64):
    """返回 [R,G] optical thickness；用于小批参考/梯度核查。"""
    o, d, m, r, s, alpha = [x.to(computation_dtype) for x in
                            (origins, directions, means, rotation, scales, opacity)]
    nr = len(o)
    delta = o[:, None] - m[None]
    dirs = d[:, None].expand_as(delta)
    rhs = torch.cat((delta.permute(1, 2, 0), dirs.permute(1, 2, 0)), dim=-1)
    local = torch.linalg.solve(r, rhs).permute(2, 0, 1)
    y, v = local[:nr] / s[None], local[nr:] / s[None]
    speed = torch.linalg.vector_norm(v, dim=-1)
    unit = v / speed[..., None]
    center = -(y * unit).sum(-1)
    # 正交距离由叉积得到，避免 c-b^2/a 两大数相减。
    cross = torch.linalg.cross(y, unit, dim=-1)
    transverse = cross.square().sum(-1)
    zlo = (speed * lower - center) / math.sqrt(2)
    if upper is None:
        tail = 0.5 * torch.erfc(zlo)
    else:
        zhi = (speed * upper - center) / math.sqrt(2)
        tail = 0.5 * torch.where(
            zlo >= 0, torch.erfc(zlo) - torch.erfc(zhi),
            torch.where(zhi <= 0, torch.erfc(-zhi) - torch.erfc(-zlo),
                        torch.erf(zhi) - torch.erf(zlo)))
    amplitude = -torch.log1p(-alpha.flatten()) / (
        math.sqrt(2 * math.pi) * s.amin(-1))
    return amplitude[None] * torch.exp(-0.5 * transverse) * (
        math.sqrt(2 * math.pi) / speed) * tail


def factor_transmittance(origins, directions, means, rotation, scales, opacity,
                         gaussian_chunk=512, computation_dtype=torch.float64):
    """无粗筛、保持全部太阳方向的分块参考；不含训练专用 tau 上限。"""
    tau = torch.zeros(len(origins), device=origins.device, dtype=computation_dtype)
    for start in range(0, len(means), gaussian_chunk):
        end = start + gaussian_chunk
        tau = tau + factor_integral(origins, directions, means[start:end],
            rotation[start:end], scales[start:end], opacity[start:end],
            computation_dtype=computation_dtype).sum(-1)
    return torch.exp(-tau)
