"""评估指标：视觉质量 + 驾驶动态一致性。

动态一致性指标（静态漂移、物体轨迹平滑度）是本方案基准测试的核心
（第 10 节）。FVD 需要 I3D 特征提取器和外部权重；这里我们暴露一个清晰的
接口钩子，以及一个帧级的 FID 代理。
"""
from __future__ import annotations

import numpy as np
import torch

from ..auditor.state import MotionState, Track
from ..projector.energies import e_obj, e_static


# ---------------------------------------------------------------- 动态指标
def static_drift(state: MotionState) -> float:
    """在可靠的静态像素上，经过自车运动补偿后的平均静态漂移（越低越好）。"""
    return float(e_static(state))


def track_acceleration(tracks: list[Track]) -> float:
    """物体轨迹的加速度 / 拐折分数（越低越好）。"""
    return float(e_obj(tracks))


# ----------------------------------------------------------------- 视觉质量
def lpips_distance(a: torch.Tensor, b: torch.Tensor, net: str = "alex", _cache={}) -> float:
    """逐帧 LPIPS 的均值。``a,b``：``[T,3,H,W]``，取值范围 [-1,1]。"""
    import lpips

    if net not in _cache:
        _cache[net] = lpips.LPIPS(net=net).eval()
    model = _cache[net].to(a.device)
    with torch.no_grad():
        d = model(a, b)
    return float(d.mean())


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """``a,b`` 取值范围 [-1,1] -> PSNR（dB）。"""
    a01 = (a + 1) / 2
    b01 = (b + 1) / 2
    mse = (a01 - b01).pow(2).mean().clamp_min(1e-10)
    return float(10 * torch.log10(1.0 / mse))


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    """使用 skimage 计算逐帧 SSIM 的均值（``a,b``：``[T,3,H,W]``，取值范围 [-1,1]）。"""
    from skimage.metrics import structural_similarity as sk_ssim

    a = ((a + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    b = ((b + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    vals = [sk_ssim(a[t], b[t], channel_axis=-1, data_range=1.0) for t in range(a.shape[0])]
    return float(sum(vals) / len(vals))


def frechet_feature_distance(real_features, fake_features) -> float:
    """对已经抽取的特征计算标准 Fréchet 距离。"""
    from scipy.linalg import sqrtm

    real = np.asarray(real_features, dtype=np.float64)
    fake = np.asarray(fake_features, dtype=np.float64)
    if real.ndim != 2 or fake.ndim != 2 or real.shape[1] != fake.shape[1]:
        raise ValueError("Fréchet 输入必须为特征维一致的 [N,D]")
    if len(real) < 2 or len(fake) < 2:
        raise ValueError("Fréchet 距离每侧至少需要两个样本")
    mu_real, mu_fake = real.mean(0), fake.mean(0)
    cov_real, cov_fake = np.cov(real, rowvar=False), np.cov(fake, rowvar=False)
    product = sqrtm(cov_real @ cov_fake)
    if np.iscomplexobj(product):
        if not np.allclose(product.imag, 0, atol=1e-6):
            raise ValueError("Fréchet covariance sqrt 产生显著复数分量")
        product = product.real
    value = ((mu_real - mu_fake) ** 2).sum() + np.trace(cov_real + cov_fake - 2 * product)
    return float(max(value, 0.0))


def fid_future(real_frame_features, fake_frame_features) -> float:
    return frechet_feature_distance(real_frame_features, fake_frame_features)


def fvd8(real_i3d_features, fake_i3d_features) -> float:
    """8 帧 clip 的 I3D 特征 FVD；特征提取器及权重由评估配置显式提供。"""
    return frechet_feature_distance(real_i3d_features, fake_i3d_features)


def fvd(real_features, fake_features):
    """兼容旧接口；调用方必须传入已抽取的 I3D 特征。"""
    return fvd8(real_features, fake_features)
