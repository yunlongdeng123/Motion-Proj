"""评估指标：视觉质量 + 驾驶动态一致性。

动态一致性指标（静态漂移、物体轨迹平滑度）是本方案基准测试的核心
（第 10 节）。FVD 需要 I3D 特征提取器和外部权重；这里我们暴露一个清晰的
接口钩子，以及一个帧级的 FID 代理。
"""
from __future__ import annotations

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


def fvd(real_videos, fake_videos):  # pragma: no cover - needs I3D weights
    """Frechet Video Distance 的接口钩子。需要 I3D 特征提取器。

    V1 中未实现（需要外部 I3D 权重）。可在此接入一个预训练的 I3D，
    并计算片段特征的 Frechet 距离。在 V1 阶段，请改用 LPIPS/SSIM 以及
    动态一致性指标来做合理性检查。
    """
    raise NotImplementedError("FVD requires an I3D feature extractor; not bundled in V1")
