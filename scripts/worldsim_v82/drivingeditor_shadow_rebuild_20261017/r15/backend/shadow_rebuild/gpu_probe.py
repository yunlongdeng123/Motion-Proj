"""GPU 启动后第一道检查：实际二进制 ABI、像素坐标、alpha*T 与反传。

本模块只有显式调用 probe 时才分配 CUDA；导入/CPU 预检不会触发。
"""
import torch
from .raster import Camera, reference_raster, cuda_raster


def probe(config):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU 未由用户启动")
    camera = Camera([[28, 0, 15.5], [0, 28, 15.5], [0, 0, 1]], torch.eye(4), 32, 32, "cuda")
    xyz = torch.tensor([[0., 0., 3.], [.03, .02, 4.]], device="cuda", requires_grad=True)
    cov = torch.diag(torch.tensor([.04, .06, .01], device="cuda")).expand(2, 3, 3).clone().requires_grad_(True)
    opacity = torch.tensor([[.7], [.8]], device="cuda", requires_grad=True)
    features = torch.tensor([[1., 1., 0., 3., -1., .5, .3, -.2, .7, .1, -.9, .8, .2],
                             [1., 0., 1., 4., .5, -1., .8, .4, -.3, .9, .6, -.4, .3]], device="cuda", requires_grad=True)
    reference = reference_raster(camera, xyz, cov, opacity, features)
    result = cuda_raster(camera, xyz, cov, opacity, features, config.max_features)
    error = (reference-result).abs().mean()
    if not bool(error < .015):
        raise RuntimeError(f"主光栅器 CPU/CUDA 语义或投影不一致: MAE={float(error)}")
    alpha = result[..., 0]
    if not bool((alpha <= 1.0001).all()) or not bool((result[..., 1:3].sum(-1)-alpha).abs().max() < 2e-5):
        raise RuntimeError("全场景可见身份通道不满足同一 alpha*T")
    loss = (result*torch.linspace(0, 1, 32, device="cuda")[None, :, None]).sum()
    loss.backward()
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) or not bool(p.grad.abs().sum() > 0) for p in (xyz, cov, opacity, features)):
        raise RuntimeError("实际 CUDA 几何/材质反向传播失败")
    torch.cuda.synchronize()
    return dict(status="GPU_ABI_GRADIENTS_PASSED", mean_abs_error=float(error.detach()),
                device=torch.cuda.get_device_name(), feature_capacity_requested=config.max_features)
